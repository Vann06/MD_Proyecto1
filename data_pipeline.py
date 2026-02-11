"""Utilities to load, clean and integrate Divorcios & Violencia Intrafamiliar datasets."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Datos"
DIVORCIOS_DIR = DATA_DIR / "Divorcios"
VIF_DIR = DATA_DIR / "Violacion_Intrafamiliar"
PROCESSED_DIR = DATA_DIR / "Procesados"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = PROCESSED_DIR / "panel_departamento_anio.csv"


def _normalize_text(text: str) -> str:
    """Return ASCII-ready text by stripping tildes and odd characters."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("�", "n")
    normalized = unicodedata.normalize("NFKD", cleaned)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _slugify(text: str) -> str:
    """Create a snake-case token that is safe for joins."""
    normalized = _normalize_text(text).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug


CANONICAL_DEPARTMENTS: List[str] = [
    "Alta Verapaz",
    "Baja Verapaz",
    "Chimaltenango",
    "Chiquimula",
    "El Progreso",
    "Escuintla",
    "Guatemala",
    "Huehuetenango",
    "Izabal",
    "Jalapa",
    "Jutiapa",
    "Petén",
    "Quetzaltenango",
    "Quiché",
    "Retalhuleu",
    "Sacatepéquez",
    "San Marcos",
    "Santa Rosa",
    "Sololá",
    "Suchitepéquez",
    "Totonicapán",
    "Zacapa",
]

DEPARTMENT_ALIAS: Dict[str, str] = {}
for dept in CANONICAL_DEPARTMENTS:
    key = _slugify(dept)
    DEPARTMENT_ALIAS[key] = dept

# Extra aliases for common variants.
DEPARTMENT_ALIAS.update(
    {
        "altenverapaz": "Alta Verapaz",
        "altaverapaz": "Alta Verapaz",
        "bajaverapaz": "Baja Verapaz",
        "sacatepequez": "Sacatepéquez",
        "peten": "Petén",
        "quetzaltenango": "Quetzaltenango",
        "quiche": "Quiché",
        "suchitepequez": "Suchitepéquez",
        "toto": "Totonicapán",
        "zacaba": "Zacapa",
        "sanmarcos": "San Marcos",
    }
)

DEPARTMENT_CODE_TO_NAME: Dict[int, str] = {
    1: "Guatemala",
    2: "El Progreso",
    3: "Sacatepéquez",
    4: "Chimaltenango",
    5: "Escuintla",
    6: "Santa Rosa",
    7: "Sololá",
    8: "Totonicapán",
    9: "Quetzaltenango",
    10: "Suchitepéquez",
    11: "Retalhuleu",
    12: "San Marcos",
    13: "Huehuetenango",
    14: "Quiché",
    15: "Baja Verapaz",
    16: "Alta Verapaz",
    17: "Petén",
    18: "Izabal",
    19: "Zacapa",
    20: "Chiquimula",
    21: "Jalapa",
    22: "Jutiapa",
}

YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def infer_year_from_filename(name: str) -> Optional[int]:
    """Attempt to extract a plausible year code from a file name."""
    matches = YEAR_PATTERN.findall(name)
    for match in matches:
        year = int(match)
        if 1995 <= year <= 2035:
            return year
    digits = re.findall(r"\d{4}", name)
    for candidate in digits:
        year = int(candidate)
        if 1995 <= year <= 2035:
            return year
    return None


def _standardize_department(value: object) -> Optional[str]:
    """Map free-text department names to the canonical catalog."""
    if not isinstance(value, str):
        return None
    key = _slugify(value).replace("departamento_", "").replace("_departamento", "")
    if key in DEPARTMENT_ALIAS:
        return DEPARTMENT_ALIAS[key]
    for alias_key, canonical in DEPARTMENT_ALIAS.items():
        if alias_key in key:
            return canonical
    return None


def _decode_vif_department(code_value: object) -> Optional[str]:
    """Derive the department name from the numeric MPCIO code."""
    if pd.isna(code_value):
        return None
    try:
        code = int(float(code_value)) // 100
    except (ValueError, TypeError):
        return None
    return DEPARTMENT_CODE_TO_NAME.get(code)


def _unstack_counts(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    """Pivot categorical counts into wide format columns."""
    if value_col not in df.columns:
        return pd.DataFrame()
    subset = df.dropna(subset=[value_col]).copy()
    if subset.empty:
        return pd.DataFrame()
    subset[value_col] = subset[value_col].apply(lambda x: _slugify(str(x)))
    table = (
        subset.groupby(["departamento", "anio", value_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename_map = {col: f"{prefix}{col}" for col in table.columns if col not in {"departamento", "anio"}}
    table = table.rename(columns=rename_map)
    return table


def load_divorcios_raw() -> pd.DataFrame:
    """Read and concatenate every .sav divorce record file."""
    frames: List[pd.DataFrame] = []
    for path in sorted(DIVORCIOS_DIR.glob("*.sav")):
        df = pd.read_spss(path)
        df.columns = [_slugify(col) for col in df.columns]
        df["archivo"] = path.name
        df["anio_fuente"] = infer_year_from_filename(path.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No se encontraron archivos .sav en 'Datos/Divorcios'.")
    return pd.concat(frames, ignore_index=True)


def load_vif_raw() -> Tuple[pd.DataFrame, Dict[str, Dict[float, str]]]:
    """Load VIF .sav files plus the dictionary of value labels."""
    frames: List[pd.DataFrame] = []
    label_cache: Dict[str, Dict[float, str]] = {}
    for path in sorted(VIF_DIR.glob("*.sav")):
        frame, meta = pyreadstat.read_sav(path, apply_value_formats=False)
        frame.columns = [_slugify(col) for col in frame.columns]
        frame["archivo"] = path.name
        frame["anio_fuente"] = infer_year_from_filename(path.name)
        frames.append(frame)
        for var, mapping in meta.variable_value_labels.items():
            if var not in label_cache and mapping:
                converted: Dict[float, str] = {}
                for key, value in mapping.items():
                    try:
                        converted[float(key)] = value
                    except (ValueError, TypeError):
                        continue
                if converted:
                    label_cache[var] = converted
    if not frames:
        raise FileNotFoundError("No se encontraron archivos .sav en 'Datos/Violacion_Intrafamiliar'.")
    return pd.concat(frames, ignore_index=True), label_cache


def _coalesce_year(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Stack several year columns and keep the first valid 4-digit value."""
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for column in candidates:
        if column in df.columns:
            candidate = pd.to_numeric(df[column], errors="coerce").astype("Float64")
            result = result.fillna(candidate)
    result = result.where((result >= 1995) & (result <= 2035))
    return result.astype("Int64")


def _aggregate_divorcios(df: pd.DataFrame) -> pd.DataFrame:
    """Return yearly divorce counts per department."""
    df = df.copy()
    df["anio"] = _coalesce_year(df, ["anoocu", "anoreg", "anio_fuente"])
    dep_series = pd.Series(pd.NA, index=df.index, dtype="string")
    for column in ("depocu", "depreg", "depoc", "departamento"):
        if column in df.columns:
            dep_series = dep_series.fillna(df[column].apply(_standardize_department))
    df["departamento"] = dep_series
    filtered = df.dropna(subset=["departamento", "anio"])
    summary = (
        filtered.groupby(["departamento", "anio"], as_index=False)
        .size()
        .rename(columns={"size": "divorcios_total"})
    )
    return summary


def _map_values(series: pd.Series, mapping: Dict[float, str]) -> pd.Series:
    """Translate numeric codes to their labels, keeping NaNs intact."""
    if not mapping or series is None:
        return series

    def _lookup(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        for converter in (float, int):
            try:
                key = converter(value)
            except (ValueError, TypeError):
                continue
            if key in mapping:
                return mapping[key]
        return mapping.get(value)

    return series.apply(_lookup)


def _aggregate_vif(df: pd.DataFrame, labels: Dict[str, Dict[float, str]]) -> pd.DataFrame:
    """Return yearly VIF totals plus gender/age/aggression splits."""
    df = df.copy()
    df["departamento"] = df.get("hec_deptomcpio").apply(_decode_vif_department)
    df["anio"] = _coalesce_year(df, ["hec_ano", "ano_emision", "anio_fuente"])

    sex_labels = labels.get("VIC_SEXO", {1.0: "Hombres", 2.0: "Mujeres"})
    aggression_labels = labels.get("HEC_TIPAGRE", {})

    df["victima_sexo"] = _map_values(df.get("vic_sexo"), sex_labels)
    df["tipo_agresion"] = _map_values(df.get("hec_tipagre"), aggression_labels)

    def _format_age(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        try:
            return f"{int(float(value)):02d}"
        except (ValueError, TypeError):
            return None

    df["grupo_edad"] = df.get("vic_edad").apply(_format_age)
    filtered = df.dropna(subset=["departamento", "anio"])

    base = (
        filtered.groupby(["departamento", "anio"], as_index=False)
        .size()
        .rename(columns={"size": "vif_total"})
    )
    sexo = _unstack_counts(filtered, "victima_sexo", "vif_sexo_")
    tipo = _unstack_counts(filtered, "tipo_agresion", "vif_tipo_")
    edad = _unstack_counts(filtered, "grupo_edad", "vif_edad_")

    result = base
    for extra in (sexo, tipo, edad):
        if not extra.empty:
            result = result.merge(extra, on=["departamento", "anio"], how="left")
    count_cols = [col for col in result.columns if col.startswith("vif_")]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    return result


def build_panel(force_rebuild: bool = False, only_common_years: bool = True) -> pd.DataFrame:
    """Generate the merged panel, optionally limiting to overlapping years."""
    if PANEL_PATH.exists() and not force_rebuild:
        panel = pd.read_csv(PANEL_PATH)
        panel["anio"] = panel["anio"].astype(int)
        return panel
    divorcios_raw = load_divorcios_raw()
    vif_raw, labels = load_vif_raw()
    divorcios = _aggregate_divorcios(divorcios_raw)
    vif = _aggregate_vif(vif_raw, labels)
    if only_common_years:
        common_years: Set[int] = set(divorcios["anio"].dropna().astype(int)) & set(
            vif["anio"].dropna().astype(int)
        )
        if not common_years:
            raise ValueError("No hay años en común entre divorcios y VIF.")
        divorcios = divorcios[divorcios["anio"].isin(common_years)]
        vif = vif[vif["anio"].isin(common_years)]
    # Outer merge lets us keep departments that only appear in univariate splits.
    panel = divorcios.merge(vif, on=["departamento", "anio"], how="outer")
    panel = panel.sort_values(["anio", "departamento"]).reset_index(drop=True)
    panel.to_csv(PANEL_PATH, index=False)
    return panel


def describe_columns(df: pd.DataFrame, sample_rows: int = 3) -> pd.DataFrame:
    """Show type, null share and sample value for every column."""
    records = []
    for column in df.columns:
        series = df[column]
        sample = series.dropna()
        record = {
            "columna": column,
            "dtype": str(series.dtype),
            "nulos": int(series.isna().sum()),
            "%_nulos": round(series.isna().mean() * 100, 2),
            "ejemplo": sample.iloc[0] if not sample.empty else None,
        }
        records.append(record)
    return pd.DataFrame(records)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted table with null counts per column."""
    summary = (
        df.isna()
        .sum()
        .rename("nulos")
        .to_frame()
        .assign(pct=lambda s: (s["nulos"] / len(df)) * 100)
        .query("nulos > 0")
        .sort_values("nulos", ascending=False)
    )
    return summary.reset_index().rename(columns={"index": "columna", "pct": "%_nulos"})


def split_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Separate numeric vs categorical column names for downstream analysis."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = sorted(set(df.columns) - set(numeric_cols))
    return {"numericas": numeric_cols, "categoricas": categorical_cols}


def vif_quartile_table(
    panel: pd.DataFrame,
    source_col: str = "divorcios_total",
    target_col: str = "vif_total",
    quantiles: int = 4,
) -> pd.DataFrame:
    """Summarise VIF totals by quantiles of divorce totals, handling ties safely."""
    missing_cols = {source_col, target_col} - set(panel.columns)
    if missing_cols:
        raise KeyError(f"Faltan columnas requeridas en el panel: {missing_cols}.")
    working = panel.dropna(subset=[source_col, target_col]).copy()
    if working.empty:
        raise ValueError("No hay datos suficientes para calcular cuartiles.")
    try:
        quantile_codes = pd.qcut(
            working[source_col], quantiles, labels=False, duplicates="drop"
        )
    except ValueError as exc:
        raise ValueError("No es posible formar cuantiles con los datos actuales.") from exc
    if quantile_codes.isna().all():
        raise ValueError("Los valores de divorcios son constantes; no hay cuartiles.")
    n_bins = int(quantile_codes.max()) + 1
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    working["div_q"] = quantile_codes.map(lambda idx: labels[int(idx)] if pd.notna(idx) else pd.NA)
    summary = (
        working.groupby("div_q")[target_col]
        .agg(media="mean", mediana="median", maximo="max")
        .round(1)
    )
    return summary
