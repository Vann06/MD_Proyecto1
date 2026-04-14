import nbformat as nbf

nb = nbf.v4.new_notebook()

celdas = [
    # 1. Introducción
    nbf.v4.new_markdown_cell("# Exploración Socioeconómica vs Tipos de Violencia Intrafamiliar\nEste cuaderno cruza datos demográficos estáticos (Censo 2018) con crímenes de Violencia Intrafamiliar (VIF) reportados en 2018 para evaluar si es predictivo el perfil departamental para determinar el **Tipo Predominante de Violencia (Target)**."),
    
    # 2. Bibliotecas y Configuración
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='darkgrid', palette='mako')\npd.set_option('display.max_columns', None)\n\nimport warnings\nwarnings.filterwarnings('ignore')"),
    
    # 3. Lectura de Datos
    nbf.v4.new_markdown_cell("## 1. Carga de Datos y Limpieza"),
    nbf.v4.new_code_cell("""# Cargamos nuestras tres bases
# 1. Decisiones del hogar (latin-1 para tildes)
df_hogares = pd.read_csv('../Datos/Poblacion/hogares_decisiones.csv', encoding='latin-1')

# 2. Economía y Empleo 
df_economia = pd.read_csv('../Datos/Poblacion/economia_empleo.csv', encoding='latin-1')

# 3. Panel histórico completo
df_panel = pd.read_csv('../Datos/Procesados/panel_departamento_anio.csv')

# Filtramos el panel al año de impacto censal (2018) para un cruce realista transversal
df_2018 = df_panel[df_panel['anio'] == 2018].copy()

# Estandarizamos departametos para evitar fallos de Merge
def slugify(v):
    if not isinstance(v, str): return ''
    import unicodedata, re
    c = v.replace('', 'n')
    n = unicodedata.normalize('NFKD', c)
    s = "".join(ch for ch in n if not unicodedata.combining(ch)).lower()
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')

# Agregamos llave de cruce
df_hogares['key'] = df_hogares['departamento'].apply(slugify)
df_economia['key'] = df_economia['departamento'].apply(slugify)
df_2018['key'] = df_2018['departamento'].apply(slugify)

print(f"Dimensiones Hogares: {df_hogares.shape}")
print(f"Dimensiones Economía: {df_economia.shape}")
print(f"Dimensiones Panel (2018): {df_2018.shape}")"""),

    # 4. Feature Engineering
    nbf.v4.new_markdown_cell("## 2. Ingeniería de Variables (Feature Engineering) e Integración"),
    nbf.v4.new_code_cell("""# Hacemos el Merge usando la llave estandarizada
df_master = df_2018.merge(df_hogares.drop(columns=['codigo', 'departamento']), on='key', how='left')
df_master = df_master.merge(df_economia.drop(columns=['codigo', 'departamento']), on='key', how='left')

# Drop nulls en nuestras variables obligatorias
df_master = df_master.dropna(subset=['poblacion_15_mas', 'total_de_hogares'])

# =======================
# VARIABLES PREDICTORAS
# =======================
# 1. Tasa de Desempleo (cesantes + aspirantes / PEA Total)
pea_total = df_master['pea_ocupados'] + df_master['pea_cesantes'] + df_master['pea_aspirantes']
df_master['tasa_desempleo'] = ((df_master['pea_cesantes'] + df_master['pea_aspirantes']) / pea_total) * 100

# 2. Tasa de dependencia de trabajo doméstico
df_master['tasa_quehaceres_hogar'] = (df_master['pei_quehaceres_hogar'] / df_master['poblacion_15_mas']) * 100

# 3. Empoderamiento vs Patriarcado: Ratio de toma unilateral de decisión por hombres
df_master['tasa_decision_hombre'] = (df_master['decision_hombre'] / df_master['total_de_hogares']) * 100

# 4. Contexto Socio-Legal: Tasa de divorcios per capita (por 10,000 habs grandes)
df_master['tasa_divorcios_10k'] = (df_master['divorcios_total'] / df_master['poblacion_15_mas']) * 10000

# =======================
# VARIABLE RESPUESTA (TARGET)
# =======================
# Columnas puras de tipos de VIF en nuestro panel
vif_types = [
    'vif_tipo_fisica', 
    'vif_tipo_psicologica', 
    'vif_tipo_patrimonial',
    'vif_tipo_sexual'
]

# Determinamos el tipo de VIF máximo por departamento
df_master['vif_predominante'] = df_master[vif_types].idxmax(axis=1).str.replace('vif_tipo_', '').str.capitalize()

print("Variables creadas exitosamente:")
display(df_master[['departamento', 'tasa_desempleo', 'tasa_decision_hombre', 'vif_predominante']].head())"""),

    # 5. Visualizaciones Target
    nbf.v4.new_markdown_cell("## 3. Análisis Descriptivo del Target\nRespondemos a la pregunta: *¿Qué distribución tiene el predominio de la violencia en Guatemala?*"),
    nbf.v4.new_code_cell("""plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_master, x='vif_predominante', palette='Set2')
plt.title("Distribución de Tipo de Violencia Predominante por Departamento (2018)", fontsize=14, pad=15)
plt.ylabel("Nro de Departamentos")
plt.xlabel("Tipo predominante de agresión")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')
plt.show()"""),

    # 6. Scatter de variables
    nbf.v4.new_markdown_cell("## 4. Comportamiento Sociológico vs Violencia Física\nVamos a comprobar visualmente si existe una separación (clusters) entre departamentos con alta dependencia/toma de poder unilateral y el tipo de violencia."),
    nbf.v4.new_code_cell("""plt.figure(figsize=(12, 7))

sns.scatterplot(
    data=df_master, 
    x='tasa_decision_hombre', 
    y='tasa_desempleo', 
    hue='vif_predominante', 
    size='tasa_divorcios_10k',
    sizes=(50, 400),
    alpha=0.8,
    palette='Set1'
)

plt.title("Tasa de Desempleo vs Machismo en Decisiones del Hogar", fontsize=15, pad=15)
plt.xlabel("Porcentaje de Hogares donde el Hombre decide exclusivamente (%)", fontsize=12)
plt.ylabel("Tasa de Desempleo Reportada (%)", fontsize=12)

# Añadimos nombres a los extremos
for i, row in df_master.iterrows():
    if row['tasa_decision_hombre'] > df_master['tasa_decision_hombre'].quantile(0.8) or row['tasa_desempleo'] > df_master['tasa_desempleo'].quantile(0.8):
        plt.text(row['tasa_decision_hombre']+0.2, row['tasa_desempleo'], row['departamento'], fontsize=9)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()"""),

    # 7. Correlaciones
    nbf.v4.new_markdown_cell("## 5. Matriz de Correlación sobre Variables Estructurales"),
    nbf.v4.new_code_cell("""cols_analizar = [
    'tasa_divorcios_10k', 
    'tasa_desempleo', 
    'tasa_quehaceres_hogar', 
    'tasa_decision_hombre'
]

# Normalizar la tasa absoluta de VIF para no sesgarnos por población pura
df_master['tasa_vif_total_10k'] = (df_master['vif_total'] / df_master['poblacion_15_mas']) * 10000
cols_analizar.append('tasa_vif_total_10k')

corr = df_master[cols_analizar].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title("Correlación de Variables Sociodemográficas", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()""")
]

nb.cells.extend(celdas)

with open('Analisis_Exploratorio_EDA/eda_socioeconomico_predictivo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generado exitosamente en Analisis_Exploratorio_EDA/eda_socioeconomico_predictivo.ipynb")
