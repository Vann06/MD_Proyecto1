import nbformat as nbf
import os

# Asegurar carpeta predictiva
os.makedirs('Modelos_Predictivos', exist_ok=True)

nb = nbf.v4.new_notebook()

celdas = [
    # Titulo
    nbf.v4.new_markdown_cell("# Modelo Predictivo: Clasificación de Violencia Intrafamiliar predominantemente\nEste cuaderno implementa el ensamblado de datos y 3 algoritmos de Machine Learning (Árboles, Bosques y Regresión Logística) acorde a los lineamientos del proyecto para responder a la viabilidad de la variable respuesta."),
    
    # Imports
    nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport unicodedata, re\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.tree import DecisionTreeClassifier, plot_tree\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n\nsns.set_theme(style='whitegrid')\nwarnings = __import__('warnings')\nwarnings.filterwarnings('ignore')"""),
    
    # 1. Super Dataframe Data Engineering
    nbf.v4.new_markdown_cell("## 1. Construcción del 'Super Dataframe' de Entrenamiento"),
    nbf.v4.new_code_cell("""# Función para llaves maestras
def slugify(v):
    if not isinstance(v, str): return ''
    c = v.replace('', 'n') # Limpieza especifica por si acaso
    n = unicodedata.normalize('NFKD', c)
    s = "".join(ch for ch in n if not unicodedata.combining(ch)).lower()
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')

print("1. Cargando todos los datasets censales...")
# Cargar Censo 2018 (Variables Estructurales Estáticas)
df_hogares = pd.read_csv('../Datos/Poblacion/hogares_decisiones.csv', encoding='latin-1', thousands=',')
df_economia = pd.read_csv('../Datos/Poblacion/economia_empleo.csv', encoding='latin-1', thousands=',')
df_civil = pd.read_csv('../Datos/Poblacion/estado_civil.csv')
df_edu = pd.read_csv('../Datos/Poblacion/nivel_educativo.csv')

# Cargar Serie de Tiempo Histórica
df_panel = pd.read_csv('../Datos/Procesados/panel_departamento_anio.csv')

print("2. Normalizando Claves Geográficas...")
cuadros = [df_hogares, df_economia, df_civil, df_edu, df_panel]
for df in cuadros:
    # A todos les fabricamos un 'key'
    if 'departamento' in df.columns:
        df['key'] = df['departamento'].apply(slugify)
    elif 'Departamento' in df.columns:
        df['key'] = df['Departamento'].apply(slugify)

print("3. Feature Engineering Específico Censal...")
# Calcular % Educacion Superior (como en cluster)
sup = df_edu['Superior_Licenciatura'] + df_edu.get('Superior_Maestria_Doc', 0)
df_edu['pct_educacion_superior'] = (sup / df_edu['Poblacion_4_mas']) * 100

# Calcular % Uniones de hecho
col_unido = 'Unido' if 'Unido' in df_civil.columns else 'Unida(o)'
col_pob10 = 'Poblaci\u00f3n_10_mas' if 'Poblaci\u00f3n_10_mas' in df_civil.columns else 'Total'
# Manejo preventivo de nombres extraños en pandas por codificación
cols_civil = list(df_civil.columns)
for c in cols_civil:
    if 'Unid' in c: col_unido = c
    if '10' in c and 'Pob' in c: col_pob10 = c
df_civil['pct_uniones_hecho'] = (df_civil[col_unido] / df_civil[col_pob10]) * 100

print("4. Consolidando el Panel Múltiple...")
# Filtramos Panel Histórico de 2013 a 2023 (10 años = ~220 datos para ML estable)
df_train_base = df_panel[(df_panel['anio'] >= 2013) & (df_panel['anio'] <= 2023)].copy()

# Cruce Estrella (Merge del panel temporal con datos censales demográficos)
df_master = df_train_base.merge(df_hogares.drop(columns=['codigo', 'departamento'], errors='ignore'), on='key', how='left')
df_master = df_master.merge(df_economia.drop(columns=['codigo', 'departamento'], errors='ignore'), on='key', how='left')
df_master = df_master.merge(df_civil[['key', 'pct_uniones_hecho']], on='key', how='left')
df_master = df_master.merge(df_edu[['key', 'pct_educacion_superior']], on='key', how='left')

# Forzar numéricos (Limpiar basuras de strings)
for col in df_master.columns.drop(['key', 'departamento', 'anio'], errors='ignore'):
    if df_master[col].dtype == object:
        df_master[col] = pd.to_numeric(df_master[col].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce')

df_master = df_master.dropna(subset=['pct_uniones_hecho', 'poblacion_15_mas'])
print(f"Dimensiones del Super Dataframe final: {df_master.shape}\\nListo para crear el Target.")
"""),
    
    # 2. Creacion de Target y Predictores
    nbf.v4.new_code_cell("""# =======================
# CREACIÓN DEL TARGET (Y)
# =======================
vif_types = ['vif_tipo_fisica', 'vif_tipo_psicologica', 'vif_tipo_patrimonial']
df_master['vif_predominante'] = df_master[vif_types].idxmax(axis=1).str.replace('vif_tipo_', '').str.capitalize()

# Convertimos Target a Numérico (Label Encoding) para los algoritmos
# 0 = Fisica, 1 = Patrimonial, 2 = Psicologica... (depende de cuales ganen empíricamente)
# Para simplificar y hacer un modelo dicotómico brutal, vamos a agrupar en: 'Fisica' vs 'No Fisica' 
# Pero como nos pidieron 'predominante', dejamos el label encoder:
le = LabelEncoder()
df_master['Target'] = le.fit_transform(df_master['vif_predominante'])
clases_target = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Clases de la Variable Respuesta:\\n{clases_target}\\n")

# =======================
# VARIABLES PREDICTORAS (X)
# =======================
pea_total = df_master['pea_ocupados'] + df_master['pea_cesantes'] + df_master['pea_aspirantes']
df_master['tasa_desempleo'] = ((df_master['pea_cesantes'] + df_master['pea_aspirantes']) / pea_total) * 100
df_master['tasa_quehaceres_hogar'] = (df_master['pei_quehaceres_hogar'] / df_master['poblacion_15_mas']) * 100
df_master['tasa_decision_hombre'] = (df_master['decision_hombre'] / df_master['total_de_hogares']) * 100
df_master['tasa_divorcios_10k'] = (df_master['divorcios_total'] / df_master['poblacion_15_mas']) * 10000

features = [
    'tasa_desempleo', 
    'tasa_quehaceres_hogar', 
    'tasa_decision_hombre', 
    'tasa_divorcios_10k', 
    'pct_educacion_superior', 
    'pct_uniones_hecho'
]

# Rellenar posibles nulos marginales con la mediana antes de modelar
df_master[features] = df_master[features].fillna(df_master[features].median())

# =======================
# SPLIT & SCALING
# =======================
X = df_master[features]
y = df_master['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Pipeline de Datos Completado. Entorno listo para Modelado.")"""),

    # 3. Decision Tree
    nbf.v4.new_markdown_cell("## 2. Modelo 1: Árboles de Decisión (Interpretabilidad Visual)"),
    nbf.v4.new_code_cell("""# Entrenamos el Arbol de Clasificacion
dt = DecisionTreeClassifier(max_depth=3, random_state=42) # Profundidad 3 para evitar overfitting y poder visualizarlo
dt.fit(X_train_scaled, y_train)

# Prediccion
y_pred_dt = dt.predict(X_test_scaled)

print("--- REPORTE DEL ÁRBOL DE DECISIÓN ---")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

# Visualización del Árbol (Reglas Categóricas)
plt.figure(figsize=(16, 8))
plot_tree(dt, feature_names=features, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Reglas Clave: ¿Cómo se decide si estalla Violencia Física o Psicológica?")
plt.show()"""),

    # 4. Random Forest
    nbf.v4.new_markdown_cell("## 3. Modelo 2: Random Forest (Solidez y Prevención de Overfitting)"),
    nbf.v4.new_code_cell("""# Entrenamos Random Forest con multiples estimadores
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
print("--- REPORTE DE BOSQUES ALEATORIOS ---")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Matriz de Importancia (El Insight Real)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="magma")
plt.title("Random Forest: Importancia de las Variables para Predecir VIF Predominante", fontsize=14)
plt.xlabel("Peso Relativo en la Decisión (%)")
plt.tight_layout()
plt.show()"""),

    # 5. Logistic Regression
    nbf.v4.new_markdown_cell("## 4. Modelo 3: Regresión Logística Múltiple (Impacto Probabilístico)"),
    nbf.v4.new_code_cell("""# Regresión Logística
log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
print("--- REPORTE DE REGRESIÓN LOGÍSTICA ---")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# Interpretar coeficientes de variable de interés (Divorcios) vs Clases
coef = log_reg.coef_

# DataFrame para visualizar los pesos (Log-Odds) en una de las clases, ej. la primera clase 'Fisica'
df_coef = pd.DataFrame({'Variable': features, 'Peso Logistico (Odds)': coef[0]})
df_coef = df_coef.sort_values(by='Peso Logistico (Odds)', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_coef, x='Peso Logistico (Odds)', y='Variable', palette="coolwarm")
plt.vlines(0, -1, len(features), colors='black', linestyles='solid', alpha=0.5)
plt.title(f"Afinidad Probabilística hacia desarrollar Violencia '{le.classes_[0]}'", fontsize=14)
plt.tight_layout()
plt.show()"""),
]

nb.cells.extend(celdas)

with open('Modelos_Predictivos/clasificacion_predictiva.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook Final guardado en: Modelos_Predictivos/clasificacion_predictiva.ipynb")
