
### Desempeño General

La Regresión Logística alcanzó un **accuracy de 74.55%** y un **F1 macro 
de 0.7479** en el conjunto de prueba, un desempeño ligeramente inferior a 
los modelos basados en árboles (80%). El mejor hiperparámetro encontrado fue 
`C=10`, con un F1 de validación cruzada de 0.6796. El hecho de que el mejor 
valor de `C` sea alto (poca regularización) sugiere que el modelo necesitaba 
flexibilidad para ajustarse a las cuatro clases.

A diferencia del enfoque original (donde la regresión logística colapsó a un 
esquema binario), aquí el modelo **sí logró estimar coeficientes separados 
para las cuatro clases** (matriz de coeficientes 4×7), confirmando que la 
reformulación del target permite un análisis multinomial real.

### Desempeño por Clase

La clase **Baja_conflictividad** fue la mejor predicha (F1=0.91, Precision=0.94), 
mientras que **Conflicto_visible** (F1=0.62) y **Resolucion_legal** (F1=0.62) 
fueron las más desafiantes, aunque por razones distintas:

- **Conflicto_visible** tiene alta precisión (0.89) pero bajo recall (0.47): 
  el modelo es cauteloso al predecir esta clase y deja escapar muchos casos 
  reales
- **Resolucion_legal** tiene alto recall (0.91) pero baja precisión (0.48): 
  el modelo predice esta clase en exceso, absorbiendo casos que realmente son 
  Conflicto_visible

### Análisis de la Matriz de Confusión

La confusión más notable es entre **Conflicto_visible y Resolucion_legal**: 
el modelo clasificó 8 casos de Conflicto_visible como Resolución_legal. 
Esto significa que la regresión logística **tiende a favorecer la predicción 
de Resolucion_legal cuando hay divorcios altos**, sin distinguir bien si la 
VIF es alta o baja.

Esta confusión es consistente con la naturaleza lineal del modelo: al no poder 
capturar interacciones complejas entre variables, se inclina por el patrón más 
dominante en cada clase. El árbol y el random forest, por tener lógica 
condicional, manejan mejor esta distinción.

### Interpretación de los Coeficientes por Clase

La Regresión Logística tiene una ventaja única sobre los modelos de árboles: 
**muestra la dirección y magnitud del efecto** de cada variable sobre cada clase. 
Analizando los coeficientes:

**Baja_conflictividad:**
- Único predictor positivo fuerte: `pct_decision_hombre` (+2.39)
- Predictores negativos: educación superior (-2.59), uniones de hecho (-2.46), 
  casados (-2.48), decisión mujer (-2.49)
- **Interpretación:** las zonas de baja conflictividad visible están 
  caracterizadas por **hogares donde el hombre decide** y por **menor 
  formalización de vínculos conyugales** (ni casados ni unidos por debajo de 
  la media). Esto sugiere el perfil de departamentos rurales donde las 
  denuncias son menos frecuentes.

**Conflicto_visible:**
- Predictores positivos fuertes: `pct_decision_mujer` (+3.05), `pct_casados` 
  (+2.46), `pct_uniones_hecho` (+1.83)
- Predictores negativos: decisión hombre (-0.81), quehaceres hogar (-1.13)
- **Interpretación:** departamentos donde las mujeres deciden más en el hogar 
  y hay mayor formalización conyugal tienen **más probabilidad de caer en 
  "Conflicto visible"**. Es consistente con la hipótesis de que la autonomía 
  femenina se asocia con mayor visibilización del conflicto (denuncia + 
  divorcio).

**Resolucion_legal:**
- Predictores positivos: educación superior (+1.55), decisión mujer (+1.58), 
  uniones de hecho (+1.34)
- Predictores negativos: decisión hombre (-2.12), desempleo (-1.41), 
  quehaceres hogar (-1.14)
- **Interpretación:** la educación superior y la menor dependencia femenina 
  (menos quehaceres del hogar, menor desempleo) predicen el régimen donde los 
  conflictos **se resuelven por la vía legal antes de escalar a violencia 
  reportable**. Este es el patrón más cercano al perfil "urbano educado".

**Violencia_atrapada:**
- Predictor positivo más fuerte: `tasa_quehaceres_hogar` (+2.45)
- También positivos: casados (+1.17), desempleo (+1.07)
- Predictores negativos: decisión mujer (-2.14), uniones de hecho (-0.70)
- **Interpretación:** el perfil más preocupante socialmente. Los departamentos 
  donde **muchas mujeres se dedican a quehaceres del hogar** (proxy de 
  dependencia económica) y donde **las mujeres tienen poca autonomía de 
  decisión**, tienen mayor probabilidad de caer en "Violencia atrapada": 
  se denuncia la violencia pero no se sale del matrimonio.