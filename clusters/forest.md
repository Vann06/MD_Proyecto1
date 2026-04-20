### Desempeño General

El Random Forest alcanzó un rendimiento prácticamente idéntico al del Árbol 
de Decisión, con un **accuracy de 80%** y un **F1 macro de 0.79** en el 
conjunto de prueba. Los mejores hiperparámetros encontrados por GridSearchCV 
fueron `max_depth=5`, `min_samples_leaf=5`, `min_samples_split=5` y 
`n_estimators=100`, con un F1 de validación cruzada de 0.7868.

La clase más desafiante sigue siendo **Resolución_legal**, con un F1 de 0.67. 
Esto es esperable porque es una clase en frontera: comparte una dimensión con 
Baja_conflictividad (VIF baja) y otra con Conflicto_visible (divorcios altos), 
lo que genera confusiones naturales.

### Análisis de la Matriz de Confusión

Los patrones de confusión son coherentes y no aleatorios:

- **Baja_conflictividad → Resolucion_legal (2 errores):** ambas comparten VIF baja
- **Conflicto_visible → Resolucion_legal (3 errores):** ambas comparten divorcios altos
- **Violencia_atrapada → Baja_conflictividad y Conflicto_visible (1 cada una):** 
  comparten al menos una dimensión

El modelo **nunca confunde clases diagonalmente opuestas** 
(ej: Baja_conflictividad con Conflicto_visible), lo cual indica que las 
predicciones respetan la estructura bidimensional del target.

### Hallazgos Clave

**1. La educación superior es el predictor más fuerte (23.1%)**

Es la variable más discriminante para clasificar el régimen de conflicto familiar 
de un departamento. Departamentos con mayor proporción de población con 
educación superior tienden a caer en regímenes de mayor conflictividad visible 
(Conflicto_visible o Resolucion_legal), respaldando la hipótesis de que 
**el acceso a la justicia y al sistema legal formal está mediado por el nivel 
educativo**.

**2. Las dinámicas de poder en el hogar pesan en conjunto cerca del 30%**

Al sumar `pct_decision_mujer` (16.9%) y `pct_decision_hombre` (13.2%), las 
variables relacionadas con quién toma decisiones en el hogar representan el 
segundo bloque más importante del modelo. Esto conecta directamente con la 
literatura sobre autonomía femenina y violencia intrafamiliar: **quién decide 
en el hogar se asocia fuertemente con cómo se manifiesta (o no se manifiesta) 
el conflicto**.

**3. Distribución equilibrada de importancias**

Ninguna variable domina por completo el modelo. La variable más importante 
tiene 23% y la menos importante 10%, lo que indica que **los cuatro regímenes 
son el resultado de una interacción de múltiples factores estructurales**, 
no de un único determinante. Esto respalda la validez teórica del enfoque 
multidimensional.

**4. La estructura conyugal tiene peso moderado**

Las variables de estado civil (`pct_uniones_hecho` y `pct_casados`) aportan 
conjuntamente alrededor del 23% de la capacidad predictiva, lo cual confirma 
que **la formalidad del vínculo de pareja está asociada al tipo de dinámica 
de violencia y divorcio** que presenta un departamento.