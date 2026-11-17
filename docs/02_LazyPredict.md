# Segunda etapa: LazyPredict

Copiar `notebooks/02_LazyPredict.ipynb`, `src/benchmark.py` y `requirements_lazypredict.txt` sobre la raíz del proyecto actual, conservando 01_EDA y sus correcciones.

Instalar desde la raíz: `python -m pip install -r requirements_lazypredict.txt`.
Ejecutar 01_EDA si no existen sus CSV y metadatos; después ejecutar 02_LazyPredict de principio a fin.

El módulo solo lee raw_train y metadatos del EDA. No necesita abrir X_test, y_test ni raw_test. Genera tablas 10–16 y gráfico 08. No modifica el split ni entrena el modelo final.

La lista acotada de siete modelos y la semilla son explícitas. Errores y advertencias se conservan. La codificación se ajusta sobre entrenamiento interno y LazyPredict realiza su imputación/escalado en cada modelo. CV y tuning automáticos están desactivados para evitar validaciones no agrupadas.

Fuente técnica revisada: código de LazyClassifier.fit y provide_models en lazypredict 0.4.0 instalado durante la preparación; proyecto https://github.com/shankarpandala/lazypredict.

## Validación realizada

Las cinco celdas de código se ejecutaron secuencialmente con IPython usando las 71.706 canciones de train. Siete de siete modelos completaron sin errores. Entrenamiento interno: 57.423; validación interna: 14.283. Tiempo observado: 72,7 segundos (dependiente del equipo). Formato ipynb validado y gráfico inspeccionado. No se validó el arranque externo del kernel Jupyter en este entorno; se incluyen las salidas reales de ejecución secuencial.

### Interpretación de la ejecución incluida

- **ExtraTrees:** AUC 0,875 y lift@10% 3,58; lidera el ranking, pero la brecha train-validación de 0,125 indica que hay que controlar su complejidad.
- **Random Forest:** AUC 0,869 y brecha 0,131; también requiere regularización y validación agrupada.
- **HistGradientBoosting:** AUC 0,864, lift 3,41 y brecha 0,026; es un candidato por su equilibrio, aunque el recall inicial es 0,362 y necesita estudiar la decisión y/o ponderación de clases.
- **Regresión logística:** AUC 0,843 y brecha 0,001; aporta una referencia simple competitiva.
- **GaussianNB:** recall 0,941, pero precisión 0,335 y lift 1,97. Un recall alto por sí solo no implica una buena priorización promocional.
- **Árbol individual:** AUC 0,695 y brecha 0,305; los ensambles lo superan en esta partición.

Se conservan como candidatos Regresión Logística, Random Forest, HistGradientBoosting y ExtraTrees. La selección definitiva dependerá de CV agrupada, optimización y la comparación de representaciones del género. Estos valores corresponden a la ejecución incluida; al volver a ejecutar, contrastarlos con las tablas actualizadas.
