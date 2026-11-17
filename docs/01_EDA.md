# EDA — resultados y decisiones

## Hallazgos de esta ejecución

- Se reservaron **18,034 canciones para test** y se analizaron **71,706 de train**, sin IDs ni claves de canción compartidos.
- El P80 de train es **52**; la proporción de hits es **21.01%**. Supera el 20% por los empates en el umbral.
- **10.27%** de las canciones de train tiene popularidad cero. Se conserva como no-hit y se documenta la ambigüedad.
- **18.05%** de train es multigénero. Se observan **114 géneros**: elegir una sola etiqueta perdería información.
- Las tasas de hit por género van de **0.00% a 94.55%**. Son asociaciones descriptivas con grupos solapados; motivan comparar representaciones del género, no prueban que PCA sea superior.
- En duración, log1p cambia la asimetría de **10.39 a -0.32**; en otras variables su efecto es menor. Esto apoya conservar el experimento, sin asumir normalidad.

Estos hallazgos corresponden al dataset y split registrados en esta ejecución. Al cambiar la fuente se deben actualizar las conclusiones con los resultados nuevos.

## Validación

Las diez celdas de código se ejecutaron secuencialmente con IPython y el dataset completo, sin errores. Se validó el formato ipynb y se incluyeron las salidas y gráficos reales. El entorno no permitió iniciar el kernel externo Jupyter por restricciones de sockets; no se afirma haber validado ese arranque.

Controles aprobados: grupos e IDs disjuntos, reutilización del split con hashes, orden de IDs raw/processed, ausencia de target/identificadores en X, esquema consistente, suma multi-hot igual al número de géneros en train y tratamiento de un género desconocido en un caso de prueba.

Python de validación: 3.12.14; dependencias fijadas en requirements.txt.

SHA256 de la fuente: `bb30c3feff2c427894a19d0473c2a8e7cb8c7dc5616ce9be6ff9633a6a5ac9d2`.

## Alcance

Se mantienen carpetas y reglas de EV2. No se incluyen resultados de LazyPredict, PCA, ensambles o evaluación final porque corresponden a la siguiente etapa. El informe técnico global de EV3 aún no está redactado.
