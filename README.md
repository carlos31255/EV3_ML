# EV3 — Spotify: EDA

Primera etapa de EV3, basada en https://github.com/carlos31255/EV2_ML.

## Ejecución
1. Colocar `Spotify_Tracks_Dataset.csv` en `data/raw/`.
2. Usar Python 3.12 (entorno de validación), crear un entorno e instalar `pip install -r requirements.txt`.
   Alternativa conda: `conda env create -f environment.yml` y `conda activate ml_spotify_ev3`.
3. Abrir `notebooks/01_EDA.ipynb` y ejecutar Restart Kernel & Run All.

El notebook hace la preparación y el EDA; no requiere ejecutar main antes.
`python main.py` es una alternativa para generar solo los datos preparados.

## Estructura
- `data/raw/`: fuente original; `train/` y `test/` guardan el split sin codificar.
- `data/processed/`: matrices sin escalado/PCA, etiquetas y grupos por conjunto.
- `src/pipeline.py`: limpieza, split reproducible y exportación.
- `src/graficos.py`: gráficos del EDA, derivados de EV2.
- `notebooks/01_EDA.ipynb`: análisis sobre train.
- `output/graficos/` y `output/tablas/`: evidencias generadas.
- `output/modelos/`: reservado para las siguientes etapas.
- `docs/`: decisiones metodológicas y validación.

## Diferencias respecto de EV2
El EDA no calcula métricas, distribuciones ni tasas de test. La codificación de géneros se aprende después del split usando train. El hash de la fuente y de cada split impide reutilizar archivos alterados silenciosamente. Para sustituir voluntariamente el dataset o split, revisar y ejecutar `generar_split_raw(forzar=True)`; no hacerlo para buscar una partición favorable.

Se mantiene la deduplicación por track_id, popularidad máxima entre copias y separación agrupada por artista+título. Esta clave no reconoce todos los títulos alternativos ni garantiza artistas distintos. Se mantienen las transformaciones deterministas de EV2.

Los CSV procesados son una referencia. La modelación ajustará codificación, imputación, escalado y PCA dentro de cada fold, preferiblemente partiendo de raw. El P80 de train define una etiqueta fija. El test histórico ya se examinó en EV2; esta limitación se conserva.

No se incluyen dataset, splits ni modelos entrenados en este paquete. LazyPredict y PCA se incorporarán en la siguiente etapa; no son necesarios para ejecutar 01_EDA.
