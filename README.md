# TP2 - Sistema de Deteccion y Clasificacion de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings -> Busqueda por similitud -> Clasificacion -> Deteccion -> Pipeline completo**

La catedra provee la infraestructura completa (Docker, aplicacion Gradio, base de datos
vectorial, API, scripts y herramientas de evaluacion/visualizacion). El estudiante debe
realizar un **fork** de este repositorio y completar **unicamente** las funciones indicadas
en cada etapa. Cualquier modificacion fuera de esas funciones debera estar debidamente
justificada en el informe.

## Funciones a implementar

| Etapa | Funcion | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |
| 4 | `evaluate_pipeline()` | `src/lib/services/pipeline_service.py` |
| 4 | `optimize_model()` | `src/lib/services/pipeline_service.py` |
| 4 | `generate_annotations(folder_path, output_format)` | `src/lib/services/pipeline_service.py` |

Cada funcion tiene su docstring con el comportamiento esperado y sugerencias.

## Objetivo del backend

API asincronica en Python que permite:

- Busqueda por similitud (`/search`): imagen consultada, top K similares y raza predicha
- Deteccion + clasificacion (`/detect`): bounding boxes, raza y scores de confianza
- Consultar estado de procesamiento asincronico (`/status/{job_id}`)
- Seleccion dinamica del modelo de embeddings (`/models` + campo `model` en `/search`)

La API responde `HTTP 202` con `job_id` y luego permite consultar resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tp2/
├── src/
│   ├── app/
│   │   └── main.py
│   ├── frontend/
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py
│       ├── bootstrap.py
│       ├── config.py
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py          # NDCG@10, IoU, AP/mAP, precision/recall/F1 (provisto)
│       ├── visualization/
│       │   └── draw.py             # dibujo de bounding boxes (provisto)
│       ├── services/
│       │   ├── similarity_service.py   # Etapa 1
│       │   ├── classifier_service.py   # Etapa 2
│       │   ├── detection_service.py    # Etapa 3
│       │   ├── pipeline_service.py     # Etapa 4
│       │   └── task_manager.py
│       └── storage/
│           ├── embedding_store.py      # base vectorial JSON
│           └── pgvector_store.py       # base vectorial PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py         # descarga el dataset de Kaggle
│   ├── build_index.py              # indexa el dataset en la base vectorial
│   ├── train_classifier.py         # entrena/evalua el clasificador (Etapa 2)
│   └── generate_annotations.py     # anotacion automatica YOLO/COCO (Etapa 4)
├── data/
│   ├── dataset/                    # 70 Dog Breeds Image Dataset (no se versiona)
│   ├── eval/                       # conjunto de prueba anotado (Etapa 4)
│   └── embeddings.json             # base vectorial JSON (si USE_PGVECTOR=false)
├── models/                         # checkpoints entrenados (no se versionan)
├── output/
├── train.ipynb                     # notebook de experimentacion
├── informe.ipynb                   # informe tecnico
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
└── .env.docker.example / .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker Desktop (incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)

```bash
python scripts/download_dataset.py
```

O descargarlo manualmente y descomprimir `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configura tus modelos

Entrena tus modelos (Etapa 2) y guardalos dentro de la carpeta `models`. Por defecto, el
modulo soporta modelos construidos con pytorch validando la extension **.pth**.

Si eligen utilizar otro framework, pueden exportarlo a formato **.onnx**

Recuerda actualizar las configuraciones del .env correspondiente para actualizar la ruta
hacia tus modelos (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`).

El entorno local con o sin docker reinicia la aplicacion y actualiza el codigo
automaticamente si utilizan docker compose.

Puede que el reinicio automatico no funcione en todas las versiones de Docker Desktop en
sistemas *Windows*, en tal caso deberan correr los comandos como se mencionan en el
siguiente apartado para actualizar el codigo dentro de docker.

## Opcion 1 - Corriendo todo dentro de docker

Es la forma en que la catedra evaluara el trabajo.

### 1. Buildea y corre la aplicacion.

Actualiza el archivo **.env.docker.example** y ajustalo a tus necesidades. Luego corre desde el terminal :

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000`,
postgres en `localhost:5432`.

## Opcion 2 - Configurando el ambiente local desde cero

### 1. Clona tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
```

### 2. Crea un entorno virtual con Python 3.12

Con python estandar:

```bash
# Linux / Mac
python3.12 -m venv .venv

# Windows
py -3.12 -m venv .venv
```

Alternativa con [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Instalar uv en Linux / Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar uv en Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Crear el entorno
uv venv --python 3.12 .venv
```

### 3. Activa el entorno virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

### 4. Instala las dependencias

```bash
pip install -r requirements.txt
# o, si usas uv:
uv pip install -r requirements.txt
```

### 5. Configura las variables de entorno

Copia el ejemplo local a `src/.env` y ajustalo a tus necesidades. El backend local y los
scripts de `scripts/` leen la configuracion desde ese archivo.

```bash
# Linux / Mac
cp .env.local.example src/.env
```

```powershell
# Windows
copy .env.local.example src\.env
```

### 6. Descarga el dataset

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las
tenes, descarga el dataset manualmente desde Kaggle y descomprimi `train/`, `valid/` y
`test/` dentro de `data/dataset`.

### 7. Inicia la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (default). Con `USE_PGVECTOR=false` se usa la
base vectorial JSON (`data/embeddings.json`) y no hace falta docker para trabajar localmente.

### 8. Inicia el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 9. Inicia el frontend (en otra terminal, con el venv activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 10. Verifica que todo este corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

### 11. A trabajar

1. Implementa las funciones de la Etapa 1 en `src/lib/services/similarity_service.py`.
2. Indexa el dataset en la base vectorial: `python scripts/build_index.py --split train`
   (desde la raiz del repo).
3. Proba la busqueda por similitud desde el frontend (pestaña Etapa 1).
4. Continua con las Etapas 2, 3 y 4. Documenta los experimentos en `train.ipynb` y el
   informe en `informe.ipynb`.

## Scripts provistos

Se ejecutan desde la raiz del repositorio con el ambiente local activo (y postgres corriendo
si `USE_PGVECTOR=true`). Leen la configuracion de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar el dataset en la base vectorial (requiere Etapa 1 implementada)
python scripts/build_index.py --split train

# Entrenar y evaluar el clasificador (requiere Etapa 2 implementada)
python scripts/train_classifier.py --model resnet18_finetuned

# Generar anotaciones automaticas (requiere Etapas 3 y 4 implementadas)
python scripts/generate_annotations.py data/eval --format yolo
python scripts/generate_annotations.py data/eval --format coco
```

## Configuracion

No hardcodear parametros. Configurar mediante `.env`:

1. En la ejecucion local copiar `.env.local.example` a `src/.env`
2. Ajustar variables de modelo seleccionado (`EMBEDDING_MODEL`), threshold de similitud,
   cantidad de vecinos (`TOP_K`), paths y configuracion de YOLO
3. Opcional ( habilitada por defecto ): configurar conexion a PostgreSQL + pgvector

## Endpoints

- Backend: `http://localhost:8000`
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (imagen provista por catedra): `http://localhost:8080`

| Endpoint | Descripcion |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: busqueda por similitud (acepta `model` y `top_k`) |
| `POST /detect` | Etapa 3: deteccion + clasificacion |
| `GET /status/{job_id}` | Estado del procesamiento asincronico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend |

## Pipeline provisto (base funcional)

1. Orquestacion completa de busqueda por similitud (Etapa 1) y deteccion + clasificacion (Etapa 3)
2. Seleccion dinamica del modelo de embeddings: `baseline`, `resnet18_finetuned`, `cnn_custom`
3. Busqueda por similitud configurable (`cosine` o `l2`) con manejo de desconocidos via `SIMILARITY_THRESHOLD`
4. Persistencia configurable en JSON o PostgreSQL + pgvector (`USE_PGVECTOR`)
5. Funciones auxiliares de evaluacion (`lib/evaluation/metrics.py`): NDCG@10, IoU, AP/mAP, precision/recall/F1, specificity
6. Herramientas de visualizacion (`lib/visualization/draw.py`)
7. Frontend Gradio con pestañas para Etapa 1 (imagen consultada, top K similares, raza predicha)
   y Etapa 3 (bounding boxes, razas y scores)

Las funciones de cada etapa estan marcadas con `NotImplementedError` y deben ser completadas
por el equipo para que el pipeline funcione end-to-end.

## Evaluacion del pipeline (Etapa 4)

- Construir en `data/eval` un conjunto de al menos **10 imagenes complejas** anotadas
  manualmente (bounding boxes + raza).
- Calcular mAP, IoU, precision, recall y F1 con `evaluate_pipeline()`.
- Elegir una estrategia de optimizacion (cuantizacion INT8 o exportacion ONNX/TensorRT) y
  comparar tiempo de inferencia, uso de memoria y precision con `optimize_model()`.

## Notas importantes

- La implementacion actual es una base operativa para pruebas end-to-end y debe completarse
  con las funciones de cada etapa.
- Para usar `pgvector`, levantar `postgres` y definir `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimension del embedding del modelo elegido; al
  cambiar de modelo de embeddings hay que reindexar la base (`scripts/build_index.py`).
- Colocar los modelos entrenados en `models`, los cuales deberan estar disponibles en un
  link con acceso de solo lectura publico para poder ser descargados por los docentes.

## Entregables

- Repositorio Git privado (fork de este template).
- Pull Request abierto contra el repositorio original provisto por la catedra
  (sera evaluado **sin mergear** a main o el branch principal).
- Informe en IPYNB (`informe.ipynb`).
- Notebook de experimentacion (`train.ipynb`).

Para que el trabajo practico se considere aprobado, el sistema debe andar sin errores
corriendo los siguientes comandos :

```bash
docker compose build
docker compose up
```

## Fecha de entrega

Sabado 27/06 hasta las 23:59 (via campus).

- Solo se consideraran commits hasta esa hora
- Se recomienda trabajar de forma incremental (no un unico commit final)
