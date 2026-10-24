# TF-U Backend

Backend de **PLASMA** (Plataforma de Lenguaje y Análisis de Salud Mental Automatizado), correspondiente al Trabajo Final de la carrera. Está organizado como un conjunto de microservicios en Python/FastAPI que analizan corpus textuales: estadísticas, limpieza, embeddings, emociones, tópicos, palabras, riesgo de suicidio, RAG e informes.

Todas las peticiones del cliente pasan por un **api-gateway** que valida el JWT de Supabase y deriva el tráfico al microservicio correspondiente.

## Arquitectura

```text
┌──────────────┐      HTTP       ┌──────────────┐      JWT + proxy      ┌─────────────────────┐
│  Frontend    │ ──────────────> │ api-gateway  │ ────────────────────> │  Microservicios     │
│  (Next.js)   │  :7500          │   FastAPI    │                       │ analysis-service    │
└──────────────┘                 └──────┬───────┘                       └─────┬───────┬───────┘
                                        │                                     │       │
                                 ┌──────▼───────┐                    ┌────────▼──┐  ┌─▼──────────────┐
                                 │ auth-service │                    │ Supabase  │  │ ChromaDB       │
                                 │   :5050      │                    │ Postgres  │  │ :8000          │
                                 └──────────────┘                    │ Auth      │  │ (embeddings)   │
                                                                     │ Storage   │  └────────────────┘
                                                                     └───────────┘
```

- **Supabase** se usa para autenticación, almacenamiento de datasets, estado de jobs y resultados agregados.
- **ChromaDB** almacena los embeddings de los documentos y los chunks de chat.
- Los servicios de ML cargan modelos de Hugging Face y comparten una imagen base con PyTorch (`ml-base`).
- `analysis-service` agrupa clustering, topic modeling, wordcloud, EDA, emociones y riesgo en un único runtime;
  el gateway conserva los prefijos públicos originales.

## Microservicios

| Servicio | Puerto | Prefijo en el gateway | Responsabilidad | GPU |
| --- | --- | --- | --- | --- |
| `api-gateway` | 7500 | — | Autenticación, proxy y logging de requests | No |
| `auth-service` | 5050 | `/auth` | Login, logout, sesión y administración de usuarios | No |
| `dataset-service` | 8082 | `/dataset` | Carga, parseo, chunking e indexación de datasets | Sí |
| `cleaning-service` | 8089 | `/datacleaning` | Limpieza y anonimización del texto | No |
| `analysis-service` | 8080 | `/eda`, `/clustering`, `/wordcloud`, `/emotions`, `/topicmodeling`, `/suicide` | Runtime agrupado de análisis | Sí |
| `evolution-service` | 8090 | `/evolution` | Evolución temporal de palabras y emociones | No |
| `classification-service` | 8087 | `/classification` | Entrenamiento de clasificadores de menciones | Sí |
| `rag-service` | 8088 | `/rag` | Chat RAG con LLM e historial | Sí |
| `report-service` | 8092 | `/report` | Generación de informes y gráficos | No |
| `chroma-service` | 8000 | — | Vector store (imagen oficial `chromadb/chroma`) | No |

## Stack

- **Python 3.11 / 3.12**, FastAPI + Uvicorn.
- **Supabase** (Postgres, Auth, Storage) mediante `supabase-py`.
- **ChromaDB** como base vectorial.
- **PyTorch, Transformers, sentence-transformers** para embeddings y modelos de clasificación.
- **scikit-learn, UMAP, HDBSCAN, BERTopic** para clustering y tópicos.
- **LangChain + Google Gemini / Groq** para el servicio RAG.
- **pandas, numpy, matplotlib, reportlab** para análisis, gráficos y PDF.
- **Docker Compose** para orquestación; imágenes con soporte CUDA para los servicios de ML.

## Requisitos

- Docker Engine y Docker Compose v2.
- Proyecto de Supabase con el esquema aplicado (ver [Base de datos](#base-de-datos)).
- Opcional: GPU NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) para los servicios marcados con GPU. Sin GPU funcionan en CPU (más lentos).
- Para desarrollo local sin Docker: Python 3.12 y acceso a Supabase/Chroma.

## Configuración

Cada microservicio tiene su propio `.env`. Los archivos de ejemplo están versionados como `.env.example`; los `.env` reales **no** se deben commitear.

### 1. Crear los `.env`

Desde la raíz del repositorio:

**PowerShell**

```powershell
Get-ChildItem -Recurse -Filter .env.example | ForEach-Object {
  $target = Join-Path $_.DirectoryName ".env"
  if (-not (Test-Path $target)) { Copy-Item $_.FullName $target }
}
```

**Bash**

```bash
for f in */.env.example; do cp -n "$f" "${f%.example}"; done
```

Luego completá en cada archivo las credenciales de Supabase, los tokens de Hugging Face y las claves de los proveedores de LLM.

### 2. Variables compartidas

La siguiente tabla indica **en qué servicios el código realmente lee cada variable**.

| Variable | Servicios que la usan | Descripción |
| --- | --- | --- |
| `SUPABASE_URL` | Todos los servicios Python (`api-gateway`, `auth`, `dataset`, `cleaning`, `analysis-service`, `evolution`, `classification`, `rag`, `report`) | URL del proyecto Supabase |
| `SUPABASE_ANON_KEY` | Todos menos `api-gateway` | Clave pública (`anon` / publishable) |
| `SUPABASE_SECRET_KEY` | `dataset`, `cleaning`, `analysis-service` | Clave de servicio (service role) |
| `SUPABASE_SERVICE_ROLE_KEY` | `auth` | Clave de servicio para administrar usuarios |
| `SUPABASE_DATASETS_TABLE` | `dataset`, `cleaning`, `analysis-service` | Tabla de datasets (por defecto `datasets`) |
| `CHROMA_HOST` / `CHROMA_PORT` | `dataset`, `cleaning`, `analysis-service`, `rag` | Host y puerto de Chroma. En Docker: `chroma-service` / `8000`. `classification` y `evolution` los tienen hardcodeados |
| `CHROMA_COLLECTION` | `classification`, `cleaning`, `analysis-service`, `rag` | Colección de documentos (`documents`) |
| `CHROMA_COLLECTION_NAME` | `dataset`, `analysis-service` | Nombre de la colección de documentos (equivalente a `CHROMA_COLLECTION`) |
| `CHROMA_CHUNKED_COLLECTION` | `analysis-service` | Colección de chunks de chat (`chunked_chats`) |
| `CHROMA_CHUNKED_COLLECTION_NAME` | `dataset` | Colección de chunks de chat (`chunked_chats`) |
| `CHROMA_COLLECTION_DOCUMENTS` / `CHROMA_COLLECTION_CHUNKED` | `rag` | Colecciones auxiliares usadas por el chatbot del dataset |
| `CHROMA_PAGE_SIZE` | `cleaning`, `analysis-service` | Tamaño de página al leer Chroma (`2000`) |
| `CHROMA_TOKEN` | `dataset`, `analysis-service` | Token opcional de Chroma. `analysis-service` lo declara para topic modeling aunque el cliente actual no lo envía |
| `CHROMA_FALLBACK_LOCAL` | `rag` | Si es `true`, usa Chroma persistente local cuando no hay servidor |
| `HF_TOKEN` | `dataset`, `analysis-service`, `evolution`, `classification`, `rag` | Token de Hugging Face para descargar modelos. `evolution` también acepta `HUGGING_FACE_HUB_TOKEN` |
| `TOKENIZERS_PARALLELISM` | `classification` | Desactiva el paralelismo de tokenizers (`false`) |
| `CORS_ORIGINS` | `dataset` | Orígenes permitidos, separados por coma. Es obligatoria en ese servicio |
| `LOG_LEVEL` | Todos | Nivel de logging (`INFO` por defecto) |

> Las claves de servicio otorgan privilegios elevados. Usalas solo en los servicios que las necesitan y nunca las expongas en el frontend.

### 3. Caché de modelos y VRAM

`model_cache.py` está copiado en ocho servicios, pero **no todos instancian `ModelSlot`**. Según el código, estas son las variables que tienen efecto real:

| Variable | Servicios donde tiene efecto | Default | Qué controla |
| --- | --- | --- | --- |
| `WARMUP` | `analysis-service`, `evolution`, `rag` | `true` | Precarga de modelos al iniciar. Con `false` la carga queda lazy |
| `MODEL_TTL_SECONDS` | `dataset`, `analysis-service`, `evolution`, `rag` | `600` | Segundos de inactividad antes de descargar un `ModelSlot` |
| `MODEL_VRAM_MARGIN_MB` | `dataset`, `analysis-service`, `evolution`, `rag` | `512` | VRAM adicional que se exige libre antes de cargar un modelo |
| `MODEL_EVICT_POLL_SECONDS` | `dataset`, `analysis-service`, `evolution`, `rag` | `0` (automático) | Cada cuántos segundos el hilo de evicción revisa si puede liberar el modelo |

Detalles que no son obvios:

- Los módulos `clustering` y `topicmodeling` solo llaman a `check_fits_or_raise()`, por lo que únicamente les afecta `MODEL_VRAM_MARGIN_MB`; no instancian `ModelSlot` y por eso `WARMUP`, `MODEL_TTL_SECONDS` y `MODEL_EVICT_POLL_SECONDS` no hacen nada en ellos.
- `classification` tiene una copia de `model_cache.py` pero no la importa. Su `MODEL_TTL_HOURS` es distinto: controla cuántas horas se conservan los ZIP de modelos entrenados en disco.
- `dataset` usa `ModelSlot` para el modelo de embeddings, pero no llama a `warmup_enabled()`: su `WARMUP` no tiene efecto.
- `risk` usa `ModelSlot` solo para el verificador RoBERTa; su `WARMUP` tampoco se lee.

### 4. Variables por servicio

Se listan solo las variables propias; las compartidas de las tablas anteriores también aplican.

| Servicio | Variables propias |
| --- | --- |
| `api-gateway` | `CLUSTERING_SERVICE_URL`, `DATASET_SERVICE_URL`, `TOPIC_MODELING_SERVICE_URL`, `WORDCLOUD_SERVICE_URL`, `EDA_SERVICE_URL`, `EMOTIONS_SERVICE_URL`, `CLASSIFICATION_SERVICE_URL`, `RAG_SERVICE_URL`, `DATA_CLEANING_SERVICE_URL`, `EVOLUTION_SERVICE_URL`, `SUICIDE_SERVICE_URL`, `REPORT_SERVICE_URL`, `AUTH_SERVICE_URL` |
| `auth-service` | `SUPABASE_SERVICE_ROLE_KEY` |
| `dataset-service` | `APP_NAME`, `APP_VERSION`, `MAX_UPLOAD_SIZE_MB=50`, `INTERNAL_SERVICE_TOKEN`, `EMBEDDING_PROVIDER=local`, `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE=64`, `EMBEDDING_DEVICE=auto`, `EMBEDDING_FP16=true`, `EMBEDDING_VRAM_MB=1500`, `INGEST_CHUNK_SIZE=2000`, `CHAT_CHUNK_SIZE=20`, `CHAT_CHUNK_OVERLAP=4`, `CORS_ORIGINS`, `CHROMA_CHUNKED_COLLECTION_NAME`, `MODEL_TTL_SECONDS`, `MODEL_VRAM_MARGIN_MB`, `MODEL_EVICT_POLL_SECONDS` |
| `cleaning-service` | `CLEANING_DOWNLOAD_PATH=./data/temp_downloads` |
| `analysis-service` | `WARMUP`, `MODEL_TTL_SECONDS`, `MODEL_VRAM_MARGIN_MB`, `MODEL_EVICT_POLL_SECONDS`, `VERIFY_MODEL_ID`, `VERIFY_VRAM_MB`, `GOOGLE_API_KEY`, `LLM_MODEL`, `TITLE_*`, `CHROMA_*` |
| `evolution-service` | `WARMUP`, `MODEL_TTL_SECONDS`, `MODEL_VRAM_MARGIN_MB`, `MODEL_EVICT_POLL_SECONDS`, `HUGGING_FACE_HUB_TOKEN`. Chroma está hardcodeado a `chroma-service:8000` / `documents` |
| `classification-service` | `CHROMA_COLLECTION=documents`, `EMBEDDING_MODEL_NAME`, `MAX_CLASES=7`, `MODEL_STORAGE_PATH=/app/data/modelos`, `MODEL_TTL_HOURS=24`, `CLASSIFICATION_FORCE_DEVICE` (vacío = auto), `TOKENIZERS_PARALLELISM=false` |
| `rag-service` | `CHROMA_COLLECTION=rag_docs`, `CHROMA_COLLECTION_DOCUMENTS`, `CHROMA_COLLECTION_CHUNKED`, `CHROMA_FALLBACK_LOCAL=false`, `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE=32`, `RAG_EMB_VRAM_MB=800`, `DOCS_PATH=/app/data/rag_docs`, `CHUNK_SIZE=2000`, `CHUNK_OVERLAP=200`, `RAG_INIT_ON_STARTUP=false`, `TOP_K=4`, `RETRIEVE_K=12`, `RERANK=true`, `RERANK_MODEL`, `RERANK_MIN_SCORE=-2.0`, `HISTORY_TAIL=4`, `HISTORY_MSG_CHARS=600`, `HISTORY_MAX_CHARS=2000`, `LLM_PROVIDER`, `GOOGLE_API_KEY`, `LLM_MODEL`, `GROQ_API_KEY`, `GROQ_MODEL`, `WARMUP`, `MODEL_TTL_SECONDS`, `MODEL_VRAM_MARGIN_MB`, `MODEL_EVICT_POLL_SECONDS` |
| `report-service` | Sin variables propias; usa Supabase |

El detalle completo de cada variable está documentado en el `.env.example` de cada carpeta; las variables de los
análisis agrupados están en `analysis-service/.env.example`.

## Ejecución con Docker

El `compose.yaml` de la raíz de este repositorio levanta todos los microservicios, ChromaDB y la red `microservicios`. El frontend es un proyecto separado y no forma parte de este compose.

### 1. Construir la imagen base de ML

Los servicios `dataset-service`, `classification-service`, `rag-service` y `analysis-service` usan `FROM proyecto/ml-base`, por lo que la imagen debe existir antes de construir esos servicios:

```bash
docker build -f ml-base/Dockerfile -t proyecto/ml-base ml-base
```

Alcanza con construirla una vez; Compose reutiliza la imagen y solo la reconstruye si cambiás el `Dockerfile` de `ml-base`.

### 2. Levantar el sistema

Desde la raíz del repositorio:

```bash
docker compose up -d --build
```

Servicios expuestos al host:

- API Gateway: <http://localhost:7500>
- Chroma: solo interno (`chroma-service:8000`)

El resto de los microservicios quedan en la red interna y no publican puertos al host.

### 3. Operación

```bash
docker compose ps
docker compose logs -f api-gateway
docker compose logs -f dataset-service
docker compose down
```

### Volúmenes

Rutas relativas a la raíz de este repositorio:

| Ruta local | Uso |
| --- | --- |
| `data/chroma-data` | Persistencia de ChromaDB |
| `data/hf_cache` | Caché de modelos de Hugging Face (compartida por los servicios de ML) |
| `data/modelos` | Clasificadores entrenados por `classification-service` |
| `data/rag_docs` | Documentos fuente del RAG (solo lectura) |
| `data/temp_downloads` | Archivos temporales de limpieza |

Compose crea las carpetas si no existen y `data/` está ignorado por Git. Antes de usar el RAG, colocá los documentos a indexar en `data/rag_docs`.

## Desarrollo local de un servicio

```bash
cd dataset-service
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8082
```

Cada servicio carga su `.env` desde su propio directorio (`pydantic-settings` o `os.environ`). Para desarrollo local, apuntá `CHROMA_HOST=localhost`, `CHROMA_PORT=8000` y publicá el puerto de Chroma o levantalo aparte.

## API Gateway y autenticación

- `POST /auth/login` es el único endpoint público: delega en `auth-service` y propaga la cookie `access_token`.
- El resto de rutas requiere un JWT válido, enviado como cookie `access_token` o como `Authorization: Bearer <token>`.
- La firma se valida contra el JWKS de Supabase (`{SUPABASE_URL}/auth/v1/.well-known/jwks.json`).
- El gateway agrega `X-User-Token`, `X-User-ID` y `X-Request-ID` antes de reenviar la petición.
- Las rutas `/auth/admin/*` verifican además que el usuario tenga `app_metadata.role = "admin"`.
- Los servicios no reciben tráfico directo en producción: solo el gateway publica el puerto 7500.

### Formato de rutas

```text
/{servicio}/{operación}       -> {SERVICE_URL}/{operación}
/{servicio}/jobs              -> listado de jobs del usuario
/{servicio}/jobs/{job_id}     -> estado de un job
/{servicio}/descargar/{id}    -> descarga de resultados
```

El gateway centraliza el mapa explícito de servicios en `api-gateway/config.py`.
Las URLs pueden incluir una barra final, que se normaliza al unirlas con las rutas.
Cada URL se valida al resolver el servicio; si falta o no usa `http`/`https` con
un host válido, la petición se rechaza con un error de configuración en lugar de
intentar construir una URL inválida. Los prefijos agrupados de `analysis-service`
mantienen sus URLs configuradas individualmente.

### Endpoints por servicio

| Servicio | Endpoints |
| --- | --- |
| `auth` | `POST /login`, `GET /me`, `POST /logout`, CRUD `/admin/users` |
| `dataset` | `POST /uploadDataset`, `GET /getDatasets`, `DELETE /deleteDataset`, `DELETE /deleteDatasetsByUser`, `GET /jobs` |
| `datacleaning` | `POST /cleanDataset`, `POST /download`, `GET /jobs` |
| `eda` | `POST /eda`, `POST /getResults`, `GET /jobs` |
| `clustering` | `POST /visualizeEmbeddings`, `POST /getEmbeddingsImage`, `GET /jobs` |
| `wordcloud` | `POST /visualizeWordCloud`, `POST /getWordCloudImage`, `GET /jobs` |
| `emotions` | `POST /analyzeEmotions`, `POST /getResults`, `GET /jobs` |
| `evolution` | `POST /users`, `POST /word-evolution`, `POST /emotions-evolution`, `GET /jobs` |
| `topicmodeling` | `POST /getTopics`, `POST /getTopicResults`, `GET /jobs` |
| `classification` | `POST /detectar_menciones`, `GET /descargar_modelo`, `GET /jobs` |
| `suicide` | `POST /analyzeRisk`, `POST /getResults`, `GET /jobs` |
| `rag` | `POST /chat`, `POST /message`, `GET /chat`, `GET /chats`, `DELETE /chat`, `DELETE /chats`, `POST /activate_chat` |
| `report` | `POST /datos-graficos`, `GET /disponibles`, `POST /generar`, `GET /descargar/{job_id}`, `GET /jobs` |

## Jobs asíncronos

Los análisis pesados no bloquean la request. El runtime agrupado usa helpers compartidos para persistir jobs y configurar logging, sin cambiar sus contratos:

1. El servicio crea un registro en la tabla `jobs` de Supabase (`status = queued`).
2. Responde `202 Accepted` con `{ "job_id": "...", "status": "queued" }`.
3. El trabajo corre en `BackgroundTasks` y actualiza el estado a `running`, `succeeded` o `failed`.
4. El frontend consulta `GET /<servicio>/jobs/<job_id>` hasta obtener un estado final.

Los jobs sin actualización durante **30 minutos** se consideran obsoletos. Cada fila guarda `user_id`, `service`, `job_type`, `dataset_name`, `params`, `result`, `error_code` y `error_detail`, lo que permite listar el historial por usuario.

## Base de datos

### Esquema base

`_schemas/schema_public.sql` es un dump **solo de esquema** del proyecto Supabase remoto. Contiene las 12 tablas de `public`, funciones de trigger, índices, RLS, políticas y grants. No contiene filas, usuarios de Auth ni objetos de Storage.

Para un proyecto nuevo, usalo como baseline:

```powershell
$env:SUPABASE_DB_URL = "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
psql $env:SUPABASE_DB_URL -f _schemas/schema_public.sql
```

También podés pegar el contenido en el SQL Editor de Supabase.

> El dump está pensado para Supabase: usa `auth.uid()`, los roles `anon`, `authenticated` y `service_role`, y dueños como `postgres`. No lo apliques tal cual sobre un PostgreSQL sin el esquema `auth` de Supabase.

Si el dump ya refleja el estado actual, usalo como baseline. No lo apliques dos veces sobre el mismo proyecto: las políticas y los `ALTER TABLE` no son idempotentes.

### Regenerar el dump

Desde la raíz del repositorio, con Docker corriendo y el proyecto vinculado. Requiere la Supabase CLI (`npm install -g supabase`); `supabase link` pide la contraseña de la base, no la service role key.

```powershell
supabase login
supabase link --project-ref <project-ref>
supabase db dump --linked --schema public --keep-comments -f _schemas/schema_public.sql
```

`db dump` exporta solo el esquema por defecto. Si necesitás datos aparte, usá `--data-only` y no commitees ese archivo.

### Almacenamiento

- **Supabase Postgres**: usuarios, datasets, jobs y resultados agregados.
- **Supabase Storage**: archivos originales de datasets e informes.
- **ChromaDB**: colecciones `documents` (documentos), `chunked_chats` (fragmentos para RAG por chat) y `rag_docs` (documentación del asistente general).

## Modelos y GPU

Los servicios de ML tienen una copia de `model_cache.py`. El comportamiento efectivo depende de cómo lo use cada uno:

- `dataset`, los módulos de `analysis-service`, `evolution` y `rag` instancian `ModelSlot`: carga perezosa, control de admisión por VRAM, TTL y evicción.
- Los módulos `clustering` y `topicmodeling` de `analysis-service` solo usan `check_fits_or_raise()`: verifican VRAM antes de ejecutar, pero no mantienen modelos cacheados.
- `classification` no importa `model_cache.py`.

Variables y alcance exacto: ver [Caché de modelos y VRAM](#3-caché-de-modelos-y-vram). En resumen:

- Evicción automática tras `MODEL_TTL_SECONDS` (600 s por defecto) sin uso en `dataset`, `analysis-service`, `evolution` y `rag`.
- Margen de seguridad de `MODEL_VRAM_MARGIN_MB` (512 MB por defecto) en `dataset`, `analysis-service`, `evolution` y `rag`.
- `WARMUP=false` para evitar precargar modelos al iniciar en `analysis-service`, `evolution` y `rag`.
- Respuesta **HTTP 503** (`ModelBusyError`) cuando no hay VRAM suficiente para cargar un modelo.

Si no hay GPU, los servicios funcionan en CPU. Algunos permiten forzar el dispositivo con `CLASSIFICATION_FORCE_DEVICE` o `EMBEDDING_DEVICE`.

Modelos principales:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/multilingual-e5-large`.
- Emociones: `pysentimiento/robertuito-base-uncased-emotion` (ES), `j-hartmann/emotion-english-distilroberta-base` (EN).
- Riesgo: modelo propio (`analysis-service/risk/model_artifacts/risk_model.pkl`) + verificación con `vibhorag101/roberta-base-suicide-prediction-phr`.
- Tópicos: BERTopic + títulos con Gemini.
- RAG: embeddings multilingües, reranking con cross-encoder y LLM Gemini/Groq.

## Observabilidad

- Cada servicio registra método, ruta, estado, latencia y `request_id` en formato estructurado.
- El gateway propaga `X-Request-ID` y lo devuelve en la respuesta.
- Los errores no controlados se registran con stacktrace y se mapean a respuestas HTTP claras (`502`, `503`, `504`).

## Seguridad

- No commitear archivos `.env` (están en `.gitignore`); solo se versionan los `.env.example` con valores de ejemplo.
- Rotar periódicamente las claves de Supabase, Hugging Face, Gemini y Groq.
- La `service role key` solo debe estar en `auth-service`, `dataset-service`, `cleaning-service` y `analysis-service`.
- En producción, servir el gateway detrás de HTTPS y no exponer los microservicios internos.
- Las cookies usan `httpOnly` y `samesite=lax`; para HTTPS conviene habilitar `secure`.

## Solución de problemas

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `pull access denied for proyecto/ml-base` | Falta construir la imagen base | Ejecutar el `docker build` de `ml-base` |
| `503 Servicio no disponible` | Microservicio caído o nombre DNS incorrecto | Revisar `docker compose ps` y los `*_SERVICE_URL` |
| `401 No autenticado` | Cookie/JWT ausente o vencido | Volver a iniciar sesión |
| `403 Sin permiso` en `/auth/admin/*` | El usuario no tiene rol `admin` | Asignar `app_metadata.role = "admin"` en Supabase |
| `CORS_ORIGINS` vacío en dataset | Falta la variable en el `.env` | Definir `CORS_ORIGINS=http://localhost:8080,http://localhost:7500` |
| `503` al cargar un modelo | VRAM insuficiente | Liberar GPU, reducir modelos concurrentes o usar CPU |
| Descargas de modelos lentas | Caché de Hugging Face vacía | Montar `data/hf_cache` y definir `HF_TOKEN` |

## Licencia

Proyecto académico desarrollado para el Trabajo Final de la carrera (UNSL). Consultá con los autores antes de reutilizar el código o los modelos.
