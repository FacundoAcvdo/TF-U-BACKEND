import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core import get_collection, get_supabase_client
from data import (
    MODEL_ID_RE,
    MODEL_STORAGE_PATH,
    guardar_modelo_en_disco,
    leer_nombre_archivo,
    limpiar_modelos_antiguos,
    obtener_datos_dataset,
)
from schemas import ResumenMenciones, TrainingRequest, TrainingResponse
from training import (
    entrenar_bert,
    entrenar_sklearn,
    procesar_etiquetas_automatico,
    validar_tamano_muestras,
)
import logging
from logging_utils import setup_service_logger
import jobs

logger = setup_service_logger("classification-service")


app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        n = limpiar_modelos_antiguos()
        if n:
            logger.info(f"[classification] limpieza inicial: {n} modelo(s) de +24h borrados")
    except Exception as e:
        logger.warning(f"[classification] limpieza inicial omitida: {e}")
    yield


app.router.lifespan_context = lifespan

origins = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        user_id = request.headers.get("x-user-id", "-")
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latencia_ms={latencia_ms} "
            f"req={request_id} user_id={user_id}"
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.exception(
            f"fallo no controlado {request.method} {request.url.path} "
            f"req={request_id} latencia_ms={latencia_ms} causa={type(e).__name__}: {e}"
        )
        raise


@app.post("/detectar_menciones", status_code=202)
async def start_detectar_menciones(
    data: TrainingRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    """Encola el entrenamiento/deteccion. Devuelve 202 {job_id}; el resultado
    se consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="classification", job_type="classification",
            user_id=x_user_id, dataset_name=data.dataset,
            params={"modelo": data.modelo},
        )
    except Exception as e:
        logger.exception(f"[classification] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_classification_job, job["job_id"], jsonable_encoder(data), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


async def _run_classification_job(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_detectar_menciones(TrainingRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[classification] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[classification] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
        jobs.mark_failed(supabase, job_id, 500, "Error interno del servidor.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    supabase = get_supabase_client(x_user_token)
    row = jobs.get_job(supabase, job_id, x_user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return row


@app.get("/jobs")
async def list_active_jobs(
    dataset_name: str | None = None,
    job_type: str | None = None,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    """Trabajos activos del usuario (para retomar al volver a la pantalla)."""
    supabase = get_supabase_client(x_user_token)
    return {"jobs": jobs.list_jobs(supabase, x_user_id, dataset_name, job_type, active_only=True)}


async def _compute_detectar_menciones(
    data: TrainingRequest,
    x_user_id: str,
    x_user_token: str,
):
    try:
        try:
            limpiar_modelos_antiguos()
        except Exception:
            pass
        supabase = get_supabase_client(x_user_token)
        response = (
            supabase.table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{data.dataset}' no encontrado o sin permisos.",
            )
        dataset_id = response.data[0].get("dataset_id")

        collection = get_collection()

        filtros = {
            "$and": [
                {"user_id": x_user_id},
                {"dataset_id": dataset_id},
            ]
        }

        es_bert = data.modelo == "BERT"

        clases, documentos, x_vectors = await asyncio.to_thread(
            obtener_datos_dataset,
            collection,
            filtros,
            not es_bert,
            es_bert,
        )

        y_labels, label_mapping, clases_nombres = procesar_etiquetas_automatico(clases)
        validar_tamano_muestras(y_labels, data)

        if es_bert:
            if not documentos:
                raise HTTPException(status_code=400, detail="No se encontraron textos para entrenar BERT.")
            resultado = entrenar_bert(documentos, y_labels, label_mapping, clases_nombres, data)
        else:
            if x_vectors is None or len(x_vectors) == 0:
                raise HTTPException(status_code=400, detail="No se encontraron embeddings.")
            if len(x_vectors) != len(y_labels):
                raise HTTPException(
                    status_code=500,
                    detail="Inconsistencia entre embeddings y etiquetas recuperados de ChromaDB.",
                )
            resultado = entrenar_sklearn(x_vectors, y_labels, label_mapping, clases_nombres, data)

        (
            tp,
            fp,
            tn,
            fn,
            acc,
            f1,
            prec,
            rec,
            total_muestras,
            muestras_train,
            muestras_test,
            zip_buffer,
            matriz,
        ) = resultado

        file_name = f"{data.modelo.lower().replace(' ', '_')}_modelo.zip"
        model_id = guardar_modelo_en_disco(zip_buffer, file_name)

        return TrainingResponse(
            resumen=ResumenMenciones(
                truePositive=int(tp),
                falsePositive=int(fp),
                trueNegative=int(tn),
                falseNegative=int(fn),
                accuracy=acc,
                f1_score=f1,
                precision=prec,
                recall=rec,
                total_muestras=total_muestras,
                muestras_train=muestras_train,
                muestras_test=muestras_test,
                matriz_confusion=matriz,
                clases=clases_nombres,
            ),
            model_id=model_id,
            nombre_archivo=file_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[classification] fallo en detectar_menciones | causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.get("/descargar_modelo")
async def descargar_modelo(model_id: str):
    if not MODEL_ID_RE.match(model_id or ""):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    ruta_zip = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.zip")
    if not os.path.isfile(ruta_zip):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    return FileResponse(
        path=ruta_zip,
        media_type="application/zip",
        filename=leer_nombre_archivo(model_id),
    )
