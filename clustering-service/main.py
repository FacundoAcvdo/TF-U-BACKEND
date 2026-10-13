import time
import uuid
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI, HTTPException, Header, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import asyncio

import jobs

from supabase_service import get_supabase_client, get_dataset_info, save_clustering_results, get_clustering_results
from chromadb_service import get_dataset_embeddings

from cluster_service import get_cluster_and_outlier_dataframes
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("clustering-service")


@asynccontextmanager
async def lifespan(app: FastAPI):

    pass
    try:
        import numpy as np

        try:
            import umap
            import hdbscan
            pass
        except Exception as e:
            logger.exception(f"[clustering-service] warmup pre-import fallo: {e}")

        try:
            dummy = np.random.rand(120, 384).astype(np.float32)
            await asyncio.to_thread(get_cluster_and_outlier_dataframes, dummy)
            pass
        except Exception as e:
            logger.warning(f"[clustering-service] warmup dummy clustering fallo (no crítico): {e}")
    except Exception as e:
        logger.exception(f"[clustering-service] warmup error: {e}")
    yield
    pass

app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusterRequest(BaseModel):
    dataset: str


def build_response_data(clusters_df, outliers_df, max_points: int = 8000):
    total = len(clusters_df) + len(outliers_df)


    if total > max_points:

        import pandas as pd
        full_df = pd.concat([clusters_df, outliers_df], ignore_index=True)

        sampled = full_df.sample(n=max_points, random_state=42)

        clusters_sampled = sampled[sampled["cluster"] != "-1"]
        outliers_sampled = sampled[sampled["cluster"] == "-1"]
        clusters_df = clusters_sampled
        outliers_df = outliers_sampled


    points = []
    if not clusters_df.empty:
        for _, row in clusters_df.iterrows():
            points.append({"x": float(row["x"]), "y": float(row["y"]), "cluster": str(row["cluster"])})
    if not outliers_df.empty:
        for _, row in outliers_df.iterrows():
            points.append({"x": float(row["x"]), "y": float(row["y"]), "cluster": "-1"})


    clusters_summary = []
    if not clusters_df.empty:
        counts = clusters_df["cluster"].value_counts().to_dict()
        for cid in sorted(counts.keys(), key=lambda x: int(x)):
            clusters_summary.append({"id": str(cid), "count": int(counts[cid])})

    return {
        "points": points,
        "clusters": clusters_summary,
        "total": int(total),
        "displayed": int(len(points)),
        "outliers": int(len(outliers_df)),
        "truncated": bool(total > max_points),
    }


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


@app.post("/visualizeEmbeddings", status_code=202)
async def start_visualize_embeddings(
    data: ClusterRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola el clustering. Devuelve 202 {job_id}; el resultado se consulta
    con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="clustering", job_type="embeddings",
            user_id=x_user_id, dataset_name=data.dataset, params={},
        )
    except Exception as e:
        logger.exception(f"[clustering] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_cluster_job, job["job_id"], data.model_dump(mode="json"), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


async def _run_cluster_job(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_visualize_embeddings(ClusterRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[clustering] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[clustering] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
        jobs.mark_failed(supabase, job_id, 500, "Error interno del servidor.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
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
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Trabajos activos del usuario (para retomar al volver a la pantalla)."""
    supabase = get_supabase_client(x_user_token)
    return {"jobs": jobs.list_jobs(supabase, x_user_id, dataset_name, job_type, active_only=True)}


async def _compute_visualize_embeddings(
    data: ClusterRequest,
    x_user_id: str,
    x_user_token: str,
):
    try:
        supabase = get_supabase_client(x_user_token)

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)

        embeddings = await get_dataset_embeddings(dataset_info.get("dataset_id"), x_user_id)

        if embeddings is None or len(embeddings) == 0:
            raise HTTPException(status_code=404, detail="No embeddings found for dataset")

        clusters_df, outliers_df = await asyncio.to_thread(
            get_cluster_and_outlier_dataframes, embeddings
        )

        result = build_response_data(clusters_df, outliers_df)

        try:
            save_clustering_results(
                supabase, x_user_id, dataset_info.get("dataset_id"),
                result["points"], result["clusters"], result["total"], result["displayed"], result["outliers"], result["truncated"]
            )
        except Exception as e:
            logger.warning(f"[clustering-service] warn save_clustering_results: {e}")

        return {
            "dataset_id": dataset_info.get("dataset_id"),
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error al procesar embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar embeddings.")


@app.post("/getEmbeddingsImage")
async def get_clustering_image(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
    data: ClusterRequest = None,
):
    try:
        if data is None or not data.dataset:
            raise HTTPException(status_code=400, detail="Dataset is required")
        supabase = get_supabase_client(x_user_token)
        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info.get("dataset_id")
        data_db = get_clustering_results(supabase, x_user_id, dataset_id)
        return {"dataset_id": dataset_id, **data_db}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error al obtener embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al obtener embeddings.")
