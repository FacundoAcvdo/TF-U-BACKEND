from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import asyncio

from supabase_service import get_supabase_client, get_dataset_info, save_clustering_results, get_clustering_results
from chromadb_service import get_dataset_embeddings

from cluster_service import get_cluster_and_outlier_dataframes

###########

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup eager: pre-imports y compilación Numba/UMAP para no penalizar primera petición
    print("[clustering-service] warmup start - precargando UMAP/HDBSCAN", flush=True)
    try:
        import numpy as np  # noqa
        # 1) pre-imports pesados (evita import lazy dentro de get_cluster_and_outlier_dataframes en CPU)
        try:
            import umap  # noqa: F401
            import hdbscan  # noqa: F401
            print("[clustering-service] warmup pre-import umap/hdbscan OK", flush=True)
        except Exception as e:
            print(f"[clustering-service] warmup pre-import fallo: {e}", flush=True)
        # 2) dummy clustering para compilar Numba y loguear backend GPU/CPU en startup, no en primera request
        try:
            dummy = np.random.rand(120, 384).astype(np.float32)
            await asyncio.to_thread(get_cluster_and_outlier_dataframes, dummy)
            print("[clustering-service] warmup dummy clustering done (Numba compilado)", flush=True)
        except Exception as e:
            print(f"[clustering-service] warmup dummy clustering fallo (no crítico): {e}", flush=True)
    except Exception as e:
        print(f"[clustering-service] warmup error: {e}", flush=True)
    yield
    print("[clustering-service] shutdown", flush=True)

app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############

class ClusterRequest(BaseModel):
    dataset: str


def build_response_data(clusters_df, outliers_df, max_points: int = 8000):
    total = len(clusters_df) + len(outliers_df)

    # downsample si supera max_points (muestreo uniforme para mantener forma)
    if total > max_points:
        # concatenar y samplear proporcionalmente
        import pandas as pd
        full_df = pd.concat([clusters_df, outliers_df], ignore_index=True)
        # sample aleatorio con seed fijo para reproducibilidad
        sampled = full_df.sample(n=max_points, random_state=42)
        # separar de nuevo sampled por cluster
        clusters_sampled = sampled[sampled["cluster"] != "-1"]
        outliers_sampled = sampled[sampled["cluster"] == "-1"]
        clusters_df = clusters_sampled
        outliers_df = outliers_sampled

    # puntos para front: lista de {x,y,cluster}
    points = []
    if not clusters_df.empty:
        for _, row in clusters_df.iterrows():
            points.append({"x": float(row["x"]), "y": float(row["y"]), "cluster": str(row["cluster"])})
    if not outliers_df.empty:
        for _, row in outliers_df.iterrows():
            points.append({"x": float(row["x"]), "y": float(row["y"]), "cluster": "-1"})

    # resumen por cluster
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


@app.post("/visualizeEmbeddings")
async def visualize_embeddings(
    data: ClusterRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
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
            print(f"[clustering-service] warn save_clustering_results: {e}")

        return {
            "dataset_id": dataset_info.get("dataset_id"),
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al procesar embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar embeddings: {str(e)}")


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
        print(f"Error al obtener embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener embeddings: {str(e)}")
