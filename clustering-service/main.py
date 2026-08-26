from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
import matplotlib.pyplot as plt
from pydantic import BaseModel
from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd
import matplotlib
import uuid
import io
import os

from supabase import create_client
from storage3 import create_client as create_storage_client

import chromadb
import numpy as np

matplotlib.use('Agg') 

class ClusterRequest(BaseModel):
    dataset: str

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client

def get_storage_client(user_jwt: str):
    storage_url = f"{SUPABASE_URL}/storage/v1"
    headers = {
        "apiKey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {user_jwt}",
    }
    return create_storage_client(storage_url, headers, is_async=False)

@app.post("/visualizeEmbeddings")
async def main(
    data: ClusterRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        storage = get_storage_client(x_user_token)

        response = (
            supabase
            .table("datasets")
            .select("*")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

        dataset_info = response.data[0] if response.data else None

        if not response.data:
            raise HTTPException(status_code=404, detail="Dataset not found or access denied.")

        client = chromadb.HttpClient(host="chroma-service", port=8000)
        collection = client.get_collection(name="documents")

        resultados = collection.get(
            where= {
                "$and": [
                    {"user_id": x_user_id},
                    {"dataset_id": dataset_info.get("dataset_id") }
                ]
            },
            include=["embeddings"]
        )

        embeddings = np.array(resultados["embeddings"], dtype=np.float32)

        umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom').fit(reduced_embeddings)
        clusters = hdbscan_model.labels_

        reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

        df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
        df["cluster"] = [str(c) for c in clusters]

        clusters_df = df.loc[df.cluster != "-1", :]
        outliers_df = df.loc[df.cluster == "-1", :]

        plt.figure(figsize=(10, 8))
        plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
        plt.scatter(
            clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int),
            alpha=0.6, s=2, cmap='tab20b'
        )
        plt.axis('off')

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buffer.seek(0)
        imagen_bytes = buffer.read()

        image_name = f"{x_user_id}_{uuid.uuid4()}.png"

        storage.from_("imagenes-generadas").upload(
            path=image_name,
            file=imagen_bytes,
            file_options={"content-type": "image/png"}
        )

        #public_url = storage.from_("imagenes-generadas").get_public_url(image_name) - Se debería activar en producción
        public_url = f"http://127.0.0.1:54321/storage/v1/object/public/imagenes-generadas/{image_name}"

        supabase.table("imagenes-generadas").insert({
            "user_id": x_user_id,
            "dataset_id": dataset_info.get("dataset_id"),
            "function_name": "clustering",
            "public_url": public_url,
            "image_path": image_name,
        }).execute()

        return {"public_url": public_url}

    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        plt.close('all')
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

@app.post("/getEmbeddingsImage")
async def get_clustering_image(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
    data: ClusterRequest = None
):
    try:
        supabase = get_supabase_client(x_user_token)

        response = (
            supabase
            .table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Dataset not found for the user.")

        dataset_id = response.data[0].get("dataset_id")

        response = (
            supabase
            .table("imagenes-generadas")
            .select("public_url")
            .eq("user_id", x_user_id)
            .eq("function_name", "clustering")
            .eq("dataset_id", dataset_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="No clustering image found for the user.")

        image_info = response.data[0]
        public_url = image_info.get("public_url")

        return {"public_url": public_url}

    except Exception as e:
        print(f"Error al obtener la imagen de clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener la imagen de clustering: {str(e)}")