from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from supabase_service import get_supabase_client, get_storage_client, get_dataset_info, upload_image_to_bucket, upload_image_info, get_cluster_image
from chromadb_service import get_dataset_embeddings

from cluster_service import get_cluster_and_outlier_dataframes
from plot_service import plot_clusters

###########

app = FastAPI()

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


@app.post("/visualizeEmbeddings")
async def main(
    data: ClusterRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        storage = get_storage_client(x_user_token)

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)

        embeddings = get_dataset_embeddings(dataset_info.get("dataset_id"), x_user_id)

        clusters_df, outliers_df = get_cluster_and_outlier_dataframes(embeddings)

        imagen_bytes = plot_clusters(clusters_df, outliers_df)

        image_name = upload_image_to_bucket(storage, imagen_bytes, x_user_id)

        image_info = upload_image_info(supabase, image_name, x_user_id, dataset_info.get("dataset_id"))

        return image_info

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

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)

        dataset_id = dataset_info.get("dataset_id")

        public_url = get_cluster_image(supabase, x_user_id, dataset_id)

        return {"public_url": public_url}

    except Exception as e:
        print(f"Error al obtener la imagen de clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener la imagen de clustering: {str(e)}")