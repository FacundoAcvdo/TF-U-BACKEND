from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header
from pydantic import BaseModel
import os

from stats_service import (
    get_dataset_row_analysis,
    get_estadisticas_corpus,
    get_estadisticas_clases
)

from plot_service import (
    generate_sentence_distribution_plot,
    generate_class_distribution_plot,
    generate_distribution_plot
)

from supabase_service import (
    get_supabase_client,
    get_storage_client,
    get_dataset_info,
    upload_images,
    save_image_info_to_db,
    save_eda_results_to_db,
    get_imagenes_generadas_eda,
    get_class_stats,
    get_corpus_stats
)

from utils import parse_dataset_json

###############

DATASET_STORAGE_PATH = os.environ.get("DATASET_STORAGE_PATH", "./data/datasets")

###############

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############

class EdaRequest(BaseModel):
    dataset: str

@app.post("/eda")
async def main(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)
    storage = get_storage_client(x_user_token)

    dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)

    dataset_id = dataset_info["dataset_id"]

    dataset_path = os.path.join(
        DATASET_STORAGE_PATH, "users", x_user_id, f"{dataset_id}.json"
    )
    
    dataset = parse_dataset_json(dataset_path)

    # Análisis de las filas del dataset
    dataset = get_dataset_row_analysis(dataset)
    print(dataset.head())

    # Estadísticas generales del corpus
    DF_CORPUS = get_estadisticas_corpus(dataset)
    print(DF_CORPUS.head())

    # Crear un DataFrame para almacenar las estadísticas por clase
    DF_CLASES = get_estadisticas_clases(dataset)
    print(DF_CLASES.head())

    # Generar las imágenes en buffer
    imagenes = []
    imagenes += generate_sentence_distribution_plot(dataset)
    imagenes += generate_class_distribution_plot(DF_CLASES)
    imagenes += generate_distribution_plot(dataset)

    # Subir las imágenes al bucket
    image_public_url, image_names = upload_images(storage, imagenes, x_user_id)

    # Guardar informacion de imagenes en tabla
    save_image_info_to_db(supabase, x_user_id, dataset_id, image_public_url, image_names)

    # Guardar resultados de EDA en tabla
    save_eda_results_to_db(supabase, x_user_id, dataset_id, DF_CORPUS, DF_CLASES)

    return {
        "corpus_stats": DF_CORPUS.to_dict(orient="records")[0],
        "class_stats": DF_CLASES.to_dict(orient="records"),
        "images": image_public_url,
    }

@app.post("/getResults")
async def get_results(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)

    dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
    
    dataset_id = dataset_info["dataset_id"]

    images = get_imagenes_generadas_eda(supabase, x_user_id, dataset_id)

    class_stats = get_class_stats(supabase, x_user_id, dataset_id)

    corpus_stats = get_corpus_stats(supabase, x_user_id, dataset_id)

    return {
        "images": images,
        "class_stats": class_stats,
        "corpus_stats": corpus_stats
    }