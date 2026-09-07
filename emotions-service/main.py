from urllib import response

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import matplotlib.pyplot as plt

from utils import parse_dataset_json
from emotion_service import EMOTIONS, get_empath_client, enrich_dataframe_with_emotions
from stats_service import get_corpus_emotion_stats, get_class_emotion_stats
from plot_service import (
    generate_emotion_corpus_mean_plot,
    generate_emotion_corpus_distribution_plot,
    generate_emotion_corpus_correlation_plot,
    generate_emotion_class_mean_plot,
    generate_emotion_class_std_plot,
    generate_emotion_class_median_plot,
)
from supabase_service import (
    get_supabase_client,
    get_storage_client,
    get_dataset_info,
    upload_images,
    save_image_info_to_db,
    get_imagenes_generadas_emotions,
)

###############

DATASET_STORAGE_PATH = os.environ.get("DATASET_STORAGE_PATH", "./data/datasets")

###############

app = FastAPI(title="Emotions Service")

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############

class EmotionsRequest(BaseModel):
    dataset: str

@app.post("/analyzeEmotions")
async def analyze_emotions(
    data: EmotionsRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        storage = get_storage_client(x_user_token)

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info["dataset_id"]

        dataset_path = os.path.join(
            DATASET_STORAGE_PATH, "users", x_user_id, f"{dataset_id}.json"
        )

        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset file not found at {dataset_path}")

        df = parse_dataset_json(dataset_path)

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        lexicon = get_empath_client()
        df_enriched = enrich_dataframe_with_emotions(df, categories=EMOTIONS, lexicon=lexicon)

        corpus_stats = get_corpus_emotion_stats(df_enriched, EMOTIONS)
        class_stats = get_class_emotion_stats(df_enriched, EMOTIONS)

        imagenes = []
        imagenes += generate_emotion_corpus_mean_plot(corpus_stats)
        imagenes += generate_emotion_corpus_distribution_plot(df_enriched, EMOTIONS)
        imagenes += generate_emotion_corpus_correlation_plot(df_enriched, EMOTIONS)
        imagenes += generate_emotion_class_mean_plot(class_stats, EMOTIONS)
        imagenes += generate_emotion_class_std_plot(class_stats, EMOTIONS)
        imagenes += generate_emotion_class_median_plot(class_stats, EMOTIONS)

        public_urls, image_names = upload_images(storage, imagenes, x_user_id)
        save_image_info_to_db(supabase, x_user_id, dataset_id, public_urls, image_names)

        return {
            "public_urls": public_urls,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[emotions-service] Error al procesar emociones: {str(e)}")
        try:
            plt.close('all')
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Error al procesar emociones: {str(e)}")

@app.post("/getResults")
async def get_results(
    data: EmotionsRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info["dataset_id"]

        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        
        dataset_id = dataset_info["dataset_id"]

        images = get_imagenes_generadas_emotions(supabase, x_user_id, dataset_id)

        return {
            "public_urls": images
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[emotions-service] Error al obtener resultados: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener resultados: {str(e)}")