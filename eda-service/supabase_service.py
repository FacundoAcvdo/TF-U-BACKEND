from storage3 import create_client as create_storage_client
from supabase import create_client
from fastapi import HTTPException
import uuid
import os


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

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

def get_dataset_info(supabase, dataset_name: str, user_id: str):
    response = (
        supabase
        .table("datasets")
        .select("*")
        .eq("dataset_name", dataset_name)
        .eq("user_id", user_id)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied.")

    return response.data[0] if response.data else None

def upload_images(storage, imagenes, user_id):
    image_public_url = dict()
    image_names = []

    for data_imagen, funcion in imagenes:
        image_name = f"{user_id}/{uuid.uuid4()}.png"

        storage.from_("imagenes-generadas").upload(
            path=image_name,
            file=data_imagen,
            file_options={"content-type": "image/png"}
        )

        image_public_url[funcion] = f"http://127.0.0.1:54321/storage/v1/object/public/imagenes-generadas/{image_name}"
        image_names.append(image_name)

    return image_public_url, image_names


def save_image_info_to_db(supabase, user_id, dataset_id, image_public_url, image_names):
    for funcion, public_url in image_public_url.items():
        image_name = image_names.pop(0)
        
        supabase.table("imagenes-generadas").insert({
            "user_id": user_id,
            "dataset_id": dataset_id,
            "function_name": funcion,
            "public_url": public_url,
            "image_path": image_name,
        }).execute()

def save_eda_results_to_db(supabase, user_id, dataset_id, corpus_stats, class_stats):
    corpus_stats = corpus_stats.to_dict(orient="records")[0]
    corpus_stats["dataset_id"] = dataset_id
    corpus_stats["user_id"] = user_id
    supabase.table("corpus_stats").insert(corpus_stats).execute()

    class_stats = class_stats.to_dict(orient="records")
    for class_stat in class_stats:
        class_stat["dataset_id"] = dataset_id
        class_stat["user_id"] = user_id
        supabase.table("class_stats").insert(class_stat).execute()

def get_imagenes_generadas_eda(supabase, user_id: str, dataset_id: str):
    response = (
        supabase
        .table("imagenes-generadas")
        .select("*")
        .eq("user_id", user_id)
        .eq("dataset_id", dataset_id)
        .in_("function_name", ["boxplot", "catplot", "class_distribution", "distribution"])
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="No se encontraron imágenes generadas para este dataset")

    return {img["function_name"]: img["public_url"] for img in response.data}

def get_class_stats(supabase, user_id: str, dataset_id: str):
    response = (
        supabase
        .table("class_stats")
        .select("clase, texto, vocabulary, words, word_avg, sentences, sentence_avg, sentence_min, sentence_max")
        .eq("user_id", user_id)
        .eq("dataset_id", dataset_id)
        .execute()
    )

    return response.data

def get_corpus_stats(supabase, user_id: str, dataset_id: str):
    response = (
        supabase
        .table("corpus_stats")
        .select("texto, vocabulary, words, word_avg, sentences, sentence_avg, sentence_min, sentence_max")
        .eq("user_id", user_id)
        .eq("dataset_id", dataset_id)
        .execute()
    )

    return response.data[0] if response.data else None