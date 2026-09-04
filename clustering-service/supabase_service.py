from storage3 import create_client as create_storage_client
from supabase import create_client
from fastapi import HTTPException
import uuid
import os

##########

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

##########

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

def upload_image_to_bucket(storage, image_bytes, user_id):
    image_name = f"{user_id}/{uuid.uuid4()}.png"
    storage.from_("imagenes-generadas").upload(
        path=image_name,
        file=image_bytes,
        file_options={"content-type": "image/png"}
    )
    return image_name

def upload_image_info(supabase, image_name, user_id, dataset_id):
    #public_url = storage.from_("imagenes-generadas").get_public_url(image_name) - Se debería activar en producción
    public_url = f"http://127.0.0.1:54321/storage/v1/object/public/imagenes-generadas/{image_name}"

    supabase.table("imagenes-generadas").insert({
        "user_id": user_id,
        "dataset_id": dataset_id,
        "function_name": "clustering",
        "public_url": public_url,
        "image_path": image_name,
    }).execute()

    return {"public_url": public_url}

def get_cluster_image(supabase, user_id : str, dataset_id: str):
    response = (
        supabase
        .table("imagenes-generadas")
        .select("public_url")
        .eq("user_id", user_id)
        .eq("function_name", "clustering")
        .eq("dataset_id", dataset_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="No clustering image found for the user.")

    return response.data[0].get("public_url")
