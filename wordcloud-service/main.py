from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import uuid
import io
import os
import re
import json
from pathlib import Path

from nltk.corpus import stopwords

from supabase import create_client
from storage3 import create_client as create_storage_client

from wordcloud import WordCloud

class WordCloudRequest(BaseModel):
    dataset: str

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
BASE_PATH = os.environ.get("BASE_PATH")

STOPWORDS = (
    set(stopwords.words("spanish"))
    | set(stopwords.words("english"))
)

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

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()

    texto = re.sub(r"\b\d+\b", " ", texto)

    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)

    palabras = texto.split()

    palabras = [
        palabra
        for palabra in palabras
        if palabra not in STOPWORDS
        and len(palabra) > 2
    ]

    return " ".join(palabras)

@app.post("/visualizeWordCloud")
async def createWordCloud(
    data: WordCloudRequest,
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

        if not response.data:
            print(f"Dataset not found for user {x_user_id} and dataset {data.dataset}")
            raise HTTPException(
                status_code=404,
                detail="Dataset not found or access denied."
            )

        print(f"Dataset info: {response.data[0]}")

        dataset_info = response.data[0]
        dataset_id = dataset_info.get("dataset_id")

        if not dataset_id:
            print(f"Dataset does not have a dataset_id for user {x_user_id} and dataset {data.dataset}")
            raise HTTPException(
                status_code=500,
                detail="Dataset does not have a dataset_id."
            )

        print(f"Looking for dataset directory for user {x_user_id} and dataset_id {dataset_id}")

        dataset_file = (
            Path("./data/datasets") / "users" / x_user_id / f"{dataset_id}.json"
        )

        print(f"Constructed dataset file path: {dataset_file}")

        if not dataset_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset file not found: {dataset_file}"
            )

        print(f"Dataset file: {dataset_file}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        documents = dataset_json.get("documents", [])

        textos = []

        for document in documents:
            texto = document.get("texto", "").strip()

            if texto:
                textos.append(limpiar_texto(texto))

        texto_completo = " ".join(textos)

        if not texto_completo.strip():
            raise HTTPException(
                status_code=400,
                detail="The dataset does not contain usable text."
            )

        wordcloud = WordCloud(
            width=2400,
            height=1200,
            background_color="white",
            max_words=200,
            collocations=False,
            stopwords=STOPWORDS,
            colormap="viridis",
            font_path="./Inter_24pt-SemiBold.ttf",
            prefer_horizontal=0.9,
        ).generate(texto_completo)

        plt.figure(figsize=(12, 6), dpi=200)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("")

        buffer = io.BytesIO()

        plt.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()
        buffer.seek(0)

        imagen_bytes = buffer.read()

        image_name = f"{x_user_id}/{uuid.uuid4()}.png"

        storage.from_("imagenes-generadas").upload(
            path=image_name,
            file=imagen_bytes,
            file_options={
                "content-type": "image/png"
            }
        )

        public_url = (
            f"http://127.0.0.1:54321/"
            f"storage/v1/object/public/"
            f"imagenes-generadas/{image_name}"
        )

        supabase.table("imagenes-generadas").insert({
            "user_id": x_user_id,
            "dataset_id": dataset_id,
            "function_name": "wordcloud",
            "public_url": public_url,
            "image_path": image_name,
        }).execute()

        return {
            "public_url": public_url,
            "dataset_id": dataset_id,
            "documents_processed": len(textos),
        }

    except HTTPException:
        plt.close("all")
        raise

    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        plt.close("all")

        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el archivo: {str(e)}"
        )

@app.post("/getWordCloudImage")
async def getWordCloudImage(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
    data: WordCloudRequest = None
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
            .eq("function_name", "wordcloud")
            .eq("dataset_id", dataset_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="No word cloud image found for the user.")

        image_info = response.data[0]
        public_url = image_info.get("public_url")

        return {"public_url": public_url}

    except Exception as e:
        print(f"Error al obtener la imagen de clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener la imagen de clustering: {str(e)}")