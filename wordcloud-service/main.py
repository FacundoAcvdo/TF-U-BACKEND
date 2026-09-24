from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

import os
import re
from collections import Counter

from nltk.corpus import stopwords

from supabase import create_client
from chroma_reader import fetch_paginated

class WordCloudRequest(BaseModel):
    dataset: str

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

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

def save_wordcloud_results(supabase, user_id: str, dataset_id: str, frequencies: list, total_words: int, unique_words: int, max_count: int, min_count: int, documents_processed: int):
    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "frequencies": frequencies,
        "total_words": total_words,
        "unique_words": unique_words,
        "max_count": max_count,
        "min_count": min_count,
        "documents_processed": documents_processed,
    }
    try:
        supabase.table("wordcloud_results").upsert(payload, on_conflict="dataset_id,user_id").execute()
    except Exception:
        try:
            supabase.table("wordcloud_results").delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("wordcloud_results").insert(payload).execute()

def get_wordcloud_results(supabase, user_id: str, dataset_id: str):
    response = (
        supabase.table("wordcloud_results")
        .select("frequencies, total_words, unique_words, max_count, min_count, documents_processed")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="No wordcloud results found for this dataset")
    row = response.data[0]
    return {
        "frequencies": row["frequencies"],
        "total_words": row["total_words"],
        "unique_words": row["unique_words"],
        "max_count": row["max_count"],
        "min_count": row["min_count"],
        "documents_processed": row["documents_processed"],
    }

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

def build_wordcloud_data(textos_limpios: list[str], max_words: int = 200):
    # tokens globales
    all_tokens = []
    # doc frequency por palabra
    doc_freq = Counter()
    for t in textos_limpios:
        tokens = t.split()
        all_tokens.extend(tokens)
        # doc frequency: unique por doc
        doc_freq.update(set(tokens))

    total_words = len(all_tokens)
    counter = Counter(all_tokens)
    unique_words = len(counter)

    most_common = counter.most_common(max_words)
    if not most_common:
        return {
            "total_words": 0,
            "unique_words": 0,
            "frequencies": [],
            "max_count": 0,
            "min_count": 0,
        }

    max_count = most_common[0][1]
    min_count = most_common[-1][1]

    frequencies = []
    for rank, (word, count) in enumerate(most_common, 1):
        percentage = (count / total_words * 100) if total_words else 0
        df = doc_freq.get(word, 0)
        frequencies.append({
            "text": word,
            "value": count,  # compat con wordcloud libs
            "count": count,
            "percentage": round(percentage, 2),
            "doc_frequency": df,
            "doc_percentage": round((df / len(textos_limpios) * 100) if textos_limpios else 0, 1),
            "rank": rank,
        })

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "frequencies": frequencies,
        "max_count": max_count,
        "min_count": min_count,
    }

async def load_and_process_dataset(x_user_id: str, dataset_id: str):
    # Solo pide documents, paginado.
    where = {
        "$and": [
            {"user_id": x_user_id},
            {"dataset_id": dataset_id},
        ]
    }
    data = await fetch_paginated(where, ["documents"])
    documents = data.get("documents", [])
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found for dataset in Chroma")
    textos = []
    for texto_raw in documents:
        texto = (texto_raw or "").strip()
        if texto:
            textos.append(limpiar_texto(texto))
    texto_completo = " ".join(textos)
    if not texto_completo.strip():
        raise HTTPException(status_code=400, detail="The dataset does not contain usable text.")
    return textos, len(textos)

@app.post("/visualizeWordCloud")
async def createWordCloud(
    data: WordCloudRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
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
            raise HTTPException(status_code=404, detail="Dataset not found or access denied.")

        dataset_info = response.data[0]
        dataset_id = dataset_info.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=500, detail="Dataset does not have a dataset_id.")

        textos_limpios, docs_processed = await load_and_process_dataset(x_user_id, dataset_id)

        wc_data = build_wordcloud_data(textos_limpios, max_words=200)

        try:
            save_wordcloud_results(
                supabase, x_user_id, dataset_id,
                wc_data["frequencies"], wc_data["total_words"], wc_data["unique_words"],
                wc_data["max_count"], wc_data["min_count"], docs_processed
            )
        except Exception as e:
            print(f"[wordcloud-service] warn save_wordcloud_results: {e}")

        return {
            "dataset_id": dataset_id,
            "documents_processed": docs_processed,
            **wc_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al procesar wordcloud: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar wordcloud: {str(e)}")

@app.post("/getWordCloudImage")
async def getWordCloudImage(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
    data: WordCloudRequest = None
):
    try:
        supabase = get_supabase_client(x_user_token)
        if data is None or not data.dataset:
            raise HTTPException(status_code=400, detail="Dataset is required")
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
        data_db = get_wordcloud_results(supabase, x_user_id, dataset_id)
        return {"dataset_id": dataset_id, **data_db}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al obtener wordcloud: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener wordcloud: {str(e)}")
