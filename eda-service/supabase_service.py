from supabase import create_client
from fastapi import HTTPException
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client

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

def save_eda_results(supabase, user_id: str, dataset_id: str, corpus_stats: dict, class_stats: list, sentence_stats: dict, sentence_freq: list, distribution: list):
    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "corpus_stats": corpus_stats,
        "class_stats": class_stats,
        "sentence_stats": sentence_stats,
        "sentence_freq": sentence_freq,
        "distribution": distribution,
    }
    # upsert por (dataset_id, user_id) - requiere UNIQUE constraint
    try:
        supabase.table("eda_results").upsert(payload, on_conflict="dataset_id,user_id").execute()
    except Exception:
        # fallback si el driver no soporta on_conflict: delete + insert
        try:
            supabase.table("eda_results").delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("eda_results").insert(payload).execute()

def get_eda_results(supabase, user_id: str, dataset_id: str):
    response = (
        supabase
        .table("eda_results")
        .select("corpus_stats, class_stats, sentence_stats, sentence_freq, distribution")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="No se encontraron resultados EDA para este dataset")
    row = response.data[0]
    return {
        "corpus_stats": row["corpus_stats"],
        "class_stats": row["class_stats"],
        "sentence_stats": row["sentence_stats"],
        "sentence_freq": row["sentence_freq"],
        "distribution": row["distribution"],
    }