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

def save_emotions_results(supabase, user_id: str, dataset_id: str, emotions: list, corpus_stats: dict, class_stats: list, correlation: dict, distribution: list, top_words: dict, class_top_words: dict, calidad: dict | None = None):
    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "emotions": emotions,
        "corpus_stats": corpus_stats,
        "class_stats": class_stats,
        "correlation": correlation,
        "distribution": distribution,
        "top_words": top_words,
        "class_top_words": class_top_words,
    }
    if calidad is not None:
        payload["calidad"] = calidad
    base = {k: v for k, v in payload.items() if k != "calidad"}
    try:
        supabase.table("emotions_results").upsert(payload, on_conflict="dataset_id,user_id").execute()
        return
    except Exception:
        pass
    try:
        supabase.table("emotions_results").upsert(base, on_conflict="dataset_id,user_id").execute()
    except Exception:
        try:
            supabase.table("emotions_results").delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("emotions_results").insert(base).execute()

def get_emotions_results(supabase, user_id: str, dataset_id: str):
    try:
        response = (
            supabase
            .table("emotions_results")
            .select("emotions, corpus_stats, class_stats, correlation, distribution, top_words, class_top_words, calidad")
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception:
        response = (
            supabase
            .table("emotions_results")
            .select("emotions, corpus_stats, class_stats, correlation, distribution, top_words, class_top_words")
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    if not response.data:
        raise HTTPException(status_code=404, detail="No emotions results found for this dataset")
    row = response.data[0]
    out = {
        "emotions": row["emotions"],
        "corpus_stats": row["corpus_stats"],
        "class_stats": row["class_stats"],
        "correlation": row["correlation"],
        "distribution": row["distribution"],
        "top_words": row["top_words"],
        "class_top_words": row["class_top_words"],
    }
    if row.get("calidad") is not None:
        out["calidad"] = row["calidad"]
    return out
