"""Persistencia de resultados en Supabase (tabla suicide_results)."""
import os

from fastapi import HTTPException
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")


def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client


def get_dataset_info(supabase, dataset_name: str, user_id: str) -> dict:
    response = (
        supabase.table("datasets")
        .select("dataset_id,tipo")
        .eq("dataset_name", dataset_name)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied.")
    row = response.data[0]
    row.setdefault("tipo", "dataset")
    return row


def save_suicide_results(supabase, user_id: str, dataset_id: str, payload: dict) -> None:
    row = {"dataset_id": dataset_id, "user_id": user_id, **payload}
    try:
        supabase.table("suicide_results").upsert(row, on_conflict="dataset_id,user_id").execute()
    except Exception:
        try:
            supabase.table("suicide_results").delete().eq("dataset_id", dataset_id).eq(
                "user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("suicide_results").insert(row).execute()


def get_suicide_results(supabase, user_id: str, dataset_id: str) -> dict:
    response = (
        supabase.table("suicide_results")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="No suicide-risk results for this dataset")
    return response.data[0]
