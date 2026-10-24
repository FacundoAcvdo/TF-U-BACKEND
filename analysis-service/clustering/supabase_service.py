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

def save_clustering_results(supabase, user_id: str, dataset_id: str, points: list, clusters: list, total: int, displayed: int, outliers: int, truncated: bool):
    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "points": points,
        "clusters": clusters,
        "total": total,
        "displayed": displayed,
        "outliers": outliers,
        "truncated": truncated,
    }
    try:
        supabase.table("clustering_results").upsert(payload, on_conflict="dataset_id,user_id").execute()
    except Exception:
        try:
            supabase.table("clustering_results").delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("clustering_results").insert(payload).execute()

def get_clustering_results(supabase, user_id: str, dataset_id: str):
    response = (
        supabase
        .table("clustering_results")
        .select("points, clusters, total, displayed, outliers, truncated")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="No clustering results found for this dataset")
    row = response.data[0]
    return {
        "points": row["points"],
        "clusters": row["clusters"],
        "total": row["total"],
        "displayed": row["displayed"],
        "outliers": row["outliers"],
        "truncated": row["truncated"],
    }
