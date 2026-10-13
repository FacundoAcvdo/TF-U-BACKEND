import logging
from datetime import datetime, timezone

logger = logging.getLogger("jobs")


STALE_MINUTES = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_stale(row: dict) -> bool:
    try:
        updated = datetime.fromisoformat(str(row.get("updated_at")))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - updated).total_seconds() / 60
        return age_min > STALE_MINUTES
    except Exception:
        return False


def create_job(supabase, *, service: str, job_type: str, user_id: str,
               dataset_name: str | None = None, params: dict | None = None) -> dict:
    payload = {
        "service": service,
        "job_type": job_type,
        "status": "queued",
        "user_id": user_id,
        "dataset_name": dataset_name,
        "params": params or {},
    }
    resp = supabase.table("jobs").insert(payload).execute()
    if not resp.data:
        raise RuntimeError("No se pudo crear el job.")
    row = resp.data[0]
    return {"job_id": row["job_id"], "status": row["status"]}


def mark_running(supabase, job_id: str) -> None:
    supabase.table("jobs").update({"status": "running"}).eq("job_id", job_id).execute()


def mark_succeeded(supabase, job_id: str, result: dict | None = None) -> None:
    supabase.table("jobs").update({
        "status": "succeeded",
        "result": result or {},
        "finished_at": _now_iso(),
    }).eq("job_id", job_id).execute()


def mark_failed(supabase, job_id: str, error_code: int, error_detail: str) -> None:
    try:
        supabase.table("jobs").update({
            "status": "failed",
            "error_code": error_code,
            "error_detail": error_detail,
            "finished_at": _now_iso(),
        }).eq("job_id", job_id).execute()
    except Exception as e:
        logger.warning(f"[jobs] no se pudo marcar job {job_id} como fallido: {e}")


def _public_row(row: dict) -> dict:
    return {
        "job_id": row["job_id"],
        "service": row.get("service"),
        "job_type": row.get("job_type"),
        "status": row.get("status"),
        "dataset_name": row.get("dataset_name"),
        "result": row.get("result"),
        "error_code": row.get("error_code"),
        "error_detail": row.get("error_detail"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "finished_at": row.get("finished_at"),
        "params": row.get("params"),
    }


def get_job(supabase, job_id: str, user_id: str) -> dict | None:
    resp = (
        supabase.table("jobs").select("*")
        .eq("job_id", job_id).eq("user_id", user_id).limit(1).execute()
    )
    if not resp.data:
        return None
    row = resp.data[0]
    if row.get("status") in ("queued", "running") and _is_stale(row):
        logger.warning(f"[jobs] job {job_id} abandonado (stale), se marca como fallido.")
        mark_failed(supabase, job_id, 500, "El trabajo se interrumpio. Vuelve a intentarlo.")
        row = {**row, "status": "failed", "error_code": 500,
               "error_detail": "El trabajo se interrumpio. Vuelve a intentarlo."}
    return _public_row(row)


def list_jobs(supabase, user_id: str, dataset_name: str | None = None,
              job_type: str | None = None, active_only: bool = True) -> list[dict]:
    query = supabase.table("jobs").select("*").eq("user_id", user_id)
    if dataset_name:
        query = query.eq("dataset_name", dataset_name)
    if job_type:
        query = query.eq("job_type", job_type)
    if active_only:
        query = query.in_("status", ["queued", "running"])
    query = query.order("created_at", desc=True).limit(20)
    resp = query.execute()
    rows = resp.data or []
    out = []
    for row in rows:
        if row.get("status") in ("queued", "running") and _is_stale(row):
            logger.warning(f"[jobs] job {row['job_id']} abandonado (stale), se marca como fallido.")
            mark_failed(supabase, row["job_id"], 500, "El trabajo se interrumpio. Vuelve a intentarlo.")
            row = {**row, "status": "failed", "error_code": 500,
                   "error_detail": "El trabajo se interrumpio. Vuelve a intentarlo."}
            if active_only:
                continue
        out.append(_public_row(row))
    return out
