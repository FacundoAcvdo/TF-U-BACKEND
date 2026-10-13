import time
import uuid
"""Suicide-risk service: opción A (E5+LogReg) siempre, opción B (RoBERTa) solo alto+EN."""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import explain as explain_mod
import risk_service
import verify_service
from chroma_reader import fetch_paginated
from supabase_service import (

    get_dataset_info,
    get_suicide_results,
    get_supabase_client,
    save_suicide_results,
)
import logging
from logging_utils import setup_service_logger
import jobs
logger = setup_service_logger("risk-service")

DISCLAIMER = explain_mod.DISCLAIMER
TOP_K_DEFAULT = 200
TOP_K_MAX = 500
VERIFY_CAP = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    pass
    try:
        info = await asyncio.to_thread(risk_service.warmup)
        pass
    except Exception as e:
        logger.exception(f"[risk-service] warmup fallo (reintenta lazy): {e}")
    yield


app = FastAPI(title="Suicide Risk Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskRequest(BaseModel):
    dataset: str
    with_verification: bool = True
    top_k: int = TOP_K_DEFAULT
    fuente: str | None = None


def _summarize(scores: list[float], levels: list[str]) -> dict:
    counts = dict(Counter(levels))
    for n in risk_service.NIVELES:
        counts.setdefault(n, 0)
    total = len(scores)
    crit = counts["alto"] + counts["potencial"]
    return {
        "counts": counts,
        "total": total,
        "pct_alto_potencial": round(100.0 * crit / total, 2) if total else 0.0,
        "score_medio": round(float(np.mean(scores)), 4) if scores else 0.0,
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        user_id = request.headers.get("x-user-id", "-")
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latencia_ms={latencia_ms} "
            f"req={request_id} user_id={user_id}"
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.exception(
            f"fallo no controlado {request.method} {request.url.path} "
            f"req={request_id} latencia_ms={latencia_ms} causa={type(e).__name__}: {e}"
        )
        raise


@app.post("/analyzeRisk", status_code=202)
async def start_analyze_risk(
    data: RiskRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola el analisis de riesgo. Devuelve 202 {job_id}; el resultado se
    consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="suicide", job_type="risk",
            user_id=x_user_id, dataset_name=data.dataset, params={},
        )
    except Exception as e:
        logger.exception(f"[risk-service] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_risk_job, job["job_id"], data.model_dump(mode="json"), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


async def _run_risk_job(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_analyze_risk(RiskRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[risk-service] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[risk-service] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
        jobs.mark_failed(supabase, job_id, 500, "Error interno del servidor.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)
    row = jobs.get_job(supabase, job_id, x_user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return row


@app.get("/jobs")
async def list_active_jobs(
    dataset_name: str | None = None,
    job_type: str | None = None,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Trabajos activos del usuario (para retomar al volver a la pantalla)."""
    supabase = get_supabase_client(x_user_token)
    return {"jobs": jobs.list_jobs(supabase, x_user_id, dataset_name, job_type, active_only=True)}


async def _compute_analyze_risk(
    data: RiskRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        import os as _os

        supabase = get_supabase_client(x_user_token)
        ds_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = ds_info["dataset_id"]


        tipo = str(ds_info.get("tipo", "dataset")).lower()
        es_chat = (data.fuente or ("chat" if tipo == "chat" else "mensajes")) == "chat"
        collection = (_os.environ.get("CHROMA_CHUNKED_COLLECTION", "chunked_chats")
                      if es_chat else None)
        fuente = "chat" if es_chat else "mensajes"

        chroma_data = await fetch_paginated(
            {"$and": [{"user_id": x_user_id}, {"dataset_id": dataset_id}]},
            ["documents", "metadatas", "embeddings"],
            collection,
        )
        ids = chroma_data.get("ids") or []
        if not ids:
            raise HTTPException(status_code=404, detail="No documents found for dataset in Chroma")
        docs = list(chroma_data.get("documents") or [])
        metas = list(chroma_data.get("metadatas") or [])
        raw_embs = chroma_data.get("embeddings")
        embs = list(raw_embs) if raw_embs is not None else []

        info = risk_service.get_model_info()
        dim = info["embedding_dim"]
        th = risk_service.get_thresholds()


        def _emb_ok(e) -> bool:
            if e is None:
                return False
            try:
                return len(e) == dim
            except TypeError:
                return False

        valid_idx = [i for i, e in enumerate(embs) if _emb_ok(e)]
        if not valid_idx:
            raise HTTPException(
                status_code=400,
                detail=f"Ningún embedding coincide con dim={dim} "
                       f"({info['embedding_model']}). Reingesta el dataset con E5-large.",
            )
        X = np.asarray([embs[i] for i in valid_idx], dtype=np.float32)
        scores = await asyncio.to_thread(risk_service.score_vectors, X)
        levels = [risk_service.score_to_level(p, th) for p in scores]


        by_clase: dict = {}
        evolucion: list[dict] = []
        if not es_chat:
            for k, p, lv in zip(valid_idx, scores, levels):
                clase = str((metas[k] or {}).get("clase", "sin_clase"))
                agg = by_clase.setdefault(clase, {"total": 0, "alto": 0, "potencial": 0})
                agg["total"] += 1
                if lv in ("alto", "potencial"):
                    agg[lv] += 1
        else:
            por_mes: dict = {}
            for k, lv in zip(valid_idx, levels):
                fecha = str((metas[k] or {}).get("fecha_desde", "?"))[:7]
                agg = por_mes.setdefault(fecha, {"total": 0, "alto": 0, "potencial": 0})
                agg["total"] += 1
                if lv in ("alto", "potencial"):
                    agg[lv] += 1
            evolucion = [{"mes": m, **v} for m, v in sorted(por_mes.items())]


        order = sorted(range(len(valid_idx)),
                       key=lambda j: (levels[j] in ("alto", "potencial"), scores[j]),
                       reverse=True)
        top_k = max(1, min(data.top_k or TOP_K_DEFAULT, TOP_K_MAX))
        top_items: list[dict] = []
        verify_candidates: list[int] = []
        for rank, j in enumerate(order[:top_k]):
            i = valid_idx[j]
            texto = (docs[i] if i < len(docs) else "")[:2000]
            item: dict = {
                "doc_id": ids[i],
                "texto": texto,
                "score_a": round(scores[j], 4),
                "score_b": None,
                "nivel": levels[j],
            }
            if es_chat:
                meta = metas[i] if i < len(metas) else {}
                mensajes = (docs[i] if i < len(docs) else "").split("\n")
                evidencia = []
                for n_msg, msg in enumerate(mensajes):
                    ev_m = explain_mod.explain_text(msg)
                    if ev_m["has_evidence"]:
                        evidencia.append({"n": n_msg, "texto": msg[:500],
                                          "triggers": ev_m["triggers"]})
                item.update({
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "msg_desde": 0,
                    "msg_hasta": len(mensajes) - 1,
                    "fecha_desde": (meta or {}).get("fecha_desde"),
                    "fecha_hasta": (meta or {}).get("fecha_hasta"),
                    "mensajes_evidencia": evidencia[:5],
                    "spans": [],
                    "triggers": sorted({t for e in evidencia for t in e["triggers"]}),
                })
            else:
                ev = explain_mod.explain_text(texto)
                item.update({"spans": ev["spans"], "triggers": ev["triggers"]})
            top_items.append(item)
            if (data.with_verification and levels[j] == "alto"
                    and verify_service.is_english(docs[i] if i < len(docs) else "")
                    and len(verify_candidates) < VERIFY_CAP):
                verify_candidates.append(rank)


        verified = 0
        if verify_candidates:
            b_scores = await asyncio.to_thread(
                verify_service.verify,
                [top_items[r]["texto"] for r in verify_candidates],
            )
            for r, bs in zip(verify_candidates, b_scores):
                if bs is not None:
                    top_items[r]["score_b"] = round(float(bs), 4)

                    la = top_items[r]["nivel"]
                    lb = risk_service.score_to_level(float(bs), th)
                    rank_lv = {n: k for k, n in enumerate(risk_service.NIVELES)}
                    top_items[r]["nivel"] = la if rank_lv[la] >= rank_lv[lb] else lb
                    verified += 1

        payload = {
            "fuente": fuente,
            "modelo_version": f"{info['embedding_model']}+logreg",
            "trained_at": info["trained_at"],
            "thresholds": th,
            "resumen": _summarize(scores, levels),
            "por_clase": by_clase,
            "evolucion": evolucion,
            "skipped_sin_embedding": len(ids) - len(valid_idx),
            "verificados_b": verified,
            "top_items": top_items,
            "disclaimer": DISCLAIMER,
        }
        try:
            save_suicide_results(supabase, x_user_id, dataset_id, payload)
        except Exception as e:
            logger.warning(f"[risk-service] warn save: {e}")

        return {"dataset_id": dataset_id, **payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[risk-service] fallo en POST /analyzeRisk | method=POST user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.post("/getResults")
async def get_results(
    data: RiskRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        dataset_id = get_dataset_info(supabase, data.dataset, x_user_id)["dataset_id"]
        row = get_suicide_results(supabase, x_user_id, dataset_id)
        row.setdefault("disclaimer", DISCLAIMER)
        return {"dataset_id": dataset_id, **row}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[risk-service] fallo en POST /getResults | method=POST user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")
