"""Suicide-risk service: opción A (E5+LogReg) siempre, opción B (RoBERTa) solo alto+EN."""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Header, HTTPException
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

DISCLAIMER = explain_mod.DISCLAIMER
TOP_K_DEFAULT = 200
TOP_K_MAX = 500
VERIFY_CAP = 200  # máx docs que pasan por RoBERTa por análisis


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[risk-service] warmup: cargando risk_model.pkl", flush=True)
    try:
        info = await asyncio.to_thread(risk_service.warmup)
        print(f"[risk-service] modelo OK dim={info['embedding_dim']}", flush=True)
    except Exception as e:
        print(f"[risk-service] warmup fallo (reintenta lazy): {e}", flush=True)
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
    fuente: str | None = None  # override manual: "mensajes" | "chat"


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


@app.post("/analyzeRisk")
async def analyze_risk(
    data: RiskRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        import os as _os

        supabase = get_supabase_client(x_user_token)
        ds_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = ds_info["dataset_id"]
        # Autodetección por tipo guardado en ingesta (services.py:163),
        # con override manual opcional.
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

        # Filtra filas sin embedding válido o de otra dimensión (otro encoder).
        # Ojo: los embeddings pueden venir como np.ndarray -> no usar 'or' ni
        # truthiness sobre ellos (ValueError: truth value of an array...).
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

        # Por clase (modo mensajes) o evolución por fecha (modo chat, sin 'clase')
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

        # Ranking para evidencia (alto/potencial primero, luego score)
        order = sorted(range(len(valid_idx)),
                       key=lambda j: (levels[j] in ("alto", "potencial"), scores[j]),
                       reverse=True)
        top_k = max(1, min(data.top_k or TOP_K_DEFAULT, TOP_K_MAX))
        top_items: list[dict] = []
        verify_candidates: list[int] = []  # posiciones dentro de order
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

        # Opción B solo para alto+EN (cascada)
        verified = 0
        if verify_candidates:
            b_scores = await asyncio.to_thread(
                verify_service.verify,
                [top_items[r]["texto"] for r in verify_candidates],
            )
            for r, bs in zip(verify_candidates, b_scores):
                if bs is not None:
                    top_items[r]["score_b"] = round(float(bs), 4)
                    # Nivel final: el mayor de ambos
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
            print(f"[risk-service] warn save: {e}", flush=True)

        return {"dataset_id": dataset_id, **payload}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[risk-service] error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
