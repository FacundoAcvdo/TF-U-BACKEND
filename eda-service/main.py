from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import numpy as np
from collections import Counter

from stats_service import (
    get_dataset_row_analysis,
    get_estadisticas_corpus,
    get_estadisticas_clases
)

from supabase_service import (
    get_supabase_client,
    get_dataset_info,
    save_eda_results,
    get_eda_results,
)

from utils import parse_dataset_json

###############

DATASET_STORAGE_PATH = os.environ.get("DATASET_STORAGE_PATH", "./data/datasets")

###############

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############

class EdaRequest(BaseModel):
    dataset: str

def build_sentence_stats(dataset):
    s = dataset["sentences"].astype(float)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1) if len(s) > 1 else 0.0),
        "min": int(s.min()),
        "max": int(s.max()),
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }

def build_sentence_freq(dataset, max_points=40):
    cnt = Counter(dataset["sentences"].tolist())
    # ordenado por valor de oración
    items = sorted(cnt.items())
    # limitar si hay muchos valores distintos
    if len(items) > max_points:
        # agrupar en bins si >40 valores distintos
        vals = dataset["sentences"].astype(int).tolist()
        hist, edges = np.histogram(vals, bins=max_points)
        return [{"sentences": int(edges[i]), "count": int(hist[i]), "bin_end": int(edges[i+1])} for i in range(len(hist))]
    return [{"sentences": int(k), "count": int(v)} for k, v in items]

def build_distribution_histograms(dataset, bins=20):
    clases = np.unique(dataset['clase'])
    result = []
    for clase in clases:
        grupo = dataset[dataset['clase'] == clase]
        words = grupo['words'].astype(int).tolist()
        sents = grupo['sentences'].astype(int).tolist()

        # histograma words
        if words:
            w_counts, w_edges = np.histogram(words, bins=bins)
            words_hist = [{"bin_start": int(w_edges[i]), "bin_end": int(w_edges[i+1]), "count": int(w_counts[i])} for i in range(len(w_counts))]
        else:
            words_hist = []

        # histograma sentences
        if sents:
            s_counts, s_edges = np.histogram(sents, bins=bins)
            sents_hist = [{"bin_start": int(s_edges[i]), "bin_end": int(s_edges[i+1]), "count": int(s_counts[i])} for i in range(len(s_counts))]
        else:
            sents_hist = []

        result.append({
            "clase": str(clase),
            "words_histogram": words_hist,
            "sentences_histogram": sents_hist,
            "words_stats": {
                "count": len(words),
                "mean": float(np.mean(words)) if words else 0,
                "min": int(np.min(words)) if words else 0,
                "max": int(np.max(words)) if words else 0,
            },
            "sentences_stats": {
                "count": len(sents),
                "mean": float(np.mean(sents)) if sents else 0,
                "min": int(np.min(sents)) if sents else 0,
                "max": int(np.max(sents)) if sents else 0,
            }
        })
    return result

def build_eda_response(dataset):
    dataset = get_dataset_row_analysis(dataset)
    DF_CORPUS = get_estadisticas_corpus(dataset)
    DF_CLASES = get_estadisticas_clases(dataset)

    corpus_stats = DF_CORPUS.to_dict(orient="records")[0]
    class_stats = DF_CLASES.to_dict(orient="records")

    sentence_stats = build_sentence_stats(dataset)
    sentence_freq = build_sentence_freq(dataset)
    distribution = build_distribution_histograms(dataset, bins=20)

    return {
        "corpus_stats": corpus_stats,
        "class_stats": class_stats,
        "sentence_stats": sentence_stats,
        "sentence_freq": sentence_freq,
        "distribution": distribution,
    }

@app.post("/eda")
async def main(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)

    dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
    dataset_id = dataset_info["dataset_id"]
    dataset_path = os.path.join(DATASET_STORAGE_PATH, "users", x_user_id, f"{dataset_id}.json")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at {dataset_path}")

    dataset = parse_dataset_json(dataset_path)
    if dataset.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    result = build_eda_response(dataset)
    result["dataset_id"] = dataset_id

    # guardar unificado en eda_results (corpus + clase + charts)
    try:
        save_eda_results(
            supabase, x_user_id, dataset_id,
            result["corpus_stats"], result["class_stats"],
            result["sentence_stats"], result["sentence_freq"],
            result["distribution"]
        )
    except Exception as e:
        print(f"[eda-service] warn save_eda_results: {e}")

    return result

@app.post("/getResults")
async def get_results(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)
    dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
    dataset_id = dataset_info["dataset_id"]

    # solo devuelve si existe en DB; si no, 404 -> front queda idle con botón ejecutar
    db = get_eda_results(supabase, x_user_id, dataset_id)
    return {"dataset_id": dataset_id, **db}
