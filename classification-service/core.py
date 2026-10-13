import os
import time

import chromadb
import torch
from fastapi import HTTPException
from supabase import create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def resolver_dispositivo() -> str:
    forzado = os.environ.get("CLASSIFICATION_FORCE_DEVICE", "").strip().lower()
    if forzado in ("cpu", "cuda"):
        if forzado == "cuda":
            try:
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = resolver_dispositivo()

try:
    n_hilos = os.cpu_count() or 4
    torch.set_num_threads(max(1, min(n_hilos, 8)))
    torch.set_num_interop_threads(max(1, min(n_hilos, 4)))
except Exception:
    pass

_chroma_client = None
_collection_cache = {}


def get_chroma_client(intentos: int = 10, espera_inicial: float = 1.0):
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    espera = espera_inicial
    ultimo_error: Exception | None = None
    for _ in range(intentos):
        try:
            _chroma_client = chromadb.HttpClient(host="chroma-service", port=8000)
            return _chroma_client
        except Exception as e:
            ultimo_error = e
            time.sleep(espera)
            espera = min(espera * 2, 10.0)
    raise HTTPException(
        status_code=503,
        detail=f"No se pudo conectar con ChromaDB tras {intentos} intentos: {ultimo_error}",
    )


def get_collection(nombre: str | None = None):
    if nombre is None:
        nombre = CHROMA_COLLECTION
    if nombre not in _collection_cache:
        _collection_cache[nombre] = get_chroma_client().get_collection(name=nombre)
    return _collection_cache[nombre]


def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client
