import os
import re
import glob
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from chroma_service import get_chroma_client, get_or_create_collection
from prompts import PROMPTS, PROMPT_TITLE
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("rag-service")


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.6-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("TOP_K", "4"))

RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "12"))
RERANK_ENABLED = os.environ.get("RERANK", "true").lower() in ("true", "1", "yes", "on")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "-2.0"))
HISTORY_TAIL = int(os.environ.get("HISTORY_TAIL", "4"))
HISTORY_MSG_CHARS = int(os.environ.get("HISTORY_MSG_CHARS", "600"))
HISTORY_MAX_CHARS = int(os.environ.get("HISTORY_MAX_CHARS", "2000"))

NO_EVIDENCE_MSG = (
    "No tengo información suficiente en los documentos indexados sobre ese punto."
)

DOCS_PATH = os.environ.get("DOCS_PATH", "/app/data/rag_docs")

RAG_EMB_VRAM_MB = int(os.environ.get("RAG_EMB_VRAM_MB", "800"))

RAG_INIT_ON_STARTUP = os.environ.get("RAG_INIT_ON_STARTUP", "false").lower() in ("true", "1", "yes", "on")

import model_cache
from model_cache import ModelBusyError, ModelSlot, warmup_enabled


RAG_DOCS_COLLECTION = os.environ.get("CHROMA_COLLECTION", "rag_docs")
DOCUMENTS_COLLECTION = os.environ.get("CHROMA_COLLECTION_DOCUMENTS", "documents")
CHUNKED_CHATS_COLLECTION = os.environ.get("CHROMA_COLLECTION_CHUNKED", "chunked_chats")

ALL_COLLECTIONS = {
    "rag_docs": RAG_DOCS_COLLECTION,
    "documents": DOCUMENTS_COLLECTION,
    "chunked_chats": CHUNKED_CHATS_COLLECTION,
}


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_history(chat_history: List[dict]) -> str:
    if not chat_history:
        return "Sin historial previo."
    lines = []
    for m in chat_history[-HISTORY_TAIL:]:
        autor = m.get("autor", "usuario")
        texto = str(m.get("mensaje", ""))[:HISTORY_MSG_CHARS]
        lines.append(f"{autor}: {texto}")
    texto = "\n".join(lines)
    if len(texto) > HISTORY_MAX_CHARS:
        texto = texto[-HISTORY_MAX_CHARS:]
    return texto


_reranker = None


def _get_reranker():
    """Cross-encoder singleton (carga perezosa; None si RERANK=false)."""
    global _reranker
    if not RERANK_ENABLED:
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


class E5Embeddings(HuggingFaceEmbeddings):
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(
            [
                t if t.startswith(self.passage_prefix)
                else f"{self.passage_prefix}{t}"
                for t in texts
            ]
        )

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{self.query_prefix}{text}")


class _LazyE5Embeddings:
    """Proxy con prefijos E5 que resuelve el modelo via slot (lazy+TTL)."""

    def __init__(self, service):
        self._service = service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        slot = self._service._get_emb_slot()
        slot.maybe_evict()
        with slot.use() as emb:
            return emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        slot = self._service._get_emb_slot()
        slot.maybe_evict()
        with slot.use() as emb:
            return emb.embed_query(text)


class RAGService:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.chroma_client = None
        self.stores: Dict[str, Chroma] = {}
        self.prompts: Dict[str, ChatPromptTemplate] = {}
        self._initialized = False
        self._emb_slot = None
        self._emb_slot_lock = threading.Lock()
        self._lazy_e5 = _LazyE5Embeddings(self)


    def _get_emb_slot(self):
        slot = self._emb_slot
        if slot is None:
            with self._emb_slot_lock:
                slot = self._emb_slot
                if slot is None:
                    service = self

                    def _load():
                        return service._build_embeddings()

                    def _drop(emb):
                        try:
                            del emb
                        except Exception:
                            pass

                    slot = ModelSlot(
                        name="rag-e5", vram_mb=RAG_EMB_VRAM_MB,
                        load_fn=_load, unload_fn=_drop)
                    self._emb_slot = slot
        return slot


    def _build_embeddings(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs: dict = {"device": device}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        try:
            return E5Embeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": EMBEDDING_BATCH_SIZE,
                },
            )
        except Exception as e:
            logger.warning(f"[rag_service] no se pudo usar fp16, se usa fp32: {e}")
            return E5Embeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": EMBEDDING_BATCH_SIZE,
                },
            )


    def _init_embeddings(self):
        self._get_emb_slot()
        if warmup_enabled():
            slot = self._get_emb_slot()
            slot.maybe_evict()
            with slot.use():
                logger.warning("[rag_service] embeddings precargados por warmup.")
        else:
            logger.warning("[rag_service] warmup omitido por WARMUP=false (carga lazy al primer uso).")


    def _init_llm(self):
        if LLM_PROVIDER == "groq":
            if not GROQ_API_KEY:
                logger.warning("[rag_service] LLM_PROVIDER=groq sin GROQ_API_KEY.")
                self.llm = None
                return
            try:
                from langchain_groq import ChatGroq
            except Exception as e:
                logger.error(f"[rag_service] falta paquete langchain-groq: {e}")
                self.llm = None
                return
            self.llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.2,
            )
            return
        if not GOOGLE_API_KEY:
            self.llm = None
            return
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            top_p=0.9,
        )

    def _init_stores(self):
        self.chroma_client = get_chroma_client()
        for key, coll_name in ALL_COLLECTIONS.items():
            get_or_create_collection(self.chroma_client, name=coll_name)
            self.stores[key] = Chroma(
                client=self.chroma_client,
                collection_name=coll_name,
                embedding_function=self._lazy_e5,
            )

            template = PROMPTS.get(key)
            self.prompts[key] = ChatPromptTemplate.from_template(template)
            pass

    def _discover_documents(self) -> List[str]:
        if not DOCS_PATH:
            pass
            return []
        p = Path(DOCS_PATH)
        if not p.exists():
            pass
            return []
        found_files: List[str] = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            found_files.extend(glob.glob(str(p / ext)))
        found_files = sorted(set(f for f in found_files if Path(f).is_file()))
        if not found_files:
            logger.warning(f"[rag_service] WARNING: No se encontraron documentos en {DOCS_PATH}. Monta el volumen en compose: {DOCS_PATH}:/app/data/rag_docs")
        return found_files

    def _load_documents(self) -> List[Document]:
        files = self._discover_documents()
        if not files:
            pass
            return []
        docs: List[Document] = []
        for fpath in files:
            try:
                if fpath.lower().endswith(".pdf"):
                    loader = PyPDFLoader(fpath)
                elif fpath.lower().endswith((".txt", ".md")):
                    loader = TextLoader(fpath, encoding="utf-8")
                else:
                    continue
                loaded = loader.load()
                for d in loaded:
                    d.page_content = re.sub(r'\s+', ' ', d.page_content).strip()
                    d.metadata["source"] = fpath
                docs.extend(loaded)
                pass
            except Exception as e:
                logger.exception(f"[rag_service] Error cargando {fpath}: {e}")
                continue
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        pass
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        pass
        return splits

    def _reindex_rag_docs(self):
        """Reindexa rag_docs si RAG_INIT_ON_STARTUP=true: agrega documentos nuevos encontrados en DOCS_PATH."""
        if not RAG_INIT_ON_STARTUP:
            pass
            return
        docs = self._load_documents()
        splits = self._split_documents(docs)
        if not splits:
            pass
            return
        store = self.stores.get("rag_docs")
        if store is None:
            pass
            return
        pass
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            store.add_documents(documents=batch)
        pass

    def initialize(self):
        if self._initialized:
            return
        self._init_embeddings()
        self._init_llm()
        self._init_stores()

        try:
            self._reindex_rag_docs()
        except Exception as e:
            logger.exception(f"[rag_service] Error en reindex rag_docs: {e}")
        self._initialized = True
        pass

    def is_ready(self) -> bool:
        return self._initialized and self.llm is not None and len(self.stores) == 3

    def is_store_ready(self, collection: str) -> bool:
        return self._initialized and collection in self.stores and self.llm is not None

    def get_store(self, collection: str) -> Chroma:
        if collection not in self.stores:
            raise ValueError(f"Colección '{collection}' no existe. Disponibles: {list(self.stores.keys())}")
        return self.stores[collection]

    def query(
        self,
        question: str,
        chat_history: List[dict] = None,
        collection: str = "rag_docs",
        filter: Optional[Dict] = None,
        k: Optional[int] = None,
    ) -> Tuple[str, List[Document]]:
        """Consulta por colección con filtro opcional (para documents/chunked_chats)."""
        if not self.is_store_ready(collection):
            raise RuntimeError(f"RAG no inicializado para colección '{collection}'")
        store = self.get_store(collection)
        top_k = k or TOP_K
        retrieve_k = max(RETRIEVE_K, top_k)

        try:
            candidatos = store.max_marginal_relevance_search(
                question, k=retrieve_k, filter=filter)
        except Exception:
            candidatos = store.similarity_search(question, k=retrieve_k, filter=filter)

        reranker = None
        if RERANK_ENABLED and len(candidatos) > 1:
            try:
                reranker = _get_reranker()
            except Exception as e:
                logger.warning(f"[rag_service] reranker no disponible, se usa orden MMR: {e}")
                reranker = None

        if reranker is not None:
            pares = [(question, (d.page_content or "")[:2000]) for d in candidatos]
            puntajes = [float(s) for s in reranker.predict(pares)]
            ordenados = sorted(zip(candidatos, puntajes),
                               key=lambda x: x[1], reverse=True)
            mejor = float(ordenados[0][1]) if ordenados else float("-inf")
            if mejor < RERANK_MIN_SCORE:
                logger.warning(
                    f"[rag_service] sin evidencia (mejor rerank={mejor:.2f} "
                    f"< {RERANK_MIN_SCORE}): {[round(s, 2) for _, s in ordenados[:4]]}")
                return NO_EVIDENCE_MSG, []
            relevant_docs = [d for d, _ in ordenados[:top_k]]
        else:
            relevant_docs = list(candidatos[:top_k])

        context = _format_docs(relevant_docs)
        history_str = _format_history(chat_history or [])
        prompt = self.prompts[collection]
        prompt_value = prompt.format(context=context, chat_history=history_str, question=question)
        response = self.llm.invoke(prompt_value)

        content = response.content if hasattr(response, "content") else str(response)
        return content, relevant_docs

    def generate_title(self, question: str) -> str:
        resp = self.llm.invoke(PROMPT_TITLE.format(question=question[:500]))
        titulo = resp.content.strip().strip('"') if hasattr(resp,'content') else str(resp).strip()
        return titulo[:80]

rag_service = RAGService()
