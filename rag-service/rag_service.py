import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from chroma_service import get_chroma_client, get_or_create_collection
from prompts import PROMPTS

# ----------------- Config -----------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.6-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("TOP_K", "4"))

DOCS_PATH = os.environ.get("DOCS_PATH", "/app/data/rag_docs")

RAG_INIT_ON_STARTUP = os.environ.get("RAG_INIT_ON_STARTUP", "false").lower() in ("true", "1", "yes", "on")

# Nombres de colecciones (configurables por env).
RAG_DOCS_COLLECTION = os.environ.get("CHROMA_COLLECTION", "rag_docs")
DOCUMENTS_COLLECTION = os.environ.get("CHROMA_COLLECTION_DOCUMENTS", "documents")
CHUNKED_CHATS_COLLECTION = os.environ.get("CHROMA_COLLECTION_CHUNKED", "chunked_chats")

ALL_COLLECTIONS = {
    "rag_docs": RAG_DOCS_COLLECTION,
    "documents": DOCUMENTS_COLLECTION,
    "chunked_chats": CHUNKED_CHATS_COLLECTION,
}

# ---------- helpers top-level ----------
def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_history(chat_history: List[dict]) -> str:
    if not chat_history:
        return "Sin historial previo."
    lines = []
    for m in chat_history[-6:]:
        autor = m.get("autor", "usuario")
        texto = m.get("mensaje", "")
        lines.append(f"{autor}: {texto}")
    return "\n".join(lines)


class E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings con prefijos obligatorios de E5.

    Documentos indexados -> "passage: ", consultas -> "query: ".
    Sin prefijos la calidad de recuperación cae fuerte.
    """
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


class RAGService:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.chroma_client = None
        self.stores: Dict[str, Chroma] = {}
        self.prompts: Dict[str, ChatPromptTemplate] = {}
        self._initialized = False

    # ---------- Embeddings ----------
    def _init_embeddings(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = E5Embeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": EMBEDDING_BATCH_SIZE,
            },
        )
        if device == "cuda":
            try:
                self.embeddings.client.half()
            except Exception as e:
                print(f"[rag_service] WARNING no se pudo usar fp16: {e}")
        print(f"[rag_service] Embeddings '{EMBEDDING_MODEL}' en {device}")

    # ---------- LLM ----------
    def _init_llm(self):
        if not GOOGLE_API_KEY:
            print("[rag_service] WARNING: GOOGLE_API_KEY no configurada, LLM deshabilitado")
            self.llm = None
            return
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            top_p=0.9,
        )
        print("[rag_service] LLM inicializado")

    # ---------- Chroma stores ----------
    def _init_stores(self):
        self.chroma_client = get_chroma_client()
        for key, coll_name in ALL_COLLECTIONS.items():
            get_or_create_collection(self.chroma_client, name=coll_name)
            self.stores[key] = Chroma(
                client=self.chroma_client,
                collection_name=coll_name,
                embedding_function=self.embeddings,
            )

            template = PROMPTS.get(key)
            self.prompts[key] = ChatPromptTemplate.from_template(template)
            print(f"[rag_service] Store '{key}' -> colección '{coll_name}' inicializado")

    # ---------- rag_docs: carga desde disco ----------
    def _discover_documents(self) -> List[str]:
        if not DOCS_PATH:
            print("[rag_service] DOCS_PATH no configurado -> skip")
            return []
        p = Path(DOCS_PATH)
        if not p.exists():
            print(f"[rag_service] DOCS_PATH no existe: {p.resolve()} -> skip")
            return []
        found_files: List[str] = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            found_files.extend(glob.glob(str(p / ext)))
        found_files = sorted(set(f for f in found_files if Path(f).is_file()))
        if not found_files:
            print(f"[rag_service] WARNING: No se encontraron documentos en {DOCS_PATH}. Monta el volumen en compose: {DOCS_PATH}:/app/data/rag_docs")
        return found_files

    def _load_documents(self) -> List[Document]:
        files = self._discover_documents()
        if not files:
            print(f"[rag_service] No se encontraron documentos en {DOCS_PATH}. Colección rag_docs quedará sin reindexar.")
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
                print(f"[rag_service] Cargado {len(loaded)} páginas de {fpath}")
            except Exception as e:
                print(f"[rag_service] Error cargando {fpath}: {e}")
                continue
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        print(f"[rag_service] Splitting con chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"[rag_service] Chunks generados: {len(splits)}")
        return splits

    def _reindex_rag_docs(self):
        """Reindexa rag_docs si RAG_INIT_ON_STARTUP=true: agrega documentos nuevos encontrados en DOCS_PATH."""
        if not RAG_INIT_ON_STARTUP:
            print("[rag_service] RAG_INIT_ON_STARTUP=false -> skip reindex rag_docs")
            return
        docs = self._load_documents()
        splits = self._split_documents(docs)
        if not splits:
            print("[rag_service] No hay splits para reindexar rag_docs")
            return
        store = self.stores.get("rag_docs")
        if store is None:
            print("[rag_service] Store rag_docs no inicializado, skip reindex")
            return
        print(f"[rag_service] Reindexando rag_docs con {len(splits)} chunks (append, sin borrar existentes)")
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            store.add_documents(documents=batch)
        print("[rag_service] Reindex rag_docs completado")

    def initialize(self):
        if self._initialized:
            return
        self._init_embeddings()
        self._init_llm()
        self._init_stores()
        # Solo rag_docs se reindexa condicionalmente; documents/chunked_chats consumen embeddings ya en Chroma
        try:
            self._reindex_rag_docs()
        except Exception as e:
            print(f"[rag_service] Error en reindex rag_docs: {e}")
        self._initialized = True
        print("[rag_service] Inicialización completa - 3 stores listos")

    def is_ready(self) -> bool:
        return self._initialized and self.llm is not None and len(self.stores) == 3

    def is_store_ready(self, collection: str) -> bool:
        return self._initialized and collection in self.stores and self.llm is not None

    def get_store(self, collection: str) -> Chroma:
        if collection not in self.stores:
            raise ValueError(f"Colección '{collection}' no existe. Disponibles: {list(self.stores.keys())}")
        return self.stores[collection]

    # ---------- Query genérico por colección ----------
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

        # Búsqueda con filtro server-side (Chroma where)
        if filter:
            relevant_docs = store.similarity_search(question, k=top_k, filter=filter)
        else:
            relevant_docs = store.similarity_search(question, k=top_k)

        context = _format_docs(relevant_docs)
        history_str = _format_history(chat_history or [])
        prompt = self.prompts[collection]
        prompt_value = prompt.format(context=context, chat_history=history_str, question=question)
        response = self.llm.invoke(prompt_value)
        # ChatGoogleGenerativeAI retorna AIMessage
        content = response.content if hasattr(response, "content") else str(response)
        return content, relevant_docs

rag_service = RAGService()
