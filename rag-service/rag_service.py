import os
import re
import glob
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from chroma_service import get_chroma_client, get_or_create_collection, COLLECTION_NAME

# ----------------- Config -----------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.6-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("TOP_K", "4"))

DOCS_PATH = os.environ.get("DOCS_PATH", "/app/data/rag_docs")

RAG_INIT_ON_STARTUP = os.environ.get("RAG_INIT_ON_STARTUP", "false").lower() in ("true", "1", "yes", "on")

# ---------- Prompt técnico para investigadores (conciso) ----------
SYSTEM_PROMPT_TEMPLATE = """Eres un asistente técnico para INVESTIGADORES en salud mental (depresión, ansiedad, factores de riesgo y prevención del suicidio).

Objetivo: responder de forma CONCISA, directa y sin rodeos, pero sin omitir detalles críticos.

Reglas:
- Responde EXACTAMENTE lo preguntado. No añadas introducciones largas, historia general, ni conclusiones genéricas si no se piden.
- Usa ÚNICAMENTE el contexto proporcionado. Si falta info, di: "No tengo información suficiente en los documentos indexados sobre ese punto" y sugiere qué documento/sección podría contenerla.
- Sé conciso: apunta a 120-220 palabras, párrafos cortos o bullets. Solo expande si la pregunta pide profundidad.
- Incluye solo detalles importantes: definición precisa, criterios clave (DSM-5/CIE-11), clasificaciones/subtipos y diferencias con conceptos afines cuando aporten valor. Omite ejemplos y matices secundarios salvo que sean esenciales.
- Formato OBLIGATORIO en Markdown GFM (el frontend lo renderiza bonito):
  * Usa **negrita** para términos clave.
  * Usa listas con guión `- ` para enumerar ítems (ej. Puntos de Función desglosados).
  * Para datos tabulares (ej. PF por módulo) usa SIEMPRE tabla GFM con cabecera y separador: `| Función | Tipo | Complejidad | PF |` + `|---|---|---|---|` + filas. Nunca escribas tablas como texto con `|` sin cabecera.
  * Deja una línea en blanco entre párrafos, listas y tablas.
- Cita general sin inventar (ej. "según documento X, sección Y").
- No diagnostiques, no etiquetes personas, no prescribas. Si la pregunta insinúa un caso, aclara que es información general de investigación.
- Tono técnico, objetivo, en español (salvo que pregunten en otro idioma). Nada de tono empático-terapéutico.
- No expliques tu razonamiento paso a paso; ve al resultado.
- No añadas recursos de crisis salvo pedido explícito.
- Cierra solo si corresponde con: "Nota: Información con fines educativos y de investigación, no sustituye evaluación profesional."

Contexto recuperado:
{context}

Historial (si existe):
{chat_history}

Pregunta:
{question}

Respuesta concisa:"""

# ---------- helpers top-level (evitan redefinir en cada _build_chains) ----------
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


class RAGService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self._initialized = False

    # ---------- Embeddings ----------
    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("[rag_service] Embeddings inicializados")

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
            print(f"[rag_service] No se encontraron documentos en {DOCS_PATH}. Colección quedará vacía hasta montar el volumen.")
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

    def _init_vectorstore(self, splits: List[Document] = None):
        chroma_client = get_chroma_client()
        get_or_create_collection(chroma_client)

        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        if splits:
            print(f"[rag_service] Inicializando vectorstore con {len(splits)} documentos")
            batch_size = 100
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i+batch_size]
                self.vectorstore.add_documents(documents=batch)
        else:
            print("[rag_service] Inicializando vectorstore sin documentos (colección vacía)")

        print("[rag_service] Vectorstore inicializado")
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )

    def _build_chains(self):
        if not self.llm or not self.retriever:
            return
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        self.rag_chain = (
            {
                "context": RunnableLambda(lambda x: _format_docs(self.retriever.invoke(x["question"]))),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: _format_history(x.get("chat_history", []))),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def initialize(self):
        if self._initialized:
            return
        self._init_embeddings()
        self._init_llm()
        if RAG_INIT_ON_STARTUP:
            docs = self._load_documents()
            splits = self._split_documents(docs)
            self._init_vectorstore(splits)
        else:
            self._init_vectorstore(splits=None)
        self._build_chains()
        self._initialized = True

    def is_ready(self) -> bool:
        return self._initialized and self.retriever is not None and self.llm is not None

    def query(self, question: str, chat_history: List[dict] = None) -> Tuple[str, List[Document]]:
        if not self.is_ready():
            raise RuntimeError("RAG no inicializado")
        relevant_docs = self.retriever.invoke(question)
        response = self.rag_chain.invoke({
            "question": question,
            "chat_history": chat_history or []
        })
        return response, relevant_docs

rag_service = RAGService()
