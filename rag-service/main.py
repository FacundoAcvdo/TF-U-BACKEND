import time
import uuid
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Body, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_service import rag_service
from supabase_service import (

    activate_chat_record,
    delete_all_chats,
    delete_chat,
    desactivate_all_chats,
    get_supabase_client,
    create_chat_record,
    get_active_chat,
    create_message,
    get_chat_messages,
    get_active_chat_with_messages,
    get_user_chats_with_stats,
    get_dataset_by_name,
    update_chat_title,
)
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("rag-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_service.initialize()
        pass
    except Exception as e:
        logger.exception(f"[main] Error inicializando RAG en startup: {e}")
    yield


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_supabase(x_user_token: str = Header(..., alias="X-User-Token")):
    try:
        return get_supabase_client(x_user_token)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


class ChatMessageRequest(BaseModel):
    mensaje: str
    dataset_name: str | None = None


class CreateChatRequest(BaseModel):
    dataset_name: str | None = None

class ActivateChatRequest(BaseModel):
    chat_id: str

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


@app.post("/chat")
async def create_chat(
    data: CreateChatRequest = Body(default=CreateChatRequest()),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        dataset_name = data.dataset_name.strip() if data.dataset_name and data.dataset_name.strip() else None
        dataset_id = None
        if dataset_name:
            row = get_dataset_by_name(supabase, x_user_id, dataset_name)
            if not row:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' no encontrado para el usuario")
            dataset_id = row.get("dataset_id")
        chat_id = str(uuid.uuid4())
        chat = create_chat_record(supabase, chat_id, x_user_id, dataset_id=dataset_id)
        return {
            "chat_id": chat_id,
            "fecha_creacion": chat.get("fecha_creacion") if chat else None,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "activo": True,
            "mensajes": [],
            "message": "Chat activo creado (anterior desactivado si existía en su grupo)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[main] Error creando chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno creando chat.")


@app.post("/message")
async def send_message(
    data: ChatMessageRequest = Body(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    pregunta = data.mensaje.strip()
    if not pregunta:
        raise HTTPException(status_code=422, detail="mensaje no puede estar vacío")

    dataset_name = data.dataset_name.strip() if data.dataset_name and data.dataset_name.strip() else None


    dataset_id = None
    collection = "rag_docs"
    rag_filter = None
    if dataset_name:
        try:
            row = get_dataset_by_name(supabase, x_user_id, dataset_name)
            if not row:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' no encontrado para el usuario")

            dataset_id = row.get("dataset_id")
            tipo = (row.get("tipo") or "").strip().lower()
            if tipo in ("chat", "chunked_chat", "chunked_chats", "chunked"):
                collection = "chunked_chats"
            else:
                collection = "documents"
            rag_filter = {
                "$and": [
                    {"user_id": x_user_id},
                    {"dataset_id": dataset_id }
                ]
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"[rag-service] fallo resolviendo dataset | user_id={x_user_id} dataset={dataset_name} causa={type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail="Error interno resolviendo dataset.")


    chat = get_active_chat(supabase, x_user_id, dataset_id=dataset_id)
    if not chat:
        grupo = f"dataset '{dataset_name}'" if dataset_name else "rag_docs"
        raise HTTPException(status_code=404, detail=f"No hay chat activo para {grupo}. Crea uno con POST /chat primero.")
    chat_id = chat["chat_id"]


    mensajes_previos = get_chat_messages(supabase, chat_id, x_user_id)
    historial_previo = [
        {"autor": m.get("autor"), "mensaje": m.get("mensaje")}
        for m in mensajes_previos[-4:]
    ]


    msg_usuario = create_message(supabase, chat_id, x_user_id, "usuario", pregunta)

    historial_para_rag = historial_previo


    if not rag_service.is_store_ready(collection):
        logger.error(f"[main] RAG Not Ready para colección {collection}")
        raise HTTPException(status_code=503, detail=f"RAG service not ready para colección '{collection}'. Please try again later.")

    try:
        respuesta, docs_relevantes = rag_service.query(pregunta, chat_history=historial_para_rag, collection=collection, filter=rag_filter)
    except RuntimeError as re:
        logger.exception(f"[main] RAG RuntimeError: {re}")
        raise HTTPException(status_code=503, detail="Servicio RAG no disponible. Intenta más tarde.")
    except Exception as e:
        logger.exception(f"[main] RAG query error: {e}")
        raise HTTPException(status_code=500, detail="Error interno RAG.")


    try:
        msg_asistente = create_message(supabase, chat_id, x_user_id, "asistente", respuesta)
    except Exception as e:
        logger.warning(f"[main] WARNING no se pudo guardar respuesta asistente: {e}")
        raise HTTPException(status_code=500, detail="Error interno guardando respuesta.")

    es_primer = len(mensajes_previos) == 0
    titulo = None
    if es_primer and not chat.get("titulo"):
        try:
            titulo = rag_service.generate_title(pregunta)
            update_chat_title(supabase, chat_id, x_user_id, titulo)
        except Exception as e:
            logger.warning(f"WARNING título: {e}")

    return {
        "respuesta": msg_asistente,
        "pregunta": msg_usuario,
        "titulo": titulo,
    }


@app.get("/chat")
async def get_chat(
    dataset_name: str | None = Query(default=None),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        dataset_id = None
        if dataset_name and dataset_name.strip():
            row = get_dataset_by_name(supabase, x_user_id, dataset_name.strip())
            if not row:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' no encontrado para el usuario")
            dataset_id = row.get("dataset_id")
        data = get_active_chat_with_messages(supabase, x_user_id, dataset_id)
        if data is None:
            grupo = f"dataset '{dataset_name}'" if dataset_name else "rag_docs"
            raise HTTPException(status_code=404, detail=f"No hay chat activo para {grupo}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[main] Error get_chat activo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.get("/chats")
async def list_chats(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        chats = get_user_chats_with_stats(supabase, x_user_id)
        return {
            "total": len(chats),
            "chats": chats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[main] Error list_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.delete("/chat")
async def delete_chat_record(
    data: ActivateChatRequest = Body(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        delete_chat(supabase, x_user_id, data.chat_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[main] Error delete_chat activo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

@app.delete("/chats")
async def delete_all(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        delete_all_chats(supabase, x_user_id)
        return {"message": f"Todos los chats eliminados para el usuario {x_user_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[main] Error delete_all_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

@app.post("/activate_chat")
async def activate_chat(
    data: ActivateChatRequest = Body(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    chat_id = data.chat_id

    try:
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id es requerido para activar un chat")
        chat = get_active_chat(supabase, x_user_id)
        if chat and chat["chat_id"] == chat_id:
            return {"message": "El chat ya está activo", "chat_id": chat_id}
        desactivate_all_chats(supabase, x_user_id)

        new_chat = activate_chat_record(supabase, chat_id, x_user_id)
        return {
            "message": "Chat activado",
            "chat_id": new_chat["chat_id"],
        }
    except Exception as e:
        logger.exception(f"[main] Error activating chat: {e}")
        raise HTTPException(status_code=500, detail="Error al activar el chat")
