import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_service import rag_service
from supabase_service import (
    get_supabase_client,
    create_chat_record,
    get_active_chat,
    delete_chat_record,
    create_message,
    get_chat_messages,
    get_active_chat_with_messages,
    get_user_chats_with_stats,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_service.initialize()
        print("[main] RAG inicializado en startup")
    except Exception as e:
        print(f"[main] Error inicializando RAG en startup: {e}")
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
        raise HTTPException(status_code=500, detail=str(e))


class ChatMessageRequest(BaseModel):
    mensaje: str

@app.post("/chat")
async def create_chat(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        chat_id = str(uuid.uuid4())
        chat = create_chat_record(supabase, chat_id, x_user_id)
        return {
            "chat_id": chat_id,
            "fecha_creacion": chat.get("fecha_creacion") if chat else None,
            "activo": True,
            "mensajes": [],
            "message": "Chat activo creado (anterior desactivado si existía)"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main] Error creando chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error creando chat: {str(e)}")


@app.post("/message")
async def send_message(
    data: ChatMessageRequest = Body(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    pregunta = data.mensaje.strip()
    if not pregunta:
        raise HTTPException(status_code=422, detail="mensaje no puede estar vacío")

    # 1. Inferir chat activo
    chat = get_active_chat(supabase, x_user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="No hay chat activo. Crea uno con POST /chat primero.")
    chat_id = chat["chat_id"]

    # 2. Historial previo para contexto RAG
    mensajes_previos = get_chat_messages(supabase, chat_id, x_user_id)
    historial_previo = [
        {"autor": m.get("autor"), "mensaje": m.get("mensaje")}
        for m in mensajes_previos
    ]

    # Insertar mensaje usuario y guardar copia pública para eventual uso
    msg_usuario = create_message(supabase, chat_id, x_user_id, "usuario", pregunta)

    historial_para_rag = historial_previo + [{"autor": "usuario", "mensaje": pregunta}]

    # 3. RAG - fix bug previo: variable 're' no definida
    if not rag_service.is_ready():
        print("[main] RAG Not Ready")
        raise HTTPException(status_code=503, detail="RAG service not ready. Please try again later.")

    try:
        respuesta, docs_relevantes = rag_service.query(pregunta, chat_history=historial_para_rag)
    except RuntimeError as re:
        print(f"[main] RAG RuntimeError: {re}")
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        print(f"[main] RAG query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error RAG: {str(e)}")

    # 4. Guardar respuesta
    try:
        msg_asistente = create_message(supabase, chat_id, x_user_id, "asistente", respuesta)
    except Exception as e:
        print(f"[main] WARNING no se pudo guardar respuesta asistente: {e}")
        raise HTTPException(status_code=500, detail=f"Error guardando respuesta: {str(e)}")

    return {
        "respuesta": msg_asistente,
        "pregunta": msg_usuario,
    }


@app.get("/chat")
async def get_chat(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        data = get_active_chat_with_messages(supabase, x_user_id)
        if data is None:
            raise HTTPException(status_code=404, detail="No hay chat activo para este usuario")
        return data
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main] Error get_chat activo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats")
async def list_chats(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        chats = get_user_chats_with_stats(supabase, x_user_id)
        return {
            "user_id": x_user_id,
            "total": len(chats),
            "chats": chats
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main] Error list_chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat")
async def delete_chat(
    x_user_id: str = Header(..., alias="X-User-ID"),
    supabase=Depends(get_supabase),
):
    try:
        chat = get_active_chat(supabase, x_user_id)
        if not chat:
            raise HTTPException(status_code=404, detail="No hay chat activo para eliminar")
        delete_chat_record(supabase, chat["chat_id"], x_user_id)
        return {"message": "Chat activo eliminado", "chat_id": chat["chat_id"]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main] Error delete_chat activo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
