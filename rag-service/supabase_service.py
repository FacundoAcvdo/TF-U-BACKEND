import os
from datetime import datetime, timezone, timedelta
from supabase import create_client
from fastapi import HTTPException
from typing import List, Dict, Any, Optional
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("rag-service")


try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
except Exception:
    LOCAL_TZ = timezone(timedelta(hours=-3))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")


def _to_public_mensaje(row: Dict[str, Any]) -> Dict[str, Any]:
    """Filtra solo columnas públicas para no exponer user_id/chat_id/id."""
    return {
        "autor": row.get("autor"),
        "mensaje": row.get("mensaje"),
        "created_at": row.get("created_at"),
    }


def _format_fecha_hora(iso_str: Optional[str]) -> tuple[str, str]:
    """Parsea ISO UTC '2026-09-10T18:15:35.749321+00:00' -> fecha/hora en America/Argentina/Buenos_Aires."""
    if not iso_str:
        return "", ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_local = dt.astimezone(LOCAL_TZ)
        return dt_local.strftime("%Y-%m-%d"), dt_local.strftime("%H:%M:%S")
    except Exception:
        return iso_str[:10] if len(iso_str) >= 10 else "", iso_str[11:19] if len(iso_str) > 19 else ""


def _normalize_autor(autor: str) -> str:
    return "usuario" if autor.lower().strip() in ("usuario", "user", "human") else "asistente"


def _get_user_chats(supabase, user_id: str) -> List[Dict[str, Any]]:
    """Uso interno: lista chats del usuario ordenados por creación desc."""
    try:
        resp = supabase.table("chats").select("*").eq("user_id", user_id).order("fecha_creacion", desc=True).execute()
        return resp.data or []
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR _get_user_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno listando chats.")


def _get_chat_record(supabase, chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Uso interno."""
    try:
        resp = supabase.table("chats").select("*").eq("chat_id", chat_id).eq("user_id", user_id).execute()
        if not resp.data:
            return None
        return resp.data[0]
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR _get_chat_record: {e}")
        raise HTTPException(status_code=500, detail="Error interno obteniendo chat.")


def get_supabase_client(user_jwt: str):
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY no configurados")
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client

def get_dataset_by_name(supabase, user_id: str, dataset_name: str) -> Optional[Dict[str, Any]]:
    try:

        resp = supabase.table("datasets").select("dataset_id,dataset_name,tipo").eq("user_id", user_id).eq("dataset_name", dataset_name).limit(1).execute()
        if not resp.data:
            return None
        return resp.data[0]
    except Exception as e:

        try:
            resp = supabase.table("datasets").select("dataset_id,dataset_name").eq("user_id", user_id).eq("dataset_name", dataset_name).limit(1).execute()
            if not resp.data:
                return None
            row = resp.data[0]
            row["tipo"] = None
            return row
        except Exception as e2:
            logger.exception(f"[supabase_service] ERROR get_dataset_by_name: {e2}")
            raise HTTPException(status_code=500, detail="Error interno buscando dataset '{dataset_name}'.")

def activate_chat_record(supabase, chat_id: str, user_id: str):
    try:
        resp = supabase.table("chats").update({"activo": True}).eq("chat_id", chat_id).eq("user_id", user_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' no encontrado para el usuario")
        pass
        return resp.data[0]
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR activate_chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno activando chat.")

def desactivate_all_chats(supabase, user_id: str):
    try:
        resp = supabase.table("chats").update({"activo": False}).eq("user_id", user_id).execute()
        count = len(resp.data) if resp.data else 0
        if count:
            logger.warning(f"[supabase_service] Desactivados {count} chats previos de user {user_id}")
        return count
    except Exception as e:
        logger.warning(f"[supabase_service] ERROR desactivate_all_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno desactivando chats previos.")

def deactivate_active_chats(supabase, user_id: str, dataset_id: Optional[str] = None):
    try:
        if dataset_id is None:
            resp = supabase.table("chats").update({"activo": False}).eq("user_id", user_id).eq("activo", True).is_("dataset_id", "null").execute()
        else:
            try:
                resp = supabase.table("chats").update({"activo": False}).eq("user_id", user_id).eq("activo", True).not_.is_("dataset_id", "null").execute()
            except Exception:
                active = supabase.table("chats").select("chat_id,dataset_id").eq("user_id", user_id).eq("activo", True).execute()
                ids = [r["chat_id"] for r in (active.data or []) if r.get("dataset_id") is not None]
                if not ids:
                    return 0
                resp = supabase.table("chats").update({"activo": False}).in_("chat_id", ids).execute()
        count = len(resp.data) if resp.data else 0
        if count:
            logger.warning(f"[supabase_service] Desactivados {count} chats previos de user {user_id} grupo={'rag_docs' if dataset_id is None else 'dataset'}")
        return count
    except Exception as e:
        if "activo" in str(e).lower() or "column" in str(e).lower():
            logger.warning(f"[supabase_service] WARNING deactivate: columna activo no existe, ejecutar migración: {e}")
            return 0
        logger.exception(f"[supabase_service] ERROR deactivate_active_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno desactivando chats previos.")

def create_chat_record(
    supabase,
    chat_id: str,
    user_id: str,
    dataset_id: Optional[str] = None,
    privacy_mode: bool = False,
):
    payload = {
        "chat_id": chat_id,
        "user_id": user_id,
        "activo": True,
        "dataset_id": dataset_id,
        "privacy_mode": bool(privacy_mode),
    }
    try:
        deactivate_active_chats(supabase, user_id, dataset_id)
        resp = supabase.table("chats").insert(payload).execute()
        pass
        return resp.data[0] if resp.data else None
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "idx_chats_one_active" in msg or "duplicate" in msg:
            logger.exception(f"[supabase_service] Conflicto chat activo, reintentando deactivate: {e}")
            try:
                deactivate_active_chats(supabase, user_id, dataset_id)
                resp = supabase.table("chats").insert(payload).execute()
                return resp.data[0] if resp.data else None
            except Exception as e2:
                raise HTTPException(status_code=500, detail="Error interno creando chat activo.")
        if "privacy_mode" in msg:
            logger.warning(f"[supabase_service] WARNING columna privacy_mode no existe, reintentando sin ella: {e}")
            fallback = {**payload}
            fallback.pop("privacy_mode", None)
            try:
                resp = supabase.table("chats").insert(fallback).execute()
                return resp.data[0] if resp.data else None
            except Exception as e2:
                raise HTTPException(status_code=500, detail="Error interno creando chat sin modo privacidad.")
        if "dataset_id" in msg:
            logger.warning(f"[supabase_service] WARNING columna dataset_id no existe, reintentando sin ella: {e}")
            fallback = {
                "chat_id": chat_id,
                "user_id": user_id,
                "activo": True,
                "privacy_mode": bool(privacy_mode),
            }
            try:
                deactivate_active_chats(supabase, user_id, None)
                resp = supabase.table("chats").insert(fallback).execute()
                return resp.data[0] if resp.data else None
            except Exception as e3:
                if "privacy_mode" in str(e3).lower():
                    fallback.pop("privacy_mode", None)
                    resp = supabase.table("chats").insert(fallback).execute()
                    return resp.data[0] if resp.data else None
                raise HTTPException(status_code=500, detail="Error interno creando chat fallback.")
        logger.exception(f"[supabase_service] ERROR insert chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno creando chat en Supabase.")

def get_active_chat(supabase, user_id: str, dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        q = supabase.table("chats").select("*").eq("user_id", user_id).eq("activo", True)
        if dataset_id is None:
            q = q.is_("dataset_id", "null")
        else:
            q = q.eq("dataset_id", dataset_id)
        resp = q.limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR get_active_chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno obteniendo chat activo.")

def delete_chat_record(supabase, chat_id: str, user_id: str):
    try:
        resp = supabase.table("chats").delete().eq("chat_id", chat_id).eq("user_id", user_id).execute()
        return resp.data
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR delete_chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno eliminando chat.")

def create_message(supabase, chat_id: str, user_id: str, autor: str, mensaje: str) -> Dict[str, Any]:
    """Inserta y devuelve SOLO columnas públicas: autor, mensaje, created_at."""
    try:
        autor_norm = _normalize_autor(autor)
        resp = supabase.table("chat_mensajes").insert({"chat_id": chat_id, "user_id": user_id, "autor": autor_norm, "mensaje": mensaje}).execute()
        pass
        if not resp.data:
            return None
        return _to_public_mensaje(resp.data[0])
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR insert chat_mensajes: {e}")
        raise HTTPException(status_code=500, detail="Error interno guardando mensaje.")

def get_chat_messages(supabase, chat_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Devuelve solo columnas públicas para no filtrar user_id/chat_id/id."""
    try:
        resp = supabase.table("chat_mensajes").select("autor,mensaje,created_at").eq("chat_id", chat_id).eq("user_id", user_id).order("created_at").execute()
        return [_to_public_mensaje(r) for r in (resp.data or [])]
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR get_chat_messages: {e}")
        raise HTTPException(status_code=500, detail="Error interno obteniendo mensajes.")

def get_active_chat_with_messages(supabase, user_id: str, dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    chat = get_active_chat(supabase, user_id, dataset_id)
    if not chat:
        return None
    chat_full = _get_chat_record(supabase, chat["chat_id"], user_id)
    if not chat_full:
        return None
    mensajes_raw = get_chat_messages(supabase, chat["chat_id"], user_id)
    mensajes = [_to_public_mensaje(m) for m in mensajes_raw]
    return {
        "chat_id": chat_full.get("chat_id"),
        "titulo": chat_full.get("titulo"),
        "fecha_creacion": chat_full.get("fecha_creacion"),
        "dataset_id": chat_full.get("dataset_id"),
        "privacy_mode": bool(chat_full.get("privacy_mode", False)),
        "mensajes": mensajes,
    }

def get_user_chats_with_stats(supabase, user_id: str) -> List[Dict[str, Any]]:
    chats = _get_user_chats(supabase, user_id)
    result = []
    for c in chats:
        cid = c.get("chat_id")
        try:
            msgs = supabase.table("chat_mensajes").select("autor,mensaje,created_at").eq("chat_id", cid).eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            count_resp = supabase.table("chat_mensajes").select("id", count="exact").eq("chat_id", cid).eq("user_id", user_id).execute()
            cantidad = count_resp.count if hasattr(count_resp, "count") and count_resp.count is not None else len(count_resp.data or [])
            ultimo = None
            if msgs.data:
                ultimo = _to_public_mensaje(msgs.data[0])
        except Exception as e:
            logger.warning(f"[supabase_service] WARNING stats chat {cid}: {e}")
            cantidad = 0
            ultimo = None
        result.append({"titulo": c.get("titulo"), "chat_id": cid, "fecha_creacion": c.get("fecha_creacion"), "updated_at": c.get("updated_at"), "activo": c.get("activo"), "dataset_id": c.get("dataset_id"), "cantidad_mensajes": cantidad, "ultimo_mensaje": ultimo})
    return result

def delete_chat(supabase, user_id: str, chat_id: str):
    try:
        supabase.table("chat_mensajes").delete().eq("chat_id", chat_id).eq("user_id", user_id).execute()
        supabase.table("chats").delete().eq("chat_id", chat_id).eq("user_id", user_id).execute()
        pass
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR delete_chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno eliminando chat.")

def delete_all_chats(supabase, user_id: str):
    try:
        supabase.table("chat_mensajes").delete().eq("user_id", user_id).execute()
        supabase.table("chats").delete().eq("user_id", user_id).execute()
        pass
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR delete_all_chats: {e}")
        raise HTTPException(status_code=500, detail="Error interno eliminando todos los chats.")

def update_chat_title(supabase, chat_id, user_id, titulo):
    supabase.table("chats").update({"titulo": titulo}).eq("chat_id", chat_id).eq("user_id", user_id).execute()


def update_chat_privacy_mode(supabase, chat_id: str, user_id: str, privacy_mode: bool):
    try:
        resp = (
            supabase.table("chats")
            .update({"privacy_mode": bool(privacy_mode)})
            .eq("chat_id", chat_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="Chat no encontrado.")
        return resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[supabase_service] ERROR update privacy mode: {e}")
        raise HTTPException(
            status_code=503,
            detail="Falta aplicar la migración privacy_mode en la tabla chats.",
        )
