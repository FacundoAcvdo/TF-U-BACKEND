from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, EmailStr
from supabase import create_client
from typing import Literal, Optional
import os

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

ADMIN_ROLE = "admin"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/login")
async def login(data: LoginRequest, response: Response):

    try:
        result = supabase.auth.sign_in_with_password({
            "email": data.email,
            "password": data.password
        })
    except Exception:
        raise HTTPException(401, "Email o contraseña incorrectos")

    response.set_cookie(
        key="access_token",
        value=result.session.access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=result.session.expires_in,
        path="/",
    )

    return {
        "user": {
            "id": result.user.id,
            "email": result.user.email,
        }
    }

def _admin_client():
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(503, "Administración no configurada: falta SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def _exigir_admin(x_user_id: Optional[str]):
    if not x_user_id:
        raise HTTPException(401, "No autenticado")
    try:
        resp = _admin_client().auth.admin.get_user_by_id(x_user_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(403, "Sin permiso")
    meta = {}
    if resp.user is not None and isinstance(resp.user.app_metadata, dict):
        meta = resp.user.app_metadata
    if meta.get("role") != ADMIN_ROLE:
        raise HTTPException(403, "Sin permiso: se requiere super admin")
    return resp.user

def _resumen_usuario(u) -> dict:
    meta = u.app_metadata if isinstance(getattr(u, "app_metadata", None), dict) else {}
    return {
        "id": u.id,
        "email": u.email,
        "role": meta.get("role", "user"),
        "created_at": str(getattr(u, "created_at", None)),
        "last_sign_in_at": str(getattr(u, "last_sign_in_at", None)),
    }

class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    role: Literal["admin", "user"] = "user"

class AdminRoleRequest(BaseModel):
    role: Literal["admin", "user"]

class AdminPasswordRequest(BaseModel):
    password: str

@app.get("/auth/me")
async def me(x_user_id: Optional[str] = Header(default=None, alias="X-User-ID")):
    if not x_user_id:
        raise HTTPException(401, "No autenticado")
    try:
        resp = _admin_client().auth.admin.get_user_by_id(x_user_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "No autenticado")
    if resp.user is None:
        raise HTTPException(401, "No autenticado")
    return {"user": _resumen_usuario(resp.user)}

@app.get("/auth/admin/users")
async def admin_listar_usuarios(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
    page: int = 1,
    per_page: int = 50,
):
    _exigir_admin(x_user_id)
    try:
        usuarios = _admin_client().auth.admin.list_users(page=page, per_page=per_page)
    except Exception as e:
        raise HTTPException(500, f"No se pudo listar usuarios: {str(e)}")
    return {"users": [_resumen_usuario(u) for u in usuarios]}

@app.post("/auth/admin/users")
async def admin_crear_usuario(
    data: AdminCreateUserRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
):
    _exigir_admin(x_user_id)
    try:
        resp = _admin_client().auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True,
            "app_metadata": {"role": data.role},
        })
    except Exception as e:
        raise HTTPException(400, f"No se pudo crear el usuario: {str(e)}")
    return {"message": "Usuario creado", "user": _resumen_usuario(resp.user)}

@app.patch("/auth/admin/users/{user_id}/role")
async def admin_cambiar_rol(
    user_id: str,
    data: AdminRoleRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
):
    actor = _exigir_admin(x_user_id)
    if user_id == actor.id and data.role != ADMIN_ROLE:
        raise HTTPException(400, "No podés quitarte tu propio rol de super admin")
    try:
        resp = _admin_client().auth.admin.update_user_by_id(
            user_id, {"app_metadata": {"role": data.role}}
        )
    except Exception as e:
        raise HTTPException(400, f"No se pudo cambiar el rol: {str(e)}")
    return {"message": "Rol actualizado", "user": _resumen_usuario(resp.user)}

@app.patch("/auth/admin/users/{user_id}/password")
async def admin_cambiar_password(
    user_id: str,
    data: AdminPasswordRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
):
    _exigir_admin(x_user_id)
    if not data.password or len(data.password) < 6:
        raise HTTPException(400, "La contraseña debe tener al menos 6 caracteres")
    try:
        _admin_client().auth.admin.update_user_by_id(
            user_id, {"password": data.password}
        )
    except Exception as e:
        raise HTTPException(400, f"No se pudo cambiar la contraseña: {str(e)}")
    return {"message": "Contraseña actualizada. El usuario deberá iniciar sesión de nuevo."}

@app.delete("/auth/admin/users/{user_id}")
async def admin_eliminar_usuario(
    user_id: str,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
):
    actor = _exigir_admin(x_user_id)
    if user_id == actor.id:
        raise HTTPException(400, "No podés eliminar tu propio usuario")
    try:
        existentes = _admin_client().auth.admin.list_users(page=1, per_page=100)
        admins = [
            u for u in existentes
            if isinstance(getattr(u, "app_metadata", None), dict)
            and u.app_metadata.get("role") == ADMIN_ROLE
        ]
        if any(u.id == user_id for u in admins) and len(admins) <= 1:
            raise HTTPException(400, "No se puede eliminar al último super admin")
        _admin_client().auth.admin.delete_user(user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"No se pudo eliminar el usuario: {str(e)}")
    return {"message": "Usuario eliminado"}

@app.post("/auth/logout")
async def logout(
    response: Response,
    x_user_token: Optional[str] = Header(default=None, alias="X-User-Token"),
):
    if x_user_token:
        try:
            _admin_client().auth.admin.sign_out(x_user_token)
        except Exception:
            pass
    response.delete_cookie(key="access_token", path="/")
    return {"message": "Sesión cerrada exitosamente"}

@app.get("/")
async def root():
    return {"message": "Auth Service is running"}
