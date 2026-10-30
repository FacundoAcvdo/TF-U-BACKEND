import os
from fastapi import Cookie, Header, HTTPException
from jwt import PyJWKClient
import jwt

SUPABASE_URL = os.environ.get("SUPABASE_URL")
JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

jwks_client = PyJWKClient(JWKS_URL)

# Caché en memoria de usuarios que aceptaron los términos durante la sesión actual
# (evita consultar auth-service cuando el JWT fue emitido justo antes de aceptar).
_accepted_users_cache: set[str] = set()


def mark_terms_accepted(user_id: str) -> None:
    if user_id:
        _accepted_users_cache.add(user_id)


def has_accepted_terms_claim(user: dict) -> bool:
    app_meta = user.get("app_metadata")
    if isinstance(app_meta, dict) and app_meta.get("terms_accepted") is True:
        return True
    user_id = user.get("sub")
    return bool(user_id and user_id in _accepted_users_cache)


def verify_jwt(access_token: str = Cookie(None), authorization: str = Header(None)):
    if access_token is None and authorization:
        esquema, _, credenciales = authorization.partition(" ")
        if esquema.lower() == "bearer" and credenciales:
            access_token = credenciales
    if access_token is None:
        raise HTTPException(401, "No autenticado")

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(access_token)
        payload = jwt.decode(
            access_token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Sesión expirada")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Token inválido")

    payload["raw_token"] = access_token
    return payload
