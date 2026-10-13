import os
from fastapi import Cookie, HTTPException
from jwt import PyJWKClient
import jwt

SUPABASE_URL = os.environ.get("SUPABASE_URL")
JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

jwks_client = PyJWKClient(JWKS_URL)


def verify_jwt(access_token: str = Cookie(None)):
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
