from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, EmailStr
from supabase import create_client
import os

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/login")
async def login(data: LoginRequest, response: Response):

    print(data)

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

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/register")
async def registro(data: RegisterRequest):
    try:
        response = supabase.auth.sign_up({
            "email": data.email,
            "password": data.password,
        })
    except Exception as e:
        raise HTTPException(400, f"No se pudo crear el usuario: {str(e)}")

    if response.user is None:
        raise HTTPException(400, "No se pudo crear el usuario")

    return {
        "message": "Usuario registrado exitosamente",
        "user_id": response.user.id
    }

@app.get("/")
async def root():
    return {"message": "Auth Service is running"}