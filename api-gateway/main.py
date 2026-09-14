from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_jwt
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICES = {
    "clustering": os.getenv("CLUSTERING_SERVICE_URL"),
    "dataset": os.getenv("DATASET_SERVICE_URL"),
    "eda": os.getenv("EDA_SERVICE_URL"),
    "emotions": os.getenv("EMOTIONS_SERVICE_URL"),
    "topicmodeling": os.getenv("TOPIC_MODELING_SERVICE_URL"),
    "wordcloud": os.getenv("WORDCLOUD_SERVICE_URL"),
    "classification": os.getenv("CLASSIFICATION_SERVICE_URL"),
    "rag": os.getenv("RAG_SERVICE_URL"),
    "datacleaning": os.getenv("DATA_CLEANING_SERVICE_URL"),
}

@app.post("/auth/login")
async def login_proxy(request: Request, response: Response):
    async with httpx.AsyncClient() as client:
        upstream = await client.post(
            os.getenv("AUTH_SERVICE_URL") + "/auth/login",
            json=await request.json(),
        )

    for cookie_header in upstream.headers.get_list("set-cookie"):
        response.headers.append("set-cookie", cookie_header)

    return upstream.json()


@app.api_route("/{service}/{operation}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway(service: str, operation: str, request: Request, user: dict = Depends(verify_jwt)):
    if service not in SERVICES:
        raise HTTPException(404, "Servicio no encontrado")

    target_url = f"{SERVICES[service]}/{operation}"

    EXCLUDED_HEADERS = {
        "host", "content-length", "connection",
        "transfer-encoding", "authorization",
    }

    forwarded_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in EXCLUDED_HEADERS
    }

    forwarded_headers["X-User-Token"] = user["raw_token"]
    forwarded_headers["X-User-ID"] = user["sub"]

    body = await request.body()  # bytes crudos, sirve para JSON, multipart, lo que sea

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=forwarded_headers,
            content=body if request.method in ["POST", "PUT"] else None,
            params=dict(request.query_params),
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )