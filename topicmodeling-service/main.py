from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

from fastapi import (
    Body,
    FastAPI,
    Header,
    HTTPException,
)

from topic_modeling_service import TopicModelingService
from supabase_service import SupabaseService
from chroma_service import ChromaService
from topic_modeling_provider import TopicModelingProvider

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_service = SupabaseService()
chroma_service = ChromaService()
topic_provider = TopicModelingProvider()

topic_modeling_service = TopicModelingService(
    supabase_repository=supabase_service,
    chroma_repository=chroma_service,
    topic_provider=topic_provider,
)

@app.post("/getTopics")
async def get_topics(
    dataset: str = Body(..., embed=True),
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    print(f"Obteniendo tópicos para dataset: {dataset}")

    try:
        resultado = await topic_modeling_service.get_topics(
            dataset_name=dataset,
            user_id=x_user_id,
            user_token=x_user_token,
        )

        return {
            "status": "success",
            **resultado,
        }

    except ValueError as exc:
        print(f"[VALUE ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"Error ejecutando topic modeling: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Error interno procesando el topic modeling.",
        ) from exc