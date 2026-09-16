from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

from fastapi import (
    Body,
    FastAPI,
    Header,
    HTTPException,
)

from config import get_settings
from topic_modeling_service import TopicModelingService
from supabase_service import SupabaseService
from chroma_service import ChromaService
from topic_modeling_provider import TopicModelingProvider
from storage_service import StorageService

settings = get_settings()
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
storage_service = StorageService(
    settings.dataset_storage_path
)

topic_modeling_service = TopicModelingService(
    supabase_repository=supabase_service,
    chroma_repository=chroma_service,
    topic_provider=topic_provider,
    storage_service=storage_service
)

@app.post("/getTopics")
async def get_topics(
    dataset: str = Body(..., embed=True),
    max_topics: int = Body(30, embed=True),

    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        resultado = await topic_modeling_service.get_topics(
            dataset_name=dataset,
            user_id=x_user_id,
            user_token=x_user_token,
            max_topics=max_topics,
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

        raise HTTPException(
            status_code=500,
            detail="Error interno procesando el topic modeling.",
        ) from exc