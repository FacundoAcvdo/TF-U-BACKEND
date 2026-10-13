from typing import Any

from supabase import create_client

from config import get_settings
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("topicmodeling-service")


TOPICS_TABLE = "topicmodeling_results"


TOPICS_SAVE_WARN_BYTES = 500_000

TOPICS_TRUNCATED_CHARS = 500


def _truncate_topic_texts(
    topics: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    truncated: list[dict[str, Any]] = []

    for topic in topics:
        copy = dict(topic)
        docs = []

        for doc in topic.get("representative_documents", []):
            text = str(doc.get("text", ""))
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0] + "..."
            docs.append({**doc, "text": text})

        copy["representative_documents"] = docs
        truncated.append(copy)

    return truncated


def _strip_topic_texts(
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    minimal: list[dict[str, Any]] = []

    for topic in topics:
        copy = dict(topic)
        copy["representative_documents"] = [
            {"document_id": doc.get("document_id", ""), "text": ""}
            for doc in topic.get("representative_documents", [])
        ]
        minimal.append(copy)

    return minimal


def _payload_size_bytes(topics: list[dict[str, Any]]) -> int:
    import json

    try:
        return len(json.dumps(topics, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _looks_like_size_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "413",
            "54000",
            "payload",
            "too large",
            "content too large",
            "row is too big",
            "statement timeout",
            "timeout",
        )
    )


class SupabaseService:

    def __init__(self) -> None:

        settings = get_settings()

        if not settings.supabase_url:
            raise RuntimeError(
                "SUPABASE_URL no está configurado."
            )

        self.base_url = (
            settings.supabase_url.rstrip("/")
        )

        self.headers = {
            "apikey": settings.supabase_secret_key,
            "Authorization": (
                "Bearer "
                + settings.supabase_secret_key
            ),
            "Content-Type": "application/json",
        }

        self.table = (
            settings.supabase_datasets_table
        )

        self.anon_key = (
            settings.supabase_anon_key
        )

    def get_supabase_client(
        self,
        user_jwt: str,
    ):

        client = create_client(
            self.base_url,
            self.anon_key,
        )

        client.postgrest.auth(user_jwt)

        return client

    async def get_dataset_by_name(
        self,
        *,
        dataset_name: str,
        user_id: str,
        user_token: str,
    ) -> dict[str, Any] | None:
        pass
        supabase = self.get_supabase_client(
            user_jwt=user_token,
        )
        pass
        response = (
            supabase
            .table(self.table)
            .select("dataset_id, dataset_name")
            .eq("user_id", user_id)
            .eq("dataset_name", dataset_name)
            .limit(1)
            .execute()
        )

        if not response.data:

            return None

        return response.data[0]

    async def get_topic_results(
        self,
        *,
        user_id: str,
        dataset_id: str,
        max_topics: int,
        user_token: str,
    ) -> dict[str, Any] | None:
        supabase = self.get_supabase_client(
            user_jwt=user_token,
        )

        response = (
            supabase
            .table(TOPICS_TABLE)
            .select(
                "document_count, topic_count, outlier_count, "
                "outlier_percentage, topics"
            )
            .eq("user_id", user_id)
            .eq("dataset_id", dataset_id)
            .eq("max_topics", max_topics)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        row = response.data[0]

        return {
            "document_count": row.get("document_count", 0),
            "topic_count": row.get("topic_count", 0),
            "outlier_count": row.get("outlier_count", 0),
            "outlier_percentage": row.get("outlier_percentage", 0),
            "topics": row.get("topics") or [],
        }

    async def save_topic_results(
        self,
        *,
        user_id: str,
        dataset_id: str,
        max_topics: int,
        resultado: dict[str, Any],
        user_token: str,
    ) -> None:
        """Guarda best-effort: nunca lanza, solo loguea si falla todo."""
        supabase = self.get_supabase_client(
            user_jwt=user_token,
        )

        topics_full = resultado.get("topics", [])

        levels: list[tuple[str, list[dict[str, Any]]]] = [
            ("completo", topics_full),
            (
                "truncado",
                _truncate_topic_texts(
                    topics_full,
                    TOPICS_TRUNCATED_CHARS,
                ),
            ),
            ("minimo", _strip_topic_texts(topics_full)),
        ]

        start = 0
        if _payload_size_bytes(topics_full) > TOPICS_SAVE_WARN_BYTES:
            pass
            start = 1

        base = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "max_topics": max_topics,
            "document_count": resultado.get("document_count", 0),
            "topic_count": resultado.get("topic_count", 0),
            "outlier_count": resultado.get("outlier_count", 0),
            "outlier_percentage": resultado.get("outlier_percentage", 0),
        }

        for name, topics in levels[start:]:
            try:
                self._upsert_topics(
                    supabase,
                    {**base, "topics": topics},
                )
                if name != "completo":
                    pass
                return
            except Exception as exc:
                logger.exception("[SupabaseService] save_topic_results "

                    f"nivel '{name}' falló: {exc}")
                if not _looks_like_size_error(exc):
                    break

        logger.warning("[SupabaseService] No se pudo guardar el caché de "

            "tópicos (se devuelve el resultado calculado igual).")

    @staticmethod
    def _upsert_topics(supabase, payload: dict[str, Any]) -> None:
        try:
            supabase.table(TOPICS_TABLE).upsert(
                payload,
                on_conflict="dataset_id,user_id,max_topics",
            ).execute()
        except Exception:
            supabase.table(TOPICS_TABLE).delete().eq(
                "dataset_id", payload["dataset_id"]
            ).eq(
                "user_id", payload["user_id"]
            ).eq(
                "max_topics", payload["max_topics"]
            ).execute()
            supabase.table(TOPICS_TABLE).insert(payload).execute()
