import asyncio
import json
import re
from typing import Any
import httpx
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("topicmodeling-service")


PROMPT_TEMPLATE = """Genera un título corto y representativo para cada tópico de un modelado de tópicos (BERTopic).

Reglas:
- 3 a 6 palabras, sin comillas, sin punto final.
- Sin prefijos como "Tópico", "Título" ni numeración.
- En el idioma dominante de los documentos de ese tópico (español o inglés).
- Basate en las keywords y los fragmentos. No inventes temas que no estén.
- Devuelve SOLO un array JSON válido con este formato exacto:
[{"topic_id": 0, "title": "..."}]

Tópicos:
{topics_block}
"""


def _is_transient_error(message: str) -> bool:
    msg = message.lower()
    return any(
        marker in msg
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "resource_exhausted",
            "unavailable",
            "overloaded",
            "high demand",
            "try again later",
            "quota",
            "rate limit",
            "internal",
            "deadline exceeded",
            "timeout",
        )
    )


class TitleProvider:
    """Genera títulos legibles con Groq, Gemini o modelo local (Ollama) sin pasar por rag-service."""

    def __init__(
        self,
        provider: str = "groq",
        groq_api_key: str = "",
        groq_model: str = "openai/gpt-oss-20b",
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        ollama_base_url: str = "http://ollama:11434",
        ollama_model: str = "qwen3:8b",
        ollama_keep_alive: str = "5m",
        ollama_timeout_seconds: int = 180,
        ollama_num_ctx: int = 4096,
        enabled: bool = True,
        batch_size: int = 20,
        max_chars_per_doc: int = 400,
        timeout: int = 30,
    ) -> None:
        self.provider = (provider or "groq").strip().lower()
        self.groq_api_key = (groq_api_key or "").strip()
        self.groq_model = (groq_model or "openai/gpt-oss-20b").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "gemini-2.0-flash").strip()
        self.ollama_base_url = (ollama_base_url or "http://ollama:11434").strip().rstrip("/")
        self.ollama_model = (ollama_model or "qwen3:8b").strip()
        self.ollama_keep_alive = (ollama_keep_alive or "5m").strip()
        self.ollama_timeout_seconds = max(30, int(ollama_timeout_seconds or 180))
        self.ollama_num_ctx = max(1024, int(ollama_num_ctx or 4096))
        self.enabled = enabled
        self.batch_size = max(1, batch_size)
        self.max_chars_per_doc = max(100, max_chars_per_doc)
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self.provider == "groq":
            return bool(self.groq_api_key)
        return bool(self.api_key)

    def is_available_for(self, privacy_mode: bool = False) -> bool:
        if not self.enabled:
            return False
        if privacy_mode:
            return bool(self.ollama_base_url and self.ollama_model)
        return self.is_available

    async def generate_titles(
        self,
        topics: list[dict[str, Any]],
        privacy_mode: bool = False,
    ) -> dict[int, str]:
        """Devuelve {topic_id: title}. Vacío si está deshabilitado o falla."""
        if not topics or not self.is_available_for(privacy_mode=privacy_mode):
            return {}

        titles: dict[int, str] = {}
        effective_batch_size = min(self.batch_size, 10) if privacy_mode else self.batch_size

        for i in range(0, len(topics), effective_batch_size):
            batch = topics[i : i + effective_batch_size]
            try:
                batch_titles = await self._generate_batch(batch, privacy_mode=privacy_mode)
                titles.update(batch_titles)
            except Exception as exc:
                logger.exception(f"[TitleProvider] Batch {i // effective_batch_size} falló: {exc}")

            if not privacy_mode and i + effective_batch_size < len(topics):
                await asyncio.sleep(4)

        return titles

    @staticmethod
    def fallback_title(keywords: list[str]) -> str:
        return " · ".join(keywords[:3]) if keywords else "Tópico sin título"

    async def _generate_batch(
        self,
        batch: list[dict[str, Any]],
        privacy_mode: bool = False,
    ) -> dict[int, str]:
        prompt = PROMPT_TEMPLATE.replace(
            "{topics_block}",
            self._build_topics_block(batch),
        )

        last_error: Exception | None = None
        if privacy_mode:
            call_fn = self._call_ollama_sync
            provider_label = f"ollama ({self.ollama_model})"
            call_timeout = self.ollama_timeout_seconds + 15
            max_attempts = 2
        else:
            call_fn = self._call_groq_sync if self.provider == "groq" else self._call_gemini_sync
            provider_label = self.provider
            call_timeout = self.timeout + 10
            max_attempts = 3

        for attempt in range(max_attempts):
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(call_fn, prompt),
                    timeout=call_timeout,
                )
                return self._parse_response(text, batch)
            except Exception as exc:
                last_error = exc
                msg = str(exc)

                if not privacy_mode and _is_transient_error(msg):
                    wait = 10 * (attempt + 1)
                    logger.exception(
                        f"[TitleProvider] Error transitorio en {provider_label} "
                        f"(intento {attempt + 1}/{max_attempts}), esperando {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.exception(f"[TitleProvider] Error batch ({provider_label}): {exc}")
                    break

        raise last_error or RuntimeError(f"Fallo desconocido en {provider_label}.")

    def _call_ollama_sync(self, prompt: str) -> str:
        with httpx.Client(timeout=self.ollama_timeout_seconds) as client:
            resp = client.post(
                f"{self.ollama_base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Eres un asistente experto en análisis de tópicos. Respondes únicamente con un array JSON válido.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "stream": False,
                    "keep_alive": self.ollama_keep_alive,
                    "think": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": self.ollama_num_ctx,
                    },
                },
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            message = data.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Ollama devolvió una respuesta vacía.")
            return content.strip()

    def _call_groq_sync(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente experto en análisis de tópicos. Respondes únicamente con un array JSON válido.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        candidates = [self.groq_model]
        for fallback_model in ("openai/gpt-oss-20b", "llama-3.1-8b-instant"):
            if fallback_model not in candidates:
                candidates.append(fallback_model)

        with httpx.Client(timeout=self.timeout) as client:
            last_resp = None
            for model_name in candidates:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                )
                last_resp = resp
                if resp.status_code == 404 and "model_not_found" in resp.text:
                    logger.warning(
                        f"[TitleProvider] Modelo Groq '{model_name}' no disponible en la cuenta, probando siguiente..."
                    )
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:400]}")
                self.groq_model = model_name
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                return (message.get("content") or "").strip()

            raise RuntimeError(
                f"Groq HTTP {last_resp.status_code if last_resp else 404}: "
                f"{last_resp.text[:400] if last_resp else 'model_not_found'}"
            )

    def _call_gemini_sync(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        return (response.text or "").strip()

    def _build_topics_block(self, batch: list[dict[str, Any]]) -> str:
        parts: list[str] = []

        for topic in batch:
            topic_id = topic.get("topic_id")
            keywords = topic.get("keywords", [])[:10]
            docs = topic.get("representative_documents", [])[:2]

            lines = [
                f"Topic {topic_id}:",
                f"keywords: {', '.join(keywords)}",
            ]

            for n, doc in enumerate(docs, start=1):
                text = self._clean(doc.get("text", ""))
                lines.append(f"doc{n}: {text}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) > self.max_chars_per_doc:
            text = text[: self.max_chars_per_doc].rsplit(" ", 1)[0] + "..."
        return text

    @staticmethod
    def _parse_response(
        text: str,
        batch: list[dict[str, Any]],
    ) -> dict[int, str]:
        valid_ids = set()
        for t in batch:
            try:
                valid_ids.add(int(t.get("topic_id")))
            except (TypeError, ValueError):
                continue

        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        data: Any = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match is None:
                logger.exception(f"[TitleProvider] Respuesta no-JSON del LLM: {text[:500]!r}")
                return {}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                logger.exception(f"[TitleProvider] JSON inválido: {exc}. Respuesta: {text[:500]!r}")
                return {}

        if isinstance(data, dict):
            for key in ("titles", "topics", "result", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                return {}

        titles: dict[int, str] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "topic_id" not in item:
                    continue
                try:
                    tid = int(item["topic_id"])
                except (TypeError, ValueError):
                    continue
                raw_title = item.get("title", "")
                title = str(raw_title).strip().strip('"').strip()
                if tid in valid_ids and title:
                    titles[tid] = title[:80]

        return titles
