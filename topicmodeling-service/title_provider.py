import asyncio
import json
import re
from typing import Any


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
    """Genera títulos legibles con Gemini sin pasar por rag-service."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        enabled: bool = True,
        batch_size: int = 20,
        max_chars_per_doc: int = 400,
        timeout: int = 30,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model or "gemini-2.0-flash"
        self.enabled = enabled
        self.batch_size = max(1, batch_size)
        self.max_chars_per_doc = max(100, max_chars_per_doc)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return bool(self.enabled and self.api_key)

    async def generate_titles(
        self,
        topics: list[dict[str, Any]],
    ) -> dict[int, str]:
        """Devuelve {topic_id: title}. Vacío si está deshabilitado o falla."""
        if not topics or not self.is_available:
            return {}

        titles: dict[int, str] = {}

        for i in range(0, len(topics), self.batch_size):
            batch = topics[i : i + self.batch_size]
            try:
                batch_titles = await self._generate_batch(batch)
                titles.update(batch_titles)
            except Exception as exc:
                print(f"[TitleProvider] Batch {i // self.batch_size} falló: {exc}")

            # Pausa breve para no golpear el RPM del tier gratis.
            if i + self.batch_size < len(topics):
                await asyncio.sleep(4)

        return titles

    @staticmethod
    def fallback_title(keywords: list[str]) -> str:
        return " · ".join(keywords[:3]) if keywords else "Tópico sin título"

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def _generate_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[int, str]:
        prompt = PROMPT_TEMPLATE.replace(
            "{topics_block}",
            self._build_topics_block(batch),
        )

        last_error: Exception | None = None

        for attempt in range(3):
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self._call_gemini_sync, prompt),
                    timeout=self.timeout + 10,
                )
                return self._parse_response(text, batch)
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                # Transitorios (rate limit, sobrecarga): esperar y reintentar.
                if _is_transient_error(msg):
                    wait = 10 * (attempt + 1)
                    print(
                        "[TitleProvider] Gemini transitorio "
                        f"(intento {attempt + 1}/3), esperando {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    print(f"[TitleProvider] Error batch: {exc}")
                    break

        raise last_error or RuntimeError("Fallo desconocido en Gemini.")

    def _call_gemini_sync(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        return (response.text or "").strip()

    # ------------------------------------------------------------------
    # Prompt / parsing
    # ------------------------------------------------------------------

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

        # Tolera fences ```json ... ``` y texto alrededor del JSON.
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        data: Any = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match is None:
                print(f"[TitleProvider] Respuesta no-JSON de Gemini: {text[:500]!r}")
                return {}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                print(f"[TitleProvider] JSON inválido: {exc}. Respuesta: {text[:500]!r}")
                return {}

        # Tolera envoltorios {"titles": [...]} o {"topics": [...]}.
        if isinstance(data, dict):
            for key in ("titles", "topics", "result", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                print(f"[TitleProvider] Respuesta dict sin lista: {text[:500]!r}")
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

        if not titles:
            print(f"[TitleProvider] Sin títulos parseables. Respuesta: {text[:500]!r}")

        return titles
