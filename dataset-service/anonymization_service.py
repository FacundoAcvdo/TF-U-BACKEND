"""Anonimización de documentos al momento de la ingesta.

Fase 1:
- Reemplaza los remitentes por PERSONA_1, PERSONA_2, etc.
- Reemplaza menciones de esos remitentes en el texto.
- Reemplaza emails, teléfonos, IPs y URLs por tags.
- Anonimiza valores string de metadata.

No usa modelos de NER ni persiste el diccionario. El mapa real -> tag vive
solamente durante la llamada a `anonymize_documents`.

No se anonimizan `clase`, `fecha` ni nombres de datasets/archivos: el nombre
del dataset se conserva tal cual porque es la referencia visible del usuario.
"""

import ipaddress
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from models import Document


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])"
)

_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)

_IPV4_RE = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w])(?!\.\w)"
)

_IPV6_RE = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"
)

_PHONE_RE = re.compile(
    r"(?<![\w])(?<!\d\.)\+?\d[\d\s().-]{6,20}\d(?![\w])"
)

_EMAIL_FULL_RE = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

_PHONE_KEY_RE = re.compile(
    r"^\+?[\d\s().-]{7,}$"
)

_DATE_LIKE_RE = re.compile(
    r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"
)

_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)

_DATE_INSIDE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)

_THOUSANDS_RE = re.compile(
    r"^\d{1,3}(?:\.\d{3})+$"
)

_DECIMAL_RE = re.compile(
    r"^\d+[.,]\d{1,2}$"
)

_ANON_TAG_RE = re.compile(
    r"^(?:PERSONA_\d+|\[(?:EMAIL|TEL|IP|URL)_\d+\])$",
    re.IGNORECASE,
)

_STRUCTURAL_METADATA_KEYS = {
    "formato_original",
    "formato_detectado",
}

_NON_PERSON_VALUES = {
    "sistema",
    "system",
    "desconocido",
    "desconocida",
    "unknown",
    "null",
    "none",
    "n/a",
}

_ACCENT_CLASSES = {
    "a": "aáàäâã",
    "e": "eéèëê",
    "i": "iíìïî",
    "o": "oóòöôõ",
    "u": "uúùüû",
    "n": "nñ",
    "c": "cç",
}

_PII_PREFIXES = {
    "email": "EMAIL",
    "phone": "TEL",
    "ip": "IP",
    "url": "URL",
}

_SENDER_REGEX_MAX_PATTERNS = 500
_TEXT_CACHE_MAX_ENTRIES = 4096
_TOKEN_RE = re.compile(
    r"(?<![\w@])@?[\wÀ-ÿ'’\-‑–—]+",
    re.UNICODE,
)


@dataclass
class AnonymizationStats:
    documentos: int = 0
    documentos_modificados: int = 0
    remitentes: int = 0
    menciones_remitentes: int = 0
    emails: int = 0
    telefonos: int = 0
    ips: int = 0
    urls: int = 0
    elapsed_ms: int = 0
    collect_senders_ms: int = 0
    build_sender_regex_ms: int = 0
    process_documents_ms: int = 0


@dataclass
class AnonymizationResult:
    documents: list[Document]
    stats: AnonymizationStats


@dataclass
class _Span:
    start: int
    end: int
    kind: str
    text: str
    priority: int = 0


def _normalize_key(value: str) -> str:
    """Normaliza un nombre para compararlo sin acentos ni mayúsculas."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = re.sub(
        r"[^\w\s@.+'_’\-]",
        "",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip(
        " \t\r\n.,;:!?¡¿\"'`´()[]{}<>«»“”"
    )


def _canonical_email(value: str) -> str:
    return value.strip().casefold()


def _canonical_phone(value: str) -> str:
    return re.sub(r"\D", "", value)


def _looks_like_phone(value: str) -> bool:
    raw = value.strip()
    digits = _canonical_phone(raw)

    if not 8 <= len(digits) <= 15:
        return False

    if _DATE_LIKE_RE.fullmatch(raw) or _ISO_DATE_RE.fullmatch(raw):
        return False

    if _DATE_INSIDE_RE.search(raw):
        return False

    if _THOUSANDS_RE.fullmatch(raw) or _DECIMAL_RE.fullmatch(raw):
        return False

    if (
        len(digits) > 13
        and not raw.startswith("+")
        and not re.search(r"[\s().-]", raw)
    ):
        return False

    return True


def _sender_key(value: str) -> tuple[str, str]:
    raw = str(value).strip()

    if _ANON_TAG_RE.fullmatch(raw):
        return "tag", raw.casefold()

    if _EMAIL_FULL_RE.fullmatch(raw):
        return "email", _canonical_email(raw)

    if _PHONE_KEY_RE.fullmatch(raw) and _looks_like_phone(raw):
        return "phone", _canonical_phone(raw)

    return "name", _normalize_key(raw)


def _sender_pattern(key: str) -> str:
    parts: list[str] = []

    for char in key:
        if char in _ACCENT_CLASSES:
            parts.append(f"[{_ACCENT_CLASSES[char]}]")
        elif char.isspace():
            parts.append(r"\s+")
        elif char in {"'", "’"}:
            parts.append(r"['’]")
        elif char == "-":
            parts.append(r"[-‑–—]")
        else:
            parts.append(re.escape(char))

    return "".join(parts)


def _first_token(key: str) -> str:
    return key.split(" ", 1)[0].strip()


def _valid_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False

    return True


def _valid_ipv6(value: str) -> bool:
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False

    return True


def _trim_url(value: str) -> str:
    return value.rstrip(".,;:!?)]}\"'")


def _select_non_overlapping(
    spans: list[_Span],
) -> list[_Span]:
    if not spans:
        return []

    ordered = sorted(
        spans,
        key=lambda span: (
            span.start,
            -(span.end - span.start),
            -span.priority,
        ),
    )

    selected: list[_Span] = []

    for span in ordered:
        if selected and span.start < selected[-1].end:
            continue

        selected.append(span)

    return selected


def _collect_structured_spans(text: str) -> list[_Span]:
    if not any(
        char.isdigit() or char in "@:/+."
        for char in text
    ):
        return []

    spans: list[_Span] = []
    url_spans: list[_Span] = []

    for match in _URL_RE.finditer(text):
        raw = _trim_url(match.group(0))

        if not raw:
            continue

        span = _Span(
            start=match.start(),
            end=match.start() + len(raw),
            kind="url",
            text=raw,
            priority=40,
        )

        url_spans.append(span)
        spans.append(span)

    def overlaps_url(span: _Span) -> bool:
        return any(
            span.start < url.end and url.start < span.end
            for url in url_spans
        )

    for match in _EMAIL_RE.finditer(text):
        span = _Span(
            start=match.start(),
            end=match.end(),
            kind="email",
            text=match.group(0),
            priority=30,
        )

        if not overlaps_url(span):
            spans.append(span)

    for match in _IPV4_RE.finditer(text):
        raw = match.group(0)

        if not _valid_ipv4(raw):
            continue

        span = _Span(
            start=match.start(),
            end=match.end(),
            kind="ip",
            text=raw,
            priority=20,
        )

        if not overlaps_url(span):
            spans.append(span)

    for match in _IPV6_RE.finditer(text):
        raw = match.group(0)

        if not _valid_ipv6(raw):
            continue

        span = _Span(
            start=match.start(),
            end=match.end(),
            kind="ip",
            text=raw,
            priority=20,
        )

        if not overlaps_url(span):
            spans.append(span)

    for match in _PHONE_RE.finditer(text):
        raw = match.group(0)

        if not _looks_like_phone(raw):
            continue

        span = _Span(
            start=match.start(),
            end=match.end(),
            kind="phone",
            text=raw,
            priority=10,
        )

        if not overlaps_url(span):
            spans.append(span)

    return _select_non_overlapping(spans)


class _Registry:
    """Mapa real -> tag de una sola carga de dataset."""

    def __init__(self) -> None:
        self.sender_name_tags: dict[str, str] = {}
        self.sender_email_tags: dict[str, str] = {}
        self.sender_phone_tags: dict[str, str] = {}
        self.sender_patterns: list[tuple[str, str]] = []
        self.sender_regex: re.Pattern[str] | None = None
        self.sender_sequences: dict[tuple[str, ...], str] = {}
        self.sender_max_words = 1

        self._seen_patterns: set[tuple[str, str]] = set()
        self._pii_tags: dict[tuple[str, str], str] = {}
        self._normalized_cache: dict[str, str] = {}
        self._sender_key_cache: dict[str, tuple[str, str]] = {}
        self.text_cache: dict[str, tuple[str, tuple[int, int, int, int, int]]] = {}
        self._pii_counters = {
            "email": 0,
            "phone": 0,
            "ip": 0,
            "url": 0,
        }

        self.person_count = 0

    def normalize_key(self, value: str) -> str:
        cached = self._normalized_cache.get(value)
        if cached is not None:
            return cached

        normalized = _normalize_key(value)
        self._normalized_cache[value] = normalized
        return normalized

    def sender_key(self, value: str) -> tuple[str, str]:
        cached = self._sender_key_cache.get(value)
        if cached is not None:
            return cached

        raw = str(value).strip()
        if _ANON_TAG_RE.fullmatch(raw):
            result = ("tag", raw.casefold())
        elif _EMAIL_FULL_RE.fullmatch(raw):
            result = ("email", _canonical_email(raw))
        elif _PHONE_KEY_RE.fullmatch(raw) and _looks_like_phone(raw):
            result = ("phone", _canonical_phone(raw))
        else:
            result = ("name", self.normalize_key(raw))

        self._sender_key_cache[value] = result
        return result

    def add_sender(
        self,
        kind: str,
        key: str,
    ) -> None:
        self.person_count += 1
        tag = f"PERSONA_{self.person_count}"

        if kind == "name":
            self.sender_name_tags[key] = tag
            self._add_name_pattern(key, key)
            self.sender_sequences[tuple(key.split())] = tag
            self.sender_max_words = max(
                self.sender_max_words,
                len(key.split()),
            )

            alias = _first_token(key)

            if (
                alias
                and alias != key
                and len(alias) >= 3
                and alias not in _NON_PERSON_VALUES
            ):
                existing = self.sender_name_tags.get(alias)

                if existing is None:
                    self.sender_name_tags[alias] = tag
                    self._add_name_pattern(alias, alias)
                    self.sender_sequences[tuple(alias.split())] = tag

        elif kind == "email":
            self.sender_email_tags[key] = tag

        elif kind == "phone":
            self.sender_phone_tags[key] = tag

    def _add_name_pattern(
        self,
        key: str,
        lookup: str,
    ) -> None:
        item = (_sender_pattern(key), lookup)

        if item in self._seen_patterns:
            return

        self._seen_patterns.add(item)
        self.sender_patterns.append(item)

    def build_sender_regex(self) -> None:
        if not self.sender_patterns or len(self.sender_patterns) > _SENDER_REGEX_MAX_PATTERNS:
            return

        ordered = sorted(
            self.sender_patterns,
            key=lambda item: len(item[1]),
            reverse=True,
        )

        combined = "|".join(
            f"(?:{pattern})" for pattern, _ in ordered
        )

        self.sender_regex = re.compile(
            rf"(?<![\w@])(?:@?{combined})(?!\w)",
            re.IGNORECASE,
        )

    def pii_tag(
        self,
        kind: str,
        canonical: str,
    ) -> str:
        key = (kind, canonical)
        tag = self._pii_tags.get(key)

        if tag is not None:
            return tag

        self._pii_counters[kind] += 1
        tag = f"[{_PII_PREFIXES[kind]}_{self._pii_counters[kind]}]"
        self._pii_tags[key] = tag

        return tag


def _collect_senders(
    documents: list[Document],
    registry: _Registry,
) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        autor = document.autor

        if autor is None:
            continue

        raw = str(autor).strip()

        if not raw:
            continue

        kind, key = registry.sender_key(raw)

        if kind == "tag" or not key:
            continue

        if kind == "name" and key in _NON_PERSON_VALUES:
            continue

        sender = (kind, key)

        if sender in seen:
            continue

        seen.add(sender)
        ordered.append(sender)

    return ordered


class AnonymizationService:
    """Anonimización determinística, sin modelos ni persistencia."""

    def anonymize_documents(
        self,
        documents: list[Document],
        *,
        es_chat: bool = False,
    ) -> AnonymizationResult:
        """Anonimiza in-place y devuelve los documentos con estadísticas.

        `es_chat` queda reservado para la fase de NER; en fase 1 no cambia
        el comportamiento.
        """

        started = time.perf_counter()
        stats = AnonymizationStats(documentos=len(documents))
        registry = _Registry()

        collect_started = time.perf_counter()
        for kind, key in _collect_senders(documents, registry):
            registry.add_sender(kind, key)
        stats.collect_senders_ms = int(
            (time.perf_counter() - collect_started) * 1000
        )

        regex_started = time.perf_counter()
        registry.build_sender_regex()
        stats.build_sender_regex_ms = int(
            (time.perf_counter() - regex_started) * 1000
        )
        stats.remitentes = registry.person_count

        process_started = time.perf_counter()
        for document in documents:
            changed = False

            new_autor = self._anonymize_author(
                document.autor,
                registry,
            )

            if new_autor != document.autor:
                document.autor = new_autor
                changed = True

            new_text = self._anonymize_text(
                document.texto,
                registry,
                stats,
            )

            if new_text != document.texto:
                document.texto = new_text
                changed = True

            new_metadata = self._anonymize_metadata(
                document.metadata,
                registry,
                stats,
            )

            if new_metadata != document.metadata:
                document.metadata = new_metadata
                changed = True

            if changed:
                stats.documentos_modificados += 1
        stats.process_documents_ms = int(
            (time.perf_counter() - process_started) * 1000
        )

        stats.elapsed_ms = int(
            (time.perf_counter() - started) * 1000
        )

        return AnonymizationResult(
            documents=documents,
            stats=stats,
        )

    def _anonymize_author(
        self,
        autor: str | None,
        registry: _Registry,
    ) -> str | None:
        if not autor:
            return autor

        kind, key = registry.sender_key(str(autor))

        if kind == "name":
            return registry.sender_name_tags.get(key, autor)

        if kind == "email":
            return registry.sender_email_tags.get(key, autor)

        if kind == "phone":
            return registry.sender_phone_tags.get(key, autor)

        return autor

    def _anonymize_text(
        self,
        text: str,
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> str:
        if not text:
            return text

        cached = registry.text_cache.get(text)
        if cached is not None:
            anonymized, counts = cached
            stats.emails += counts[0]
            stats.telefonos += counts[1]
            stats.ips += counts[2]
            stats.urls += counts[3]
            stats.menciones_remitentes += counts[4]
            return anonymized

        local_stats = AnonymizationStats()
        anonymized = self._replace_structured(
            text,
            registry,
            local_stats,
        )

        anonymized = self._replace_senders(
            anonymized,
            registry,
            local_stats,
        )
        counts = (
            local_stats.emails,
            local_stats.telefonos,
            local_stats.ips,
            local_stats.urls,
            local_stats.menciones_remitentes,
        )
        if len(registry.text_cache) < _TEXT_CACHE_MAX_ENTRIES:
            registry.text_cache[text] = (anonymized, counts)
        stats.emails += counts[0]
        stats.telefonos += counts[1]
        stats.ips += counts[2]
        stats.urls += counts[3]
        stats.menciones_remitentes += counts[4]
        return anonymized

    def _replace_structured(
        self,
        text: str,
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> str:
        spans = _collect_structured_spans(text)

        if not spans:
            return text

        replacements: list[tuple[int, int, str]] = []

        for span in spans:
            if span.kind == "email":
                canonical = _canonical_email(span.text)
                tag = (
                    registry.sender_email_tags.get(canonical)
                    or registry.pii_tag("email", canonical)
                )
                stats.emails += 1

            elif span.kind == "phone":
                canonical = _canonical_phone(span.text)
                tag = (
                    registry.sender_phone_tags.get(canonical)
                    or registry.pii_tag("phone", canonical)
                )
                stats.telefonos += 1

            elif span.kind == "ip":
                canonical = str(
                    ipaddress.ip_address(span.text)
                )
                tag = registry.pii_tag("ip", canonical)
                stats.ips += 1

            else:
                tag = registry.pii_tag("url", span.text)
                stats.urls += 1

            replacements.append(
                (span.start, span.end, tag)
            )

        parts: list[str] = []
        cursor = 0

        for start, end, tag in replacements:
            parts.append(text[cursor:start])
            parts.append(tag)
            cursor = end

        parts.append(text[cursor:])
        return "".join(parts)

    def _replace_senders(
        self,
        text: str,
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> str:
        if registry.sender_regex is None:
            return self._replace_senders_by_tokens(text, registry, stats)

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            key = registry.normalize_key(raw.lstrip("@"))
            tag = registry.sender_name_tags.get(key)

            if tag is None:
                return raw

            stats.menciones_remitentes += 1

            return tag

        return registry.sender_regex.sub(replace, text)

    def _replace_senders_by_tokens(
        self,
        text: str,
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> str:
        if not registry.sender_sequences:
            return text

        tokens = list(_TOKEN_RE.finditer(text))
        replacements: list[tuple[int, int, str]] = []
        index = 0

        while index < len(tokens):
            best: tuple[int, str] | None = None
            max_end = min(
                len(tokens),
                index + registry.sender_max_words,
            )

            for end in range(max_end, index, -1):
                if end > index + 1:
                    between = text[
                        tokens[end - 2].end():tokens[end - 1].start()
                    ]
                    if not between.isspace():
                        continue

                raw = text[tokens[index].start():tokens[end - 1].end()]
                key = registry.normalize_key(raw.lstrip("@"))
                tag = registry.sender_sequences.get(tuple(key.split()))

                if tag is not None:
                    best = (end, tag)
                    break

            if best is None:
                index += 1
                continue

            end, tag = best
            replacements.append(
                (
                    tokens[index].start(),
                    tokens[end - 1].end(),
                    tag,
                )
            )
            stats.menciones_remitentes += 1
            index = end

        if not replacements:
            return text

        parts: list[str] = []
        cursor = 0
        for start, end, tag in replacements:
            parts.append(text[cursor:start])
            parts.append(tag)
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    def _anonymize_metadata(
        self,
        metadata: dict[str, Any],
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return metadata

        result: dict[str, Any] = {}

        for key, value in metadata.items():
            if key in _STRUCTURAL_METADATA_KEYS:
                result[key] = value
                continue

            result[key] = self._anonymize_metadata_value(
                value,
                registry,
                stats,
            )

        return result

    def _anonymize_metadata_value(
        self,
        value: Any,
        registry: _Registry,
        stats: AnonymizationStats,
    ) -> Any:
        if isinstance(value, str):
            return self._anonymize_text(
                value,
                registry,
                stats,
            )

        if isinstance(value, list):
            return [
                self._anonymize_metadata_value(
                    item,
                    registry,
                    stats,
                )
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._anonymize_metadata_value(
                    item,
                    registry,
                    stats,
                )
                for item in value
            )

        if isinstance(value, dict):
            return {
                key: self._anonymize_metadata_value(
                    item,
                    registry,
                    stats,
                )
                for key, item in value.items()
            }

        return value
