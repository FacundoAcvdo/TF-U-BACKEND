"""Evidencia textual: spans + palabras gatillo (sin modelo extra).

V1: no re-scorea oraciones (rápido). Localiza lexicón ES/EN por
coincidencia insensible a tildes/mayúsculas y devuelve offsets
para que el front subraye con <mark>.
"""
from __future__ import annotations

import re
import unicodedata

TRIGGERS_ES = [
    "quiero morir", "quiero morirme", "quitarme la vida", "quitarme del medio",
    "no quiero seguir", "no quiero vivir", "no doy más", "no puedo más",
    "sería más fácil no despertar", "no despertar", "soy una carga",
    "no le encuentro sentido", "quiero desaparecer", "quiero terminar con todo",
    "no veo salida", "me quiero ir", "no existir", "acabar con mi vida",
    "pensando en suicidarme", "pensando en matarme", "intento de suicidio",
]

TRIGGERS_EN = [
    "kill myself", "killing myself", "want to die", "wanna die",
    "take my own life", "end my life", "suicidal thoughts", "suicide note",
    "don't want to live", "no reason to live", "better off dead",
    "don't want to wake up",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?…\n])\s+")

DISCLAIMER = (
    "Resultado generado por un modelo de IA experimental que puede cometer errores "
    "(falsos positivos y falsos negativos). Solo detecta patrones lingüísticos en el "
    "texto, no constituye diagnóstico ni evaluación clínica. Los casos marcados deben "
    "ser revisados por personal cualificado."
)


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s


_NORM_TRIGGERS = [(_norm(t), t) for t in TRIGGERS_ES + TRIGGERS_EN]


def split_sentences(text: str, max_len: int = 2000) -> list[tuple[str, int]]:
    """Devuelve [(oración, offset_en_texto)]."""
    text = (text or "")[:max_len]
    out: list[tuple[str, int]] = []
    cursor = 0
    for part in _SENT_SPLIT.split(text):
        if not part.strip():
            cursor += len(part)
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        out.append((part, start))
        cursor = start + len(part)
    return out or [(text, 0)]


def find_triggers(text: str) -> list[dict]:
    """Triggers en todo el texto con offsets de oración contenedora."""
    norm_text = _norm(text or "")
    hits: list[dict] = []
    for norm_t, orig_t in _NORM_TRIGGERS:
        idx = norm_text.find(norm_t)
        if idx >= 0:

            hits.append({"trigger": orig_t, "pos_norm": idx})
    return hits


def explain_text(text: str, max_spans: int = 3) -> dict:
    """Spans más relevantes = oraciones que contienen triggers."""
    sents = split_sentences(text or "")
    spans: list[dict] = []
    triggers_found: list[str] = []
    for sent, off in sents:
        n_sent = _norm(sent)
        matched = [orig for norm_t, orig in _NORM_TRIGGERS if norm_t in n_sent]
        if matched:
            spans.append({
                "texto": sent.strip(),
                "start": off,
                "end": off + len(sent),
                "triggers": matched,
            })
            for m in matched:
                if m not in triggers_found:
                    triggers_found.append(m)
        if len(spans) >= max_spans:
            break
    return {"spans": spans, "triggers": triggers_found, "has_evidence": bool(spans)}
