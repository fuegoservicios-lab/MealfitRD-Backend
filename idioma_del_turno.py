# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1171 · 2026-11-23] El idioma de ESTE mensaje, dicho en cada turno.

Un usuario diario charló en inglés a las 6:40 p. m. («whats the name of the yogurt…») y a las 10:54 p. m. escribió en
español («agrega al desayuno 4 huevos…»): el coach le contestó en INGLÉS. La regla F del prompt («responde en el idioma
en que te escribe ESTE mensaje») vive en el system prompt, lejos del mensaje; el historial en inglés pesaba más.

Esto detecta el idioma del último mensaje del usuario con palabras frecuentes (sin IA, sin coste) y, solo cuando es
claro, añade una línea al contexto de ese turno. Si no está claro (un «ok», un nombre de plato), no dice nada y manda
la regla F como siempre. Knob `MEALFIT_COACH_IDIOMA_DEL_TURNO` (True).
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

_PALABRAS = {
    "es": {
        "el", "la", "los", "las", "de", "del", "que", "y", "en", "con", "para", "por", "una", "un", "lo", "al", "mi",
        "mis", "tu", "me", "te", "se", "es", "hoy", "ayer", "agrega", "anota", "comí", "desayuno", "almuerzo", "cena",
        "merienda", "cuánto", "cuántas", "qué", "cómo", "pero", "sin", "más", "también", "eso", "esto", "registré",
        "solo", "quiero", "tengo", "puedes", "gracias", "sí", "no", "ya",
    },
    "en": {
        "the", "and", "you", "your", "is", "are", "for", "to", "of", "with", "it", "that", "what", "whats", "what's",
        "my", "i", "on", "in", "add", "yesterday", "today", "how", "much", "many", "can", "please", "this", "was",
        "did", "ate", "had", "name", "sent", "breakfast", "lunch", "dinner", "snack", "do", "have", "thanks", "yes",
    },
}
_NOTA = {
    "es": "Idioma de ESTE mensaje: español. Responde en español aunque antes se hablara en otro idioma.",
    "en": "Language of THIS message: English. Reply in English even if earlier messages were in another language "
          "(dish names stay in Spanish).",
}


def _activo() -> bool:
    return os.environ.get("MEALFIT_COACH_IDIOMA_DEL_TURNO", "true").strip().lower() not in ("0", "false", "no", "off")


def idioma_del_mensaje(texto: str) -> Optional[str]:
    """'es' o 'en' si el mensaje lo dice claro (≥2 palabras y ventaja ≥2); si no, None."""
    palabras = re.findall(r"[a-záéíóúüñ']+", (texto or "").lower())
    votos = {k: sum(1 for p in palabras if p in v) for k, v in _PALABRAS.items()}
    ganador = max(votos, key=votos.get)
    otro = min(votos.values())
    if votos[ganador] >= 2 and votos[ganador] - otro >= 2:
        return ganador
    return None


def _texto(msg: Any) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def nota_de_idioma(mensaje_usuario: Any) -> Optional[str]:
    """La línea para el contexto del turno, o None si no hay idioma claro o el knob está apagado."""
    if not _activo():
        return None
    lang = idioma_del_mensaje(_texto(mensaje_usuario))
    return _NOTA.get(lang) if lang else None
