"""[P1-PLAN-LOTE-1160 · 2026-11-21] Recuperar un pote NO es habérselo tomado.

Batería real (caso R1160, gpt-6-luna, la conversación del dueño del 21-nov): con la herramienta nueva el coach recupera
bien los tres potes, pero al «Agrega esos 3 y ya» de después anotó el Atlas Gainer como TOMADO (190 kcal en el diario).
La regla del manual no bastó: «agrega» tras hablar de suplementos se lee como registrar. Esta guarda es determinista y
vive en `agent.execute_tools`, junto a las demás de `log_consumed_meal`:

  · la llamada anota un SUPLEMENTO (`meal_type='suplemento'` o `suplemento=`),
  · en esta conversación reciente se habló de recuperar/borrar la Despensa (la herramienta de recuperar ya corrió,
    o el usuario lo dijo con esas palabras),
  · y el último mensaje del usuario NO dice que se lo tomó («me tomé», «tomé», «bebí», «comí»…).

Entonces no se registra y el coach recibe por qué. Si el usuario sí dice que se lo tomó, pasa como siempre.
tooltip-anchor: P1-PLAN-LOTE-1160-NO-ES-TOMAR
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional

_VENTANA_MENSAJES = 14

_RE_TOMA = re.compile(
    r"\b(?:me\s+(?:lo|la|los|las)?\s*(?:tom|beb|com)\w*|tom[eo]|tomamos|tomaste|beb[io]|comi|consum[io]\w*|"
    r"desayun\w*|almorc\w*|almorz\w*|cen[eo]|merend\w*|ingeri\w*|me\s+met[io])\b",
    re.IGNORECASE,
)
_RE_RECUPERAR = re.compile(r"\b(?:recuper\w*|restaur\w*|borr[eoa]\w*|elimin\w*|devuelv\w*|deshac\w*|papelera)\b",
                           re.IGNORECASE)
_HERRAMIENTAS_DE_RECUPERAR = {"recuperar_de_la_nevera"}   # guardar un pote nuevo (foto) sí puede ir con su toma

MENSAJE = (
    "NO REGISTRADO: el usuario está hablando de devolver sus potes a la Despensa (recuperarlos o guardarlos), no dijo "
    "que se los tomó — recuperar un pote NO es habérselo tomado. No lo anotes en su diario. Dile en una frase que sus "
    "suplementos ya están en su Despensa (o recupéralos con recuperar_de_la_nevera si aún no lo hiciste) y, si se tomó "
    "alguno hoy, que te lo diga («me tomé…») para anotarlo."
)


def _sin_acentos(t: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(ch))


def _texto(m: Any) -> str:
    c = getattr(m, "content", "") or ""
    if isinstance(c, list):
        c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _tipo(m: Any) -> str:
    return getattr(m, "type", None) or getattr(m, "role", None) or ""


def _es_suplemento(tool_args: dict) -> bool:
    return bool(str((tool_args or {}).get("suplemento") or "").strip()) or \
        str((tool_args or {}).get("meal_type") or "").strip().lower() in ("suplemento", "supplement")


def rechazar_toma_tras_recuperar(messages: Iterable[Any], tool_args: dict) -> Optional[str]:
    """El texto del rechazo, o None si la anotación puede seguir."""
    if not _es_suplemento(tool_args):
        return None
    recientes = list(messages or [])[-_VENTANA_MENSAJES:]
    ultimo_usuario = next((_texto(m) for m in reversed(recientes)
                           if _tipo(m) in ("human", "user") and not _texto(m).startswith("[")), "")
    if _RE_TOMA.search(_sin_acentos(ultimo_usuario)):
        return None   # lo dijo: «me tomé la creatina»
    hubo_recuperacion = False
    for m in recientes:
        if _tipo(m) == "tool" and getattr(m, "name", None) in _HERRAMIENTAS_DE_RECUPERAR:
            hubo_recuperacion = True
        for tc in (getattr(m, "tool_calls", None) or []):
            nombre = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if nombre in _HERRAMIENTAS_DE_RECUPERAR:
                hubo_recuperacion = True
        if _tipo(m) in ("human", "user") and _RE_RECUPERAR.search(_sin_acentos(_texto(m))):
            hubo_recuperacion = True
    return MENSAJE if hubo_recuperacion else None
