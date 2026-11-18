# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1077 · 2026-11-18] «Lo mismo» sin día = lo último que anotó HOY: el sistema busca ese registro.

Caso vivo del dueño (18-nov, 00:25 y 00:31): a las 00:25 anotó con foto «jugo de toronja con muy poca azúcar»; a las
00:31, otra foto de un vaso, el escáner preguntó «¿Qué bebida es exactamente? (Limonada dulce · Agua con limón…)» y él
respondió «Lo mismo». El coach anotó «bebida con limón»: lo que aparentaba la foto, no lo que dijo.

El lote 988 (`referencia_a_registro`) cubre «lo mismo DE AYER»; esto es su gemelo para «lo mismo» a secas, que solo
puede ser algo de hoy y reciente. Determinista: si un trozo del mensaje (cada respuesta va separada por «·» o por línea)
ES «lo mismo», «la misma», «igual», «lo de antes»… y hay un registro de hoy de hace ≤3 h, el prompt recibe ese
registro con sus ingredientes y la orden de usar el equivalente (una bebida → esa bebida).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

VENTANA_H = 3.0

# El trozo ENTERO es la referencia (respuesta a una duda): «Lo mismo», «la misma», «igual que antes», «lo de antes».
_TROZO = re.compile(
    r"^(?:(?:es|era|fue|otra vez|tambien)\s+)?"
    r"(?:(?:lo|la|el|los|las)\s+mism[oa]s?|igual(?:es)?|(?:lo|la|el)\s+de\s+(?:antes|hace\s+(?:un\s+)?rato))"
    r"(?:\s+(?:que|de)\s+(?:antes|hace\s+(?:un\s+)?rato|la\s+otra|el\s+otro|lo\s+otro))?"
    r"(?:\s+(?:de|que)\s+(?:la|el)\s+(?:foto|vaso|plato)\s+(?:anterior|de\s+antes))?$")
# En medio de una frase solo con ancla temporal de HOY: «me tomé lo mismo que hace rato».
_EN_FRASE = re.compile(r"\b(?:lo|la|el)\s+mism[oa]\s+(?:que|de)\s+(?:antes|hace\s+(?:un\s+)?rato)\b")
_OTRO_DIA = re.compile(r"\b(?:ayer|anoche|antier|anteayer|semana|lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b")


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[¡!¿?.,;:\"«»()]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def detectar(texto: str) -> Optional[str]:
    """El trozo que es «lo mismo» (tal cual lo escribió), o None. «Lo mismo de ayer» no es suyo (lote 988)."""
    crudo = str(texto or "")
    if _OTRO_DIA.search(_norm(crudo)):
        return None
    for trozo in re.split(r"[\n·•|]+", crudo):
        if _TROZO.match(_norm(trozo)):
            return trozo.strip()
    m = _EN_FRASE.search(_norm(crudo))
    return m.group(0) if m else None


def _dt(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:                                                          # noqa: BLE001
        return None


def elegir(filas: list, ahora_utc: datetime) -> Optional[dict]:
    """El registro de hoy anotado más recientemente (por `created_at`: la hora en que lo DIJO), si fue hace ≤3 h."""
    mejor, mejor_at = None, None
    for f in filas or []:
        if not isinstance(f, dict):
            continue
        at = _dt(f.get("created_at") or f.get("consumed_at"))
        if not at or at > ahora_utc + timedelta(minutes=5) or ahora_utc - at > timedelta(hours=VENTANA_H):
            continue
        if mejor_at is None or at > mejor_at:
            mejor, mejor_at = f, at
    return mejor


def _filas_de_hoy(user_id: str, tz_offset_min: Optional[int]) -> list:
    from db_facts import get_consumed_meals_today
    return get_consumed_meals_today(user_id, tz_offset_mins=tz_offset_min) or []


def bloque_para_chat(user_id, texto: str, tz_offset_min: Optional[int] = None, *, ahora_utc: Optional[datetime] = None,
                     filas: Optional[list] = None) -> str:
    """Texto «Para el asistente» o "". Nunca lanza."""
    try:
        if not user_id or user_id == "guest":
            return ""
        trozo = detectar(texto)
        if not trozo:
            return ""
        ahora = ahora_utc or datetime.now(timezone.utc)
        fila = elegir(filas if filas is not None else _filas_de_hoy(user_id, tz_offset_min), ahora)
        if not fila:
            return ""
        at = _dt(fila.get("created_at") or fila.get("consumed_at"))
        off = 240 if tz_offset_min is None else int(tz_offset_min)
        hora = (at - timedelta(minutes=off)).strftime("%H:%M") if at else "hace un rato"
        ings = "; ".join(str(i) for i in (fila.get("ingredients") or [])[:8] if i)
        return (f"\n\n🔁 «{trozo}» = SU ÚLTIMO REGISTRO DE HOY [P1-PLAN-LOTE-1077] (buscado por el sistema): a las {hora} "
                f"anotó «{fila.get('meal_name')}»" + (f" ({ings})" if ings else "") + ". Donde respondió "
                f"«{trozo}» (a una duda de la foto o del mensaje) se refiere al alimento EQUIVALENTE de ese registro —una "
                "bebida a esa bebida, un plato a ese plato—, con su misma preparación y su porción salvo que diga otra: "
                "regístralo con ESE nombre, NO con lo que aparenta la foto ni con las opciones sugeridas. Si en ese "
                "registro no hay nada del mismo tipo, pregúntale en una frase a qué se refiere.\n")
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-1077] «lo mismo» no resuelto: {type(e).__name__}")
        return ""
