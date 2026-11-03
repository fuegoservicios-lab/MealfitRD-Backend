"""[P1-BIO-DATE-RESOLVER · 2026-11-03]
Resolución biológica de fechas conversacionales (paradoja 'Ayer / Hoy' en madrugada).

Problema resuelto:
Con el rollover a las 4:00 AM (BB-016), de 00:00 a 03:59 el día biológico activo sigue
siendo la jornada anterior (ej. lunes a las 01:30 AM pertenece al lunes biológico).
Si un usuario dice "ayer cené..." o "anoche comí...", para la mente humana que ya pasó
las 12:00 AM la cena fue "ayer" o "anoche" (lunes).
Si el backend resta 24 horas a ciegas (days_ago=1), la comida cae en el DOMINGO biológico,
perdiéndose la cena del lunes.

Este módulo arbitra la intención real del usuario entre las 00:00 y las 03:59:59 local.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Palabras clave de tiempo relativo
_RE_HOY = re.compile(r"\b(?:hoy|esta ma[ñn]ana|ahorita|reci[eé]n|hace poco|en este momento)\b", re.IGNORECASE)
_RE_ANOCHE = re.compile(r"\b(?:anoche|esta noche|de noche)\b", re.IGNORECASE)
_RE_AYER = re.compile(r"\bayer\b", re.IGNORECASE)
_RE_ANTEAYER = re.compile(r"\b(?:anteayer|antier|antes de ayer)\b", re.IGNORECASE)

# Slots de comida
_RE_SLOT_CENA = re.compile(r"\b(?:cen[eé]|cenas?|cenar|cena|dinner)\b", re.IGNORECASE)
_RE_SLOT_ALMUERZO = re.compile(r"\b(?:almor[zc][eé]|almuerzos?|almorzar|lunch)\b", re.IGNORECASE)
_RE_SLOT_DESAYUNO = re.compile(r"\b(?:desayun[eé]|desayunos?|desayunar|breakfast)\b", re.IGNORECASE)

# Días de la semana
_DIAS_SEMANA_MAP = {
    "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2,
    "jueves": 3, "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_RE_DIAS_SEMANA = re.compile(r"\b(lunes|martes|mi[eé]rcoles|miercoles|jueves|viernes|s[aá]bado|sabado|domingo)\b", re.IGNORECASE)

# Regex para extraer contexto temporal de los SystemMessage
_RE_HORA_PROMPT = re.compile(r"La hora local es (\d{1,2}):(\d{2})", re.IGNORECASE)
_RE_DIA_PROMPT = re.compile(r"Hoy es (Lunes|Martes|Mi[eé]rcoles|Jueves|Viernes|S[aá]bado|Domingo)", re.IGNORECASE)


def es_ventana_madrugada(hora_local: Optional[float]) -> bool:
    """Retorna True si la hora local se encuentra en la ventana 00:00 a 03:59:59 (antes del corte a las 4 AM)."""
    if hora_local is None:
        return False
    return 0.0 <= float(hora_local) < 4.0


def extraer_contexto_temporal_de_mensajes(messages: List[Any]) -> Tuple[Optional[float], Optional[int]]:
    """Extrae (hora_local_float, weekday_int) a partir de los SystemMessage presentes en el historial."""
    hora_res: Optional[float] = None
    weekday_res: Optional[int] = None

    for msg in messages or []:
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("system", "SystemMessage") or msg.__class__.__name__ == "SystemMessage":
            c = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if not isinstance(c, str):
                continue
            if "CONTEXTO TEMPORAL ACTUAL:" in c:
                m_h = _RE_HORA_PROMPT.search(c)
                if m_h:
                    hh, mm = int(m_h.group(1)), int(m_h.group(2))
                    hora_res = hh + mm / 60.0
                m_d = _RE_DIA_PROMPT.search(c)
                if m_d:
                    d_txt = m_d.group(1).lower().replace("é", "e").replace("á", "a")
                    weekday_res = _DIAS_SEMANA_MAP.get(d_txt)
                if hora_res is not None:
                    break

    return hora_res, weekday_res


def tiene_slot_registrado_hoy(
    user_id: Optional[str],
    slot: str,
    tz_offset: Optional[int] = None,
) -> bool:
    """Consulta en Postgres si el usuario ya tiene registrado ese slot (cena, almuerzo, desayuno)
    en el día biológico en curso."""
    if not user_id or user_id == "guest":
        return False
    try:
        from db_facts import get_consumed_meals_today
        meals = get_consumed_meals_today(user_id, tz_offset_mins=tz_offset) or []
        slot_norm = slot.strip().lower()
        for m in meals:
            if isinstance(m, dict):
                m_type = (m.get("meal_type") or "").strip().lower()
                if m_type == slot_norm:
                    return True
        return False
    except Exception as e:
        logger.debug(f"[P1-BIO-DATE-RESOLVER] No se pudo verificar slot '{slot}' para {user_id}: {e}")
        return False


def _extraer_texto_humano_reciente(messages: List[Any]) -> str:
    """Extrae el texto del último mensaje humano."""
    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user") or msg.__class__.__name__ == "HumanMessage":
            c = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def corregir_days_ago_madrugada(
    messages: List[Any],
    days_ago_actual: Optional[int],
    hora_local: Optional[float] = None,
    user_id: Optional[str] = None,
    slot_pedido: Optional[str] = None,
    tz_offset: Optional[int] = None,
    weekday_actual: Optional[int] = None,
) -> Optional[int]:
    """Arbitra si un days_ago en la ventana 00:00 - 03:59 debe corregirse a 0 (día biológico actual)
    o mantenerse en 1/2."""
    if hora_local is None:
        hora_ext, wk_ext = extraer_contexto_temporal_de_mensajes(messages)
        hora_local = hora_ext
        if weekday_actual is None:
            weekday_actual = wk_ext

    # Si no estamos en la ventana de madrugada (00:00 - 03:59:59), no aplica la paradoja
    if not es_ventana_madrugada(hora_local):
        return days_ago_actual

    texto = _extraer_texto_humano_reciente(messages)
    if not texto:
        return days_ago_actual

    # Si dice explícitamente "hoy" o "ahorita", sin duda es 0
    if _RE_HOY.search(texto):
        return 0

    # Si menciona un día de la semana explícito (ej. "el lunes cené", "el domingo almorcé")
    m_dia = _RE_DIAS_SEMANA.search(texto)
    if m_dia:
        dia_norm = m_dia.group(1).lower().replace("é", "e").replace("á", "a")
        if dia_norm in _DIAS_SEMANA_MAP and weekday_actual is not None:
            named_wk = _DIAS_SEMANA_MAP[dia_norm]
            # En madrugada, el día de la semana biológico es weekday_actual - 1 día
            # (ej. martes 01:30 AM calendario tiene día biológico lunes)
            bio_wk = (weekday_actual - 1) % 7
            distancia = (bio_wk - named_wk) % 7
            logger.info(
                f"🌙 [P1-BIO-DATE-RESOLVER] Día nombrado '{dia_norm}' ({named_wk}) vs "
                f"biológico {bio_wk} (calendario {weekday_actual}) → distancia={distancia}"
            )
            return distancia

    # "antier" o "anteayer" en madrugada:
    # A las 01:30 AM del martes, "antier" para el usuario es el domingo (2 días calendario atrás).
    # Respecto al lunes biológico, el domingo es 1 día atrás (days_ago=1).
    if _RE_ANTEAYER.search(texto):
        return 1

    # "anoche" en madrugada:
    # A la 01:30 AM, "anoche" es la noche que acaba de terminar (hace unas 3-5 horas).
    # Pertenece inequívocamente al día biológico que sigue abierto (days_ago=0).
    if _RE_ANOCHE.search(texto):
        return 0

    # "ayer" en madrugada:
    if _RE_AYER.search(texto):
        slot = slot_pedido or ("cena" if _RE_SLOT_CENA.search(texto) else ("almuerzo" if _RE_SLOT_ALMUERZO.search(texto) else None))
        
        # Si se refiere a la cena o a una comida principal
        if slot:
            ya_registrado = tiene_slot_registrado_hoy(user_id, slot, tz_offset)
            if not ya_registrado:
                # El slot de hoy está libre: el usuario está completando su cena/comida de la jornada
                return 0
            else:
                # El slot de hoy ya estaba lleno: se refiere a la cena de ayer real
                return 1

        # Comida sin slot específico (ej. "ayer me comí un sándwich")
        # Si no tiene cena hoy, asumimos la jornada en curso
        if not tiene_slot_registrado_hoy(user_id, "cena", tz_offset):
            return 0
        return 1

    return days_ago_actual
