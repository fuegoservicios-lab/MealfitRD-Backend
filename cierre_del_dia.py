# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1076 · 2026-11-17] Al cerrar el día, la proteína muy corta se dice UNA vez, sin regañar.

Caso del dueño (17-nov, 22:26): anotó la cena y el día quedó en 68 de 134 g de proteína (la mitad); el coach solo
recitó el total. La regla D (lote 1057) ya cubre el día que se PASA de kcal; esto es su gemelo para la proteína corta,
determinista: la cifra la pone el servidor y el coach solo la dice en una frase con una idea para mañana — a esa hora
no tiene sentido mandarle a comer más.

Cuándo: hoy, con la CENA ya anotada (el día está cerrando) y desde las 19:00 (o de madrugada), con meta de proteína
conocida y lo anotado por debajo del 60 % de ella.
"""
from __future__ import annotations

from typing import Optional

UMBRAL = 0.60


def nota_proteina_corta(metas: Optional[dict], consumido: Optional[dict], tipos_del_dia, hora_local: Optional[float]) -> str:
    """Texto «Para el asistente» o "" si no aplica. Nunca lanza."""
    try:
        meta = float((metas or {}).get("protein_g") or 0)
        lleva = float((consumido or {}).get("protein_g") or 0)
        tipos = {str(t or "").strip().lower() for t in (tipos_del_dia or [])}
        if meta <= 0 or "cena" not in tipos or hora_local is None:
            return ""
        h = float(hora_local)
        if not (h >= 19.0 or h < 4.0):
            return ""
        if lleva >= UMBRAL * meta:
            return ""
        falta = int(round(meta - lleva))
        return (f" CIERRE DEL DÍA [P1-PLAN-LOTE-1076]: con la cena anotada, hoy quedó en ~{int(round(lleva))} de "
                f"~{int(round(meta))} g de proteína (le faltaron ~{falta} g). Díselo en UNA frase, sin regañar ni culpa y "
                "SIN proponer comer más ahora (ya cerró el día): una idea concreta para mañana (ej. huevos o yogur en el "
                "desayuno, una porción de pollo, pescado o habichuelas en el almuerzo). Una sola vez: si ya se lo dijiste "
                "hoy en esta conversación, no lo repitas.")
    except Exception:  # noqa: BLE001
        return ""


# [P1-PLAN-LOTE-1079 · 2026-11-18] El gemelo positivo: el registro que CRUZA la meta de proteína se celebra en una frase.
# Caso (cuenta de prueba, 17-nov): con la cena llegó a 162 de 157 g y el coach solo recitó el total. «Una vez» sin
# memoria: solo el registro con el que se cruza (antes < meta ≤ después); los siguientes del día ya no la cruzan.
def _dt(v):
    from datetime import datetime, timezone
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _prot(fila) -> float:
    try:
        return float(fila.get("protein") or fila.get("protein_g") or 0)
    except (TypeError, ValueError):
        return 0.0


def nota_proteina_cumplida(metas: Optional[dict], filas) -> str:
    """Texto «Para el asistente» si el registro MÁS RECIENTE (por `created_at`; sin él, cuenta como el último) es el que
    lleva la proteína del día a su meta; si no, "". Nunca lanza."""
    try:
        meta = float((metas or {}).get("protein_g") or 0)
        filas = [f for f in (filas or []) if isinstance(f, dict)]
        if meta <= 0 or not filas:
            return ""
        from datetime import datetime, timezone
        tope = datetime.max.replace(tzinfo=timezone.utc)
        ultima = max(range(len(filas)), key=lambda i: (_dt(filas[i].get("created_at")) or tope, i))
        total = sum(_prot(f) for f in filas)
        antes = total - _prot(filas[ultima])
        if not (antes < meta <= total):
            return ""
        return (f" PROTEÍNA CUMPLIDA [P1-PLAN-LOTE-1079]: con este registro llegó a ~{int(round(total))} de "
                f"~{int(round(meta))} g de proteína de hoy. Díselo en UNA frase corta y alegre, sin exagerar (ej. «con "
                "esto ya cumpliste tu proteína de hoy»); no le insistas más en proteína hoy.")
    except Exception:  # noqa: BLE001
        return ""


# [P1-PLAN-LOTE-1080 · 2026-11-18] Dos cierres que contradecían lo que el usuario ve (cuenta de prueba, 17-nov):
# 1. a las 12:47 anotó desayuno y almuerzo en un mensaje y el coach, tras anotarlos, cerró con «No me mandaste el
#    desayuno de hoy; si lo comiste, mándame una foto»: el bloque «COMIDAS DE HOY» se calcula ANTES del registro y le
#    daba esa frase hecha. La nota tras el registro dice ahora qué franjas ya pasadas quedaron anotadas.
# 2. cerró en 1380 de 1300 kcal (+6 %) y le dijo «superaste tu meta» con consejo para mañana, mientras la app pinta
#    «meta cumplida» de 90 a 110 %. Hasta +10 % se dice en positivo: cerró en su meta.
TOLERANCIA_META = 0.10


def nota_comidas_al_dia(tipos_del_dia, hora_local: Optional[float]) -> str:
    """Texto «Para el asistente» con las franjas cuya hora ya pasó y que ya están anotadas, o "". Nunca lanza."""
    try:
        from coach_day_context import FIN_DE_FRANJA, franjas_anotadas
        if hora_local is None:
            return ""
        h = float(hora_local) % 24
        if h < 4.5:
            return ""
        anotadas = franjas_anotadas([{"meal_type": t} for t in (tipos_del_dia or [])])
        ya = [f for f in ("desayuno", "almuerzo", "merienda") if f in anotadas and h >= FIN_DE_FRANJA[f]]
        if not ya:
            return ""
        lista = " y ".join(ya) if len(ya) <= 2 else ", ".join(ya[:-1]) + " y " + ya[-1]
        return (f" COMIDAS YA ANOTADAS tras este registro [P1-PLAN-LOTE-1080]: {lista}. El bloque «COMIDAS DE HOY» se "
                "calculó ANTES de este registro: de estas NO digas «No me mandaste…» ni le pidas su foto.")
    except Exception:  # noqa: BLE001
        return ""


def nota_kcal_en_meta(metas: Optional[dict], consumido: Optional[dict]) -> str:
    """Si se pasó de su meta de kcal por ≤10 %, la orden de decirlo en positivo; si no (o sin datos), "". Nunca lanza."""
    try:
        meta = float((metas or {}).get("kcal") or 0)
        lleva = float((consumido or {}).get("kcal") or 0)
        if meta <= 0 or lleva <= meta or lleva > meta * (1 + TOLERANCIA_META):
            return ""
        return (f" CERRÓ EN SU META [P1-PLAN-LOTE-1080]: ~{int(round(lleva))} de ~{int(round(meta))} kcal, solo "
                f"~{int(round(lleva - meta))} por encima (dentro del ±10 % que su app marca como meta cumplida). Dilo "
                "en positivo («quedaste en tu meta de hoy»), sin «superaste» ni «te pasaste» y sin consejos para mañana.")
    except Exception:  # noqa: BLE001
        return ""
