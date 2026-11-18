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
