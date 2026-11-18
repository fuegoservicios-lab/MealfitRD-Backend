# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1079 · 2026-11-18] El aviso de la cena propone un plato del tamaño de lo que le queda del día.

Caso (cuenta de prueba, 17-nov, objetivo ganar músculo, meta 3350 kcal): a las 21:30 le quedaban ~1246 kcal y solo la
cena, y el aviso propuso «una pechuga de unos 180 g con vegetales al vapor y una batata mediana» (~500 kcal). Cerró el
día en 2969 (89 %). `bloque_del_dia` ya le pasaba «le quedan 1246» al prompt, pero solo tenía reglas para el margen
ESTRECHO; nada decía que la cena, la última comida principal, debía cubrir un margen grande.

Determinista: en el aviso de la CENA, con ≥600 kcal por cubrir, la regla da la cifra del plato (lo que queda, con un
techo por objetivo) y ejemplos de porción. Para ganar músculo o rendimiento, quedarse corto es el error; para perder
grasa el techo es más bajo y nunca se empuja a «recuperar» el día.
"""
from __future__ import annotations

UMBRAL_KCAL = 600
_TECHO = {"gain_muscle": 1100, "performance": 1000, "maintenance": 900, "lose_fat": 750}
_GANAR = {"gain_muscle", "performance"}


def _redondear(v: float) -> int:
    return int(round(v / 50.0) * 50)


def regla(comida, quedan_kcal, objetivo) -> str:
    """Línea para el bloque del aviso, o "" si no aplica. Nunca lanza."""
    try:
        if str(comida or "").strip().lower() != "cena":
            return ""
        quedan = float(quedan_kcal or 0)
        if quedan < UMBRAL_KCAL:
            return ""
        obj = str(objetivo or "").strip().lower()
        plato = _redondear(min(quedan, _TECHO.get(obj, 900)))
        linea = (f"- CENA A LA MEDIDA [P1-PLAN-LOTE-1079]: es su última comida principal y le quedan ~{int(round(quedan))} "
                 f"kcal. Propón un plato de ~{plato} kcal con porciones CONCRETAS de ese tamaño (p. ej. 180-220 g de "
                 "proteína, 1½-2 tazas de arroz, yuca, batata o plátano, y una grasa: aguacate o aceite); NO una cena "
                 "ligera de ~500 kcal.")
        if obj in _GANAR:
            linea += (" Su objetivo es ganar: quedarse corto es lo que frena su progreso; si no le entra todo en el plato, "
                      "sugiere completar con un batido o un vaso de leche después.")
        else:
            linea += " Sin decirle que «recupere» el día ni que coma de más."
        return linea
    except Exception:  # noqa: BLE001
        return ""
