"""[P1-COUNT-HALF-RANGE · 2026-07-27] "3½ ciruelas" pide partir la cuarta ciruela por la mitad.

## Lo que veía el owner

    3½ ciruelas          (merienda del lunes)
    1½ ciruelas          (desayuno del martes)

Nadie parte una ciruela a la mitad para completar la ración. Para frutas pequeñas que se comen
por unidad, el display muestra un rango honesto: **"3–4 ciruelas"**.

## Los dos recortes deliberados

- `ingredients_raw` queda INTACTO: la compra y los macros siguen usando el 3.5 exacto
  (lección display↔raw 2026-07-24: el display es para el humano, el raw para las máquinas).
- Los ajíes morrones ("1½ ajíes") quedan FUERA a propósito: medio pimiento picado es cocina
  normal; media ciruela de postre no.

tooltip-anchor: P1-COUNT-HALF-RANGE
"""
from __future__ import annotations

import copy

import pytest

import humanize_ingredients as H


# ───────────── 1. los casos del owner ─────────────

@pytest.mark.parametrize("linea,esperado", [
    ("3½ ciruelas", "3–4 ciruelas"),
    ("1½ ciruelas", "1–2 ciruelas"),
    ("1½ guayabas", "1–2 guayabas"),
    ("7½ fresas", "7–8 fresas"),
])
def test_conteo_mixto_a_rango(linea, esperado):
    assert H.half_count_range(linea) == esperado


# ───────────── 2. lo que NO se toca ─────────────

@pytest.mark.parametrize("linea", [
    "½ ciruela",             # fracción sola: sí se parte una para picar
    "2 ciruelas",            # entero limpio
    "1¾ ajíes morrones",     # pimiento: medio ají picado es cocina normal (exclusión deliberada)
    "1½ tazas de kale",      # unidad de volumen, no conteo de fruta
    "1½ pechuga de pollo",   # proteína: fuera del alcance
])
def test_fuera_del_alcance_queda_intacto(linea):
    assert H.half_count_range(linea) == linea


@pytest.mark.parametrize("basura", [None, 123, "", []])
def test_fail_safe(basura):
    H.half_count_range(basura)


def test_idempotente():
    una = H.half_count_range("3½ ciruelas")
    assert H.half_count_range(una) == una


# ───────────── 3. pipeline: display cambia, raw no ─────────────

def test_display_cambia_y_raw_queda_exacto():
    plan = {"days": [{"day": 1, "meals": [{
        "name": "Merienda", "meal": "Merienda",
        "ingredients": ["3.5 ciruelas", "0.75 taza de yogurt griego"],
        "recipe": ["Montaje: sirve."],
    }]}]}
    out = H.humanize_plan_ingredients(copy.deepcopy(plan))
    m = out["days"][0]["meals"][0]
    assert any("3–4 ciruelas" in s for s in m["ingredients"]), m["ingredients"]
    assert "3.5 ciruelas" in m["ingredients_raw"], "el raw alimenta compra y macros: intacto"
