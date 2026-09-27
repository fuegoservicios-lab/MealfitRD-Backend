# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-461 · 2026-09-27] La pareja en `ingredients_raw` del fresco sustituido se busca por ALIMENTO.

`_single_trip_fresh_substitute` escribía `raw[idx] = nueva` por ÍNDICE, y `ingredients_raw` no sigue el orden de la lista
visible. En el plan vivo del dueño (6594aae1, día 2) escribió las sardinas encima de otra línea y dejó «1¼ filetes de
pescado»: la lista del mes compraba pescado fresco para un plato que ya no lo usaba (y perdía la línea pisada). Forzando
la compra única sobre el corpus: 57 de 150 sustituciones dejaban el fresco en `raw`. El inventario del ratchet de `raw` la
tenía como «0 medido 07-sep»: la sustitución sólo corre desde el día 4 de un ciclo largo y la sonda nunca la vio activa.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced"}}


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _plato_duenyo():
    visibles = ["1 filete de pescado", "½ pedazo de yautía (≈107 g)", "½ ají morrón", "½ tomate", "½ cebolla",
                "1 diente de ajo", "½ limón", "½ aguacate"]
    raw = ["0.5 ají morrón", "0.5 tomate", "0.5 cebolla", "1 diente de ajo", "0.5 limón", "0.5 aguacate",
           "1 filete de pescado", "½ pedazo de yautía (≈107 g)"]
    return {"meal": "Almuerzo", "name": "Bowl de pescado", "ingredients": visibles, "ingredients_raw": raw, "recipe": []}


def test_la_pareja_se_busca_por_alimento_y_no_pisa_otra_linea(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(9)]
    days.append({"day": 10, "meals": [_plato_duenyo()]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced", contexto={})
    m = days[-1]["meals"][0]
    raw = m["ingredients_raw"]
    assert not any("pescado" in r for r in raw), raw
    assert "0.5 ají morrón" in raw, "la línea de al lado sigue en la lista"
    assert len(raw) == 8, raw
    nueva = m["ingredients"][0]
    assert raw.count(nueva) == 1, (nueva, raw)


def test_duplicados_del_mismo_fresco_salen_y_sin_pareja_se_anyade():
    m = {"ingredients": ["1 filete de pescado", "1 limón"],
         "ingredients_raw": ["185 g de filete de pescado blanco", "1 limón", "1¼ filetes de pescado"]}
    assert sf.parear_raw(m, "1 filete de pescado", "150 g de sardinas en lata") == "pareado"
    assert m["ingredients_raw"] == ["150 g de sardinas en lata", "1 limón"], m["ingredients_raw"]
    m = {"ingredients": ["½ pechuga de pollo", "1 limón"], "ingredients_raw": ["1 limón"]}
    assert sf.parear_raw(m, "½ pechuga de pollo", "100 g de atún en agua") == "añadido"
    assert m["ingredients_raw"] == ["1 limón", "100 g de atún en agua"]


def test_no_toca_lo_que_ya_es_duradero_ni_la_pareja_de_otra_linea():
    m = {"ingredients": ["150 g de pechuga de pollo", "80 g de atún en agua", "1 muslo de pollo congelado"],
         "ingredients_raw": ["80 g de atún en agua", "1 muslo de pollo congelado", "150 g de pechuga de pollo"]}
    sf.parear_raw(m, "150 g de pechuga de pollo", "150 g de sardinas en lata")
    assert m["ingredients_raw"] == ["80 g de atún en agua", "1 muslo de pollo congelado", "150 g de sardinas en lata"]
