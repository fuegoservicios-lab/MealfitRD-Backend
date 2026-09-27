# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-468 · 2026-09-27] Batería REAL del bloque 2 (días 4-7 de 30 sin congelador, alérgico al pescado).

Tres defectos que el replay no podía ver porque salen de la IA:
1. «1 chuleta (≈89 g)» —la chuleta de cerdo fresca— caía en el default de categoría (7 días) y se servía el día 7 sin
   sustituir; sólo «chuleta de cerdo» salía a 3 días.
2. «300 g de edamame cocido» tres días seguidos: el edamame del súper es congelado de fábrica y el usuario no tiene
   congelador; nada lo sustituía.
3. El «Bowl frío de garbanzos» declaraba 98 g de proteína —las de 1¾ pechugas— porque el recálculo completo de macros se
   niega si una línea con nombre no trae cantidad sumable («Ajo, 1 diente picado»). Ahora el plato cambia sus macros con
   la línea: − vieja + nueva.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import pantry_durability as pdu  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_REQ = {"need_days": 4, "allow_frozen": False, "freezer_mode": "none"}


def test_la_chuleta_a_secas_es_cerdo_fresco():
    assert pdu.classify("1 chuleta (≈89 g)")["cls"] == "freezable"
    assert pdu.classify("1 chuleta (≈89 g)")["days_fresh"] == 3
    assert pdu.classify("2 chuletas ahumadas")["cls"] == "cold", "la ahumada sigue curada"
    r = cu.sustituir_linea("1 chuleta (≈89 g)", 3, _REQ, semilla=0)
    assert r and r[1] in ("atun en agua", "sardinas en lata"), r


def test_el_edamame_sin_congelador_pasa_a_garbanzos():
    r = cu.sustituir_linea("300 g de edamame cocido", 4, {"need_days": 5, "allow_frozen": False})
    assert r and r[0] == "300 g de garbanzos cocidos", r
    assert cu.sustituir_linea("300 g de edamame cocido", 4, {"need_days": 5, "allow_frozen": True}) is None, \
        "con congelador el edamame aguanta"


class _DB:
    _T = {"pechuga": {"protein": 88.0, "carbs": 0.0, "fats": 8.0, "kcal": 450.0},
          "garbanzo": {"protein": 19.0, "carbs": 59.0, "fats": 6.0, "kcal": 363.0}}

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        for k, v in self._T.items():
            if k in low:
                return dict(v)
        return None


def test_las_macros_cambian_con_la_linea():
    m = {"name": "Bowl frío caribeño de pollo", "protein": 98, "carbs": 76, "fats": 23, "cals": 899,
         "ingredients": ["1¾ pechugas de pollo (≈286 g)", "Ajo, 1 diente picado"],
         "ingredients_raw": ["1¾ pechugas de pollo (≈286 g)", "Ajo, 1 diente picado"], "recipe": []}
    sf.sustituir_en_plato(m, 0, "1¾ pechugas de pollo (≈286 g)", "286 g de garbanzos cocidos", "garbanzos cocidos", db=_DB())
    assert (m["protein"], m["carbs"], m["fats"], m["cals"]) == (29, 135, 21, 812), m
    assert m["macros"] == ["P:29g", "C:135g", "G:21g"]


def test_sin_db_no_toca_las_macros():
    m = {"protein": 98, "carbs": 76, "fats": 23, "cals": 899, "ingredients": ["x"], "recipe": []}
    assert sf.ajustar_macros(m, "1 pechuga", "150 g de atún", None) is False
    assert m["protein"] == 98
