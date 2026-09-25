# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-234 · 2026-09-25] «Actualizar platos» y la modificación del coach respetan el tiempo de cocina.

Auditoría del 25-sep: el `meal_form` de regenerar-día es un dict de claves EXPLÍCITAS y no llevaba `cookingTime` (el
lote 220 lo hidrató en el perfil y lo inyectó en `swap_meal`, que aquí recibía «»); la modificación del coach no lo
recibía nunca, y sus rechazos eran solo los chips.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(f):
    return (_BACKEND / f).read_text(encoding="utf-8")


def _meal_form_literal():
    s = _src("routers/plans.py")
    i = s.index('            meal_form = {\n                "user_id": user_id,')
    j = s.index("\n            }\n", i)
    return s[i:j]


def test_regenerar_dia_lleva_el_tiempo_de_cocina():
    lit = _meal_form_literal()
    assert '"cookingTime": data.get("cookingTime"),' in lit
    assert '"otherMedications": data.get("otherMedications"),' in lit
    assert "tooltip-anchor: P1-PLAN-LOTE-234-TIEMPO-EN-REGENERAR-DIA" in lit


def test_la_regla_que_lee_swap_meal_no_esta_vacia_con_ese_formulario():
    import horizon
    assert horizon.cooking_time_rule({"cookingTime": "none"})
    assert horizon.cooking_time_rule({}) == ""


def test_el_coach_recibe_tiempo_y_rechazos_tecleados():
    s = _src("tools.py")
    i = s.index("def execute_modify_single_meal(")
    j = s.index("\ndef ", i + 10)
    cuerpo = s[i:j]
    assert re.search(r'cooking_time_rule\(\s*\{"cookingTime": _hp\.get\("cookingTime"\)', cuerpo)
    assert "_pwft_dl(_hp).get(\"dislikes\")" in cuerpo
    assert "tooltip-anchor: P1-PLAN-LOTE-234-TIEMPO-EN-EL-COACH" in cuerpo


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 234
