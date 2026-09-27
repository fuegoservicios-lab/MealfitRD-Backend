# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-558 · 2026-09-27] Los topes de plan entero también al cambiar platos.

Auditoría del formulario: swap, «Actualizar platos» y el coach trabajan sobre un mini-plan, así que el pescado de
embarazo (≤340 g/semana), el casabe en DM2 y las yemas con colesterol no se medían tras el cambio.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import topes_plan_entero as tpe  # noqa: E402


class _Db:
    def grams_from_ingredient_string(self, linea):
        import re
        m = re.match(r"^\s*(\d+)\s*g\b", str(linea))
        return float(m.group(1)) if m else 0.0

    def lookup(self, nombre):
        return None

    def macros_from_ingredient_string(self, s):
        return None


def _dia(n, pescado):
    return {"day": n, "meals": [{"meal": "Almuerzo", "name": f"Tilapia al limón con arroz {n}",
                                 "ingredients": [f"{pescado} g de tilapia", "80 g de arroz blanco"],
                                 "ingredients_raw": [f"{pescado} g de tilapia", "80 g de arroz blanco"],
                                 "recipe": ["El Toque de Fuego: cocina la tilapia 4 min por lado."]}]}


def test_el_tope_del_pescado_mira_el_plan_entero(monkeypatch):
    llamadas = []
    monkeypatch.setattr(__import__("embarazo_pescado"), "limitar_pescado",
                        lambda plan, fd, db=None: llamadas.append(len(plan["days"])) or 1)
    plan = {"days": [_dia(1, 150), _dia(2, 150), _dia(3, 150)]}
    assert tpe.aplicar(plan, {"medicalConditions": ["Embarazo"], "dietType": "balanced"}, db=_Db()) >= 1
    assert llamadas == [3], "el tope se aplica sobre los 3 días, no sobre el plato"


def test_sin_contexto_no_hace_nada():
    assert tpe.aplicar({"days": [_dia(1, 150)]}, {}, db=_Db()) == 0


def test_cableado_en_las_tres_superficies():
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert '__import__("topes_plan_entero").aplicar(plan_data, _micro_form)' in plans
    assert '__import__("topes_plan_entero").aplicar(pd, _rp_ctx554, _db)' in plans
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert '__import__("topes_plan_entero").aplicar(plan_data, _micro_form_cm)' in tools
    assert '__import__("topes_plan_entero").aplicar(plan_data_fresh, _micro_form_cm)' in tools
