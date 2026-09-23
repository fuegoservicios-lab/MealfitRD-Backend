# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-179 · 2026-09-23] El recorte de grasa del día respeta lo que da nombre al plato… hasta su piso, no más.

Métrica sobre los planes REALES de la batería: 3 de 48 días salían más de un 10 % sobre su grasa, uno al +24 % (familia de
4: aguacate nombrado en tres comidas del mismo día). `_trim_day_fats_to_target` se saltaba ENTERA cualquier línea que el
nombre del plato menciona (lote 46, para que el guacamole no se quede sin aguacate), así que con la grasa del día en lo
nombrado no había de dónde recortar. Ahora lo nombrado cede lo que tiene por ENCIMA de su piso de identidad (el mismo
criterio que el reequilibrio desde el 177): «½ aguacate» puede quedar en láminas, nunca en 5 g."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import identidad_plato as idp  # noqa: E402


class _DB:
    _POR_G = (("aguacate", (1.60, 0.15)), ("aceite", (8.84, 1.0)))

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\s+de\s+(.+)$", str(s).strip(), re.IGNORECASE)
        if not m:
            return None
        g, nombre = float(m.group(1)), idp._sa(m.group(2)).strip()
        kpg, fpg = next((v for k, v in self._POR_G if k in nombre), (1.0, 0.0))
        return {"name": nombre, "grams": g, "kcal": kpg * g, "protein": 0.01 * g, "carbs": 0.02 * g, "fats": fpg * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def category_of(self, s):
        return "Frutas" if "aguacate" in idp._sa(s) else ""

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _g(lineas, palabra):
    return next(float(re.match(r"\s*([\d.]+)", x).group(1)) for x in lineas if palabra in x)


def test_lo_nombrado_cede_hasta_su_piso():
    import graph_orchestrator as go
    comida = {"meal": "Desayuno", "name": "Tostadas con aguacate", "cals": 250, "protein": 2, "carbs": 3, "fats": 25,
              "ingredients": ["100 g de aguacate", "10 g de aceite de oliva"],
              "ingredients_raw": ["100 g de aguacate", "10 g de aceite de oliva"]}
    assert go._trim_day_fats_to_target([comida], 10.0, _DB(), tol=0.05)
    agu = _g(comida["ingredients"], "aguacate")
    assert 30 <= agu < 100, comida["ingredients"]
    assert _g(comida["ingredients_raw"], "aguacate") == agu, "la compra sigue a la lista"


def test_lo_nombrado_ya_en_su_piso_no_baja():
    import graph_orchestrator as go
    comida = {"meal": "Desayuno", "name": "Tostadas con aguacate", "cals": 140, "protein": 1, "carbs": 1, "fats": 14.5,
              "ingredients": ["30 g de aguacate", "10 g de aceite de oliva"],
              "ingredients_raw": ["30 g de aguacate", "10 g de aceite de oliva"]}
    go._trim_day_fats_to_target([comida], 5.0, _DB(), tol=0.05)
    assert _g(comida["ingredients"], "aguacate") == 30


def test_fraccion_sobre_piso():
    db = _DB()
    assert idp.fraccion_sobre_piso("100 g de aguacate", db.macros_from_ingredient_string("100 g de aguacate"), db) == 0.7
    assert idp.fraccion_sobre_piso("20 g de aguacate", db.macros_from_ingredient_string("20 g de aguacate"), db) == 0.0


def test_el_revisor_mide_la_proteina_como_la_guarda_el_guardado():
    """Batería real (embarazo): la puerta de proteína midió 98/116 g y marcó el plan como degradado; el guardado, que
    re-encuadra la proteína DESPUÉS, lo dejaba en 114 g. Ahora el mismo re-encuadre corre al entrar en el revisor."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def review_plan_node(state: PlanState) -> dict:")
    assert '__import__("protein_floor_last_word").reencuadrado(state["plan_result"])' in src[i:i + 500]
    import protein_floor_last_word as pf
    plan = {"days": "roto"}
    assert pf.reencuadrado(plan) is plan and pf.reencuadrado(None) is None, "nunca lanza y devuelve el mismo plan"


def test_los_consejos_de_absorcion_no_son_rechazo():
    """Batería real (embarazo): rechazo CRÍTICO por «el yogur se sirve junto con lentejas… separar los lácteos del
    hierro 2 horas». Es un consejo de absorción, no una violación de seguridad."""
    from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT as p
    assert "ABSORCIÓN" in p and "no son motivo de rechazo" in p


def test_mani_tostadas_tras_el_swap_de_almendras():
    import graph_orchestrator as go
    assert go._fix_name_gender_agreement("Ricotta batida con mandarina, maní tostadas y queso mozzarella") == \
        "Ricotta batida con mandarina, maní tostado y queso mozzarella"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 179 and m.group(2) >= "2026-09-23"
