# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1046 · A-12] Salida compacta del escáner: mismos datos que `_MealVisionResult`, menos tokens."""
from pathlib import Path

import vision_agent as va
import vision_salida_compacta as v

_BACKEND = Path(__file__).resolve().parent.parent


def _compacto():
    return v.ResultadoCompacto(
        k="plato", f=True, nm="Arroz con pollo", d="arroz blanco, pollo guisado", m=[650, 40, 80, 15],
        i=[v._CItem(n="arroz blanco", q=1, u="taza", m=[240, 4, 52, 0.5]),
           v._CItem(n="pollo guisado", q=1, u="unidad", m=[410, 36, 28, 14.5])],
        q=[v._CDuda(o="pollo guisado", p="¿Pollo o cerdo?",
                    x=[v._COpcion(t="Pollo", s=True, a=[0, 0, 0, 0]), v._COpcion(t="Cerdo", n="Arroz con cerdo", a=[60, -2, 0, 6])])])


def test_apagado_por_defecto(monkeypatch):
    monkeypatch.delenv("MEALFIT_VISION_SALIDA_COMPACTA", raising=False)
    assert v.activo() is False


def test_expandir_da_la_forma_larga_de_siempre():
    d = v.expandir(_compacto())
    largo = va._MealVisionResult.model_validate(d).model_dump()
    assert largo["calories"] == 650 and largo["protein"] == 40 and largo["healthy_fats"] == 15
    assert largo["items"][1] == {"name": "pollo guisado", "quantity": 1.0, "unit": "unidad",
                                 "calories": 410.0, "protein": 36.0, "carbs": 28.0, "healthy_fats": 14.5}
    op = largo["dudas"][0]["opciones"][1]
    assert op["texto"] == "Cerdo" and op["nombre_plato"] == "Arroz con cerdo" and op["ajuste"]["calories"] == 60
    assert largo["dudas"][0]["opciones"][0]["supuesta"] is True


def test_el_saneo_de_siempre_lo_acepta():
    out = va._coerce_meal_scan(v.expandir(_compacto()))
    assert out["calories"] == 650 and out["is_food"] is True
    assert out["items"][0]["macros"]["calories"] == 240.0
    assert out["dudas"]


def test_listas_cortas_o_raras_no_rompen():
    d = v.expandir({"k": "items", "f": True, "m": [], "i": [{"n": "leche", "q": 2, "u": "botella", "m": ["x"]}], "q": []})
    assert d["calories"] == 0 and d["items"][0]["calories"] == 0.0 and d["items"][0]["quantity"] == 2


def test_cableado_en_el_dispatch():
    src = (_BACKEND / "vision_agent.py").read_text(encoding="utf-8")
    assert "_vsc.ResultadoCompacto if _compacta else _MealVisionResult" in src
    assert "+ (_vsc.LEYENDA if _compacta else \"\")," in src
