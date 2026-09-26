# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-365 · 2026-09-26] Cambiar un ingrediente del escáner por TEXTO («Queso» → «Queso mozzarella»).

El dueño: «quiero la opción de cambiar un ingrediente: el queso no lo pude cambiar y nombrar su marca, mozzarella».
La lista de «Ingredientes que detectamos» solo dejaba marcar/desmarcar y la cantidad. Ahora cada fila tiene «Cambiar»:
el nombre nuevo va al servidor con la cantidad y la unidad que ya tiene, y una llamada de TEXTO (flash) devuelve las
macros del ingrediente nuevo Y del anterior en esa cantidad —el frontend, sin desglose, aplica la diferencia—.
Tooltip-anchor: P1-PLAN-LOTE-365
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

_REQ = {
    "plato": "Sándwich de jamón y queso",
    "anterior": "Queso",
    "nuevo": "Queso mozzarella",
    "cantidad": 2,
    "unidad": "lasca",
}


def test_el_modelo_ve_lo_anterior_lo_nuevo_y_la_cantidad():
    import ingrediente_corregido as ic
    msg = ic.mensaje_para_el_modelo(ic.PeticionIngrediente(**_REQ))
    assert "Sándwich de jamón y queso" in msg
    assert "Queso" in msg and '"Queso mozzarella"' in msg
    assert "2 lasca" in msg
    assert "es un dato" in msg


def test_normaliza_macros_nuevas_y_anteriores_acotadas_y_no_negativas():
    import ingrediente_corregido as ic
    r = ic.normalizar({"calories": 160.4, "protein": 11.26, "carbs": -3, "healthy_fats": 12,
                       "calories_anterior": 140, "protein_anterior": 9, "carbs_anterior": 1, "healthy_fats_anterior": 11})
    assert r["macros"] == {"calories": 160, "protein": 11.3, "carbs": 0.0, "healthy_fats": 12.0}
    assert r["anteriores"] == {"calories": 140, "protein": 9.0, "carbs": 1.0, "healthy_fats": 11.0}
    assert ic.normalizar({"calories": 99999})["macros"]["calories"] == 3000


def test_endpoint_exento_de_cuota_con_su_limitador_y_soft_fail():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/scan/ingrediente")')
    bloque = src[i:i + 1600]
    assert "_INGREDIENTE_LIMITER" in bloque[:300] and "verify_api_quota" not in bloque
    assert '"ingredient_unavailable"' in bloque
    assert "import ingrediente_corregido" in src[:4000]
