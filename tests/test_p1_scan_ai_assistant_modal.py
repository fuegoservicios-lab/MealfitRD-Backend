# -*- coding: utf-8 -*-
"""[P1-SCAN-AI-ASSISTANT-MODAL · 2026-10-20] Tests para el asistente IA y adición de ingredientes en el modal de escaneo.
"""
from __future__ import annotations

from pathlib import Path
import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def test_normalizar_componente_clamps_and_types():
    import ajustar_plato as ap

    raw = {
        "name": "  Pechuga de pollo a la plancha  ",
        "display": "Pechuga a la plancha",
        "qty": 150.456,
        "unit": "g",
        "calories": 248.9,
        "protein": 46.22,
        "carbs": -2.0,
        "healthy_fats": 5.4,
    }
    norm = ap.normalizar_componente(raw)
    assert norm["name"] == "Pechuga de pollo a la plancha"
    assert norm["display"] == "Pechuga a la plancha"
    assert norm["qty"] == 150.46
    assert norm["unit"] == "g"
    assert norm["macros"]["calories"] == 249
    assert norm["macros"]["protein"] == 46.2
    assert norm["macros"]["carbs"] == 0.0
    assert norm["macros"]["healthy_fats"] == 5.4


def test_normalizar_componente_con_topes_maximos():
    import ajustar_plato as ap

    raw = {
        "name": "Super batido",
        "qty": 99999,
        "unit": "ml",
        "calories": 999999,
        "protein": 9999,
        "carbs": 9999,
        "healthy_fats": 9999,
    }
    norm = ap.normalizar_componente(raw)
    assert norm["qty"] == 5000.0
    assert norm["macros"]["calories"] == 4000
    assert norm["macros"]["protein"] == 350.0
    assert norm["macros"]["carbs"] == 500.0
    assert norm["macros"]["healthy_fats"] == 300.0


def test_peticiones_validan_longitud():
    import ajustar_plato as ap
    from pydantic import ValidationError

    # Instrucción muy corta debe fallar
    with pytest.raises(ValidationError):
        ap.PeticionAjustarPlato(instruccion="a")

    # Texto de ingrediente muy corto debe fallar
    with pytest.raises(ValidationError):
        ap.PeticionAgregarIngrediente(texto="x")

    # Válidos
    p1 = ap.PeticionAjustarPlato(instruccion="quítale la ensalada y ponle 2 huevos fritos")
    assert "ensalada" in p1.instruccion

    p2 = ap.PeticionAgregarIngrediente(texto="1 vaso de jugo de naranja")
    assert "jugo" in p2.texto


def test_endpoints_declarados_en_diary_con_sus_limitadores():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert '@router.post("/scan/ajustar-plato")' in src
    assert '@router.post("/scan/agregar-ingrediente")' in src
    assert "_AJUSTAR_PLATO_LIMITER" in src
    assert "_AGREGAR_INGREDIENTE_LIMITER" in src
    assert "adjust_plate_unavailable" in src
    assert "add_ingredient_unavailable" in src
