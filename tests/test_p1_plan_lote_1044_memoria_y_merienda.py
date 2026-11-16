# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1044 · 2026-11-16] Decisiones del dueño: primero lo que el coach sabe, al final lo no anotado (A-19);
la merienda nace sin aviso (A-29)."""
import coach_day_context as cdc
import proactive_agent as pa

PLAN = {"calories": 2000, "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}}   # el del lote 132


def test_con_el_diario_corto_contesta_primero_y_pregunta_al_final():
    out = cdc.build_day_gap_context({}, PLAN, [], 21.2)
    assert "le falta REGISTRAR" in out
    assert "Contesta PRIMERO lo que pregunta usando lo que sabes de él" in out
    assert "AL FINAL, pregúntale en una línea" in out
    assert "Antes de recomendarle comer más, pregúntale" not in out


def test_la_merienda_nace_sin_aviso_y_se_puede_encender():
    assert pa.comida_con_aviso({}, "Merienda") is False
    assert pa.comida_con_aviso({"avisos_por_comida": {"merienda": {"activo": True}}}, "Merienda") is True
    for comida in ("Desayuno", "Almuerzo", "Cena"):
        assert pa.comida_con_aviso({}, comida) is True, comida
