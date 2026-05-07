"""
Test P0-4: Cuando el inventory proxy actúa como señal débil de aprendizaje,
el sistema debe:
  1. Inyectar weak_signal y banned_proteins en _chunk_lessons.
  2. Reducir la temperatura base del LLM en 0.1.
  3. El prompt builder debe generar la directiva de rotación de proteínas.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


# ─── Test 1: build_chunk_lessons_context genera la regla weak_signal ───
def test_prompt_builder_emits_weak_signal_directive():
    from prompts.plan_generator import build_chunk_lessons_context

    chunk_lessons = {
        "weak_signal": True,
        "banned_proteins": ["Pechuga de pollo", "Huevos", "Salami"],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)

    assert "SEÑAL DÉBIL DE INVENTARIO" in ctx, (
        "El prompt builder debe incluir la directiva de señal débil."
    )
    assert "Pechuga de pollo" in ctx, (
        "Las proteínas baneadas deben aparecer en el prompt."
    )
    assert "al menos 2 de las 3 proteínas principales" in ctx, (
        "La regla de rotación 2-de-3 debe estar presente."
    )


# ─── Test 2: Sin weak_signal NO genera la directiva ───
def test_prompt_builder_no_weak_signal_no_directive():
    from prompts.plan_generator import build_chunk_lessons_context

    chunk_lessons = {
        "ingredient_base_repeat_pct": 50.0,
        "repeated_bases": [{"bases": ["Pollo"]}],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)

    assert "SEÑAL DÉBIL DE INVENTARIO" not in ctx, (
        "Sin weak_signal la directiva de rotación de proteínas NO debe aparecer."
    )


# ─── Test 3: Reducción de temperatura con weak_signal ───
def test_temperature_reduction_with_weak_signal():
    """Simula el cálculo de base_temp que hace graph_orchestrator."""
    # Normal (no re-roll, attempt 1)
    normal_temp = 0.7

    # Con weak_signal
    form_data_weak = {"_chunk_lessons": {"weak_signal": True}}
    weak_signal_mod = -0.1 if form_data_weak.get("_chunk_lessons", {}).get("weak_signal") else 0.0
    weak_temp = normal_temp + weak_signal_mod

    assert weak_temp == pytest.approx(0.6, abs=1e-9), (
        f"Con weak_signal la temp debería ser 0.6, got {weak_temp}"
    )

    # Sin weak_signal
    form_data_normal = {}
    normal_mod = -0.1 if form_data_normal.get("_chunk_lessons", {}).get("weak_signal") else 0.0
    assert normal_mod == 0.0, "Sin weak_signal el mod debe ser 0."


# ─── Test 4: Inyección de banned_proteins desde prior_days ───
def test_banned_proteins_extracted_from_prior_days():
    """Simula la lógica de extracción de proteínas del chunk previo."""
    prior_days = [
        {"day": 1, "protein_pool": ["Pechuga de pollo", "Huevos"], "meals": [{"name": "A"}]},
        {"day": 2, "protein_pool": ["Huevos", "Res molida"], "meals": [{"name": "B"}]},
        {"day": 3, "protein_pool": ["Pechuga de pollo", "Cerdo"], "meals": [{"name": "C"}]},
    ]

    form_data = {"_inventory_activity_proxy_used": True}

    if form_data.get("_inventory_activity_proxy_used"):
        form_data["_force_variety"] = True
        _banned_proteins = set()
        for _d in prior_days:
            for _p in _d.get("protein_pool", []):
                _banned_proteins.add(_p)

        if "_chunk_lessons" not in form_data:
            form_data["_chunk_lessons"] = {}
        form_data["_chunk_lessons"]["weak_signal"] = True
        form_data["_chunk_lessons"]["banned_proteins"] = list(_banned_proteins)

    assert form_data["_force_variety"] is True
    assert form_data["_chunk_lessons"]["weak_signal"] is True

    banned = set(form_data["_chunk_lessons"]["banned_proteins"])
    assert banned == {"Pechuga de pollo", "Huevos", "Res molida", "Cerdo"}, (
        f"Proteínas baneadas inesperadas: {banned}"
    )


# ─── Test 5: Prompt builder genera lecciones incluso si SOLO hay weak_signal ───
def test_prompt_builder_weak_signal_only_produces_output():
    """
    Si _chunk_lessons tiene SOLAMENTE weak_signal + banned_proteins
    (sin repeated_bases, sin violaciones), el builder debe aún así
    generar un bloque de contexto no vacío.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    chunk_lessons = {
        "weak_signal": True,
        "banned_proteins": ["Pescado", "Atún"],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)

    assert len(ctx) > 0, "Debe generar contexto no vacío con solo weak_signal."
    assert "Pescado" in ctx
    assert "Atún" in ctx
