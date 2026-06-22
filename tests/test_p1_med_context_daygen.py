"""[P1-MED-CONTEXT-DAYGEN · 2026-06-22] (audit fresco P1-1) Las directivas clínicas deterministas
(condición + interacción fármaco-alimento) deben llegar NO SOLO al esqueleto sino TAMBIÉN al
day-generator de producción (`generate_days_parallel_node`), que es el nodo que elige los
ingredientes/recetas reales (toronja↔estatina, hoja verde↔warfarina, sustituto de sal↔IECA).

Pre-fix: `build_medical_condition_context` + `build_medication_context` se anexaban SOLO a
`variety_prompt`, que únicamente se interpola en `plan_skeleton_node`. El day-gen no las veía →
un perfil medicado en tier gratis (Flash; el day-gen escala a PRO solo por tier) recibía el prompt
del plato SIN la directiva.

Fix: bloque propio `clinical_directives` → expuesto en ctx como `clinical_directives_context` →
interpolado en el `dynamic_day_prompt`. Sigue anexándose a `variety_prompt` (esqueleto sin
regresión). No-op (`""`) para perfiles sin condición/medicamento → day-gen byte-equivalente.

Parser-based (robusto a venv sin langgraph) + funcional ligero sobre los builders (sin DB ni LLM).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _func_body(src: str, signature_prefix: str) -> str:
    """Cuerpo de la primera función cuyo `def` empieza con `signature_prefix`,
    hasta el siguiente `\\ndef ` a nivel módulo (captura funciones anidadas)."""
    start = src.find(signature_prefix)
    assert start >= 0, f"No se encontró la función: {signature_prefix}"
    nxt = src.find("\ndef ", start + 1)
    return src[start: nxt if nxt > 0 else len(src)]


# ─────────────────────────── A. Bloque clínico aislado ───────────────────────────

def test_clinical_directives_isolated_block():
    # Las directivas clínicas se computan en su PROPIO acumulador (no solo anexadas a variety).
    assert 'clinical_directives = ""' in _GRAPH
    assert "clinical_directives += build_medical_condition_context(form_data)" in _GRAPH
    assert "clinical_directives += build_medication_context(form_data)" in _GRAPH


def test_skeleton_still_receives_clinical_via_variety():
    # Anti-regresión del esqueleto: variety_prompt sigue incluyendo las directivas clínicas
    # (el esqueleto las consume vía ctx['variety_prompt']).
    assert 'variety_prompt = (variety_prompt or "") + clinical_directives' in _GRAPH
    # El esqueleto interpola variety_prompt (no debe perderse). Firma completa para no
    # colisionar con el ejemplo del docstring de `_node_label`.
    skel = _func_body(_GRAPH, "async def plan_skeleton_node(state: PlanState)")
    assert "ctx['variety_prompt']" in skel, (
        "El esqueleto debe seguir interpolando variety_prompt (que contiene las directivas clínicas)."
    )


def test_clinical_directives_exposed_in_ctx():
    assert '"clinical_directives_context": clinical_directives,' in _GRAPH


# ─────────────────────────── B. Llega al day-generator ───────────────────────────

def test_clinical_directives_reach_day_generator_prompt():
    body = _func_body(_GRAPH, "async def generate_days_parallel_node(")
    assert "ctx['clinical_directives_context']" in body, (
        "Las directivas clínicas DEBEN interpolarse en el prompt del day-generator "
        "(el nodo que elige ingredientes/recetas reales)."
    )
    # Debe estar dentro de la construcción del dynamic_day_prompt (el prompt del plato).
    i_prompt = body.find("dynamic_day_prompt = (")
    i_clin = body.find("ctx['clinical_directives_context']")
    assert i_prompt >= 0, "No se encontró la construcción de dynamic_day_prompt."
    assert i_clin > i_prompt, (
        "La directiva clínica debe interpolarse DENTRO del dynamic_day_prompt, no antes."
    )


def test_tooltip_anchor_present():
    # Si alguien renombra la clave/marker, este test falla antes de romper prod.
    assert _GRAPH.count("P1-MED-CONTEXT-DAYGEN") >= 3


# ─────────────────────────── C. Funcional ligero (builders, sin DB/LLM) ───────────────────────────

def test_medication_builder_emits_directive_for_warfarin():
    from prompts.plan_generator import build_medication_context
    out = build_medication_context({"medications": ["Warfarina"]})
    assert out and "INTERACCIÓN FÁRMACO-ALIMENTO" in out
    # No-op para perfil sin medicamento.
    assert build_medication_context({"medications": []}) == ""


def test_condition_builder_noop_for_no_condition():
    from prompts.plan_generator import build_medical_condition_context
    # Sin condición cubierta → "" (no contamina el prompt del day-gen de un perfil sano).
    assert build_medical_condition_context({}) == ""
