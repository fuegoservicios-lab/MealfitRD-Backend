"""[P1-RENAL-CAP-FAILHARD-GATE · 2026-06-15] Cap renal de proteína como GATE, no telemetría.

Bug original (gap-audit 2026-06-15, G3):
    El cap renal (KDIGO 0.8 g/kg, seguridad iatrogénica en ERC) trimaba proteína per-comida y
    marcaba `renal_protein_cap.meals_enforced` (¿convergió el trim?) y `.cap_complete` (¿toda la
    proteína-dominante resolvió?). PERO si alguno era False, el plan sobre-proteico se ENTREGABA
    igual al paciente renal: `should_retry` nunca leía esos flags, `review_plan_node` no añadía un
    gate de techo renal. Era telemetría sin acción correctiva — justo en la población de mayor
    riesgo iatrogénico.

Cierre:
    `review_plan_node` escala (meals_enforced is False) OR (cap_complete is False) en un plan
    renal-capeado a rechazo CRÍTICO → `should_retry` retorna "end" (severity critical) →
    `_apply_critical_review_guardrails` (rama `needs_critical_fallback`) sustituye por el fallback
    matemático, que está renal-capeado POR CONSTRUCCIÓN (`_apply_renal_cap_to_nutrition` capa la
    nutrition de origen). Se marca `_had_renal_critical` para que el soft-reject DM2 (comórbido
    DM2+ERC frecuente) NO degrade este critical a 'high'. Gate: MEALFIT_RENAL_CAP_FAILHARD_GATE (default True).

Test parser-based (sin DB/LLM): ancla la cadena estructural. La ruta crítico→fallback en sí ya
está cubierta por los tests de rechazo crítico existentes (allergen/schema); este cambio solo
enruta el caso renal hacia esa ruta probada vía severity=critical. Si un refactor rompe el gate,
la exención DM2, o baja la severidad, este test falla.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _review_body() -> str:
    start = _SRC.find("async def review_plan_node")
    assert start != -1, "No se encontró `async def review_plan_node`."
    # Fin del cuerpo = siguiente def a nivel de módulo (col 0).
    m = re.search(r"\n(async def |def )", _SRC[start + 10:])
    end = (start + 10 + m.start()) if m else len(_SRC)
    return _SRC[start:end]


def test_knob_defined_default_true():
    assert re.search(
        r'RENAL_CAP_FAILHARD_GATE\s*=\s*_env_bool\(\s*"MEALFIT_RENAL_CAP_FAILHARD_GATE"\s*,\s*True\s*\)',
        _SRC,
    ), "Falta el knob RENAL_CAP_FAILHARD_GATE (default True) — safety gate, debe defaultear ON."


def test_gate_present_and_escalates_to_critical():
    body = _review_body()
    assert "P1-RENAL-CAP-FAILHARD-GATE" in body, "Falta el tooltip-anchor del gate en review_plan_node."
    # El gate se gatea por el knob + plan renal-capeado.
    assert re.search(r"if\s+RENAL_CAP_FAILHARD_GATE\s+and\s+_renal_capped_plan\s*:", body), (
        "El gate renal debe gatearse por `RENAL_CAP_FAILHARD_GATE and _renal_capped_plan`."
    )
    # Lee AMBAS señales de inseguridad del cap.
    assert "meals_enforced" in body and "cap_complete" in body, (
        "El gate debe leer meals_enforced (trim convergió) Y cap_complete (proteína resuelta)."
    )
    # Escala a critical (no a high/minor) → única severidad que dispara el fallback renal-capeado.
    assert re.search(r'_severity_max\(\s*severity\s*,\s*"critical"\s*\)', body), (
        "El gate renal debe escalar a 'critical' — high/minor NO disparan el fallback seguro."
    )


def test_gate_sets_no_degrade_flag():
    body = _review_body()
    # `_had_renal_critical` se inicializa en False y el gate lo pone True.
    assert "_had_renal_critical = False" in body, "Falta la init `_had_renal_critical = False`."
    assert "_had_renal_critical = True" in body, (
        "El gate debe setear `_had_renal_critical = True` para blindar el critical contra degradación."
    )
    # Orden: init ANTES del gate ANTES de leerse en el degrade DM2.
    idx_init = body.find("_had_renal_critical = False")
    idx_set = body.find("_had_renal_critical = True")
    idx_read = body.find("not _had_renal_critical")
    assert idx_init != -1 and idx_set != -1 and idx_read != -1
    assert idx_init < idx_set, "La init debe preceder al set del flag."
    assert idx_set < idx_read, "El flag debe setearse antes de leerse en el degrade DM2."


def test_dm2_degrade_exempts_renal_critical():
    """Comórbido DM2+ERC: el soft-reject glucémico NO debe degradar el critical renal a 'high'
    (eso reintroduciría la entrega del plan sobre-proteico al paciente renal-diabético)."""
    body = _review_body()
    # Localiza la condición del degrade DM2 y exige que excluya el critical renal.
    m = re.search(r"if\s*\(\s*DM2_GLYCEMIC_SOFT_REJECT.*?\):", body, re.DOTALL)
    assert m, "No se encontró la condición del degrade DM2 (DM2_GLYCEMIC_SOFT_REJECT)."
    cond = m.group(0)
    assert "not _had_renal_critical" in cond, (
        "El degrade DM2 debe incluir `not _had_renal_critical` — sin esto, un paciente comórbido "
        "DM2+ERC degradaría el techo renal critical a 'high' y recibiría el plan sobre-proteico."
    )
    assert "not _had_allergen_critical" in cond, "Regresión: la exención de alérgenos debe seguir presente."
