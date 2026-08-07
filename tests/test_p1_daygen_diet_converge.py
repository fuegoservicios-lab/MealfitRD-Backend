"""[P1-DAYGEN-DIET-CONVERGE · 2026-08-07] Convergencia clínica en GENERACIÓN.

QUÉ PASÓ. El benchmark del landing (issue #9) midió 13/20 perfiles con restricciones
rechazados por el guard crítico; el diagnóstico (X-Bioboros-Review-Diag) mostró planes
vegan/vegetarian NACIENDO con camarones/atún/lácteos, y perfiles HTA con «Sal al gusto»
fabricada determinísticamente en cada comida. Los guards funcionaban; el generador
desobedecía — y el abort a cero-retries garantizaba la no-convergencia.

QUÉ ANCLA (4 capas, cada una con knob de rollback):
  1. Scrub de DIETA en pools del skeleton (`_diet_pool_item_banned` reusa
     `_scan_diet_violations` — mismo matcher, cero 4ª tabla P1-DIET-CANON-SSOT).
  2. Directiva de DIETA en skeleton + day-gen (`_build_diet_directive_context`,
     "" para balanced = cache preservado).
  3. Gate clínico del splitter de sal (HTA/renal → solo pimienta).
  4. UN retry informado para críticos de dieta/alérgeno (severidad SIGUE critical —
     jamás se entrega la violación; attempt==1 + budget).

tooltip-anchor: P1-DAYGEN-DIET-CONVERGE
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import graph_orchestrator as go

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Knobs: existen y nacen ON (son capas de seguridad/convergencia)
# ---------------------------------------------------------------------------
def test_knobs_exist_and_default_on():
    assert go.SKELETON_DIET_SCRUB_ENABLED is True
    assert go.DIET_DIRECTIVE_BLOCK_ENABLED is True
    assert go.SALT_LINE_CONDITION_GATE is True
    assert go.DIET_CRITICAL_REGEN_ENABLED is True
    for knob in ("MEALFIT_SKELETON_DIET_SCRUB", "MEALFIT_DIET_DIRECTIVE_BLOCK",
                 "MEALFIT_SALT_LINE_CONDITION_GATE", "MEALFIT_DIET_CRITICAL_REGEN"):
        assert knob in _SRC, f"knob {knob} desapareció del source"


# ---------------------------------------------------------------------------
# 2. Scrub de pools: matcher reusado, sin falsos positivos, insertado ANTES del fallback
# ---------------------------------------------------------------------------
def test_pool_item_banned_reuses_scanner_semantics():
    assert go._diet_pool_item_banned("Camarones", "vegan")
    assert go._diet_pool_item_banned("Yogur griego", "vegan")
    assert go._diet_pool_item_banned("Atún enlatado", "vegetarian")
    assert go._diet_pool_item_banned("Pechuga de pollo", "vegetarian")
    # Sin falsos positivos: 'Fresas' NO matchea 'res' (word-boundary del scanner).
    assert not go._diet_pool_item_banned("Fresas", "vegan")
    # Análogos plant-based excusados por adyacencia (mismo contrato del guard).
    assert not go._diet_pool_item_banned("Carne de soya", "vegan")
    assert not go._diet_pool_item_banned("Leche de coco", "vegan")
    # Vegetariano permite huevo/lácteo; balanced no restringe nada.
    assert not go._diet_pool_item_banned("Huevos", "vegetarian")
    assert not go._diet_pool_item_banned("Pollo", "balanced")


def test_skeleton_scrub_runs_before_empty_pool_fallback():
    """El orden importa: un pool vaciado por dieta debe caer al fallback diet+allergy-aware
    del paso 3 (que puede inyectar leguminosa o dejarlo vacío a propósito)."""
    scrub_idx = _SRC.find("SKELETON DIET SCRUB")
    fallback_idx = _SRC.find("# 3. Fallback: si algún pool quedó vacío tras scrub")
    assert 0 < scrub_idx < fallback_idx, (
        "P1-DAYGEN-DIET-CONVERGE: el scrub de dieta debe ejecutarse ANTES del fallback de "
        "pools vacíos del skeleton (paso 3)."
    )
    assert "_diet_pool_item_banned" in _SRC[scrub_idx - 2000: fallback_idx], (
        "el scrub debe usar _diet_pool_item_banned (matcher del guard reusado, no una 4ª tabla)")


# ---------------------------------------------------------------------------
# 3. Directiva de dieta: contenido, cache-neutralidad y doble inyección
# ---------------------------------------------------------------------------
def test_diet_directive_content_per_canon():
    vegan = go._build_diet_directive_context({"dietType": "vegan"})
    assert "PRIORIDAD 1" in vegan and "camarones" in vegan and "lácteo" in vegan and "huevos" in vegan.lower()
    assert "tofu" not in vegan.lower(), "P3-TOFU-REMOVE: no sugerir tofu (no se vende)"
    veget = go._build_diet_directive_context({"dietType": "vegetarian"})
    assert "Huevos y lácteos SÍ" in veget
    # Acepta variantes legacy ES (canonicaliza vía SSOT, no otra tabla).
    assert go._build_diet_directive_context({"dietType": "vegana"}) == vegan
    # balanced → "" (byte-equivalencia: el prompt-cache de la mayoría no se invalida).
    assert go._build_diet_directive_context({"dietType": "balanced"}) == ""
    assert go._build_diet_directive_context({}) == ""


def test_diet_directive_injected_in_both_prompts():
    assert '"diet_directive_context":' in _SRC, (
        "P1-DAYGEN-DIET-CONVERGE: la clave diet_directive_context desapareció del ctx compartido.")
    hits = _SRC.count("ctx['diet_directive_context']")
    assert hits >= 2, (
        f"P1-DAYGEN-DIET-CONVERGE: la directiva debe inyectarse en los DOS prompts (skeleton + "
        f"day-gen); encontradas {hits} inyecciones. El bug original era que la dieta solo viajaba "
        "enterrada en el JSON de form_data."
    )


# ---------------------------------------------------------------------------
# 4. Gate de sal
# ---------------------------------------------------------------------------
def test_salt_restricted_profile_detection():
    assert go._salt_restricted_profile({"medicalConditions": ["Hipertensión"]})
    assert not go._salt_restricted_profile({"medicalConditions": ["Diabetes T2"]})
    assert not go._salt_restricted_profile({"medicalConditions": ["Ninguna"]})
    assert not go._salt_restricted_profile({})


def test_salt_splitter_gated_before_historic_append():
    m = re.search(r"if _re\.search\(r'\\bsal\\b', _sl\) and _re\.search\(r'\\bpimienta\\b', _sl\):(.*?)continue",
                  _SRC, re.DOTALL)
    assert m, "no encuentro el splitter P3-SALT-SEPARATE-LINE"
    block = m.group(1)
    assert "SALT_LINE_CONDITION_GATE" in block and "_salt_restricted_profile" in block, (
        "P1-DAYGEN-DIET-CONVERGE: el splitter de sal perdió el gate clínico — para HTA/renal "
        "volvería a fabricar 'Sal al gusto' en cada comida (razón #1 de rechazo del reviewer "
        "en perfiles HTA, benchmark issue #9)."
    )


# ---------------------------------------------------------------------------
# 5. Retry informado para críticos de dieta/alérgeno
# ---------------------------------------------------------------------------
def _critical_state(reasons, attempt=1):
    return {
        "review_passed": False,
        "_rejection_severity": "critical",
        "rejection_reasons": list(reasons),
        "attempt": attempt,
        "pipeline_start": time.time(),  # budget completo restante
        "plan_result": {"days": [{"meals": []}]},
    }


def test_diet_critical_gets_one_informed_retry(monkeypatch):
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)
    state = _critical_state(["DIETA INCOMPATIBLE (rechazo de restricción declarada): ..."])
    assert go.should_retry(state) == "retry", (
        "un crítico de DIETA en attempt 1 con budget debe obtener UN retry informado")
    # Alérgeno: misma clase regenerable-una-vez.
    state = _critical_state(["ALÉRGENO DETECTADO (rechazo de seguridad clínica): ..."])
    assert go.should_retry(state) == "retry"


def test_diet_critical_retry_is_single_and_guard_stays_terminal(monkeypatch):
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)
    # attempt 2 (el retry ya corrió y reincidió) → fallback terminal idéntico al de hoy.
    state = _critical_state(["DIETA INCOMPATIBLE: ..."], attempt=2)
    assert go.should_retry(state) == "end"
    # Un crítico NO-dieta/alérgeno (p.ej. schema) conserva el abort inmediato.
    state = _critical_state(["SCHEMA INVÁLIDO: el plan no cumple la estructura esperada"])
    assert go.should_retry(state) == "end"
    # Sin budget → no se intenta el retry.
    state = _critical_state(["DIETA INCOMPATIBLE: ..."])
    state["pipeline_start"] = time.time() - go.GLOBAL_PIPELINE_TIMEOUT_S  # budget agotado
    assert go.should_retry(state) == "end"
    # Kill switch.
    monkeypatch.setattr(go, "DIET_CRITICAL_REGEN_ENABLED", False)
    state = _critical_state(["DIETA INCOMPATIBLE: ..."])
    assert go.should_retry(state) == "end"
