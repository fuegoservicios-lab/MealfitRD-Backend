"""[P3-COLDSTART-E2E · 2026-05-29] Regresión integrada del escenario cold-start:
"primer plan / usuario sin (o con muy pocas) preferencias / cero comidas
consumidas → el sistema genera un plan VÁLIDO, SEGURO y NO-VACÍO".

Contexto (audit de autonomía + costo, 2026-05-29): el audit confirmó que la
generación cold-start funciona porque la arquitectura es ADITIVA — hay una base
determinística (metas del calculador + anti-mode-collapse round-robin sobre
catálogos dominicanos) y todas las señales de aprendizaje se SUMAN encima solo
`if data:`. El único gap real hallado en esa dimensión fue de TESTING: las
defensas individuales estaban cubiertas (zero-log, quality gate, fallback
alérgeno) pero NO había un test que ENSAMBLARA el escenario cold-start completo
de punta a punta. Este archivo cierra ese gap.

Diseño (mismo patrón que test_p0_orch_audit_impl.py):
  - PARSER-BASED por defecto: lee el source de prod y ancla cada defensa por su
    patrón clave, de modo que un rename de producción rompa el test ANTES de
    cambiar el comportamiento.
  - SECCIÓN FUNCIONAL guarded por import (skip si los módulos no importan en el
    entorno local sin deps): ejercita las funciones puras del path cold-start
    (builders de contexto con inputs vacíos, variedad determinística sin
    historial, fallback matemático sin preferencias).

Además ancla el cleanup P3-COLDSTART-E2E-JIT-DEAD (marcar como muerto el JIT
Rolling Window de proactive_agent.py — el disparo real del próximo chunk vive en
plan_chunk_queue + process_plan_chunk_queue).

Cobertura:
  COLD-1  build_unified_behavioral_profile degrada con gracia sin preferencias
          (emite "No hay suficientes preferencias..." + alergias SIEMPRE).
  COLD-2  Builders de aprendizaje retornan "" con data vacía (capa aditiva).
  COLD-3  get_deterministic_variety_prompt produce variedad sin historial / sin
          user_id (base anti-repetición independiente de datos).
  COLD-4  Fallback matemático genera un plan COMPLETO sin preferencias.
  COLD-5  Fallback respeta alergias SIEMPRE, incluso en cold-start (P0-ORCH-1).
  COLD-6  Gate _quality_data_sufficient (consumed>=3) impide persistir score
          basura cuando no hay consumo (no envenena ciclos futuros).
  JIT     Banner DEAD del JIT Rolling Window presente + callsite comentado.
  MARKER  _LAST_KNOWN_PFIX bumpeado.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PLAN_GEN = (_BACKEND / "prompts" / "plan_generator.py").read_text(encoding="utf-8")
_AI_HELPERS = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PROACTIVE = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


# ===========================================================================
# PARSER-BASED — defensas del path cold-start ancladas al source de prod
# ===========================================================================
def test_coldstart_1_unified_profile_graceful_no_prefs_anchor():
    # Degradación con gracia cuando no hay preferencias establecidas.
    assert "No hay suficientes preferencias establecidas aún" in _PLAN_GEN
    # Alergias SIEMPRE renderizadas ("Ninguna" si vacío) — prioridad absoluta.
    assert (
        'alergias_str = ", ".join(allergies_list) if allergies_list else "Ninguna"'
        in _PLAN_GEN
    )


def test_coldstart_2_learning_builders_empty_returns_blank_anchor():
    # build_chunk_lessons_context: data vacía → "".
    assert "if not chunk_lessons or not isinstance(chunk_lessons, dict):" in _PLAN_GEN
    # build_success_patterns_context: data vacía → "".
    assert (
        "if not successful_techniques and not abandoned_techniques and not cold_start_recs:"
        in _PLAN_GEN
    )
    # build_fatigue_context: data vacía → "".
    assert "if not fatigued_ingredients:" in _PLAN_GEN


def test_coldstart_3_variety_guard_clause_anchor():
    # Guard clause: si las restricciones vacían los catálogos → libertad al LLM
    # (retorna "" en vez de matriz vacía). Base anti-mode-collapse.
    assert "if not available_proteins or not available_carbs:" in _AI_HELPERS
    assert "Dejando libertad al LLM" in _AI_HELPERS


def test_coldstart_5_fallback_allergen_filter_anchor():
    # Fallback alérgeno-aware activo por default (P0-ORCH-1).
    assert (
        'FALLBACK_ALLERGEN_FILTER = _env_bool("MEALFIT_FALLBACK_ALLERGEN_FILTER", True)'
        in _GO_SRC
    )
    # Pools con plantilla neutral garantizada (cierra "siempre hay meal seguro").
    assert "_FALLBACK_MEAL_POOLS" in _GO_SRC
    assert "def _select_safe_fallback_meal(" in _GO_SRC


def test_coldstart_6_quality_gate_anchor():
    # Gate de datos suficientes: el score basura (~0.18) NO se persiste sin
    # consumo (>=3). Degradación con gracia, no envenena ciclos futuros.
    assert (
        "_quality_data_sufficient = bool(consumed_records) and len(consumed_records) >= 3"
        in _CRON
    )
    assert "if not _quality_data_sufficient:" in _CRON


def test_jit_rolling_window_marked_dead_anchor():
    # Cleanup: el JIT Rolling Window quedó marcado DEAD (no reactivar sin migrar).
    assert "P3-COLDSTART-E2E-JIT-DEAD" in _PROACTIVE
    assert "CÓDIGO MUERTO" in _PROACTIVE
    # El callsite real sigue comentado en run_proactive_checks.
    assert "# check_and_trigger_jit_rolling_windows()" in _PROACTIVE


def test_marker_present_in_app_py():
    assert '_LAST_KNOWN_PFIX = "P3-COLDSTART-E2E · 2026-05-29"' in _APP


# ===========================================================================
# SECCIÓN FUNCIONAL (skip si el módulo no importa en este entorno)
# ===========================================================================
try:
    from prompts import plan_generator as _PG  # noqa: E402
except Exception:  # pragma: no cover - entorno sin deps
    _PG = None

try:
    import ai_helpers as _AI  # noqa: E402
except Exception:  # pragma: no cover
    _AI = None

try:
    import graph_orchestrator as _GO  # noqa: E402
except Exception:  # pragma: no cover - entorno sin langgraph
    _GO = None

_needs_pg = pytest.mark.skipif(_PG is None, reason="prompts.plan_generator no importable")
_needs_ai = pytest.mark.skipif(_AI is None, reason="ai_helpers no importable")
_needs_go = pytest.mark.skipif(_GO is None, reason="graph_orchestrator no importable (deps ausentes)")


# --- COLD-1: perfil unificado degrada con gracia sin preferencias ---
@_needs_pg
def test_func_coldstart_1_profile_no_prefs_does_not_crash():
    # Todo vacío: user_facts="", sin fatiga, sin likes, sin flavors, sin
    # cold_start_recs, sin alergias.
    ctx = _PG.build_unified_behavioral_profile("", [], [], [], [], [])
    assert isinstance(ctx, str) and ctx.strip()
    # Alergias siempre presentes ("Ninguna").
    assert "Ninguna" in ctx
    # Mensaje explícito de cold-start (sin datos suficientes).
    assert "No hay suficientes preferencias establecidas aún" in ctx


@_needs_pg
def test_func_coldstart_1_profile_renders_allergies_when_present():
    ctx = _PG.build_unified_behavioral_profile("", [], [], [], [], ["Maní", "Mariscos"])
    assert "Maní" in ctx and "Mariscos" in ctx


# --- COLD-2: builders de aprendizaje vacíos → "" (capa aditiva) ---
@_needs_pg
def test_func_coldstart_2_learning_builders_blank_when_empty():
    assert _PG.build_chunk_lessons_context({}) == ""
    assert _PG.build_chunk_lessons_context(None) == ""
    assert _PG.build_success_patterns_context([], [], None) == ""
    assert _PG.build_fatigue_context([]) == ""


# --- COLD-3: variedad determinística sin historial ni user_id ---
@_needs_ai
def test_func_coldstart_3_variety_nonempty_without_history():
    # Guest sin form_data ni user_id: catálogos completos → matriz no vacía.
    prompt = _AI.get_deterministic_variety_prompt("", form_data=None, user_id=None)
    assert isinstance(prompt, str) and prompt.strip(), (
        "La variedad determinística debe producir guía no vacía en cold-start"
    )


@_needs_ai
def test_func_coldstart_3_variety_empty_form_data_safe():
    # form_data vacío {} → no crash, retorna str (path guest).
    prompt = _AI.get_deterministic_variety_prompt("", form_data={}, user_id=None)
    assert isinstance(prompt, str)


@_needs_ai
def test_func_coldstart_3_variety_heavy_restrictions_degrades_safely():
    # Dieta vegana + muchas alergias: o produce variedad filtrada o "" (libertad
    # al LLM). Ambos son seguros — lo crítico es que NO crashea.
    form_data = {
        "diet": "vegana",
        "allergies": ["soya", "gluten", "maní", "nueces", "lácteos"],
        "dislikes": ["lentejas", "garbanzos", "frijoles"],
    }
    prompt = _AI.get_deterministic_variety_prompt("", form_data=form_data, user_id=None)
    assert isinstance(prompt, str)  # nunca None / nunca excepción


# --- COLD-4/5: fallback matemático genera plan completo + alérgeno-seguro ---
@_needs_go
def test_func_coldstart_4_fallback_day_complete_no_prefs():
    day = _GO._build_fallback_day({"target_calories": 2000, "macros": {}}, 1, frozenset())
    meals = day["meals"]
    assert len(meals) == 3
    for m in meals:
        assert m["name"]
        assert m["ingredients"]
        assert m["cals"] > 0


@_needs_go
def test_func_coldstart_4_extreme_fallback_plan_complete():
    # Sin preferencias (form_data vacío → restricted vacío): plan completo de N días.
    restricted = _GO._fallback_restricted_tokens({})
    assert restricted == frozenset()
    plan = _GO._get_extreme_fallback_plan(
        {"target_calories": 2000, "macros": {}}, "mantener", num_days=3,
        restricted_tokens=restricted,
    )
    assert isinstance(plan, dict)
    days = plan.get("days")
    assert isinstance(days, list) and len(days) == 3
    for d in days:
        assert len(d["meals"]) == 3
    assert plan.get("_is_fallback") is True


@_needs_go
def test_func_coldstart_5_fallback_respects_allergies_in_coldstart():
    # Cold-start + múltiples alergias declaradas: ninguno de esos tokens aparece.
    restricted = _GO._detect_restricted_tokens(
        {"allergies": ["huevo", "pescado", "pollo", "leche", "maní"]}
    )
    assert restricted  # detecta al menos uno
    plan = _GO._get_extreme_fallback_plan(
        {"target_calories": 2000, "macros": {}}, "mantener", num_days=3,
        restricted_tokens=restricted,
    )
    blob = str(plan).lower()
    for token in ("huevo", "pescado", "pollo", "leche", "mani"):
        assert token not in blob, f"el fallback cold-start no debe contener {token}: {blob}"
    assert plan.get("_allergen_filtered") is True


@_needs_go
def test_func_coldstart_5_no_prefs_detects_no_restrictions():
    # Usuario sin ninguna preferencia/alergia declarada → cero tokens restringidos.
    assert _GO._detect_restricted_tokens({}) == frozenset()
    assert _GO._detect_restricted_tokens({"allergies": []}) == frozenset()
