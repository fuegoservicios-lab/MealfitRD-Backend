"""[P0-ORCH-AUDIT-IMPL · 2026-05-28] Regresión del bundle de 19 fixes del audit
production-readiness de `graph_orchestrator.py` (1 P0 + 4 P1 + 14 P2).

Diseño: PARSER-BASED por defecto (lee el source de prod y ancla cada fix por su
tooltip-anchor / patrón clave), de modo que un rename de producción rompa el test
ANTES de cambiar el comportamiento. Adicionalmente, una sección FUNCIONAL guarded
por import (skip si `graph_orchestrator` no importa en el entorno local sin
langgraph) ejercita las funciones puras nuevas (allergen filter, detector de
monotonía, risk-tier, idempotencia).

Cobertura de los 19:
  P0-ORCH-1  allergen-aware fallback
  P1-ORCH-1  _record_cb_failure_unless_transient en todos los nodos LLM
  P1-ORCH-2  latch spend-cap + detección symptom-aware
  P1-ORCH-3  self_critique except → degrade no-fatal (sin raise)
  P1-ORCH-4  surgical_marker_regen fallback a state['plan_skeleton']
  P2-ORCH-1  cache stagger default + clamp
  P2-ORCH-2  semantic-cache threshold knob + telemetría
  P2-ORCH-3  día fallido → _build_fallback_day + marker
  P2-ORCH-4  predicado tenacity excluye spend-cap
  P2-ORCH-5  floor-days primero en el slice
  P2-ORCH-6  detector cross-day de proteína pesada (gap del pescado)
  P2-ORCH-7  reviewer/fact-checker risk-tiered
  P2-ORCH-8  LLM_MAX_PER_USER default >= PLAN_CHUNK_SIZE + warning
  P2-ORCH-9  'stable' en _FORM_HINT_ENUMS
  P2-ORCH-10 generation_status='complete' en insert no-chunking (services.py)
  P2-ORCH-11 cota del busy-poll local
  P2-ORCH-12 recursion_limit explícito en astream
  P2-ORCH-13 _DB_EXECUTOR max_workers knob
  P2-ORCH-14 idempotencia del emit de usage-events
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_G = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_S = (_BACKEND / "services.py").read_text(encoding="utf-8")
_DAYGEN_PROMPT = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
_SCHEMAS = (_BACKEND / "schemas.py").read_text(encoding="utf-8")


# ===========================================================================
# P0-ORCH-1 — allergen-aware fallback
# ===========================================================================
def test_p0_orch_1_helpers_present():
    assert "def _detect_restricted_tokens(form_data: dict) -> frozenset:" in _G
    assert "def _fallback_restricted_tokens(form_data: dict) -> frozenset:" in _G
    assert "def _select_safe_fallback_meal(" in _G
    assert "_FALLBACK_MEAL_POOLS" in _G
    assert 'FALLBACK_ALLERGEN_FILTER = _env_bool("MEALFIT_FALLBACK_ALLERGEN_FILTER", True)' in _G


def test_p0_orch_1_fallback_threads_restricted_tokens():
    # Builders aceptan restricted_tokens.
    assert "def _build_fallback_day(nutr: dict, day_number: int," in _G
    assert "restricted_tokens: frozenset = frozenset()) -> dict:" in _G
    assert "restricted_tokens: frozenset = frozenset()) -> dict:" in _G  # _get_extreme_fallback_plan
    # La rama CRÍTICA (rechazo médico) pasa los tokens del usuario.
    assert "restricted_tokens=_fallback_restricted_tokens(actual_form_data)" in _G
    # _repair_partial_plan también propaga.
    assert "restricted_tokens: frozenset = frozenset()) -> bool:" in _G


def test_p0_orch_1_historic_menu_preserved_as_pool_head():
    # pool[0] de cada slot = menú histórico (restricted vacío → comportamiento idéntico).
    assert '"Huevos y Avena"' in _G
    assert '"Pollo y Arroz"' in _G
    assert '"Pescado y Batata"' in _G


# ===========================================================================
# P1-ORCH-1 — CB transient-exclusion helper en TODOS los nodos LLM
# ===========================================================================
def test_p1_orch_1_helper_defined():
    assert "async def _record_cb_failure_unless_transient(cb, exc) -> None:" in _G
    # [P1-ORCH-1-DBPOOL · 2026-05-28] errores de pool de DB ("couldn't get a
    # connection") NO deben abrir el CB del modelo (salud DB ≠ salud modelo).
    assert "if _is_pool_timeout_error(exc):" in _G


def test_p1_orch_1_applied_to_all_nodes():
    # Pre-fix solo planner+day-gen excluían transient (2 callsites de
    # _is_transient_upstream_error). El helper ahora cubre los ~11 nodos.
    n_helper = _G.count("_record_cb_failure_unless_transient(")
    # 1 def + >= 11 callsites (planner, day, pro-critique, compressor, judge,
    # evaluator, 2 correctores, fact-checker, reviewer, reflector).
    assert n_helper >= 12, f"esperaba >=12 ocurrencias del helper, hubo {n_helper}"


def test_p1_orch_1_failclosed_tool_timeout_left_alone():
    # El timeout de la TOOL clínica (fail-closed) NO debe usar el helper.
    assert "await _fact_checker_cb.arecord_failure()" in _G  # sigue en el branch tool-timeout


# ===========================================================================
# P1-ORCH-2 — latch spend-cap + detección symptom-aware
# ===========================================================================
def test_p1_orch_2_latch_and_helpers():
    assert "PLAN_SPEND_CAP_BACKOFF_S = _env_int(" in _G
    assert "def _is_plan_spend_cap_error(exc: BaseException) -> bool:" in _G
    assert "def _note_plan_spend_cap() -> None:" in _G
    assert "def _plan_spend_cap_active() -> bool:" in _G


def test_p1_orch_2_global_handler_consults_latch():
    assert "_spend_cap_hit = _is_plan_spend_cap_error(e) or _plan_spend_cap_active()" in _G


# ===========================================================================
# P2-ORCH-4 — predicado tenacity excluye spend-cap (3 decoradores)
# ===========================================================================
def test_p2_orch_4_retry_predicate():
    assert "retry_if_exception" in _G.split("\n")[13]  # import en la línea del tenacity import
    assert _G.count("retry=retry_if_exception(lambda e: not _is_plan_spend_cap_error(e))") >= 3


# ===========================================================================
# P1-ORCH-3 — self_critique except → degrade no-fatal
# ===========================================================================
def test_p1_orch_3_no_raise_in_self_critique():
    assert "raise e  # Bubble up to trigger graph fallback" not in _G
    assert "[P1-ORCH-3" in _G
    # Tras el except del self_critique se conserva el plan ya generado.
    assert _G.count('return {"plan_result": partial}') >= 2  # success path + degrade path


# ===========================================================================
# P1-ORCH-4 — surgical_marker_regen fallback a state['plan_skeleton']
# ===========================================================================
def test_p1_orch_4_skeleton_fallback():
    assert 'skeleton = plan_result.get("_skeleton") or state.get("plan_skeleton") or {}' in _G
    assert "[P1-ORCH-4" in _G


# ===========================================================================
# P2-ORCH-1 — cache stagger default + clamp
# ===========================================================================
def test_p2_orch_1_stagger_default_and_clamp():
    assert '"MEALFIT_DAY_GEN_CACHE_STAGGER_MS",     1500' in _G
    assert "validator=lambda v: 0 <= v <= 10000" in _G


# ===========================================================================
# P1-COST-THINKING-CAP — cap del thinking budget en day-gen + correctores
# ===========================================================================
def test_p1_cost_thinking_cap():
    assert "DAYGEN_THINKING_BUDGET" in _G
    assert "def _thinking_budget_kwargs(model_name: str) -> dict:" in _G
    assert '"thinking_budget": DAYGEN_THINKING_BUDGET' in _G
    # flash-lite excluido (no soporta thinking config).
    assert '"lite" in model_name.lower()' in _G
    # Aplicado al day-gen (80% del costo) + los 3 correctores (1 def + >=4 callsites).
    assert _G.count("_thinking_budget_kwargs(") >= 5
    assert "**_thinking_budget_kwargs(day_model)" in _G


# ===========================================================================
# P2-ORCH-2 — semantic-cache threshold knob + telemetría
# ===========================================================================
def test_p2_orch_2_threshold_knob_and_stats():
    assert "SEMANTIC_CACHE_COSINE_THRESHOLD = _env_float(" in _G
    assert "def get_semantic_cache_stats_snapshot() -> dict:" in _G
    # El callsite usa el knob, no el 0.98 hardcoded.
    assert "search_similar_plan, profile_embedding, 0.98, 10" not in _G
    assert "SEMANTIC_CACHE_COSINE_THRESHOLD, 10)" in _G


# ===========================================================================
# P2-ORCH-3 — día fallido → math fallback + marker (no clon verbatim)
# ===========================================================================
def test_p2_orch_3_no_verbatim_clone():
    assert "valid_day_template = generated_days[0]" not in _G
    assert "fb_day = _build_fallback_day(nutrition, f_day, _restricted)" in _G
    assert 'fb_day["_day_fallback"] = True' in _G


# ===========================================================================
# P2-ORCH-5 — floor-days primero en el slice
# ===========================================================================
def test_p2_orch_5_floor_first():
    assert "mentioned = list(dict.fromkeys(deterministic_days + mentioned))" in _G
    # El patrón viejo (floor al final) ya no está en producción.
    assert "mentioned = list(dict.fromkeys(mentioned + missing))" not in _G


# ===========================================================================
# P2-ORCH-6 — detector cross-day de proteína pesada (gap del pescado)
# ===========================================================================
def test_p2_orch_6_cross_day_detector():
    assert "def _count_cross_day_heavy_protein_repetition(" in _G
    assert "heavy_protein_monotony = _count_cross_day_heavy_protein_repetition(days)" in _G
    assert "not heavy_protein_monotony" in _G


# ===========================================================================
# P2-ORCH-7 — reviewer/fact-checker risk-tiered
# ===========================================================================
def test_p2_orch_7_risk_tier():
    assert "def _profile_has_medical_risk(form_data) -> bool:" in _G
    assert '_REVIEWER_RISK_TIER_DEFAULT = "gemini-3.5-flash"' in _G
    assert "def _reviewer_model_name(form_data=None) -> str:" in _G
    assert "def _fact_checker_model_name(form_data=None) -> str:" in _G
    assert "_reviewer_model = _reviewer_model_name(form_data)" in _G
    assert "_fact_checker_model = _fact_checker_model_name(form_data)" in _G


# ===========================================================================
# P2-ORCH-8 — per-user semaphore default + warning
# ===========================================================================
def test_p2_orch_8_per_user_default_and_warning():
    assert '"MEALFIT_LLM_MAX_PER_USER",            3)' in _G
    assert "if LLM_PER_USER_ENABLED and LLM_MAX_PER_USER < PLAN_CHUNK_SIZE:" in _G


# ===========================================================================
# P2-ORCH-9 — 'stable' en _FORM_HINT_ENUMS
# ===========================================================================
def test_p2_orch_9_stable_enum():
    assert '"temporary_dip", "drastic_change", "improving", "stable",' in _G


# ===========================================================================
# P2-ORCH-10 — generation_status='complete' en insert no-chunking (services.py)
# ===========================================================================
def test_p2_orch_10_generation_status_stamp():
    assert '"generation_status": plan_data.get("generation_status", "complete")' in _S
    assert "P2-ORCH-10" in _S


# ===========================================================================
# P2-ORCH-11 — cota del busy-poll local
# ===========================================================================
def test_p2_orch_11_local_wait_bound():
    assert 'LLM_LOCAL_MAX_WAIT_S        = _env_int  ("MEALFIT_LLM_LOCAL_MAX_WAIT_S",        120,' in _G
    assert "_inc_budget_stat(\"local_wait_timeout\")" in _G
    assert "_inc_budget_stat(\"local_wait_timeout_user\")" in _G
    # La cota se aplica en ambos busy-polls.
    assert _G.count("_deadline = time.monotonic() + LLM_LOCAL_MAX_WAIT_S") >= 2


# ===========================================================================
# P2-ORCH-12 — recursion_limit explícito
# ===========================================================================
def test_p2_orch_12_recursion_limit():
    assert 'GRAPH_RECURSION_LIMIT       = _env_int  ("MEALFIT_GRAPH_RECURSION_LIMIT",       50,' in _G
    assert 'config={"recursion_limit": GRAPH_RECURSION_LIMIT}' in _G


# ===========================================================================
# P2-ORCH-13 — _DB_EXECUTOR max_workers knob
# ===========================================================================
def test_p2_orch_13_db_executor_knob():
    assert 'DB_EXECUTOR_MAX_WORKERS     = max(1, min(64, _env_int("MEALFIT_DB_EXECUTOR_MAX_WORKERS", 8)))' in _G
    assert "max_workers=DB_EXECUTOR_MAX_WORKERS" in _G
    assert "max_workers=8, thread_name_prefix=\"db-io\"" not in _G


# ===========================================================================
# P2-ORCH-14 — idempotencia del emit de usage-events
# ===========================================================================
def test_p2_orch_14_usage_idempotency():
    assert "_USAGE_EMIT_SEEN" in _G
    assert "def _usage_was_emitted(result) -> bool:" in _G
    assert "def _mark_usage_emitted(result) -> None:" in _G
    assert "if result is not None and _usage_was_emitted(result):" in _G


# ===========================================================================
# COST LEVERS (deep audit) — Z1 / Z2 / Z3 / L1
# ===========================================================================
def test_z1_prompt_no_longer_orders_nutrition_tool():
    # [Z1-PROMPT-CONTRADICTION] rule #13 ya NO ordena llamar la tool.
    assert "[Z1-PROMPT-CONTRADICTION]" in _DAYGEN_PROMPT
    assert "MÁXIMO 3 llamadas" not in _DAYGEN_PROMPT, "rule #13 todavía ordena llamar la tool"
    assert "NUNCA invoques `consultar_nutricion`" in _DAYGEN_PROMPT


def test_z2_z3_schema_fields_optional():
    # Campos muertos ahora Optional (el LLM deja de emitirlos de forma fiable).
    assert "time: Optional[str] = Field(default=None" in _SCHEMAS
    assert "macros: Optional[List[str]] = Field(default=None" in _SCHEMAS
    # Backfill robusto contra None (Optional emite None, no ausencia).
    assert 'if not m.get("time"): m["time"] = "Flexible"' in _G
    assert 'if not m.get("macros"): m["macros"] = ["Plan Matemático"]' in _G


def test_l1_bind_nutrition_tool_knob():
    # [L1-UNBIND-NUTRITION-TOOL] bind_tools gateado por knob (default True).
    assert 'DAYGEN_BIND_NUTRITION_TOOL  = _env_bool ("MEALFIT_DAYGEN_BIND_NUTRITION_TOOL",   True)' in _G
    assert "if DAYGEN_BIND_NUTRITION_TOOL:" in _G
    assert "day_llm_with_tools = day_llm.bind_tools(NUTRITION_TOOLS)" in _G
    assert "day_llm_with_tools = day_llm\n" in _G  # rama unbound


def test_z2_z3_meal_model_optional_standalone():
    """[Z2/Z3] MealModel construye SIN time/macros (schemas importa sin langgraph)."""
    import importlib
    schemas = importlib.import_module("schemas")
    m = schemas.MealModel(meal="Desayuno", name="Test", desc="d", prep_time="15 min",
                          cals=400, ingredients=["x"], recipe=["Mise en place: y"])
    dumped = m.model_dump()
    assert dumped["time"] is None and dumped["macros"] is None  # Optional default
    day = schemas.SingleDayPlanModel(day=1, meals=[m])
    assert day.model_dump()["meals"][0]["name"] == "Test"


# ===========================================================================
# Marker anchor
# ===========================================================================
def test_marker_present_in_app_py():
    # [P1-PROD-AUDIT-2 · 2026-05-30] Relajado de hardcode `P0-ORCH-AUDIT-IMPL`
    # (este bundle, 2026-05-28) a un floor de fecha: `_LAST_KNOWN_PFIX` avanza con
    # cada bundle posterior (ya pasó por P1-CHAT-GUEST-IDOR → P1-PROD-AUDIT-2).
    # Anclar el string exacto convertía este test en falso-rojo permanente tras
    # cualquier bump. La frescura la valida `test_p3_1_last_known_pfix_freshness.py`.
    import re
    app_py = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_py)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    md = re.search(r"(\d{4})-(\d{2})-(\d{2})", m.group(1))
    assert md and (int(md.group(1)), int(md.group(2)), int(md.group(3))) >= (2026, 5, 28), (
        f"_LAST_KNOWN_PFIX {m.group(1)!r} es anterior al bundle P0-ORCH-AUDIT-IMPL (2026-05-28)."
    )


# ===========================================================================
# SECCIÓN FUNCIONAL (skip si el módulo no importa en este entorno)
# ===========================================================================
try:
    import graph_orchestrator as _GO  # noqa: E402
except Exception:  # pragma: no cover - entorno local sin langgraph/deps
    _GO = None

_needs_module = pytest.mark.skipif(_GO is None, reason="graph_orchestrator no importable (deps ausentes)")


@_needs_module
def test_func_p0_orch_1_detects_allergens():
    assert "egg" in _GO._detect_restricted_tokens({"allergies": ["huevo"]})
    assert "fish" in _GO._detect_restricted_tokens({"medicalConditions": "alergia al pescado"})
    assert _GO._detect_restricted_tokens({}) == frozenset()
    assert _GO._detect_restricted_tokens({"allergies": ["ninguna"]}) == frozenset()


@_needs_module
def test_func_p0_orch_1_empty_preserves_historic_menu():
    day = _GO._build_fallback_day({"target_calories": 2000, "macros": {}}, 1, frozenset())
    names = [m["name"] for m in day["meals"]]
    assert names == ["Huevos y Avena", "Pollo y Arroz", "Pescado y Batata"]


@_needs_module
def test_func_p0_orch_1_egg_allergy_removes_egg():
    restricted = _GO._detect_restricted_tokens({"allergies": ["huevo"]})
    day = _GO._build_fallback_day({"target_calories": 2000, "macros": {}}, 1, restricted)
    blob = str(day).lower()
    assert "huevo" not in blob, f"el fallback no debe contener huevo: {blob}"


@_needs_module
def test_func_p0_orch_1_multi_allergy_safe():
    # Alergia a huevo + pescado + pollo → ninguno de esos tokens en el plan.
    restricted = _GO._detect_restricted_tokens(
        {"allergies": ["huevo", "pescado", "pollo"]}
    )
    plan = _GO._get_extreme_fallback_plan(
        {"target_calories": 2000, "macros": {}}, "mantener", num_days=3,
        restricted_tokens=restricted,
    )
    blob = str(plan).lower()
    for token in ("huevo", "pescado", "pollo"):
        assert token not in blob, f"{token} no debe aparecer: {blob}"
    assert plan.get("_allergen_filtered") is True


@_needs_module
def test_func_p2_orch_6_fish_monotony_detected():
    days = [
        {"meals": [{"name": "Salmón a la plancha", "ingredients": ["filete de pescado"]}]},
        {"meals": [{"name": "Pescado al horno", "ingredients": ["tilapia"]}]},
        {"meals": [{"name": "Cena de pescado", "ingredients": ["salmon"]}]},
    ]
    rep = _GO._count_cross_day_heavy_protein_repetition(days)
    assert rep.get("fish", 0) >= 3, f"monotonía de pescado debe detectarse: {rep}"


@_needs_module
def test_func_p2_orch_7_risk_tier_routing():
    assert _GO._profile_has_medical_risk({"allergies": ["maní"]}) is True
    assert _GO._profile_has_medical_risk({"medicalConditions": ["diabetes"]}) is True
    assert _GO._profile_has_medical_risk({}) is False
    assert _GO._profile_has_medical_risk({"allergies": []}) is False
    # Sin override de env: perfil con riesgo → risk-tier; sin riesgo → flash-lite.
    import os
    if not os.environ.get("MEALFIT_REVIEWER_MODEL"):
        assert _GO._reviewer_model_name({"allergies": ["maní"]}) == _GO._REVIEWER_RISK_TIER_DEFAULT
        assert _GO._reviewer_model_name({}) == _GO._FLASH_LITE_DEFAULT


@_needs_module
def test_func_p2_orch_2_stats_snapshot_shape():
    snap = _GO.get_semantic_cache_stats_snapshot()
    for k in ("hit", "miss", "anti_repetition_reject", "total", "hit_rate", "threshold"):
        assert k in snap


@_needs_module
def test_func_p1_cost_thinking_cap_gating():
    """[P1-COST-THINKING-CAP] El kwarg solo se aplica a modelos thinking-capable
    (gemini-3.5-flash); flash-lite y vacío → sin kwarg."""
    b = _GO.DAYGEN_THINKING_BUDGET
    kw = _GO._thinking_budget_kwargs("gemini-3.5-flash")
    if b >= 0:
        assert kw == {"thinking_budget": b}, f"esperaba budget {b}, vino {kw}"
    else:
        assert kw == {}, "knob -1 (dynamic) → sin kwarg"
    assert _GO._thinking_budget_kwargs("gemini-3.1-flash-lite") == {}, "flash-lite no soporta thinking"
    assert _GO._thinking_budget_kwargs("") == {}
    # La librería soporta el campo (verificado en vivo).
    from langchain_google_genai import ChatGoogleGenerativeAI as _C
    assert "thinking_budget" in getattr(_C, "model_fields", {})


@_needs_module
def test_func_p1_orch_1_dbpool_error_not_recorded():
    """[P1-ORCH-1-DBPOOL] Un error de pool de DB NO debe abrir el CB del modelo;
    un error real del modelo SÍ."""
    import asyncio

    class _CB:
        def __init__(self):
            self.calls = 0
        async def arecord_failure(self):
            self.calls += 1

    cb = _CB()
    # "couldn't get a connection" → salud de DB, no del modelo → NO se registra.
    asyncio.run(_GO._record_cb_failure_unless_transient(
        cb, Exception("couldn't get a connection after 12.00 sec")))
    assert cb.calls == 0, "pool-timeout no debe abrir el CB del modelo"
    # 5xx transitorio → tampoco se registra.
    asyncio.run(_GO._record_cb_failure_unless_transient(cb, Exception("503 Service Unavailable")))
    assert cb.calls == 0, "5xx transitorio no debe abrir el CB del modelo"
    # Error real del modelo → SÍ se registra.
    asyncio.run(_GO._record_cb_failure_unless_transient(cb, Exception("invalid response schema")))
    assert cb.calls == 1, "un error real del modelo SÍ debe registrarse"
