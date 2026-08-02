"""[P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] "Colocación consciente de sodio" en el swap/regen-day.

Caso real que motiva el fix (owner, 2026-08-01): regenerar la cena de un día que YA llevaba
ricotta armó "Berenjenas con Camarones" → día en 2140/2000mg (7% sobre techo), banner "Menor"
descubierto DESPUÉS del hecho. Decisión del owner: el sistema que prescribe la lista/nevera debe
ELEGIR mejor el pareo del día — sin gates duros (evidencia medida: gatear el queso same-day
quemó 3 reintentos/plan). Diseño: REPARTIR, no PROHIBIR.

  1. ANTES de generar: presupuesto de sodio restante del día inyectado como directiva INFORMATIVA
     (nunca un hard-gate) en el prompt del swap.
  2. DESPUÉS de cada candidato: si candidato+resto excede el techo, UN reintento con directiva
     explícita — comparte el presupuesto de 3 intentos de tenacity con TODOS los demás guards de
     `swap_meal` (pantry/coherencia/macros/clínico/slot/etc.), no tiene un budget aparte. Si el
     reintento tampoco baja el sodio → se ACEPTA el candidato (jamás falla el swap por sodio).
  3. El techo es la MISMA fuente que el banner/panel (`micronutrients.dri_targets`), NO un literal
     nuevo. El estimador por-comida (`_meal_sodium_mg`/`_line_sodium_mg`) es el mismo primitivo que
     ya usaba `_day_sodium_autofix` (P1-SODIUM-DAY-AUTOFIX) — extraído a nivel de módulo, SSOT.
  4. regen-day (routers/plans.py) computa el resto del día EN VIVO (ve los platos ya regenerados en
     la request actual, que la BD todavía no tiene) y lo pasa via `sodium_resto_override_mg` — el
     mismo patrón de precedencia que `pantry_override` (P2-REGEN-DAY-PANTRY-OVERRIDE). El swap
     standalone (`/swap-meal`) deriva su propio resto desde el plan en BD (fallback fiel, sin loop
     concurrente que lo deje stale).

Test organizados:
  Section A — parser/estructural (wiring, orden, SSOT, knob, presupuesto compartido).
  Section B — funcional real (`_meal_sodium_mg` / `_sodium_day_ceiling_mg_for_banner`, caso real
              simulado: ¿el presupuesto informado habría evitado camarones?).
  Section C — funcional end-to-end con mocks sobre `swap_meal` (retry único, aceptación tras 2º
              exceso, knob OFF = cero cambio de flujo).
  Section D — regen-day pasa el override en vivo (parser).
  Section E — marker de P-fix.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]


def _agent_source() -> str:
    return (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _go_source() -> str:
    return (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _plans_source() -> str:
    return (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _swap_meal_block() -> str:
    src = _agent_source()
    m = re.search(r"def swap_meal\(form_data:.*?(?=^def |^# =+\n# ORQ)", src, re.DOTALL | re.MULTILINE)
    assert m, "No se localizó el cuerpo de swap_meal."
    return m.group(0)


# =====================================================================
# Section A — Parser/estructural
# =====================================================================

def test_knob_exists_default_true():
    src = _swap_meal_block()
    assert 'os.environ.get(\n        "MEALFIT_SODIUM_AWARE_SWAP", "true"\n    )' in src or (
        'MEALFIT_SODIUM_AWARE_SWAP' in src and '"true"' in src
    ), "el knob MEALFIT_SODIUM_AWARE_SWAP debe existir con default 'true'"


def test_directive_injected_before_prompt_format_when_ceiling_exists():
    """La directiva informativa debe inyectarse a `context_extras` ANTES de construir
    `prompt_text` (para que el primer intento YA la incluya, no solo los reintentos)."""
    src = _swap_meal_block()
    i_budget = src.find("PRESUPUESTO DE SODIO")
    i_format = src.find("prompt_text = SWAP_MEAL_PROMPT_TEMPLATE.format(")
    assert i_budget > 0, "directiva de presupuesto de sodio ausente"
    assert i_format > 0, "construcción de prompt_text ausente"
    assert i_budget < i_format, "la directiva debe inyectarse ANTES de formatear prompt_text"


def test_directive_is_informative_not_a_hard_gate():
    """La directiva debe leerse como preferencia ('prefiere'/'evita'), nunca como prohibición dura
    tipo 'PROHIBIDO'/'TERMINANTEMENTE' — eso es lo que el owner pidió NO repetir (el gate duro del
    queso same-day quemó 3 reintentos/plan)."""
    src = _swap_meal_block()
    i = src.find("PRESUPUESTO DE SODIO")
    block = src[i: i + 700]
    assert "prefiere" in block.lower() or "preferencia" in block.lower()
    assert "no es una prohibición" in block.lower() or "preferencia de colocación" in block.lower()


def test_postgen_guard_wired_after_other_guards_before_return():
    """El guard post-generación debe correr DESPUÉS de todos los demás guards de calidad (P1-SWAP-
    BASE-REPEAT-GATE, P1-SWAP-SAMEDAY-PROTEIN-GATE, pantry, recipe-coherence, macros, clínico, slot,
    appetibility, dish-quality, raw-staple, sameday-variety) y justo ANTES del `return res` final —
    evalúa el candidato YA reparado por todos los closers/solvers deterministas."""
    src = _swap_meal_block()
    anchors_before = [
        "P1-SWAP-BASE-REPEAT-GATE", "P1-SWAP-SAMEDAY-PROTEIN-GATE",
        "P1-SWAP-RECIPE-COHERENCE", "P1-SWAP-MACROS · 2026-05-22] Validación post-gen",
        "P0-UPDATE-CLINICAL-GUARD", "P1-SLOT-APPROPRIATENESS",
        "P1-UPDATE-APPETIBILITY", "P2-UPDATE-DISHQUALITY-PRESSURE",
        "P2-AUDIT-V5-BATCH-RAW-STAPLE-SWAP", "P2-UPDATE-SAMEDAY-VARIETY",
    ]
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    assert i_sodium > 0, "guard post-generación de sodio ausente"
    for anchor in anchors_before:
        i_anchor = src.find(anchor)
        assert 0 < i_anchor < i_sodium, f"el guard de sodio debe correr DESPUÉS de {anchor!r}"
    i_return = src.find("return res", i_sodium)
    assert i_return > i_sodium, "el guard de sodio debe preceder al `return res` final"
    # nada de otro guard debe colarse ENTRE el guard de sodio y el return
    between = src[i_sodium:i_return]
    assert "raise ValueError" in between  # el propio guard de sodio


def test_shared_retry_budget_not_a_separate_one():
    """El reintento de sodio NO tiene su propio `@retry`/`stop_after_attempt` — comparte el único
    decorator de `invoke_with_retry` (3 intentos) con TODOS los demás guards. Documentado inline."""
    src = _swap_meal_block()
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    assert i_sodium > 0
    # Solo debe existir UN @retry en toda la función (el de invoke_with_retry) — el bloque de
    # sodio no introduce un segundo decorator/stop_after_attempt propio. Se cuenta el CALLSITE real
    # (`stop_after_attempt(`), no la palabra suelta — el fix-review la menciona en un comentario
    # explicativo (SSOT de `_SWAP_MAX_LLM_ATTEMPTS`), lo cual es legítimo y no debe contarse como
    # un 2º decorator.
    assert src.count("stop_after_attempt(") == 1, (
        "debe existir un ÚNICO callsite `stop_after_attempt(...)` (el de invoke_with_retry) — el "
        "guard de sodio "
        "comparte ese presupuesto, no inventa uno propio"
    )
    block = src[i_sodium: i_sodium + 2600]
    assert "comparte el presupuesto de" in block and "NO tiene su propio @retry" in block, (
        "debe documentarse explícitamente que el reintento de sodio comparte el budget compartido"
    )


def test_single_retry_marker_pattern_mirrors_siblings():
    """Mismo patrón marker-based que P1-SWAP-BASE-REPEAT-GATE / P2-UPDATE-SAMEDAY-VARIETY: un
    marker string en el prompt evita un 2º retry por la MISMA causa → se acepta tras 1 reintento."""
    src = _swap_meal_block()
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    block = src[i_sodium: i_sodium + 9200]
    assert '_SOD_MARKER = "🧂 RETRY PRESUPUESTO DE SODIO"' in block
    assert "_SOD_MARKER not in str(_current_prompt[0])" in block
    assert "raise ValueError" in block
    # Cuando el marker YA está presente, NO debe re-raise (se acepta) — debe existir la rama de
    # aceptación con un log de "aceptado" en vez de un segundo `raise`.
    i_marker_check = block.find("_SOD_MARKER not in str(_current_prompt[0])")
    after_if = block[i_marker_check:]
    i_else_ish = after_if.find("aceptado")
    assert i_else_ish > 0, "debe existir la rama de aceptación tras el reintento agotado"


def test_never_fails_the_swap_for_sodium():
    """El guard jamás debe convertirse en un `SWAP_*_EXHAUSTED`/422 dedicado a sodio — solo emite
    ValueError (que tenacity reintenta) y, agotado el único reintento, ACEPTA. No debe existir un
    'raise' fuera del `if _day_total_sod > _sodium_ceiling_mg:` en el bloque."""
    src = _swap_meal_block()
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    i_return = src.find("return res", i_sodium)
    block = src[i_sodium:i_return]
    assert block.count("raise ValueError") == 1, (
        "debe existir EXACTAMENTE un raise (el del primer exceso) — el camino de aceptación no "
        "debe volver a levantar"
    )


def test_ceiling_reuses_banner_source_not_a_new_literal():
    """El techo debe venir de `_sodium_day_ceiling_mg_for_banner` (misma fuente que el banner/panel
    vía `micronutrients.dri_targets`), NO de `SODIUM_DAY_CEILING_MG` (el knob del autofix, un SSOT
    hermano pero DISTINTO) ni de un literal 2000 hardcodeado en agent.py."""
    src = _swap_meal_block()
    assert "_sodium_day_ceiling_mg_for_banner" in src
    assert "SODIUM_DAY_CEILING_MG" not in src, (
        "swap_meal NO debe importar el knob del autofix determinista — son SSOT hermanos que "
        "pueden driftear; el swap debe ver el MISMO número que el banner del usuario"
    )


def test_estimator_reused_from_day_autofix_ssot():
    """`_meal_sodium_mg`/`_line_sodium_mg` deben ser las funciones YA usadas por
    `_day_sodium_autofix` (P1-SODIUM-DAY-AUTOFIX) — extraídas a nivel de módulo, no reinventadas."""
    swap_src = _swap_meal_block()
    assert "_meal_sodium_mg" in swap_src, "swap_meal debe reusar el estimador SSOT"

    go_src = _go_source()
    # Las closures viejas de _day_sodium_autofix ahora DELEGAN a las funciones de módulo.
    i_autofix = go_src.find("def _day_sodium_autofix(")
    assert i_autofix > 0
    autofix_block = go_src[i_autofix: i_autofix + 4000]
    assert "return _line_sodium_mg(_s, db)" in autofix_block, (
        "_day_sodium_autofix debe delegar en el estimador de módulo (SSOT único, sin duplicar el "
        "primitivo db.micros_from_ingredient_string)"
    )
    assert "return sum(_meal_sodium_mg(_m, db)" in autofix_block


def test_micros_from_ingredient_string_is_the_only_primitive():
    """Tanto el estimador de módulo como el autofix legado usan el MISMO primitivo del catálogo —
    ninguno de los dos reimplementa una lectura de sodio por su cuenta."""
    go_src = _go_source()
    i_line = go_src.find("def _line_sodium_mg(")
    block = go_src[i_line: i_line + 400]
    assert "db.micros_from_ingredient_string" in block
    assert "sodium_mg" in block


# =====================================================================
# Section B — Funcional real (sin mocks): estimador + techo + caso real
# =====================================================================

class _FakeIngredientDB:
    """DB determinista: mapea keywords → mg de sodio por línea, sin depender de Postgres."""

    _TABLE = (
        ("camaron", 450.0),
        ("queso curado", 300.0),
        ("queso duro", 300.0),
        ("salami", 400.0),
        ("ricotta", 350.0),
        ("pollo", 60.0),
        ("pechuga", 60.0),
        ("arroz", 5.0),
        ("aguacate", 2.0),
        ("berenjena", 3.0),
    )

    def micros_from_ingredient_string(self, s: str) -> dict:
        low = str(s).lower()
        for token, mg in self._TABLE:
            if token in low:
                return {"sodium_mg": mg}
        return {"sodium_mg": 0.0}


def test_meal_sodium_mg_sums_ingredient_lines():
    from graph_orchestrator import _meal_sodium_mg

    db = _FakeIngredientDB()
    meal = {"name": "Berenjenas con Camarones",
            "ingredients": ["300 g de camarones", "2 berenjenas medianas", "50 g de queso curado"]}
    assert _meal_sodium_mg(meal, db) == pytest.approx(450.0 + 3.0 + 300.0)


def test_meal_sodium_mg_prefers_ingredients_raw():
    from graph_orchestrator import _meal_sodium_mg

    db = _FakeIngredientDB()
    meal = {
        "ingredients": ["texto limpio sin sodio"],
        "ingredients_raw": ["300 g de camarones"],
    }
    assert _meal_sodium_mg(meal, db) == pytest.approx(450.0)


def test_meal_sodium_mg_failsafe_non_dict():
    from graph_orchestrator import _meal_sodium_mg
    assert _meal_sodium_mg(None, _FakeIngredientDB()) == 0.0
    assert _meal_sodium_mg("not a dict", _FakeIngredientDB()) == 0.0


def test_ceiling_matches_banner_dri_targets_exactly():
    """El techo del swap debe ser BIT-A-BIT el mismo que `micronutrients.dri_targets` — la fuente
    que alimenta `micronutrient_report.gaps` y por ende el banner/`_maybe_mark_panel_degraded`."""
    from graph_orchestrator import _sodium_day_ceiling_mg_for_banner
    from micronutrients import dri_targets

    for form in ({}, {"gender": "male", "age": 45}, {"gender": "female", "age": 30, "pregnant": True}):
        expected = dri_targets(
            sex=form.get("gender"), age=form.get("age"), pregnant=bool(form.get("pregnant"))
        )["sodium_mg"]["ceiling"]
        assert _sodium_day_ceiling_mg_for_banner(form) == pytest.approx(expected)


def test_ceiling_failsafe_default_2000():
    from graph_orchestrator import _sodium_day_ceiling_mg_for_banner
    assert _sodium_day_ceiling_mg_for_banner(None) == 2000.0
    assert _sodium_day_ceiling_mg_for_banner({}) == 2000.0


def test_real_case_simulation_budget_would_flag_shrimp_and_allow_fresh_chicken():
    """El caso real de hoy: día con ricotta (resto≈350mg via la db fake) + candidato de camarones+
    queso curado (750mg) → 1100mg, MUY por debajo del techo real de 2000... así que replicamos el
    caso reportado con proporciones fieles: resto=1800 (día ya cargado de sodio) + candidato
    camarones=750 → 2550 > 2000 → EXCEDE. Con el presupuesto informado (200mg restantes), el
    prompt habría empujado a una proteína fresca: pollo+arroz≈65mg → 1865 ≤ 2000 → DENTRO."""
    from graph_orchestrator import _meal_sodium_mg, _sodium_day_ceiling_mg_for_banner

    db = _FakeIngredientDB()
    ceiling = _sodium_day_ceiling_mg_for_banner({})
    resto = 1800.0  # día ya cargado (p.ej. ricotta + otros)

    shrimp_candidate = {"name": "Berenjenas con Camarones",
                         "ingredients": ["300 g de camarones", "50 g de queso curado", "2 berenjenas"]}
    fresh_candidate = {"name": "Pollo a la Plancha con Arroz",
                        "ingredients": ["200 g de pechuga de pollo fresca", "1 taza de arroz integral"]}

    shrimp_mg = _meal_sodium_mg(shrimp_candidate, db)
    fresh_mg = _meal_sodium_mg(fresh_candidate, db)

    assert resto + shrimp_mg > ceiling, "el candidato real (camarones+queso) debe exceder el techo"
    assert resto + fresh_mg <= ceiling, (
        "con el presupuesto informado, una proteína fresca SÍ habría evitado el exceso — "
        "confirma que el diseño habría corregido el caso real"
    )


# =====================================================================
# Section C — Funcional end-to-end con mocks sobre swap_meal()
# =====================================================================

class _FakeCircuitBreaker:
    def can_proceed(self):
        return True

    def record_success(self):
        pass

    def record_failure(self):
        pass


def _fake_invoke_result(meal_kwargs: dict) -> dict:
    """Envelope {"raw","parsed","parsing_error"} — mismo contrato REAL (un dict plano) que
    `.with_structured_output(MealModel, include_raw=True).invoke(...)` retorna en producción.
    Debe ser un `dict` de verdad: `swap_meal` gatea con `isinstance(_res_env, dict)`."""
    from schemas import MealModel
    return {"raw": None, "parsed": MealModel(**meal_kwargs), "parsing_error": None}


def _meal_kwargs(name, ingredients):
    return dict(
        meal="Cena", name=name, desc="Descripción de prueba", prep_time="15 min",
        cals=350, protein=20, carbs=30, fats=10, ingredients=ingredients,
        recipe=["Mise en place: prepara los ingredientes.",
                "El Toque de Fuego: cocina.", "Montaje: sirve."],
    )


class _FakeSwapLLM:
    def __init__(self, envelopes):
        self._envelopes = list(envelopes)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        env = self._envelopes[min(self.calls, len(self._envelopes)) - 1]
        return env


class _FakeChatDeepSeekInstance:
    def __init__(self, envelopes):
        self._envelopes = envelopes
        self.swap_llm = None

    def with_structured_output(self, *a, **kw):
        self.swap_llm = _FakeSwapLLM(self._envelopes)
        return self.swap_llm


@pytest.fixture
def _sodium_swap_env(monkeypatch):
    """Neutraliza los guards HERMANOS de swap_meal (irrelevantes a este test — cada uno tiene su
    propia suite) para aislar el guard de sodio. Guest + strict_pantry + sin nevera hace que la
    mayoría (slot/appetibility/dish-quality/raw-staple/sameday-variety) skip-en-silencio por diseño
    (`not (strict_pantry and not clean_ingredients)`); solo hace falta apagar explícitamente los que
    NO comparten ese guard."""
    import agent
    import nutrition_db

    monkeypatch.setattr(agent, "UPDATE_CLINICAL_GUARD", False)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *a, **kw: _FakeCircuitBreaker())
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeIngredientDB)
    monkeypatch.setenv("MEALFIT_SWAP_BASE_REPEAT_GATE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_MACRO_TRUTHUP", "false")
    monkeypatch.setenv("MEALFIT_SWAP_DETERMINISTIC_RESCALE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_PROTEIN_CLOSER", "false")
    monkeypatch.setenv("MEALFIT_SWAP_FATS_TRIM", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_SUPERPERS", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_CONDITION_DIRECTIVES", "false")
    return agent


def _base_form_data(**overrides):
    data = {
        "rejected_meal": "Ensalada vieja",
        "meal_type": "Cena",
        "swap_reason": "dislike",
        "diet_type": "balanced",
        "sodium_resto_override_mg": 1800.0,
    }
    data.update(overrides)
    return data


def test_e2e_exceeds_then_fresh_candidate_accepted(_sodium_swap_env, monkeypatch):
    """1er candidato (camarones+queso) excede el presupuesto → 1 reintento con directiva; 2º
    candidato (pollo fresco) entra en presupuesto → aceptado. El LLM debe invocarse exactamente 2×."""
    agent = _sodium_swap_env
    envelopes = [
        _fake_invoke_result(_meal_kwargs(
            "Berenjenas con Camarones",
            ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
        )),
        _fake_invoke_result(_meal_kwargs(
            "Pollo a la Plancha con Arroz",
            ["200 g de pechuga de pollo fresca", "1 taza de arroz integral"],
        )),
    ]
    fake_instance_holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekInstance(envelopes)
        fake_instance_holder["inst"] = inst
        return inst

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    result = agent.swap_meal(_base_form_data())

    assert fake_instance_holder["inst"].swap_llm.calls == 2, (
        "debe haber EXACTAMENTE 1 reintento (2 llamadas al LLM) por el exceso de sodio"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert "camaron" not in str(name).lower(), "el candidato final debe ser el fresco, no camarones"


def test_e2e_still_exceeds_after_retry_is_accepted_not_failed(_sodium_swap_env, monkeypatch):
    """Si el candidato del reintento TAMBIÉN excede, el swap se ACEPTA igual (nunca falla por
    sodio) — el marker ya usado impide un 2º retry dedicado a esta causa."""
    agent = _sodium_swap_env
    envelopes = [
        _fake_invoke_result(_meal_kwargs(
            "Berenjenas con Camarones",
            ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
        )),
        _fake_invoke_result(_meal_kwargs(
            "Salami con Queso Duro",
            ["200 g de salami", "50 g de queso duro"],
        )),
    ]
    fake_instance_holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekInstance(envelopes)
        fake_instance_holder["inst"] = inst
        return inst

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    # No debe lanzar excepción — se acepta el 2º candidato pese a seguir sobre el techo.
    result = agent.swap_meal(_base_form_data())

    assert fake_instance_holder["inst"].swap_llm.calls == 2, (
        "el reintento de sodio es ÚNICO — tras el 2º exceso se acepta, no se vuelve a intentar"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert "salami" in str(name).lower(), "el swap debe devolver el 2º candidato aunque siga alto"


def test_e2e_knob_off_zero_flow_change(_sodium_swap_env, monkeypatch):
    """Knob OFF → el candidato de camarones (que excedería el presupuesto) se acepta en el 1er
    intento SIN retry — cero cambio de flujo respecto al swap_meal pre-fix."""
    agent = _sodium_swap_env
    monkeypatch.setenv("MEALFIT_SODIUM_AWARE_SWAP", "false")
    envelopes = [
        _fake_invoke_result(_meal_kwargs(
            "Berenjenas con Camarones",
            ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
        )),
    ]
    fake_instance_holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekInstance(envelopes)
        fake_instance_holder["inst"] = inst
        return inst

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    result = agent.swap_meal(_base_form_data())

    assert fake_instance_holder["inst"].swap_llm.calls == 1, (
        "con el knob OFF, el candidato debe aceptarse en el PRIMER intento (cero retry de sodio)"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert "camaron" in str(name).lower()


# =====================================================================
# Section C.2 — fix-review 2026-08-02: findings #1 (presupuesto imposible) y
# #2 (nunca lanzar en el último intento compartido). Reproduce el escenario
# COMPUESTO del review adversarial: pantry/macros SIN neutralizar, exactamente
# el patrón "los logs de hoy" citado en el finding (pantry quema intentos 1-2).
# =====================================================================

def _fail_once_then_pass_pantry():
    """Fake de `validate_ingredients_against_pantry`: rechaza la 1ª llamada
    (simula pantry fallando en el intento 1, patrón real de logs de prod), pasa
    el resto. Firma real: (ingreds, clean_ingredients, allow_external_count=0)."""
    state = {"n": 0}

    def _fake(ingreds, clean, allow_external_count=0):
        state["n"] += 1
        if state["n"] == 1:
            return "PANTRY_VIOLATION: 'camarones' no está en tu nevera."
        return True

    _fake.calls = lambda: state["n"]
    return _fake


def _fail_once_then_pass_macros():
    """Fake de `nutrition_calculator.validate_meal_macros_against_targets`: rechaza
    su 1ª llamada (simula macros fallando en el intento donde le toque correr —
    aquí el intento 2, porque pantry ya consumió el intento 1), pasa el resto.
    Firma real: (meal_dump, targets_dict) -> (passed, drifts, summary)."""
    state = {"n": 0}

    def _fake(meal_dump, targets):
        state["n"] += 1
        if state["n"] == 1:
            return False, {"protein": {"delta_pct": 0.5, "actual": 8, "target": 15}}, (
                "⚠️ MACROS FUERA DE OBJETIVO: protein drift 50%"
            )
        return True, {}, ""

    _fake.calls = lambda: state["n"]
    return _fake


@pytest.fixture
def _composite_swap_env(monkeypatch):
    """Espejo de `_sodium_swap_env` PERO deja pantry y macros LIBRES de correr
    (no los neutraliza) — necesario para el escenario compuesto del finding #2.
    Como `clean_ingredients` deja de estar vacío (necesario para que el check de
    pantry se ejecute), el short-circuit `not (strict_pantry and not
    clean_ingredients)` que protegía a slot/appetibility/dish-quality/raw-staple
    dejar de aplicar SOLO — hay que apagarlos explícitamente."""
    import agent
    import nutrition_db
    import nutrition_calculator
    import shopping_calculator

    monkeypatch.setattr(agent, "UPDATE_CLINICAL_GUARD", False)
    monkeypatch.setattr(agent, "SLOT_APPROPRIATENESS_GATE_ENABLED", False)
    monkeypatch.setattr(agent, "UPDATE_APPETIBILITY_GUARD", False)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *a, **kw: _FakeCircuitBreaker())
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeIngredientDB)
    # Identity: `clean_ingredients` queda poblado con el texto crudo (sin depender de
    # catálogo/DB real para el parsing — determinismo total del test).
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda lst: list(lst))
    monkeypatch.setenv("MEALFIT_SWAP_BASE_REPEAT_GATE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_MACRO_TRUTHUP", "false")
    monkeypatch.setenv("MEALFIT_SWAP_DETERMINISTIC_RESCALE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_PROTEIN_CLOSER", "false")
    monkeypatch.setenv("MEALFIT_SWAP_FATS_TRIM", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_SUPERPERS", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_CONDITION_DIRECTIVES", "false")
    monkeypatch.setenv("MEALFIT_SWAP_DISH_QUALITY_PRESSURE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_RAW_STAPLE_PRESSURE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_SAMEDAY_PROTEIN_GATE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_MACROS_VALIDATE", "true")
    pantry_fake = _fail_once_then_pass_pantry()
    macros_fake = _fail_once_then_pass_macros()
    monkeypatch.setattr(agent, "validate_ingredients_against_pantry", pantry_fake)
    monkeypatch.setattr(nutrition_calculator, "validate_meal_macros_against_targets", macros_fake)
    return agent, pantry_fake, macros_fake


def test_e2e_composite_pantry_then_macros_then_sodium_final_attempt_never_fails(
    _composite_swap_env, monkeypatch
):
    """[FINDING #2 · fix-review] El escenario REAL citado por el review: intento 1 pantry
    rechaza, intento 2 macros rechaza, intento 3 (ÚLTIMO del presupuesto compartido) trae un
    candidato que EXCEDE el presupuesto de sodio. Pre-fix esto habría propagado
    `SODIUM_BUDGET_EXCEEDED` -> `reraise=True` -> `SWAP_LLM_RETRIES_EXHAUSTED` -> 422 (el swap
    ENTERO fallando por un candidato que, sin el guard de sodio, se habría aceptado). Post-fix:
    el guard detecta que es el último intento disponible y ACEPTA sin lanzar — el swap TRIUNFA."""
    agent, pantry_fake, macros_fake = _composite_swap_env

    # Los 3 intentos devuelven el MISMO candidato alto en sodio (camarones+queso) — lo que hace
    # fallar los intentos 1 y 2 es el fake de pantry/macros, NO el contenido del candidato. Lo
    # único que le importa al guard de sodio es lo que llega al intento 3.
    shrimp_meal = _meal_kwargs(
        "Berenjenas con Camarones",
        ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
    )
    envelopes = [_fake_invoke_result(shrimp_meal) for _ in range(3)]
    fake_instance_holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekInstance(envelopes)
        fake_instance_holder["inst"] = inst
        return inst

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    form_data = _base_form_data(
        # Puebla `clean_ingredients` (vía el fallback "current_pantry_ingredients" + el
        # `aggregate_shopping_list` identidad monkeypatcheado) para que el check de pantry
        # REALMENTE corra (sin esto, `if clean_ingredients:` lo salta y pantry nunca fallaría).
        current_pantry_ingredients=["200 g de pollo", "1 taza de arroz", "1 aguacate"],
        sodium_resto_override_mg=1800.0,  # techo 2000 - resto 1800 = 200mg de presupuesto
    )

    # No debe lanzar — el swap debe TRIUNFAR con el candidato del intento 3 pese a exceder sodio.
    result = agent.swap_meal(form_data)

    assert fake_instance_holder["inst"].swap_llm.calls == 3, (
        "deben agotarse los 3 intentos compartidos: 1=pantry, 2=macros, 3=sodio(final, aceptado)"
    )
    assert pantry_fake.calls() == 3, "pantry debe evaluarse en los 3 intentos (falla solo el 1º)"
    assert macros_fake.calls() == 2, (
        "macros solo corre cuando pantry ya pasó — intentos 2 y 3 (falla solo el 1º de ESOS, "
        "que es el intento 2 global)"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert "camaron" in str(name).lower(), (
        "el swap debe devolver el candidato del intento 3 (camarones) aceptado pese a exceder "
        "sodio — NUNCA debe propagar SWAP_LLM_RETRIES_EXHAUSTED por esto"
    )


def test_e2e_composite_final_attempt_accept_does_not_raise_swap_llm_retries_exhausted(
    _composite_swap_env, monkeypatch
):
    """Variante negativa explícita del mismo escenario: confirma que NO se lanza
    `SWAP_LLM_RETRIES_EXHAUSTED` (la contradicción exacta que el review reportó) — usa
    `pytest.raises` de control (debe fallar si alguien lo intenta) envuelto en un chequeo que el
    call NO levanta ninguna excepción."""
    agent, _, _ = _composite_swap_env
    shrimp_meal = _meal_kwargs(
        "Berenjenas con Camarones",
        ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
    )
    envelopes = [_fake_invoke_result(shrimp_meal) for _ in range(3)]

    def _fake_chat_deepseek(*a, **kw):
        return _FakeChatDeepSeekInstance(envelopes)

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    form_data = _base_form_data(
        current_pantry_ingredients=["200 g de pollo", "1 taza de arroz", "1 aguacate"],
        sodium_resto_override_mg=1800.0,
    )
    try:
        agent.swap_meal(form_data)
    except ValueError as e:
        if "SWAP_LLM_RETRIES_EXHAUSTED" in str(e):
            pytest.fail(
                f"REGRESIÓN (finding #2 del review): el swap propagó SWAP_LLM_RETRIES_EXHAUSTED "
                f"cuando el ÚLTIMO intento solo se rechazaría por sodio -> {e}"
            )
        raise  # cualquier otra excepción es un fallo genuino distinto, no la enmascaramos


def test_impossible_budget_skips_postcheck_without_consuming_retry(monkeypatch):
    """[FINDING #1 · fix-review] Cuando `resto_del_día >= techo` ANTES de sumar el candidato,
    ningún candidato puede caber — el post-check debe hacer skip TOTAL (no lanzar, no consumir
    ningún intento del presupuesto compartido). Verificado con el candidato de camarones (que
    normalmente dispararía el retry): debe aceptarse en el PRIMER intento (1 sola llamada LLM)."""
    import agent
    import nutrition_db

    monkeypatch.setattr(agent, "UPDATE_CLINICAL_GUARD", False)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *a, **kw: _FakeCircuitBreaker())
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeIngredientDB)
    for env_key in (
        "MEALFIT_SWAP_BASE_REPEAT_GATE", "MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE",
        "MEALFIT_UPDATE_MACRO_TRUTHUP", "MEALFIT_SWAP_DETERMINISTIC_RESCALE",
        "MEALFIT_SWAP_PROTEIN_CLOSER", "MEALFIT_SWAP_FATS_TRIM",
        "MEALFIT_UPDATE_SUPERPERS", "MEALFIT_UPDATE_CONDITION_DIRECTIVES",
    ):
        monkeypatch.setenv(env_key, "false")

    envelopes = [_fake_invoke_result(_meal_kwargs(
        "Berenjenas con Camarones",
        ["300 g de camarones", "50 g de queso curado", "2 berenjenas medianas"],
    ))]
    fake_instance_holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekInstance(envelopes)
        fake_instance_holder["inst"] = inst
        return inst

    monkeypatch.setattr(agent, "ChatDeepSeek", _fake_chat_deepseek)

    # Resto YA excede el techo (2200 > 2000) antes de sumar ningún candidato.
    result = agent.swap_meal(_base_form_data(sodium_resto_override_mg=2200.0))

    assert fake_instance_holder["inst"].swap_llm.calls == 1, (
        "presupuesto imposible -> el post-check debe hacer skip total; el candidato se acepta "
        "en el PRIMER intento sin que el guard de sodio consuma ningún retry"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert "camaron" in str(name).lower()


def test_impossible_budget_logs_skip_imposible_decision():
    """La telemetría del skip debe ser reconocible/parseable (`decision=skip_imposible`)."""
    src = _swap_meal_block()
    assert "decision=skip_imposible" in src


def test_final_attempt_never_raises_structural():
    """Ancla estructural del finding #2: el guard debe leer `attempt_number` desde
    `invoke_with_retry.statistics` (NO `.retry.statistics`, que en tenacity 9.x es un dict
    compartido siempre vacío — verificado empíricamente durante el fix) y solo debe hacer
    `raise ValueError` cuando NO es el último intento."""
    src = _swap_meal_block()
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    assert i_sodium > 0, "guard post-generación de sodio ausente"
    block = src[i_sodium: i_sodium + 9200]
    assert "invoke_with_retry.statistics" in block, (
        "debe leer el contador de intentos desde `invoke_with_retry.statistics` "
        "(NO `.retry.statistics`, que está vacío en tenacity 9.x)"
    )
    assert "_is_final_attempt" in block
    assert "decision=accept_final" in block
    # El raise de sodio debe estar gateado por `not _is_final_attempt` (vía el `elif` que solo se
    # alcanza cuando `_is_final_attempt` es False) — nunca incondicional.
    i_raise = block.find("raise ValueError")
    i_final_check = block.find("_is_final_attempt:")
    assert 0 < i_final_check < i_raise, (
        "el chequeo de último-intento debe evaluarse ANTES del raise, gateándolo"
    )


# =====================================================================
# Section D — regen-day pasa el override EN VIVO (parser)
# =====================================================================

def test_regen_day_passes_live_sodium_override():
    src = _plans_source()
    i_compute = src.find("_meal_sodium_mg as _sod_mm_rd")
    assert i_compute > 0, "regen-day debe reusar el estimador SSOT (no reinventar uno)"
    i_key = src.find('"sodium_resto_override_mg": _sodium_resto_this_meal,')
    assert i_key > i_compute, "el override debe pasarse en meal_form usando el valor YA computado"
    # Debe computarse sobre el MISMO universo que same_day_other_meal_blobs (new_meals + pendientes).
    i_blobs = src.find('"same_day_other_meal_blobs": [')
    region = src[i_compute:i_blobs]
    assert "new_meals + meals[len(new_meals) + 1:]" in region


def test_regen_day_override_is_failsafe_none_on_error():
    src = _plans_source()
    i = src.find("_sodium_resto_this_meal = sum(")
    block = src[max(0, i - 300): i + 400]
    assert "except Exception:" in block and "_sodium_resto_this_meal = None" in block


# =====================================================================
# Section E — Marker de P-fix
# =====================================================================

def test_pfix_marker_bumped_in_app_py():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-SODIUM-AWARE-PLACEMENT" in src, (
        "_LAST_KNOWN_PFIX debe bumpearse al cerrar este P-fix (contrato de "
        "test_p3_1_last_known_pfix_freshness / test_p2_hist_audit_14_marker_test_link)"
    )
