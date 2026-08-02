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
    # sodio no introduce un segundo decorator/stop_after_attempt propio.
    assert src.count("stop_after_attempt") == 1, (
        "debe existir un ÚNICO `stop_after_attempt` (el de invoke_with_retry) — el guard de sodio "
        "comparte ese presupuesto, no inventa uno propio"
    )
    block = src[i_sodium: i_sodium + 2600]
    assert "comparte el presupuesto de 3 intentos" in block or "NO tiene\n        # su propio @retry" in block, (
        "debe documentarse explícitamente que el reintento de sodio comparte el budget compartido"
    )


def test_single_retry_marker_pattern_mirrors_siblings():
    """Mismo patrón marker-based que P1-SWAP-BASE-REPEAT-GATE / P2-UPDATE-SAMEDAY-VARIETY: un
    marker string en el prompt evita un 2º retry por la MISMA causa → se acepta tras 1 reintento."""
    src = _swap_meal_block()
    i_sodium = src.find("P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista")
    block = src[i_sodium: i_sodium + 4200]
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
