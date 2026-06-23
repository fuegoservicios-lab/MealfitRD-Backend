"""[P2-SWAP-CONSISTENCY · 2026-05-22] Test parser-based + funcional del
bundle de coherencia del modal "¿Por qué quieres cambiar?".

Cierra 4 inconsistencias detectadas en el audit del flujo Cambiar Plato:

  1. Branch ``similar`` redundante en el elif chain de ``agent.py::swap_meal``
     (el anti-mode-collapse de _pick_by_inverse_freq + el filtro
     ``available_proteins = [x for x in filtered if x not in rejected]`` ya
     cubrían el efecto deterministically).
  2. Branch ``pantry_first`` huérfana en el elif chain — duplicado
     conceptual de ``budget`` tras P3-SWAP-BUDGET-COPY del mismo día.
     Eliminada del prompt pero PRESERVADA en el tuple ``strict_pantry``
     para back-compat con cualquier caller legacy.
  3. ``cravings`` / ``weekend`` colisionaban con el guardrail pantry
     estricto: hints "indulgente"/"premium" sin posibilidad de ingredientes
     externos producía retries innecesarios cuando la nevera era
     limitada. Solución: kwarg ``allow_external_count`` en
     ``validate_ingredients_against_pantry`` + wiring en
     ``agent.py::swap_meal`` con knob
     ``MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED`` (default 2).
  4. ``swap_reason='time'`` ("No tengo tiempo hoy") era soft-only —
     el prompt inyectaba "<20 min" como hint pero NO había validator
     post-gen que rechazara recetas >20 min. Solución:
     ``validate_meal_prep_time_against_target`` + wiring en el retry
     loop tenacity de ``agent.py::swap_meal``.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: el slug del
marker ``P2-SWAP-CONSISTENCY`` ↔ filename ``test_p2_swap_consistency.py``.
"""
import pathlib
import re

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
NUTR_PY = (BACKEND_ROOT / "nutrition_calculator.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Branch 'similar' eliminado del elif chain
# ---------------------------------------------------------------------------

def test_similar_branch_removed_from_swap_meal_prompt():
    """El elif chain en swap_meal NO debe tener ``elif swap_reason == 'similar'``
    — la deduplicación venía del anti-mode-collapse, el hint era redundante."""
    assert "elif swap_reason == 'similar'" not in AGENT_PY, (
        "Branch 'similar' aún presente en agent.py. Era duplicado del "
        "anti-mode-collapse (líneas ~522-576) que ya excluye proteína/carb/"
        "veggie del rechazado deterministically."
    )
    # Sanity: el marker explicativo del por qué fue removido sigue presente
    assert "Branch 'similar' eliminado" in AGENT_PY, (
        "El comentario que documenta la decisión de eliminar el branch "
        "'similar' debe permanecer como ancla narrativa para futuros lectores."
    )


def test_similar_modal_option_still_supported_as_passive_reason():
    """``swap_reason='similar'`` sigue siendo un valor válido — el modal
    Dashboard.jsx aún expone la opción. El branch SOLO se removió del
    elif chain del prompt (no context_extras extra para ese reason); el
    anti-mode-collapse aplica deterministicamente sin hint.

    [P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Tras flip a strict-by-default,
    'similar' SÍ activa strict-pantry (es uno de los reasons base). El
    test ahora verifica la nueva derivación (inversa): cravings/weekend
    son las únicas opt-out."""
    # [P4-UPDATE-DISHES-STRICT-ALL · 2026-06-23] La derivación quedó anidada bajo el knob
    # strict-all: `strict_pantry = True if _strict_all else (swap_reason not in (...))`.
    # El tuple opt-out (cravings/weekend) se preserva en el `else` (default OFF = legacy).
    m = re.search(
        r"else \(swap_reason\s+not\s+in\s+\(([^)]+)\)",
        AGENT_PY,
    )
    assert m, (
        "No se encontró `strict_pantry = swap_reason not in (...)` en agent.py. "
        "Post-P3-SWAP-PANTRY-DEFAULT la derivación es inversa."
    )
    tuple_contents = m.group(1)
    assert "'similar'" not in tuple_contents and '"similar"' not in tuple_contents, (
        "'similar' NO debería estar en el tuple de opt-out; pertenece al "
        "default strict (junto con variety/time/dislike)."
    )


# ---------------------------------------------------------------------------
# Section B — Branch 'pantry_first' eliminado del prompt PERO preservado en strict_pantry
# ---------------------------------------------------------------------------

def test_pantry_first_and_budget_branches_removed_from_swap_meal_prompt():
    """[P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Tras eliminar la opción del
    modal "Usar solo lo que tengo", AMBOS branches ('budget' y 'pantry_first')
    salen del elif chain del prompt. La intención "respeta la nevera" pasa
    a ser hint genérico inyectado para todos los reasons no-indulgentes."""
    assert "elif swap_reason == 'pantry_first'" not in AGENT_PY, (
        "Branch 'pantry_first' aún presente en agent.py."
    )
    assert "elif swap_reason == 'budget'" not in AGENT_PY, (
        "Branch 'budget' debió eliminarse junto con 'pantry_first' tras "
        "P3-SWAP-PANTRY-DEFAULT — ambos son hint redundante con el "
        "nuevo default strict-pantry."
    )
    # El comentario explicativo del removal debe permanecer como ancla narrativa
    assert "P3-SWAP-PANTRY-DEFAULT" in AGENT_PY, (
        "El marker explicativo del removal debe permanecer en agent.py "
        "como ancla para futuros lectores."
    )


def test_budget_and_pantry_first_still_strict_via_default_inversion():
    """[P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Back-compat: callers legacy
    que aún emiten ``swap_reason='budget'`` o ``'pantry_first'`` siguen
    cayendo en strict-pantry — pero ahora via la inversión del default,
    no via tuple explícito. El test ancla el contrato."""
    # [P4-UPDATE-DISHES-STRICT-ALL · 2026-06-23] La derivación quedó anidada bajo el knob
    # strict-all: `strict_pantry = True if _strict_all else (swap_reason not in (...))`.
    # El tuple opt-out (cravings/weekend) se preserva en el `else` (default OFF = legacy).
    m = re.search(
        r"else \(swap_reason\s+not\s+in\s+\(([^)]+)\)",
        AGENT_PY,
    )
    assert m, (
        "No se encontró `strict_pantry = swap_reason not in (...)` — "
        "post-P3-SWAP-PANTRY-DEFAULT la derivación es inversa (opt-out)."
    )
    tuple_contents = m.group(1)
    # El tuple opt-out debe ser exactamente {cravings, weekend}; cualquier
    # otra cosa rompe el contrato back-compat de budget/pantry_first.
    assert ("'budget'" not in tuple_contents) and ('"budget"' not in tuple_contents), (
        "'budget' NO debe estar en el opt-out — es back-compat strict."
    )
    assert ("'pantry_first'" not in tuple_contents) and ('"pantry_first"' not in tuple_contents), (
        "'pantry_first' NO debe estar en el opt-out — es back-compat strict."
    )
    assert ("'cravings'" in tuple_contents) or ('"cravings"' in tuple_contents), (
        "'cravings' debe estar en el opt-out (indulgencia explícita)."
    )
    assert ("'weekend'" in tuple_contents) or ('"weekend"' in tuple_contents), (
        "'weekend' debe estar en el opt-out (premium explícito)."
    )


# ---------------------------------------------------------------------------
# Section C — allow_external_count kwarg en validate_ingredients_against_pantry
# ---------------------------------------------------------------------------

def test_validator_accepts_allow_external_count_kwarg():
    """``validate_ingredients_against_pantry`` debe aceptar el kwarg
    ``allow_external_count`` (default 0). Verificado por inspección de la
    signature en constants.py."""
    pytest.importorskip("langchain_google_genai", reason="constants.py requiere langchain")
    import inspect
    from constants import validate_ingredients_against_pantry
    sig = inspect.signature(validate_ingredients_against_pantry)
    assert "allow_external_count" in sig.parameters, (
        "Falta kwarg `allow_external_count` en validate_ingredients_against_pantry. "
        "Necesario para que swap_reason cravings/weekend toleren externos."
    )
    default = sig.parameters["allow_external_count"].default
    assert default == 0, f"Default debe ser 0 (legacy strict), no {default!r}"


def test_external_tolerance_relaxes_unauthorized_only_not_over_limit():
    """Funcional: con allow_external_count=N, hasta N ingredientes
    "unauthorized" (no en pantry) son permitidos. ``over_limit`` (cantidades
    excedidas) NUNCA se relaja — son problema cuantitativo distinto."""
    pytest.importorskip("langchain_google_genai", reason="constants.py requiere langchain")
    from constants import validate_ingredients_against_pantry

    # Sin tolerancia: 1 ingrediente externo → falla
    result_strict = validate_ingredients_against_pantry(
        generated_ingredients=["100g pollo", "50g espárragos"],  # espárragos NO en pantry
        pantry_ingredients=["500g pollo", "1 libra arroz"],
        allow_external_count=0,
    )
    assert result_strict is not True, (
        "Sin tolerancia, un ingrediente externo debería rechazar."
    )

    # Con tolerancia=1: mismo input → True (relajado)
    result_relaxed = validate_ingredients_against_pantry(
        generated_ingredients=["100g pollo", "50g espárragos"],
        pantry_ingredients=["500g pollo", "1 libra arroz"],
        allow_external_count=1,
    )
    assert result_relaxed is True, (
        f"Con allow_external_count=1, 1 ingrediente externo debería pasar. "
        f"Retorno: {result_relaxed!r}"
    )


def test_external_tolerance_caps_at_n():
    """Con allow_external_count=2, hasta 2 externos pasan. 3 externos siguen
    fallando (cap respetado)."""
    pytest.importorskip("langchain_google_genai", reason="constants.py requiere langchain")
    from constants import validate_ingredients_against_pantry

    # 3 externos con tolerancia=2 → falla (excede el cap)
    result = validate_ingredients_against_pantry(
        generated_ingredients=[
            "100g pollo",
            "50g espárragos",
            "30g aceitunas",
            "10g trufa negra",
        ],
        pantry_ingredients=["500g pollo", "1 libra arroz"],
        allow_external_count=2,
    )
    assert result is not True, (
        "3 externos exceden el cap=2; debe seguir rechazando."
    )


def test_agent_py_wires_external_tolerance_from_knob_for_cravings_weekend():
    """``agent.py::swap_meal`` debe derivar ``_external_tolerance`` desde el
    knob ``MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED`` SOLO para
    ``swap_reason in ('cravings', 'weekend')``. Los demás reasons usan 0."""
    # Knob referenciado
    assert "MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED" in AGENT_PY, (
        "Knob `MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED` no aparece en agent.py — "
        "el wiring de tolerancia para cravings/weekend está incompleto."
    )
    # Branch condicional sobre cravings/weekend
    assert (
        "swap_reason in (\"cravings\", \"weekend\")" in AGENT_PY
        or "swap_reason in ('cravings', 'weekend')" in AGENT_PY
    ), (
        "agent.py no enmarca el tolerance lookup detrás de "
        "`swap_reason in ('cravings', 'weekend')` — sin la guarda el knob "
        "se aplicaría globalmente, regresion del strict-pantry para budget."
    )
    # El callsite del validador debe pasar el kwarg
    assert "allow_external_count=_external_tolerance" in AGENT_PY, (
        "El callsite de validate_ingredients_against_pantry debe pasar "
        "`allow_external_count=_external_tolerance`."
    )


# ---------------------------------------------------------------------------
# Section D — prep_time validator
# ---------------------------------------------------------------------------

def test_prep_time_helpers_exist():
    """``validate_meal_prep_time_against_target`` + knobs deben estar
    importables desde nutrition_calculator."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import (
        validate_meal_prep_time_against_target,
        _swap_prep_time_target_minutes,
        _swap_prep_time_validate_enabled,
        _parse_prep_time_minutes,
    )
    assert callable(validate_meal_prep_time_against_target)
    assert callable(_swap_prep_time_target_minutes)
    assert callable(_swap_prep_time_validate_enabled)
    assert callable(_parse_prep_time_minutes)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("15 min", 15.0),
        ("20 minutos", 20.0),
        ("aprox 30 mins", 30.0),
        ("1 hora", 60.0),
        ("2 horas", 120.0),
        ("15-20 min", 15.0),  # toma el primer número
        (25, 25.0),
        (25.5, 25.5),
        (None, None),
        ("", None),
        ("rápido", None),  # sin número parseable
        (True, None),  # bool no debe colarse como int
        (-5, None),  # negativos rechazados
    ],
)
def test_parse_prep_time_minutes(raw, expected):
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import _parse_prep_time_minutes
    assert _parse_prep_time_minutes(raw) == expected


def test_prep_time_validator_passes_when_within_target():
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_prep_time_against_target
    passed, actual, summary = validate_meal_prep_time_against_target(
        {"prep_time": "15 min"}, target_minutes=20
    )
    assert passed is True
    assert actual == 15.0
    assert summary == ""


def test_prep_time_validator_fails_when_exceeds_target():
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_prep_time_against_target
    passed, actual, summary = validate_meal_prep_time_against_target(
        {"prep_time": "40 min"}, target_minutes=20
    )
    assert passed is False
    assert actual == 40.0
    assert "FUERA DE OBJETIVO" in summary
    assert "40" in summary  # actual reportado
    assert "20" in summary  # target reportado


def test_prep_time_validator_passthrough_when_unparseable():
    """Si prep_time es None o formato exótico, el validator NO debe abortar
    — passthrough True. Preferimos un meal posiblemente lento a un falso
    positivo que rompa el swap."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_prep_time_against_target
    passed, actual, summary = validate_meal_prep_time_against_target(
        {"prep_time": None}, target_minutes=20
    )
    assert passed is True
    assert actual is None
    assert summary == ""


def test_prep_time_knobs_clamp_correctly(monkeypatch):
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import _swap_prep_time_target_minutes, _swap_prep_time_validate_enabled

    # Default
    monkeypatch.delenv("MEALFIT_SWAP_PREP_TIME_TARGET_MIN", raising=False)
    assert _swap_prep_time_target_minutes() == 20

    # Clamp inferior
    monkeypatch.setenv("MEALFIT_SWAP_PREP_TIME_TARGET_MIN", "1")
    assert _swap_prep_time_target_minutes() == 5

    # Clamp superior
    monkeypatch.setenv("MEALFIT_SWAP_PREP_TIME_TARGET_MIN", "999")
    assert _swap_prep_time_target_minutes() == 120

    # Kill switch
    monkeypatch.delenv("MEALFIT_SWAP_PREP_TIME_VALIDATE", raising=False)
    assert _swap_prep_time_validate_enabled() is True
    monkeypatch.setenv("MEALFIT_SWAP_PREP_TIME_VALIDATE", "false")
    assert _swap_prep_time_validate_enabled() is False


def test_agent_py_wires_prep_time_validator_only_for_time_reason():
    """El callsite del prep_time validator en swap_meal DEBE estar gateado
    por ``swap_reason == 'time'`` — otros reasons no consultan este check."""
    assert "validate_meal_prep_time_against_target as _validate_prep_time" in AGENT_PY, (
        "agent.py no importa validate_meal_prep_time_against_target alias _validate_prep_time."
    )
    assert "swap_reason == 'time'" in AGENT_PY, (
        "agent.py debe condicionar el prep_time validator a swap_reason='time' — "
        "otros reasons no deben ser bloqueados por prep_time."
    )
    # El callsite debe invocar el alias importado
    assert "_validate_prep_time(meal_dump)" in AGENT_PY, (
        "agent.py debe llamar _validate_prep_time(meal_dump) dentro del retry loop."
    )
    # Y debe inyectar el summary al retry prompt si falla
    assert "[P2-SWAP-PREP-TIME]" in AGENT_PY, (
        "Falta el log marker [P2-SWAP-PREP-TIME] que documenta el path de drift."
    )


# ---------------------------------------------------------------------------
# Section E — Marker bump + cross-link
# ---------------------------------------------------------------------------

def test_last_known_pfix_bumped():
    """``_LAST_KNOWN_PFIX`` en app.py debe estar en ``P2-SWAP-CONSISTENCY`` o
    posterior con fecha 2026-05-22 o posterior."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"',
        APP_PY,
    )
    assert m, "No se encontró `_LAST_KNOWN_PFIX = \"...\"` en app.py"
    marker = m.group(1)
    # Formato Pn-X · YYYY-MM-DD
    fmt = re.match(r"P\d+-[A-Z0-9-]+\s*·\s*(\d{4}-\d{2}-\d{2})$", marker)
    assert fmt, f"Marker no respeta formato `Pn-X · YYYY-MM-DD`: {marker!r}"
    # Floor: 2026-05-22
    assert fmt.group(1) >= "2026-05-22", (
        f"Marker date {fmt.group(1)} es anterior al floor 2026-05-22."
    )


def test_marker_slug_matches_this_filename():
    """Cross-link enforcement: el slug del marker
    (P2-SWAP-CONSISTENCY → p2_swap_consistency) DEBE matchear el archivo de
    test que ancla este P-fix. Espejo de
    test_p2_hist_audit_14_marker_test_link.py."""
    expected_slug = "p2_swap_consistency"
    this_file = pathlib.Path(__file__).name
    assert expected_slug in this_file, (
        f"Filename {this_file} no contiene slug {expected_slug!r} — "
        f"el cross-link marker↔test rompe."
    )
