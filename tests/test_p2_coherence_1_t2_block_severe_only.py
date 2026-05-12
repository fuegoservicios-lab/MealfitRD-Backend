"""[P2-COHERENCE-1 · 2026-05-11] T2 escala warn→block selectivo + warnings UI.

Cierre del audit 2026-05-11 P2: las surfaces auxiliares (T2, recalc, agent
tool) corrian en mode=warn puro tras P1-NEXT-2. Si la LLM producía un swap
o un chunk T2 con `cap_swallowed_modifier` (receta dice pollo, lista omite
pollo) o magnitudes severas (>50%), el usuario veía la lista divergente
hasta que el cron diario 04:00 UTC la contara para post-mortem.

Cierre:
    1. Helper `run_shopping_coherence_guard_and_append_history` acepta
       `block_severe_only: bool = False`. Cuando True + mode warn +
       divergencias severas (cap_swallowed_modifier OR delta_pct > 0.50)
       → escala el plan a `_shopping_coherence_block = True` y retorna
       `block_set=True`.
    2. Knob `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` (default True) kill
       switch sin redeploy.
    3. `_chunk_worker T2` pasa `block_severe_only=True` y, si block_set,
       re-raise dentro del retry loop para reintentar via `_SHOP_MAX_RETRIES`.
    4. `/recalculate-shopping-list` y `tools.modify_single_meal` capturan
       `divergences` y retornan `_coherence_warnings` summary (no bloquean).
    5. Helper nuevo `summarize_divergences_for_ui(divergences, max_items=5)`
       compacta para UI.

Tests parser-based + funcional con mocks.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHOPCALC_FP = _REPO_ROOT / "backend" / "shopping_calculator.py"
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"
_PLANS_FP = _REPO_ROOT / "backend" / "routers" / "plans.py"
_TOOLS_FP = _REPO_ROOT / "backend" / "tools.py"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def shopcalc_src() -> str:
    return _SHOPCALC_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claude_md() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


def test_knob_helper_present(shopcalc_src: str):
    assert "_get_coherence_t2_block_severe_only_knob" in shopcalc_src, (
        "P2-COHERENCE-1 regresión: helper `_get_coherence_t2_block_severe_only_knob` "
        "desapareció. Sin el knob, no hay rollback rápido si el escalado "
        "warn→block produce retry storms en T2."
    )
    m = re.search(
        r'_knob_env_bool\(\s*"MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY"\s*,\s*(\w+)',
        shopcalc_src,
    )
    assert m, (
        "P2-COHERENCE-1: el knob `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` ya "
        "no se lee vía `_knob_env_bool`. Si moviste a otra fuente, ajustar "
        "el contrato del helper."
    )
    assert m.group(1) == "True", (
        f"P2-COHERENCE-1: default del knob cambió a `{m.group(1)}`. El "
        f"contrato es `default=True` (opt-out): por defecto T2 escala "
        f"divergencias severas. Flip a False solo via env var."
    )


def test_severity_helper_present(shopcalc_src: str):
    assert "def _has_severe_divergence(" in shopcalc_src, (
        "P2-COHERENCE-1 regresión: helper `_has_severe_divergence` desapareció."
    )
    assert "_COHERENCE_SEVERE_MAGNITUDE_THRESHOLD" in shopcalc_src, (
        "P2-COHERENCE-1: constante `_COHERENCE_SEVERE_MAGNITUDE_THRESHOLD` "
        "ya no aparece. Era el threshold (0.50) para considerar magnitudes "
        "como severas."
    )


def test_summarize_helper_for_ui(shopcalc_src: str):
    assert "def summarize_divergences_for_ui(" in shopcalc_src, (
        "P2-COHERENCE-1 regresión: helper `summarize_divergences_for_ui` "
        "desapareció. Sin él, los callers de UI tienen que duplicar la "
        "lógica de compactación + max_items cap."
    )


def test_helper_signature_has_block_severe_only(shopcalc_src: str):
    """`run_shopping_coherence_guard_and_append_history` acepta el nuevo kwarg."""
    sig = re.search(
        r"def run_shopping_coherence_guard_and_append_history\([^)]*\)",
        shopcalc_src,
        re.DOTALL,
    )
    assert sig, "signature del helper no parsea"
    assert "block_severe_only" in sig.group(0), (
        "P2-COHERENCE-1 regresión: el kwarg `block_severe_only` desapareció "
        "de la signature. Sin él, T2 no puede activar el escalado selectivo "
        "y se vuelve al warn-only puro."
    )


def test_t2_passes_block_severe_only(cron_src: str):
    """`_chunk_worker T2` invoca el helper con `block_severe_only=True`.

    Nota: `warn_only_chunk_t2` aparece en MÚLTIPLES sitios (cron_tasks.py
    incluye P3-B aggregator que enumera buckets como string). Usamos
    `rfind` para agarrar la ÚLTIMA ocurrencia, que es la del worker T2
    (la real). Patrón documentado en lección P1-CHAT-TTS-1 (find vs rfind
    cuando un string aparece en docstrings/aggregators y en call sites)."""
    t2_block_idx = cron_src.rfind("warn_only_chunk_t2")
    assert t2_block_idx > 0, "marker `warn_only_chunk_t2` no encontrado en cron_tasks.py"
    surrounding = cron_src[max(0, t2_block_idx - 2000): t2_block_idx + 2000]
    assert "block_severe_only=True" in surrounding, (
        "P2-COHERENCE-1 regresión: `_chunk_worker T2` ya no pasa "
        "`block_severe_only=True` al helper coherence. Sin esto, divergencias "
        "severas en multi-week plans no fuerzan retry — el usuario ve la "
        "lista divergente hasta que el cron diario la cuente post-hoc."
    )


def test_t2_reraises_on_block_set(cron_src: str):
    """T2 re-raise cuando block_set=True para activar `_SHOP_MAX_RETRIES`."""
    surrounding_idx = cron_src.find("[P2-COHERENCE-1] T2 coherence block_severe_only")
    assert surrounding_idx > 0, (
        "P2-COHERENCE-1 regresión: el RuntimeError `[P2-COHERENCE-1] T2 "
        "coherence block_severe_only escaló warn→block` desapareció. Sin "
        "el re-raise, el block_set se ignora y T2 finaliza el chunk con "
        "lista divergente."
    )
    # Debe estar dentro del retry loop, no fuera
    assert "_SHOP_MAX_RETRIES" in cron_src[surrounding_idx - 500: surrounding_idx + 500], (
        "P2-COHERENCE-1: el RuntimeError debe estar cerca del comentario "
        "`_SHOP_MAX_RETRIES` (dentro del retry loop)."
    )


def test_recalc_returns_coherence_warnings(plans_src: str):
    """`/recalculate-shopping-list` retorna `_coherence_warnings` en response."""
    handler_idx = plans_src.find("def api_recalculate_shopping_list(")
    assert handler_idx > 0, "handler api_recalculate_shopping_list no encontrado"
    # Body hasta el siguiente def top-level
    rest = plans_src[handler_idx:]
    next_def = re.search(r"\n@router\.|\ndef\s+\w+\(", rest[1:])
    end = next_def.start() if next_def else len(rest)
    body = rest[:end]
    assert "_coherence_warnings" in body, (
        "P2-COHERENCE-1 regresión: response de /recalculate-shopping-list "
        "ya no incluye `_coherence_warnings`. Sin él, el frontend no puede "
        "renderear toast cuando el guard detectó drift post-recalc."
    )
    assert "summarize_divergences_for_ui" in body, (
        "P2-COHERENCE-1: el handler ya no usa `summarize_divergences_for_ui`. "
        "Si construyes el summary inline, asegurar el cap max_items=5 para "
        "evitar payloads gigantes en planes con muchas divergencias."
    )


def test_modify_single_meal_returns_warnings(tools_src: str):
    """`tools.execute_modify_single_meal` incluye `_coherence_warnings` en JSON."""
    fn_idx = tools_src.find("def execute_modify_single_meal(")
    assert fn_idx > 0, "execute_modify_single_meal no encontrado"
    # Body hasta la siguiente `def` o `@tool` decorator (boundary del top-level).
    # Función real ~14K chars; usamos boundary explícito en lugar de un slice
    # arbitrario que pueda truncar el path donde el campo se inserta.
    end_marker = tools_src.find("\n@tool\ndef modify_single_meal(", fn_idx)
    assert end_marker > 0, "boundary `@tool def modify_single_meal` no encontrado"
    body = tools_src[fn_idx: end_marker]
    assert "_coherence_warnings" in body, (
        "P2-COHERENCE-1 regresión: el JSON de respuesta de "
        "`execute_modify_single_meal` ya no incluye `_coherence_warnings`. "
        "Sin esto, el agente no propaga warnings al chat tras un swap."
    )


def test_claude_md_documents_t2_change(claude_md: str):
    """CLAUDE.md surface table refleja la escalación selectiva."""
    # La tabla debe mencionar el knob y "block_severe_only"
    surfaces_section = claude_md[claude_md.find("Surfaces que escriben"):claude_md.find("Surfaces que escriben") + 6000]
    assert "block_severe_only" in surfaces_section, (
        "P2-COHERENCE-1 regresión: la tabla 'Surfaces que escriben' en "
        "CLAUDE.md ya no menciona `block_severe_only`. Sin esto, un revisor "
        "futuro no entiende por qué T2 ahora puede bloquear (la tabla decía "
        "'No bloquea' pre-fix)."
    )
    knob_section = claude_md[claude_md.find("MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY"):]
    assert "MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY" in knob_section, (
        "P2-COHERENCE-1: el knob `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` "
        "no aparece en la tabla de knobs de CLAUDE.md. Operadores no sabrán "
        "cómo hacer rollback."
    )


# --------------------------------------------------------------------------
# Functional tests del helper con mocks
# --------------------------------------------------------------------------


@pytest.fixture
def helper_callable():
    """Importa el helper con stubs mínimos. Reusa el patrón monkeypatch
    auto-clean del test P0-AGENT-1 (lección sys.modules pollution)."""
    try:
        from shopping_calculator import (
            run_shopping_coherence_guard_and_append_history,
            _has_severe_divergence,
            summarize_divergences_for_ui,
        )
    except Exception as e:
        pytest.skip(f"No se pudo importar shopping_calculator: {e}")
    return (
        run_shopping_coherence_guard_and_append_history,
        _has_severe_divergence,
        summarize_divergences_for_ui,
    )


def test_severe_helper_flags_cap_swallowed(helper_callable):
    """`_has_severe_divergence` retorna True ante cap_swallowed_modifier."""
    _, has_severe, _ = helper_callable
    divs = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier"}]
    assert has_severe(divs) is True


def test_severe_helper_flags_high_magnitude(helper_callable):
    """delta_pct > 0.50 cuenta como severe."""
    _, has_severe, _ = helper_callable
    divs = [{"food": "Arroz", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.75}]
    assert has_severe(divs) is True


def test_severe_helper_does_not_flag_minor(helper_callable):
    """delta_pct <= 0.50 NO es severe."""
    _, has_severe, _ = helper_callable
    divs = [{"food": "Sal", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.30}]
    assert has_severe(divs) is False


def test_severe_helper_does_not_flag_unknown(helper_callable):
    """`unknown` (food extra en lista, no en recetas) NO es severe."""
    _, has_severe, _ = helper_callable
    divs = [{"food": "Cilantro", "hypothesis": "unknown", "magnitude": False}]
    assert has_severe(divs) is False


def test_summarize_caps_at_max_items(helper_callable):
    """`summarize_divergences_for_ui` respeta max_items."""
    _, _, summarize = helper_callable
    divs = [{"food": f"Item{i}", "hypothesis": "unknown"} for i in range(20)]
    out = summarize(divs, max_items=5)
    assert len(out) == 5
    assert out[0]["food"] == "Item0"
    assert out[4]["food"] == "Item4"


def test_summarize_skips_non_dicts(helper_callable):
    """Resilient a items inválidos (no-dict)."""
    _, _, summarize = helper_callable
    divs = [{"food": "OK", "hypothesis": "unknown"}, "string-malformada", None, {"food": "OK2", "hypothesis": "unknown"}]
    out = summarize(divs, max_items=10)
    assert len(out) == 2
    assert {x["food"] for x in out} == {"OK", "OK2"}


def test_helper_escalates_with_block_severe_only_and_severe_div(monkeypatch, helper_callable):
    """Smoke: con block_severe_only=True + cap_swallowed → block_set=True."""
    helper_fn, _, _ = helper_callable
    # Mock el guard inner para retornar 1 divergencia severa
    import shopping_calculator as sc

    def _fake_guard(plan_result, **_kwargs):
        # Simula que mode=warn no setea el flag por su cuenta
        return [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False}]
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", _fake_guard)
    # Knob está en True por default; aseguramos por env
    monkeypatch.setenv("MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY", "true")

    plan = {"days": [], "aggregated_shopping_list": [], "calc_household_multiplier": 1.0}
    divs, block_set = helper_fn(
        plan,
        mode_override="warn",
        block_severe_only=True,
        action_taken="warn_only_chunk_t2",
    )
    assert len(divs) == 1
    assert block_set is True, (
        "P2-COHERENCE-1 funcional: el helper debió escalar warn→block "
        "(cap_swallowed_modifier es severe + knob ON + block_severe_only=True). "
        "Si block_set quedó False, la escalación no está conectada."
    )
    assert plan.get("_shopping_coherence_block") is True


def test_helper_does_not_escalate_with_minor_div(monkeypatch, helper_callable):
    """Sin divergencias severas, NO escala (warn-only puro)."""
    helper_fn, _, _ = helper_callable
    import shopping_calculator as sc

    def _fake_guard(plan_result, **_kwargs):
        return [{"food": "Cilantro", "hypothesis": "unknown", "magnitude": False}]
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", _fake_guard)
    monkeypatch.setenv("MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY", "true")

    plan = {"days": [], "aggregated_shopping_list": [], "calc_household_multiplier": 1.0}
    _, block_set = helper_fn(
        plan,
        mode_override="warn",
        block_severe_only=True,
        action_taken="warn_only_chunk_t2",
    )
    assert block_set is False
    assert plan.get("_shopping_coherence_block") is None


def test_helper_does_not_escalate_when_knob_off(monkeypatch, helper_callable):
    """Knob OFF → no escala incluso con cap_swallowed_modifier presente
    (rollback rápido sin redeploy)."""
    helper_fn, _, _ = helper_callable
    import shopping_calculator as sc

    def _fake_guard(plan_result, **_kwargs):
        return [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier"}]
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", _fake_guard)
    monkeypatch.setenv("MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY", "false")

    plan = {"days": [], "aggregated_shopping_list": [], "calc_household_multiplier": 1.0}
    _, block_set = helper_fn(
        plan,
        mode_override="warn",
        block_severe_only=True,
        action_taken="warn_only_chunk_t2",
    )
    assert block_set is False, (
        "P2-COHERENCE-1: con knob OFF, el helper NO debe escalar. Si "
        "el knob no se respeta, el rollback queda roto."
    )
