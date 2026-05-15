"""[P3-FACT-SHADOW-AB · 2026-05-14] Tests para el shadow A/B PRO→FLASH
en `fact_extractor.py`.

Objetivo del feature:
  Los 2 callsites de structured extraction (`extract_facts` line ~190 +
  `_run_fact_pipeline` batch contradiction line ~350) estaban hardcoded
  a `gemini-3.1-pro-preview`. Los `with_structured_output(...)` constrain
  el output al schema — históricamente FLASH iguala PRO en ese sweet spot,
  pero migrar a ciegas arriesga regresiones silenciosas (omisión de
  facts, canonicalización mala, merges incorrectos).

  Mecánica: el helper `_invoke_with_shadow` corre PRO sync (UX truth)
  y, si knob activado + user pasa sampling determinístico, dispara FLASH
  en daemon thread. Compara outputs estructuralmente y persiste a
  `pipeline_metrics(node='fact_extractor_shadow_diff')`. Cero impacto UX.

Cobertura:
  - test_knobs_registered_in_registry
  - test_should_run_shadow_off_when_knob_empty
  - test_should_run_shadow_off_when_no_user_id
  - test_should_run_shadow_off_when_sample_rate_zero
  - test_should_run_shadow_on_when_sample_rate_one
  - test_should_run_shadow_deterministic_by_user_id
  - test_diff_facts_model_exact_match
  - test_diff_facts_model_count_mismatch
  - test_diff_facts_model_category_mismatch
  - test_diff_contradiction_result_exact_match
  - test_diff_contradiction_result_ids_mismatch
  - test_extract_facts_signature_has_user_id_param
  - test_extract_facts_uses_invoke_with_shadow
  - test_run_fact_pipeline_uses_invoke_with_shadow
  - test_async_extract_propagates_user_id
  - test_process_single_extraction_propagates_user_id
  - test_no_inline_pro_model_in_refactored_callsites
"""
import importlib
import inspect
import re
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def fact_extractor():
    """Import del módulo bajo test (recargado para asegurar default knobs)."""
    import fact_extractor as fe
    return fe


# ---------------------------------------------------------------------------
# 1. Knobs auto-registrados en _KNOBS_REGISTRY.
# ---------------------------------------------------------------------------
def test_knobs_registered_in_registry(fact_extractor, monkeypatch):
    """Los 2 knobs nuevos deben aparecer en `_KNOBS_REGISTRY` tras import."""
    # Reload con env limpio para verificar registro.
    monkeypatch.delenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", raising=False)
    importlib.reload(fact_extractor)

    from knobs import _KNOBS_REGISTRY

    assert "MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL" in _KNOBS_REGISTRY, (
        "Knob MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL no registrado en _KNOBS_REGISTRY"
    )
    assert "MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE" in _KNOBS_REGISTRY, (
        "Knob MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE no registrado en _KNOBS_REGISTRY"
    )
    # Defaults esperados.
    assert _KNOBS_REGISTRY["MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL"]["default"] == ""
    assert _KNOBS_REGISTRY["MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE"]["default"] == 0.1


# ---------------------------------------------------------------------------
# 2. `_should_run_shadow` — gate de sampling determinístico.
# ---------------------------------------------------------------------------
def test_should_run_shadow_off_when_knob_empty(fact_extractor, monkeypatch):
    """Knob vacío (default) → shadow off, sin importar user_id."""
    monkeypatch.delenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", raising=False)
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", "1.0")
    importlib.reload(fact_extractor)
    assert fact_extractor._should_run_shadow("any-user-id") is False
    assert fact_extractor._should_run_shadow(None) is False


def test_should_run_shadow_off_when_no_user_id(fact_extractor, monkeypatch):
    """Sin user_id no podemos samplear estable → shadow off."""
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", "1.0")
    importlib.reload(fact_extractor)
    assert fact_extractor._should_run_shadow(None) is False
    assert fact_extractor._should_run_shadow("") is False


def test_should_run_shadow_off_when_sample_rate_zero(fact_extractor, monkeypatch):
    """Sample 0 → todos los users skip."""
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", "0.0")
    importlib.reload(fact_extractor)
    assert fact_extractor._should_run_shadow("user-abc") is False


def test_should_run_shadow_on_when_sample_rate_one(fact_extractor, monkeypatch):
    """Sample 1.0 + knob set + user_id → siempre on."""
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", "1.0")
    importlib.reload(fact_extractor)
    assert fact_extractor._should_run_shadow("user-abc") is True
    assert fact_extractor._should_run_shadow("user-xyz") is True


def test_should_run_shadow_deterministic_by_user_id(fact_extractor, monkeypatch):
    """Mismo user_id → mismo resultado en múltiples llamadas (stable A/B)."""
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE", "0.5")
    importlib.reload(fact_extractor)
    # Mismo input → mismo output (determinismo SHA-256).
    for uid in ("user-a", "user-b", "user-c", "user-d"):
        first = fact_extractor._should_run_shadow(uid)
        for _ in range(5):
            assert fact_extractor._should_run_shadow(uid) == first


# ---------------------------------------------------------------------------
# 3. Diff functions — la lógica de comparación.
# ---------------------------------------------------------------------------
def _make_fact(category, ingrediente_canonico):
    """Helper para fabricar un FactItem mock."""
    md = MagicMock()
    md.category = category
    md.ingrediente_canonico = ingrediente_canonico
    f = MagicMock()
    f.metadata = md
    return f


def _make_facts_result(items):
    r = MagicMock()
    r.facts = items
    return r


def test_diff_facts_model_exact_match(fact_extractor):
    """Mismo conjunto de (category, canon) → exact match."""
    pro = _make_facts_result([
        _make_fact("alergia", "peanut"),
        _make_fact("preferencia", "chicken"),
    ])
    flash = _make_facts_result([
        _make_fact("alergia", "peanut"),
        _make_fact("preferencia", "chicken"),
    ])
    diff = fact_extractor._diff_facts_model(pro, flash)
    assert diff["count_match"] is True
    assert diff["category_canonico_set_match"] is True
    assert diff["only_in_pro"] == []
    assert diff["only_in_flash"] == []


def test_diff_facts_model_count_mismatch(fact_extractor):
    """Diferente cantidad de facts → count_match=False."""
    pro = _make_facts_result([
        _make_fact("alergia", "peanut"),
        _make_fact("preferencia", "chicken"),
    ])
    flash = _make_facts_result([_make_fact("alergia", "peanut")])
    diff = fact_extractor._diff_facts_model(pro, flash)
    assert diff["pro_count"] == 2
    assert diff["flash_count"] == 1
    assert diff["count_match"] is False
    assert diff["only_in_pro"] == [["preferencia", "chicken"]]


def test_diff_facts_model_category_mismatch(fact_extractor):
    """Mismo count pero diferentes categorías → set_match=False."""
    pro = _make_facts_result([_make_fact("alergia", "peanut")])
    flash = _make_facts_result([_make_fact("preferencia", "peanut")])
    diff = fact_extractor._diff_facts_model(pro, flash)
    assert diff["count_match"] is True
    assert diff["category_canonico_set_match"] is False


def _make_contradiction(new_fact, ids_to_delete):
    c = MagicMock()
    c.new_fact = new_fact
    c.ids_to_delete = ids_to_delete
    return c


def _make_merge(merged, ids_to_delete, skip_new):
    m = MagicMock()
    m.merged_fact = merged
    m.ids_to_delete = ids_to_delete
    m.skip_new_fact = skip_new
    return m


def _make_contradiction_result(contradictions, merges):
    r = MagicMock()
    r.contradictions = contradictions
    r.merges = merges
    return r


def test_diff_contradiction_result_exact_match(fact_extractor):
    """Mismo set de ids_to_delete → match."""
    pro = _make_contradiction_result(
        [_make_contradiction("nuevo", ["id-1", "id-2"])],
        [_make_merge("fusionado", ["id-3"], "skip-text")],
    )
    flash = _make_contradiction_result(
        [_make_contradiction("nuevo", ["id-1", "id-2"])],
        [_make_merge("fusionado", ["id-3"], "skip-text")],
    )
    diff = fact_extractor._diff_contradiction_result(pro, flash)
    assert diff["pro_contradictions_count"] == 1
    assert diff["flash_contradictions_count"] == 1
    assert diff["ids_to_delete_set_match"] is True
    assert diff["only_in_pro_ids"] == []
    assert diff["only_in_flash_ids"] == []


def test_diff_contradiction_result_ids_mismatch(fact_extractor):
    """ids diferentes → set_match=False + only_in_* poblados."""
    pro = _make_contradiction_result(
        [_make_contradiction("nuevo", ["id-1", "id-2"])],
        [],
    )
    flash = _make_contradiction_result(
        [_make_contradiction("nuevo", ["id-1", "id-99"])],
        [],
    )
    diff = fact_extractor._diff_contradiction_result(pro, flash)
    assert diff["ids_to_delete_set_match"] is False
    assert "id-2" in diff["only_in_pro_ids"]
    assert "id-99" in diff["only_in_flash_ids"]


# ---------------------------------------------------------------------------
# 4. Parser-based: ambos callsites usan el helper, no ChatGoogleGenerativeAI inline.
# ---------------------------------------------------------------------------
def test_extract_facts_signature_has_user_id_param(fact_extractor):
    """`extract_facts` debe aceptar `user_id` (Optional) para habilitar
    sampling determinístico del shadow."""
    sig = inspect.signature(fact_extractor.extract_facts)
    assert "user_id" in sig.parameters, (
        "extract_facts debe tener parámetro user_id para sampling determinístico"
    )
    # Default debe ser None (callers legacy sin user_id siguen funcionando).
    assert sig.parameters["user_id"].default is None


def test_extract_facts_uses_invoke_with_shadow(fact_extractor):
    """El cuerpo de `extract_facts` debe invocar `_invoke_with_shadow`
    en lugar de hardcodear `ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview"`."""
    src = inspect.getsource(fact_extractor.extract_facts)
    assert "_invoke_with_shadow" in src, (
        "extract_facts debe usar _invoke_with_shadow para activar el A/B"
    )
    assert 'callsite_tag="extract_facts"' in src, (
        "callsite_tag debe ser 'extract_facts' para distinguir en pipeline_metrics"
    )


def test_run_fact_pipeline_uses_invoke_with_shadow(fact_extractor):
    """`_run_fact_pipeline` (batch contradiction) debe usar el helper también."""
    src = inspect.getsource(fact_extractor._run_fact_pipeline)
    assert "_invoke_with_shadow" in src, (
        "_run_fact_pipeline debe usar _invoke_with_shadow para el batch contradiction"
    )
    assert 'callsite_tag="contradiction_merge"' in src, (
        "callsite_tag debe ser 'contradiction_merge' para distinguir en pipeline_metrics"
    )


def test_no_inline_pro_model_in_refactored_callsites(fact_extractor):
    """Ninguno de los 2 callsites refactorizados debe tener
    `ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview"` literal — toda
    llamada PRO debe pasar por el helper para que el shadow funcione."""
    extract_src = inspect.getsource(fact_extractor.extract_facts)
    pipeline_src = inspect.getsource(fact_extractor._run_fact_pipeline)

    pattern = r'ChatGoogleGenerativeAI\s*\([^)]*model\s*=\s*[\'"]gemini-3\.1-pro-preview[\'"]'
    extract_hits = re.findall(pattern, extract_src, re.DOTALL)
    pipeline_hits = re.findall(pattern, pipeline_src, re.DOTALL)

    assert not extract_hits, (
        f"extract_facts: PRO inline detectado, debe usar helper. Matches: {extract_hits}"
    )
    assert not pipeline_hits, (
        f"_run_fact_pipeline: PRO inline detectado, debe usar helper. Matches: {pipeline_hits}"
    )


def test_async_extract_propagates_user_id(fact_extractor):
    """`async_extract_and_save_facts` debe pasar `user_id=user_id` cuando
    invoca `extract_facts` — sin eso el shadow nunca corre."""
    src = inspect.getsource(fact_extractor.async_extract_and_save_facts)
    # Buscar el call a extract_facts con user_id keyword.
    pattern = r'extract_facts\s*\([^)]*user_id\s*=\s*user_id'
    assert re.search(pattern, src, re.DOTALL), (
        "async_extract_and_save_facts debe pasar user_id=user_id a extract_facts"
    )


def test_process_single_extraction_propagates_user_id(fact_extractor):
    """`_process_single_extraction` (drena cola pending) también debe
    propagar user_id."""
    src = inspect.getsource(fact_extractor._process_single_extraction)
    pattern = r'extract_facts\s*\([^)]*user_id\s*=\s*user_id'
    assert re.search(pattern, src, re.DOTALL), (
        "_process_single_extraction debe pasar user_id=user_id a extract_facts"
    )
