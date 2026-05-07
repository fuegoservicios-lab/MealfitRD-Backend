"""[P0-5] Per-meal pantry violation marker para chunk 1 sync.

Contexto:
    Chunks 2+ (worker async) PAUSAN a `pending_user_action` cuando los retries
    de pantry validation se agotan, refusing-to-serve. Eso honra estrictamente
    el contrato "platos solo con alimentos de la nevera".

    Chunk 1 (sync) NO pausa porque el usuario está esperando síncrono. Antes
    del fix, simplemente seteaba el flag plan-level
    `_initial_chunk_pantry_degraded=True` y servía el plan completo, dejando al
    frontend con un warning genérico sin saber QUÉ platos exactos son
    incocibles.

    Después del fix (`_mark_meals_violating_pantry`):
        - Cada comida cuyos ingredientes NO estén en pantry queda marcada con
          `_pantry_violated=True` y `_pantry_violated_reason=<violation string>`.
        - El audit retorna `meals_marked_violated=<count>` para telemetría.
        - El frontend puede renderizar warning específico en cada plato malo.

NOTA SOBRE MOCKING:
    El validador real `validate_ingredients_against_pantry` (constants.py) usa
    matching semántico (Vector Search, fuzzy, sinónimos), por lo que no es
    determinístico para tests aislados — "salmón" puede aceptarse como variante
    de algo en pantry según el estado del embedding cache. Estos tests MOCKEAN
    el validador para verificar la LÓGICA DEL HELPER (iteración por meals,
    propagación del flag, idempotencia), no la del validador en sí. El
    validador tiene sus propios tests en `test_pantry_validation_runs_in_llm_path.py`.
"""
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vip_mock_factory(invalid_ingredients: set):
    """Factory para crear un mock de _vip que rechaza si CUALQUIER ingrediente
    pertenece a `invalid_ingredients`.

    Replica el contrato real: True si todo OK, string con violaciones si no.
    """
    def _mock(generated, pantry, strict_quantities=True, tolerance=1.30):
        bad = [g for g in generated if g in invalid_ingredients]
        if not bad:
            return True
        return f"INEXISTENTES: {', '.join(bad)}."
    return _mock


# ---------------------------------------------------------------------------
# Tests del helper aislado
# ---------------------------------------------------------------------------

def test_mark_meals_violating_pantry_marks_offending_meals_only():
    """Comidas con ingredientes no-pantry → marcadas. Comidas con todo en
    pantry → intactas. Mutación in-place."""
    from cron_tasks import _mark_meals_violating_pantry

    result = {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "OK", "ingredients": ["pollo", "arroz"]},
                    {"name": "BAD-salmon", "ingredients": ["salmón", "arroz"]},
                ],
            },
            {
                "day": 2,
                "meals": [
                    {"name": "BAD-quinoa", "ingredients": ["quinoa", "pollo"]},
                ],
            },
        ]
    }
    pantry = ["pollo", "arroz", "habichuelas"]
    invalid = {"salmón", "quinoa"}

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory(invalid),
    ):
        marked = _mark_meals_violating_pantry(result, pantry)

    assert marked == 2, f"Esperaba 2 comidas marcadas, obtuve {marked}"

    # Día 1, comida 0: OK, sin marcar
    m_ok = result["days"][0]["meals"][0]
    assert "_pantry_violated" not in m_ok

    # Día 1, comida 1: salmón → marcada con reason que menciona el ingrediente
    m_bad1 = result["days"][0]["meals"][1]
    assert m_bad1.get("_pantry_violated") is True
    assert "salm" in m_bad1.get("_pantry_violated_reason", "").lower()

    # Día 2, comida 0: quinoa → marcada
    m_bad2 = result["days"][1]["meals"][0]
    assert m_bad2.get("_pantry_violated") is True
    assert "quinoa" in m_bad2.get("_pantry_violated_reason", "").lower()


def test_mark_meals_violating_pantry_returns_zero_when_all_valid():
    """Sin violaciones → 0 marcadas, ninguna comida modificada."""
    from cron_tasks import _mark_meals_violating_pantry

    result = {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "Pollo", "ingredients": ["pollo", "arroz"]},
                    {"name": "Arroz blanco", "ingredients": ["arroz"]},
                ],
            }
        ]
    }
    pantry = ["pollo", "arroz", "habichuelas", "res"]

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory(set()),  # nada inválido
    ):
        marked = _mark_meals_violating_pantry(result, pantry)

    assert marked == 0
    for m in result["days"][0]["meals"]:
        assert "_pantry_violated" not in m


def test_mark_meals_violating_pantry_handles_empty_pantry_safely():
    """Pantry vacío → no marca nada (early-return sin invocar validador)."""
    from cron_tasks import _mark_meals_violating_pantry

    result = {"days": [{"day": 1, "meals": [{"ingredients": ["foo"]}]}]}
    assert _mark_meals_violating_pantry(result, []) == 0
    assert _mark_meals_violating_pantry(result, None) == 0
    assert "_pantry_violated" not in result["days"][0]["meals"][0]


def test_mark_meals_violating_pantry_handles_malformed_input():
    """Inputs malformados (no dict, days no list, meal no dict, ingredients no
    list) NO deben raisar — el guardrail tiene que ser tolerante."""
    from cron_tasks import _mark_meals_violating_pantry

    pantry = ["pollo"]
    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory({"foo"}),
    ):
        # No dict
        assert _mark_meals_violating_pantry("not a dict", pantry) == 0
        assert _mark_meals_violating_pantry(None, pantry) == 0
        # days no es lista
        assert _mark_meals_violating_pantry({"days": "nope"}, pantry) == 0
        # day no es dict
        assert _mark_meals_violating_pantry({"days": ["string"]}, pantry) == 0
        # meal no es dict
        assert _mark_meals_violating_pantry(
            {"days": [{"meals": ["string"]}]}, pantry
        ) == 0
        # ingredients no es lista
        res = {"days": [{"meals": [{"ingredients": None}]}]}
        assert _mark_meals_violating_pantry(res, pantry) == 0
        # ingredients vacíos
        res = {"days": [{"meals": [{"ingredients": []}]}]}
        assert _mark_meals_violating_pantry(res, pantry) == 0


# ---------------------------------------------------------------------------
# Tests de integración con _validate_and_retry_initial_chunk_against_pantry
# ---------------------------------------------------------------------------

def _build_result_with_violation():
    """Result con un plato OK y otro que viola pantry."""
    return {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "OK", "ingredients": ["pollo", "arroz"]},
                    {"name": "BAD", "ingredients": ["salmón", "arroz"]},
                ],
            }
        ]
    }


def test_initial_validator_marks_meals_when_retries_exhausted():
    """[P0-5] Path: existencia falla, max_attempts=0 → degraded=True con
    per-meal markers."""
    from cron_tasks import _validate_and_retry_initial_chunk_against_pantry

    pantry = ["pollo", "arroz"]
    initial = _build_result_with_violation()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory({"salmón"}),
    ), patch(
        "cron_tasks.run_plan_pipeline",
        side_effect=AssertionError("no debería llamarse"),
    ), patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
        final, audit = _validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id="test-user",
        )

    assert audit.get("degraded") is True
    assert audit.get("meals_marked_violated") == 1, (
        f"Esperaba 1 comida marcada, audit={audit}"
    )
    assert final["days"][0]["meals"][0].get("_pantry_violated") is None
    assert final["days"][0]["meals"][1].get("_pantry_violated") is True


def test_initial_validator_does_not_mark_when_validation_passes():
    """Si el plan original NO viola pantry → no marker, no degraded."""
    from cron_tasks import _validate_and_retry_initial_chunk_against_pantry

    pantry = ["pollo", "arroz", "salmón"]
    initial = _build_result_with_violation()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory(set()),  # todo OK
    ), patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
        final, audit = _validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id="test-user",
        )

    assert audit.get("validated_ok") is True
    assert audit.get("degraded") is False
    assert audit.get("meals_marked_violated") is None
    for m in final["days"][0]["meals"]:
        assert "_pantry_violated" not in m


def test_initial_validator_marks_in_off_mode_no_retry():
    """[P0-5] qty_mode=off + violación → degrada SIN reintentar pero MARCA per-meal."""
    from cron_tasks import _validate_and_retry_initial_chunk_against_pantry

    pantry = ["pollo", "arroz"]
    initial = _build_result_with_violation()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory({"salmón"}),
    ), patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", "off"), patch(
        "cron_tasks.run_plan_pipeline",
        side_effect=AssertionError("no debería llamarse en modo off"),
    ):
        final, audit = _validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id="test-user",
        )

    assert audit.get("degraded") is True
    assert audit.get("meals_marked_violated") == 1
    assert final["days"][0]["meals"][1].get("_pantry_violated") is True


def test_initial_validator_marks_when_pipeline_exception_during_retry():
    """[P0-5] Si el pipeline LLM raisa durante un retry de existencia, el
    `current_result` retornado tiene violaciones de existencia → marcar."""
    from cron_tasks import _validate_and_retry_initial_chunk_against_pantry

    pantry = ["pollo", "arroz"]
    initial = _build_result_with_violation()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_mock_factory({"salmón"}),
    ), patch(
        "cron_tasks.run_plan_pipeline", side_effect=RuntimeError("LLM down")
    ), patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 2):
        final, audit = _validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id="test-user",
        )

    assert audit.get("degraded") is True
    assert audit.get("meals_marked_violated") == 1
    assert final["days"][0]["meals"][1].get("_pantry_violated") is True
