r"""[P1-PANTRY-GUARD-INITIAL-SKIP · 2026-05-18] El pantry guard de la generación
inicial NO debe activarse cuando la nevera del usuario tiene <N items.

Bug reportado por user 2026-05-18 (log local):
    El usuario tenía nevera con ~3 items y dio "regenerar plan". El guard
    rechazaba ingredientes legítimos del plan nuevo (yautía, piña, vainitas,
    res, queso ricotta) porque NO estaban en su nevera de 3 items. Cada
    rechazo disparaba retry del pipeline LLM completo (`run_plan_pipeline`),
    sumando 2-3 ciclos de 405s c/u + costo Gemini ~$0.30 acumulado.

    User: "como voy a llenar la nevera si estoy dandole a evaluar de nuevo?
    no tiene sentido eso. no le debe tomar en cuenta cuando es evaluar de
    nuevo (las dos opciones de evaluar de nuevo)".

Causa raíz arquitectónica:
    En generación inicial o regeneración manual, la lista de compras del plan
    ES LA QUE DEFINE el inventario futuro. El usuario aún no ha comprado.
    Validar el plan contra una nevera casi vacía RECHAZA ingredientes
    legítimos. El guard solo es útil cuando ya existe ciclo de compras vivo
    (nevera poblada) y un swap/refill DEBE respetar lo que el user compró.

Fix:
    Knob `MEALFIT_PANTRY_GUARD_MIN_ITEMS` (default 10) en constants.py. Si
    `len(pantry_ingredients) < PANTRY_GUARD_MIN_ITEMS`, el guard hace
    early-return igual que cuando la nevera está completamente vacía.

    El path `_run_pantry_validation_for_initial_chunk` (routers/plans.py:1000)
    aplica el corto-circuito y registra log explicativo del SKIP para
    operadores que quieran auditar.

Cobertura:
    1. Knob existe en constants.py con clamp razonable.
    2. Caller en routers/plans.py importa y aplica el knob.
    3. Marker P1-PANTRY-GUARD-INITIAL-SKIP anclado para drift detection.
    4. Funcional: con nevera de 3 items, el caller retorna result sin
       invocar `_validate_and_retry_initial_chunk_against_pantry`.
    5. Funcional: con nevera de 15 items (≥10), el caller SÍ invoca la
       validación (comportamiento legacy preservado).
    6. Funcional: con nevera vacía, sigue el corto-circuito original.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CONSTANTS = (_BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
_ROUTERS_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Parser-based
# ─────────────────────────────────────────────────────────────────────────────


def test_marker_present_in_constants():
    assert "P1-PANTRY-GUARD-INITIAL-SKIP" in _CONSTANTS, (
        "Marker P1-PANTRY-GUARD-INITIAL-SKIP ausente en constants.py. "
        "Un revert silente reintroduciría los retries inútiles."
    )


def test_marker_present_in_routers_plans():
    assert "P1-PANTRY-GUARD-INITIAL-SKIP" in _ROUTERS_PLANS, (
        "Marker P1-PANTRY-GUARD-INITIAL-SKIP ausente en routers/plans.py. "
        "El caller debe citar el marker en el corto-circuito."
    )


def test_knob_defined_with_clamp():
    """El knob PANTRY_GUARD_MIN_ITEMS debe existir con clamp [0, 500] y default 10."""
    match = re.search(
        r"PANTRY_GUARD_MIN_ITEMS\s*=\s*max\(0,\s*min\(500,\s*int\(os\.environ\.get\([\"']MEALFIT_PANTRY_GUARD_MIN_ITEMS[\"'],\s*[\"'](\d+)[\"']\)\)\)",
        _CONSTANTS,
    )
    assert match, (
        "Knob PANTRY_GUARD_MIN_ITEMS no encontrado con shape `max(0, min(500, ...))`. "
        "Sin clamp, valores patológicos no se protegen."
    )
    default_value = int(match.group(1))
    assert default_value == 10, (
        f"Default debería ser 10 (≈'nevera mínimamente poblada'), "
        f"actual: {default_value}. Si cambió, revisar la racionalización."
    )


def test_caller_applies_knob_before_full_validation():
    """El caller `_run_pantry_validation_for_initial_chunk` debe leer
    PANTRY_GUARD_MIN_ITEMS y aplicar early-return ANTES de invocar
    `_validate_and_retry_initial_chunk_against_pantry`."""
    # Extraer body del caller.
    m = re.search(
        r"def _run_pantry_validation_for_initial_chunk\([\s\S]+?(?=\ndef |\Z)",
        _ROUTERS_PLANS,
    )
    assert m, "_run_pantry_validation_for_initial_chunk no encontrado"
    body = m.group(0)

    # Anchor del check: importa PANTRY_GUARD_MIN_ITEMS de constants.
    assert "PANTRY_GUARD_MIN_ITEMS" in body, (
        "Caller no usa PANTRY_GUARD_MIN_ITEMS. El bug del user reaparece para "
        "neveras con 1-9 items."
    )

    # Anchor del early-return: `if len(pantry_ingredients) < _PANTRY_MIN:` ANTES
    # del try block que llama _validate_and_retry_initial_chunk_against_pantry.
    early_return_idx = body.find("len(pantry_ingredients) <")
    full_validation_idx = body.find("_validate_and_retry_initial_chunk_against_pantry")
    assert early_return_idx > 0, (
        "Caller no tiene `len(pantry_ingredients) < N` check. "
        "El early-return del skip no está implementado."
    )
    assert early_return_idx < full_validation_idx, (
        "El check de longitud DEBE estar antes de la invocación a "
        "_validate_and_retry_initial_chunk_against_pantry. Sino, el retry "
        "loop se ejecuta incluso con neveras casi vacías."
    )


def test_caller_logs_skip_for_diagnosis():
    """El skip DEBE loggear (INFO) para que SRE pueda auditar cuándo el guard
    se está saltando — útil para detectar misconfigs del knob o usuarios que
    están en loop infinito de generación sin llenar nevera."""
    m = re.search(
        r"def _run_pantry_validation_for_initial_chunk\([\s\S]+?(?=\ndef |\Z)",
        _ROUTERS_PLANS,
    )
    body = m.group(0)
    assert "Pantry guard skip" in body or "SKIP] Pantry guard" in body, (
        "El skip no emite log INFO. Sin observabilidad, un knob mal calibrado "
        "(e.g. PANTRY_GUARD_MIN_ITEMS=1000) skipearía silenciosamente todas "
        "las validaciones legítimas."
    )


def test_marker_bumped_in_app_py():
    """El marker P-fix global debe estar bumpeado al cierre de este fix."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"',
        _APP_PY,
    )
    assert m
    marker = m.group(1)
    # Aceptamos cualquier marker de fecha 2026-05-18 o posterior (puede haber
    # un bundle posterior que sobrescriba).
    date_match = re.search(r"·\s*(\d{4}-\d{2}-\d{2})$", marker)
    assert date_match, f"Marker sin fecha: {marker!r}"
    from datetime import date
    assert date.fromisoformat(date_match.group(1)) >= date(2026, 5, 18), (
        f"Marker stale: {marker!r}. Este fix es 2026-05-18."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Funcional: simular escenarios con mocks
# ─────────────────────────────────────────────────────────────────────────────


def test_skip_when_pantry_below_threshold():
    """Con nevera de 3 items (< default 10), el guard debe hacer early-return
    SIN invocar `_validate_and_retry_initial_chunk_against_pantry`."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": [{"meals": [{"name": "test"}]}]}

    with patch(
        "cron_tasks._validate_and_retry_initial_chunk_against_pantry"
    ) as mock_validate:
        result = _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user-uuid",
            pantry_ingredients=["pollo", "arroz", "tomate"],  # 3 items < 10
            transport_label="TEST-SKIP",
        )

    assert result is fake_result, (
        "Caller debió retornar el result intacto sin modificar."
    )
    mock_validate.assert_not_called(), (
        "Caller invocó _validate_and_retry_initial_chunk_against_pantry "
        "pese a nevera < threshold. El skip NO se está ejecutando."
    )


def test_validation_runs_when_pantry_above_threshold():
    """Con nevera de 15 items (≥ default 10), el guard SÍ debe invocar
    `_validate_and_retry_initial_chunk_against_pantry` (comportamiento legacy)."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": [{"meals": [{"name": "test"}]}]}
    fake_pantry = [f"item-{i}" for i in range(15)]  # 15 items >= 10

    with patch(
        "cron_tasks._validate_and_retry_initial_chunk_against_pantry",
        return_value=(fake_result, {"validated_ok": True, "attempts": 1, "degraded": False,
                                     "last_violation": None, "mode": "advisory",
                                     "pantry_size": 15, "missing_list": []}),
    ) as mock_validate:
        _result = _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user-uuid",
            pantry_ingredients=fake_pantry,
            transport_label="TEST-VALIDATE",
        )

    mock_validate.assert_called_once(), (
        "Con nevera ≥10 items el guard debe invocar la validación. "
        "Si skip, perdemos el comportamiento legacy de proteger swaps/refills "
        "contra ingredientes no comprados."
    )


def test_skip_when_pantry_empty():
    """Comportamiento original preservado: nevera vacía → early-return."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": [{"meals": [{"name": "test"}]}]}

    with patch(
        "cron_tasks._validate_and_retry_initial_chunk_against_pantry"
    ) as mock_validate:
        result = _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user-uuid",
            pantry_ingredients=[],  # vacía
            transport_label="TEST-EMPTY",
        )

    assert result is fake_result
    mock_validate.assert_not_called()


def test_knob_can_disable_guard_entirely_by_raising_threshold():
    """Si se sube el knob a 1000, ninguna nevera realista (typical 15-50 items)
    activa el guard. Útil para debugging o emergencia operacional."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": []}
    fake_pantry_50 = [f"item-{i}" for i in range(50)]  # 50 items, normalmente activa guard

    with patch("constants.PANTRY_GUARD_MIN_ITEMS", 1000), \
         patch(
            "cron_tasks._validate_and_retry_initial_chunk_against_pantry"
        ) as mock_validate:
        _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user",
            pantry_ingredients=fake_pantry_50,
            transport_label="TEST-DISABLED",
        )

    mock_validate.assert_not_called(), (
        "Con PANTRY_GUARD_MIN_ITEMS=1000 y nevera de 50, el guard debió skipear. "
        "Knob no se está leyendo correctamente."
    )
