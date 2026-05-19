r"""[P1-PANTRY-GUARD-REGEN-SKIP · 2026-05-18] El pantry guard de la generación
inicial DEBE saltarse cuando el cliente envía `update_reason` truthy.

Contexto:
    El fix P1-PANTRY-GUARD-INITIAL-SKIP (mañana del 2026-05-18) introdujo un
    threshold `PANTRY_GUARD_MIN_ITEMS=10` que skipea el guard cuando la nevera
    tiene menos items que el threshold. Pero esto seguía gateando por COUNT,
    no por INTENT: si el usuario con nevera llena (≥10 items) clickeaba
    "Renovar Plan Actual" o "Actualizar plan", el guard SÍ corría y podía
    degradar/retry el chunk porque rechazaba ingredientes del plan nuevo
    contra la nevera vieja.

    User reportó tras ver la descripción del fix matutino:
        "estos dos botones no deben tomar en cuenta si hay alimentos en la
        nevera ya que es para cambiar los alimentos, lo de la nevera es para
        cuando se actualizan platos"

Causa raíz arquitectónica:
    El threshold por count es un proxy POOR de "intent de regen". El flag
    correcto es la presencia de `update_reason` en el payload del SSE:

    - frontend/src/hooks/useRegeneratePlan.js inyecta `update_reason: reason`
      al navigate('/plan', {state: ...}) cada vez que el user clickea
      "Renovar Plan Actual" (Settings) o "Actualizar plan" (Dashboard).
    - frontend/src/pages/Plan.jsx:620 forwarda el reason al payload del SSE.
    - Reasons posibles: 'variety', 'time', 'budget', 'cravings', 'weekend',
      'similar', 'dislike' (todos full-plan regen via SSE).

    En generación inicial first-time o single-meal swap (otro endpoint),
    `update_reason` es None/ausente → guard se evalúa por el path original
    (vacío → skip, <10 → skip, ≥10 → valida).

Fix:
    `_run_pantry_validation_for_initial_chunk` ahora acepta `update_reason:
    Optional[str] = None`. Al inicio del cuerpo:

        if update_reason:
            logger.info("⏭️ [SKIP-REGEN] ...")
            return result

    Los 2 callsites (sync `/api/plans/analyze` y SSE `/api/plans/analyze/stream`)
    propagan `update_reason=data.get("update_reason")`.

Cobertura:
    1. Marker presente en el archivo de producción.
    2. Signature del helper acepta `update_reason: Optional[str]`.
    3. Branch de skip por update_reason existe ANTES del resto de checks.
    4. Los 2 callsites propagan `update_reason=data.get("update_reason")`.
    5. Funcional: con update_reason='variety' y nevera llena (15 items), el
       guard NO invoca `_validate_and_retry_initial_chunk_against_pantry`.
    6. Funcional: con update_reason='dislike' (otra reason), también skip.
    7. Funcional: con update_reason=None y nevera llena, el guard SÍ valida
       (comportamiento legacy del path "first-time generation con pantry
       manualmente poblada" preservado).
    8. Funcional: con update_reason='' (string vacío, falsy), NO skip
       — solo truthy gatea.
    9. Cross-link con marker bumpeado en app.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ROUTERS_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Parser-based
# ─────────────────────────────────────────────────────────────────────────────


def test_marker_present_in_routers_plans():
    assert "P1-PANTRY-GUARD-REGEN-SKIP" in _ROUTERS_PLANS, (
        "Marker P1-PANTRY-GUARD-REGEN-SKIP ausente en routers/plans.py. "
        "Un revert silente reintroduciría el guard sobre Renovar/Actualizar."
    )


def test_helper_signature_accepts_update_reason():
    """`_run_pantry_validation_for_initial_chunk` debe aceptar el kwarg
    `update_reason: Optional[str] = None` para que los callsites lo puedan
    propagar sin romper signatures."""
    m = re.search(
        r"def _run_pantry_validation_for_initial_chunk\(([\s\S]+?)\)\s*->",
        _ROUTERS_PLANS,
    )
    assert m, "Signature del helper no encontrada"
    signature = m.group(1)
    assert "update_reason" in signature, (
        "Signature no acepta `update_reason`. El kwarg debe declararse para "
        "permitir el skip explícito por intent."
    )
    # Default debe ser None — sino guards de tests/legacy callers se rompen.
    assert re.search(r"update_reason\s*:\s*Optional\[str\]\s*=\s*None", signature), (
        "`update_reason` debe declararse como `Optional[str] = None`. Sin "
        "default, llamadas legacy que NO pasan update_reason romperían."
    )


def test_skip_branch_runs_before_pantry_size_checks():
    """El branch `if update_reason: return result` DEBE estar antes del
    branch `if not pantry_ingredients` y antes del threshold check. Si va
    después, el threshold lo cortocircuita y el skip por intent nunca aplica
    cuando pantry está vacía o tiene <10 items."""
    m = re.search(
        r"def _run_pantry_validation_for_initial_chunk\([\s\S]+?(?=\ndef |\Z)",
        _ROUTERS_PLANS,
    )
    body = m.group(0)

    regen_skip_idx = body.find("if update_reason:")
    empty_skip_idx = body.find("if not pantry_ingredients:")
    # Anchor del threshold check: el `from constants import` que carga el knob.
    # Usar este en vez de `PANTRY_GUARD_MIN_ITEMS` porque el docstring del
    # helper también menciona el nombre del knob.
    threshold_idx = body.find("from constants import PANTRY_GUARD_MIN_ITEMS")
    # Anchor de la invocación al validador: la línea `result, _initial_audit =`
    # que SOLO existe en el callsite real, no en docstring.
    full_validation_idx = body.find("result, _initial_audit = _validate_and_retry_initial_chunk_against_pantry")

    assert regen_skip_idx > 0, (
        "Branch `if update_reason:` no encontrado. El skip por intent no "
        "está implementado."
    )
    assert regen_skip_idx < empty_skip_idx, (
        "El skip por update_reason DEBE evaluarse antes del check de pantry "
        "vacía. Si va después, una nevera vacía cortocircuita el path y el "
        "marker SKIP-REGEN no se loggea — observabilidad rota."
    )
    assert threshold_idx > 0, (
        "Anchor `from constants import PANTRY_GUARD_MIN_ITEMS` no encontrado. "
        "El threshold check del fix P1-PANTRY-GUARD-INITIAL-SKIP fue removido?"
    )
    assert regen_skip_idx < threshold_idx, (
        "El skip por update_reason DEBE evaluarse antes del threshold check."
    )
    assert full_validation_idx > 0, (
        "Invocación a `_validate_and_retry_initial_chunk_against_pantry` no "
        "encontrada. El path completo de validación fue removido?"
    )
    assert regen_skip_idx < full_validation_idx, (
        "El skip por update_reason DEBE estar antes de la invocación a "
        "_validate_and_retry_initial_chunk_against_pantry."
    )


def test_skip_branch_logs_for_observability():
    """El skip-regen DEBE emitir log INFO. Sin observabilidad un operador no
    puede auditar qué fracción de planes están saltando guard por intent
    (signal valioso si queremos calibrar el threshold por count en el
    futuro)."""
    m = re.search(
        r"def _run_pantry_validation_for_initial_chunk\([\s\S]+?(?=\ndef |\Z)",
        _ROUTERS_PLANS,
    )
    body = m.group(0)
    # Anchor explícito del log SKIP-REGEN.
    assert "SKIP-REGEN" in body, (
        "El skip por update_reason no loggea con tag `SKIP-REGEN`. Sin tag "
        "buscable, queries de log y dashboards no pueden separar el skip por "
        "intent del skip por count."
    )


def test_both_callsites_propagate_update_reason():
    """Los 2 callsites (sync + SSE) DEBEN pasar `update_reason=data.get(
    "update_reason")` al helper. Sin propagación, el frontend manda el
    reason pero el backend lo ignora — bug silencioso.

    Sync invoca directamente `_run_pantry_validation_for_initial_chunk(...)`.
    SSE invoca via `asyncio.to_thread(_run_pantry_validation_for_initial_chunk,
    ...)` para no bloquear el event loop. El kwarg debe aparecer en AMBOS.
    """
    # Eliminar la definición del helper (donde el nombre también aparece pero
    # NO es una invocación) para evitar matches falsos.
    body_without_def = re.sub(
        r"def _run_pantry_validation_for_initial_chunk\([\s\S]+?(?=\ndef |\Z)",
        "",
        _ROUTERS_PLANS,
        count=1,
    )

    # Verificación 1: hay ≥2 propagaciones literales del kwarg.
    propagations = re.findall(
        r'update_reason\s*=\s*data\.get\("update_reason"\)',
        body_without_def,
    )
    assert len(propagations) >= 2, (
        f"Se esperaban ≥2 propagaciones de `update_reason=data.get(\"update_reason\")` "
        f"(sync + SSE), encontradas: {len(propagations)}. Si solo hay 1, uno de "
        f"los callsites quedó sin propagar y el frontend manda el reason "
        f"pero el backend lo ignora silenciosamente."
    )

    # Verificación 2: el helper se invoca al menos 2 veces (directo o via
    # to_thread). Anchor: el nombre seguido por `(` (direct) o por `,` (cuando
    # es primer arg posicional de to_thread).
    invocations = re.findall(
        r"_run_pantry_validation_for_initial_chunk[\(,]",
        body_without_def,
    )
    assert len(invocations) >= 2, (
        f"Se esperaban ≥2 invocaciones del helper (sync directo + SSE via "
        f"to_thread), encontradas: {len(invocations)}. El callsite SSE puede "
        f"haber sido removido o el patrón de invocación cambió."
    )


def test_marker_bumped_in_app_py():
    """El marker P-fix global debe reflejar este fix (fecha ≥ 2026-05-18)."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"',
        _APP_PY,
    )
    assert m
    marker = m.group(1)
    date_match = re.search(r"·\s*(\d{4}-\d{2}-\d{2})$", marker)
    assert date_match, f"Marker sin fecha: {marker!r}"
    from datetime import date
    assert date.fromisoformat(date_match.group(1)) >= date(2026, 5, 18), (
        f"Marker stale: {marker!r}. Este fix es 2026-05-18."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Funcional: simular escenarios con mocks
# ─────────────────────────────────────────────────────────────────────────────


def test_skip_when_update_reason_variety_with_full_pantry():
    """Caso principal: user clickea "Renovar Plan Actual" con nevera llena
    (15 items, normalmente activaría el guard). Con update_reason='variety',
    el helper debe skipear sin invocar la validación."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": [{"meals": [{"name": "test"}]}]}
    fake_pantry = [f"item-{i}" for i in range(15)]  # 15 ≥ 10, normalmente activa guard

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
            pantry_ingredients=fake_pantry,
            transport_label="TEST-REGEN-VARIETY",
            update_reason="variety",
        )

    assert result is fake_result, (
        "Helper debió retornar el result intacto."
    )
    mock_validate.assert_not_called(), (
        "Pese a update_reason='variety' + nevera ≥10, el guard SE EJECUTÓ. "
        "El skip por intent NO está funcionando — el user volvería a ver "
        "retries inútiles cuando da 'Renovar'."
    )


def test_skip_when_update_reason_dislike_with_full_pantry():
    """Otras reasons (no solo 'variety') también deben skipear. Coverage de
    todas las opciones del modal Actualizar Plan en Dashboard.jsx."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": []}
    fake_pantry = [f"item-{i}" for i in range(20)]

    for reason in ("variety", "time", "budget", "cravings", "weekend",
                   "similar", "dislike"):
        with patch(
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
                pantry_ingredients=fake_pantry,
                transport_label="TEST",
                update_reason=reason,
            )
        mock_validate.assert_not_called(), (
            f"Reason {reason!r} no activó el skip. Cobertura completa del "
            f"modal Actualizar rota."
        )


def test_validates_when_update_reason_is_none_and_pantry_full():
    """Path legacy preservado: si update_reason=None (first-time generation
    o caller que no propaga el campo), el guard SÍ se aplica cuando hay
    pantry suficiente. Sin esto, perdemos protección sobre el caso edge
    'first plan con pantry manualmente poblada'."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": [{"meals": [{"name": "test"}]}]}
    fake_pantry = [f"item-{i}" for i in range(15)]

    with patch(
        "cron_tasks._validate_and_retry_initial_chunk_against_pantry",
        return_value=(fake_result, {
            "validated_ok": True, "attempts": 1, "degraded": False,
            "last_violation": None, "mode": "advisory",
            "pantry_size": 15, "missing_list": []
        }),
    ) as mock_validate:
        _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user-uuid",
            pantry_ingredients=fake_pantry,
            transport_label="TEST-NO-REASON",
            update_reason=None,
        )

    mock_validate.assert_called_once(), (
        "Sin update_reason el guard debió validar normalmente. Si skipea, "
        "perdemos el path legacy de first-time generation con pantry "
        "pre-poblada (edge case real)."
    )


def test_empty_string_update_reason_does_not_skip():
    """`update_reason=''` (string vacío, falsy) NO debe activar el skip.
    Solo truthy strings. Defensa contra payload con campo presente pero
    vacío que un cliente podría enviar por bug."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": []}
    fake_pantry = [f"item-{i}" for i in range(15)]

    with patch(
        "cron_tasks._validate_and_retry_initial_chunk_against_pantry",
        return_value=(fake_result, {
            "validated_ok": True, "attempts": 1, "degraded": False,
            "last_violation": None, "mode": "advisory",
            "pantry_size": 15, "missing_list": []
        }),
    ) as mock_validate:
        _run_pantry_validation_for_initial_chunk(
            result=fake_result,
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_ctx="",
            background_tasks=MagicMock(),
            actual_user_id="test-user",
            pantry_ingredients=fake_pantry,
            transport_label="TEST-EMPTY-REASON",
            update_reason="",
        )

    mock_validate.assert_called_once(), (
        "`update_reason=''` (falsy) activó el skip — debió tratarse como "
        "ausencia y caer al path legacy."
    )


def test_skip_when_update_reason_with_empty_pantry():
    """Edge case: update_reason='variety' + nevera vacía. El skip por intent
    debe ganar — emitimos el log SKIP-REGEN y retornamos sin tocar el path
    de pantry vacía. Esto da observabilidad limpia de 'cuántos planes
    skipean por intent' (incluso cuando coincidió con pantry vacía)."""
    from routers.plans import _run_pantry_validation_for_initial_chunk

    fake_result = {"days": []}

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
            actual_user_id="test-user",
            pantry_ingredients=[],
            transport_label="TEST-REGEN-EMPTY",
            update_reason="variety",
        )

    assert result is fake_result
    mock_validate.assert_not_called()
