"""[P2-A-ESCALATION-MODIFIED-AT · 2026-05-10] El UPDATE de
`meal_plans` en `_escalate_unrecoverable_chunk` (cron_tasks.py) DEBE
sellar `plan_data._plan_modified_at` Y `updated_at = NOW()` cuando
añade `_user_action_required`.

Bug original (audit 2026-05-10):
    `_escalate_unrecoverable_chunk` añadía `_recovery_exhausted_chunks`
    + `_user_action_required` al `plan_data` con `||` (jsonb merge),
    pero NO actualizaba `_plan_modified_at`. El Historial
    (`P1-AUDIT-HIST-4`, `routers/plans.py:/history-list`) ordena planes
    por `plan_data->>_plan_modified_at` (sort string ISO-8601). Resultado:
    un plan que adquirió un dead-letter HOY (estado visible para el
    usuario, requiere su acción) quedaba debajo de planes intactos del
    mismo día porque su sello semántico no se bumpeó.

    Asimetría con el espejo: retry-chunk (`routers/plans.py:4152-4162`,
    P0-3 · 2026-05-10) SÍ sellaba ambos timestamps con la misma razón
    documentada — el bug era exclusivo del path de escalation desde
    cron, no del path autenticado por el usuario.

Fix (P2-A · 2026-05-10):
    El UPDATE envuelve el merge `||` original en `jsonb_set(...,
    '{_plan_modified_at}', to_jsonb(NOW()::text), true)` Y añade
    `updated_at = NOW()` como SET adicional. El trigger
    `trg_meal_plans_set_updated_at` (P0-2) cubriría `updated_at` igual,
    pero el SET explícito documenta la intención del callsite y previene
    regresiones donde alguien remueva el SET asumiendo que el trigger
    es suficiente — patrón establecido en retry-chunk
    (test_p0_2_meal_plans_updated_at::test_retry_chunk_explicitly_sets_updated_at).

Cobertura de este test (parser-based vía `inspect.getsource`, no DB):
    1. El cuerpo de `_escalate_unrecoverable_chunk` contiene un UPDATE
       de meal_plans que sella `_plan_modified_at`.
    2. El UPDATE incluye `updated_at = NOW()` explícito.
    3. El UPDATE sigue añadiendo `_user_action_required` y
       `_recovery_exhausted_chunks` (no rompió el contrato pre-existente).
    4. Hay exactamente UN UPDATE de meal_plans en la función (sanity:
       no duplicamos por accidente).

Out of scope:
    - Test E2E con DB real que verifique reordenamiento del Historial:
      cubierto por el flujo P1-AUDIT-HIST-4 (sort por
      `_plan_modified_at`); este test asegura que el sello SE GENERE.
"""
from __future__ import annotations

import inspect
import re

import pytest


def _get_source() -> str:
    """Extrae el source de `_escalate_unrecoverable_chunk` vía
    `inspect.getsource` — más robusto que regex contra docstrings
    largos que mencionan SQL como prosa."""
    from cron_tasks import _escalate_unrecoverable_chunk
    return inspect.getsource(_escalate_unrecoverable_chunk)


def _extract_meal_plans_updates(src: str) -> list[str]:
    """Captura los bloques `UPDATE meal_plans ... WHERE id = %s`
    pasados como SQL real a `execute_sql_write` (triple-quoted strings
    cuya primera línea no-vacía es exactamente `UPDATE meal_plans`).

    Filtra docstrings/comentarios que mencionan SQL como prose
    (e.g. el docstring de `_escalate_unrecoverable_chunk` lista
    "UPDATE meal_plans.plan_data._recovery_exhausted_chunks (lista
    de week_numbers...)" como descripción operacional)."""
    blocks = []
    for m in re.finditer(r'"""(.*?)"""', src, re.DOTALL):
        body = m.group(1)
        # Real SQL: la primera línea no-vacía es `UPDATE meal_plans`.
        first_non_empty = next(
            (line.strip() for line in body.splitlines() if line.strip()),
            "",
        )
        if re.match(r"UPDATE\s+meal_plans\b", first_non_empty, re.IGNORECASE):
            blocks.append(body)
    return blocks


# ---------------------------------------------------------------------------
# 1. Sello _plan_modified_at presente.
# ---------------------------------------------------------------------------
def test_escalation_seals_plan_modified_at():
    """Núcleo del fix P2-A: sin este sello el plan dead-lettered NO
    sube al top del Historial, contradiciendo el contrato de
    P1-AUDIT-HIST-4."""
    src = _get_source()
    upds = _extract_meal_plans_updates(src)
    assert upds, "No se encontró ningún UPDATE meal_plans en la función."

    has_seal = any("_plan_modified_at" in u for u in upds)
    assert has_seal, (
        "El UPDATE de `meal_plans` en `_escalate_unrecoverable_chunk` "
        "DEBE sellar `_plan_modified_at` para que el Historial "
        "(P1-AUDIT-HIST-4) reordene el plan al top tras dead-letter. "
        "Asimetría espejo con retry-chunk (`routers/plans.py:4152-4162`, "
        "P0-3) — ambos paths que añaden `_user_action_required` deben "
        "bumpear el sello semántico."
    )

    # Defense-in-depth adicional: el sello debe usar `to_jsonb(NOW()::text)`
    # (mismo patrón que retry-chunk), no un literal stale.
    has_now_seal = any(
        re.search(
            r"to_jsonb\s*\(\s*NOW\s*\(\s*\)\s*::\s*text\s*\)",
            u,
            re.IGNORECASE,
        )
        for u in upds
    )
    assert has_now_seal, (
        "El sello debe usar `to_jsonb(NOW()::text)` para que el ISO sea "
        "fresh. Patrón espejo de retry-chunk; un literal stale rompería "
        "el sort del Historial."
    )


# ---------------------------------------------------------------------------
# 2. updated_at = NOW() explícito.
# ---------------------------------------------------------------------------
def test_escalation_explicitly_sets_updated_at():
    """`updated_at` es cubierta por el trigger `trg_meal_plans_set_updated_at`
    (P0-2), pero el SET explícito documenta la intención del callsite y
    previene regresiones donde alguien remueva el SET asumiendo que el
    trigger basta. Mismo patrón que retry-chunk
    (test_p0_2_meal_plans_updated_at::test_retry_chunk_explicitly_sets_updated_at)."""
    src = _get_source()
    upds = _extract_meal_plans_updates(src)
    assert upds

    has_explicit_updated_at = any(
        re.search(r"updated_at\s*=\s*NOW\s*\(\s*\)", u, re.IGNORECASE)
        for u in upds
    )
    assert has_explicit_updated_at, (
        "El UPDATE de `meal_plans` en escalation debe incluir "
        "`updated_at = NOW()` explícito (defense-in-depth contra remoción "
        "del trigger; documenta intención del callsite)."
    )


# ---------------------------------------------------------------------------
# 3. Contrato pre-existente preservado.
# ---------------------------------------------------------------------------
def test_escalation_preserves_user_action_required_payload():
    """Introducir el sello no debe romper el contrato P1-CHUNKS-1
    (frontend lee `_user_action_required` del plan_data tras dead-letter).
    Este test asegura que el refactor mantuvo el payload completo."""
    src = _get_source()
    upds = _extract_meal_plans_updates(src)
    assert upds

    # El payload completo debe vivir en al menos uno de los UPDATEs.
    combined = "\n".join(upds)
    for required_key in (
        "_recovery_exhausted_chunks",
        "_user_action_required",
        "reason_code",
        "chunk_id",
        "title",
        "body",
        "cta",
        "url",
    ):
        assert required_key in combined, (
            f"El UPDATE perdió la clave `{required_key}`. El payload "
            f"P1-CHUNKS-1 enriquecido debe seguir intacto — el frontend "
            f"depende de title/body/cta/url para renderizar el banner sin "
            f"acudir a otro endpoint."
        )


# ---------------------------------------------------------------------------
# 4. UN solo UPDATE de meal_plans (sanity: no duplicamos por accidente).
# ---------------------------------------------------------------------------
def test_escalation_has_exactly_one_meal_plans_update():
    """El bloque tiene exactamente UN `UPDATE meal_plans`. Si añadiste
    un segundo (e.g. para sellar `_plan_modified_at` por separado en
    lugar de envolver el merge), revisa el contrato — la idempotency
    de `_recovery_exhausted_chunks` (append jsonb) podría duplicar
    entries en el segundo statement."""
    src = _get_source()
    upds = _extract_meal_plans_updates(src)

    assert len(upds) == 1, (
        f"`_escalate_unrecoverable_chunk` debe tener exactamente 1 "
        f"`UPDATE meal_plans` (encontrados: {len(upds)}). Si necesitas "
        f"más, considerar el riesgo de doble-append a "
        f"`_recovery_exhausted_chunks`."
    )
