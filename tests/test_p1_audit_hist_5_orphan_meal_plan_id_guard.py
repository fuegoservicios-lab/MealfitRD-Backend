"""[P1-AUDIT-HIST-5 · 2026-05-09] Tests del guard
``meal_plan_id IS NOT NULL`` en crons que iteran chunks/deferrals
huérfanos (cuando el plan padre fue eliminado).

Bug original (audit Historial 2026-05-09):
    Tras un DELETE de `meal_plans`, las FK de las tablas chunk_*
    eligieron políticas distintas:
      - `plan_chunk_queue.meal_plan_id`: ON DELETE CASCADE (la fila
        del chunk se borra con el plan).
      - `chunk_deferrals.meal_plan_id`: ON DELETE SET NULL (preserva
        audit trail post P0-HIST-3).
      - `chunk_lesson_telemetry.meal_plan_id`: ON DELETE SET NULL.
      - `plan_chunk_metrics.meal_plan_id`: ON DELETE SET NULL.

    Los crons que iteran `pending_user_action` y `chunk_deferrals`
    no filtraban por `meal_plan_id IS NOT NULL`. Síntomas:
      - `_alert_chronic_deferrals` agrupaba TODOS los deferrals con
        `meal_plan_id=NULL` bajo la tupla `(user_id, NULL,
        week_number)`, generando alerta `chronic_deferrals` con
        `plan_id=None` en metadata y push confuso.
      - `_recover_pantry_paused_chunks` y
        `_alert_chunks_paused_indefinitely` procesaban rows fantasma
        cuando un INSERT corrupto/legacy dejaba un chunk sin
        meal_plan_id. Sin meal_plan_id no se puede rehidratar
        plan_data — ciclos LLM gastados.

Fix:
    `AND meal_plan_id IS NOT NULL` en el WHERE de los 3 crons.

Cobertura:
    1. Anchor del marker.
    2. `_recover_pantry_paused_chunks` SQL incluye filtro.
    3. `_alert_chunks_paused_indefinitely` SQL incluye filtro.
    4. `_alert_chronic_deferrals` SQL incluye filtro.
    5. Comentario load-bearing cita la motivación (CASCADE vs
       SET NULL, audit trail, ciclos LLM).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PATH = _BACKEND_ROOT / "cron_tasks.py"


def _cron_source() -> str:
    return _CRON_PATH.read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_cron_tasks():
    """El marker `P1-AUDIT-HIST-5` debe aparecer en `cron_tasks.py`
    en al menos los 3 sitios fixed (uno por cron afectado)."""
    text = _cron_source()
    occurrences = text.count("P1-AUDIT-HIST-5")
    assert occurrences >= 3, (
        f"Esperaba al menos 3 ocurrencias del marker `P1-AUDIT-HIST-5` "
        f"(uno por cron fixed: pantry_paused, paused_indefinitely, "
        f"chronic_deferrals). Encontradas: {occurrences}."
    )


# ---------------------------------------------------------------------------
# 2. _recover_pantry_paused_chunks
# ---------------------------------------------------------------------------
def test_recover_pantry_paused_chunks_filters_meal_plan_id_not_null():
    """El SELECT del cron de pantry recovery debe filtrar
    `meal_plan_id IS NOT NULL`. Sin esto, un row con meal_plan_id
    NULL pasaría al loop de procesamiento donde el branch
    `missing_start_date_no_anchor` requiere `meal_plan_id_str` para
    re-leer plan_data — el row sería un consumidor de ciclos sin
    output útil.
    """
    from cron_tasks import _recover_pantry_paused_chunks
    src = inspect.getsource(_recover_pantry_paused_chunks)
    norm = _normalize(src)
    # Buscar el SELECT principal (status='pending_user_action').
    # El filtro `meal_plan_id IS NOT NULL` debe estar en el mismo
    # WHERE.
    assert re.search(
        r"WHERE\s+status\s*=\s*'pending_user_action'\s+AND\s+meal_plan_id\s+IS\s+NOT\s+NULL",
        norm,
        re.IGNORECASE,
    ), (
        "El SELECT de `_recover_pantry_paused_chunks` debe filtrar "
        "`meal_plan_id IS NOT NULL` junto al status. Filtro actual: "
        f"{norm[:600]!r}..."
    )


def test_recover_pantry_paused_chunks_cites_marker():
    """Comentario load-bearing dentro del helper cita el marker —
    sin esto, un refactor podría remover el filtro perdiendo la
    motivación."""
    from cron_tasks import _recover_pantry_paused_chunks
    src = inspect.getsource(_recover_pantry_paused_chunks)
    assert "P1-AUDIT-HIST-5" in src


# ---------------------------------------------------------------------------
# 3. _alert_chunks_paused_indefinitely
# ---------------------------------------------------------------------------
def test_alert_chunks_paused_indefinitely_filters_meal_plan_id_not_null():
    """El SELECT del cron de pause indefinida (alerta + auto-unblock)
    debe filtrar `q.meal_plan_id IS NOT NULL`. Sin esto, alert_key
    sería `chunk_paused_indefinitely:None:N`, el LEFT JOIN no
    encuentra plan_data, y el reintento de unblock falla con
    ciclos LLM gastados.
    """
    from cron_tasks import _alert_chunks_paused_indefinitely
    src = inspect.getsource(_alert_chunks_paused_indefinitely)
    norm = _normalize(src)
    # El SELECT principal del cron tiene un alias `q.` para
    # plan_chunk_queue.
    assert re.search(
        r"q\.status\s*=\s*'pending_user_action'\s+AND\s+q\.meal_plan_id\s+IS\s+NOT\s+NULL",
        norm,
        re.IGNORECASE,
    ), (
        "El SELECT de `_alert_chunks_paused_indefinitely` debe "
        "filtrar `q.meal_plan_id IS NOT NULL`. Filtro actual: "
        f"{norm[:1500]!r}..."
    )


def test_alert_chunks_paused_indefinitely_cites_marker():
    from cron_tasks import _alert_chunks_paused_indefinitely
    src = inspect.getsource(_alert_chunks_paused_indefinitely)
    assert "P1-AUDIT-HIST-5" in src


# ---------------------------------------------------------------------------
# 4. _alert_chronic_deferrals
# ---------------------------------------------------------------------------
def test_alert_chronic_deferrals_filters_meal_plan_id_not_null():
    """El SELECT del cron de chronic deferrals debe filtrar
    `meal_plan_id IS NOT NULL`. Sin esto, GROUP BY agruparía TODOS
    los deferrals huérfanos (post-DELETE plan, FK SET NULL del
    P0-HIST-3) bajo la tupla `(user_id, NULL, week_number)` y
    reportaría una alerta con `plan_id=None` en metadata.
    """
    text = _cron_source()
    norm = _normalize(text)
    # Buscar el SELECT específico de chunk_deferrals con
    # reason='temporal_gate' y verificar que tiene el filtro.
    m = re.search(
        r"FROM\s+chunk_deferrals\s+WHERE[^;]*?'temporal_gate'[^;]*?GROUP\s+BY",
        norm,
        re.IGNORECASE | re.DOTALL,
    )
    assert m is not None, (
        "No pude localizar el SELECT de chunk_deferrals + "
        "reason='temporal_gate' en cron_tasks.py."
    )
    where_block = m.group(0)
    assert re.search(
        r"meal_plan_id\s+IS\s+NOT\s+NULL",
        where_block,
        re.IGNORECASE,
    ), (
        "El SELECT de chunk_deferrals (reason='temporal_gate') debe "
        "filtrar `meal_plan_id IS NOT NULL` antes del GROUP BY. "
        f"WHERE block actual: {where_block!r}"
    )


# ---------------------------------------------------------------------------
# 5. Comentario load-bearing cita motivación
# ---------------------------------------------------------------------------
def test_comments_explain_orphan_motivation():
    """Los 3 sitios fixed deben tener comentario explicativo que
    incluya al menos uno de los conceptos clave: CASCADE, SET NULL,
    huérfano, audit trail, ciclos. Sin contexto, un refactor
    posterior puede simplificar y reintroducir el bug.
    """
    text = _cron_source()
    # Buscar todas las regiones marker P1-AUDIT-HIST-5 y verificar
    # que cada una cita ≥1 concepto.
    motivations_per_site = {
        "_recover_pantry_paused_chunks": ("CASCADE", "rehidratar", "ciclos", "huérfan"),
        "_alert_chunks_paused_indefinitely": ("CASCADE", "alert_key", "LEFT JOIN", "huérfan"),
        "_alert_chronic_deferrals_or_chunk_deferrals": ("SET NULL", "audit", "huérfan", "GROUP BY"),
    }

    # Localizar cada región del marker en orden de aparición.
    marker_idxs = [
        m.start() for m in re.finditer(r"P1-AUDIT-HIST-5", text)
    ]
    assert len(marker_idxs) >= 3, (
        f"Necesitamos ≥3 ocurrencias del marker (una por cron fixed). "
        f"Encontradas: {len(marker_idxs)}."
    )

    # Para cada ocurrencia, capturar 1500 chars siguientes (block del
    # comentario + SQL) y verificar que CITA al menos UN concepto del
    # set agregado.
    all_concepts = (
        "CASCADE", "SET NULL", "huérfan", "audit", "ciclos",
        "rehidratar", "LEFT JOIN", "alert_key", "GROUP BY",
    )
    for idx in marker_idxs:
        block = text[idx:idx + 1500]
        matches = [c for c in all_concepts if c.lower() in block.lower()]
        assert matches, (
            f"Marker en pos {idx} no cita ningún concepto clave. "
            f"Esperado al menos uno de: {all_concepts}. Block:\n"
            f"{block[:400]!r}..."
        )
