"""[P2-NEXT-3 · 2026-05-11] Cron de sweep semanal que marca planes
huérfanos (status='in_progress'/'generating_next' sin chunks vivos en
`plan_chunk_queue` y antiguos >=7 días) como `abandoned`.

Hermano simétrico de `_cleanup_orphan_chunks` (P2-3) pero en la
dirección opuesta del FK: el primero cancela chunks vivos con
meal_plan_id ya borrado; este marca planes vivos sin chunks asociados.

Caso real cerrado:
    Audit 2026-05-11 detectó plan `75be68b8-…` (test fixture inyectado
    por P2-LIVE-2 con 8 chunks sintéticos) que tras cleanup de chunks
    quedó residual en `meal_plans` con `generation_status='in_progress'`
    sin cron que lo limpiase. Sin sweep, ese plan permanecería para
    siempre en estado mid-generation que nunca completará.

Drift detection:
    - Función `_sweep_meal_plans_without_chunks` borrada/renombrada → falla.
    - Cron no registrado en `register_plan_chunk_scheduler` → falla.
    - Schedule frequency cambia de weekly (sweep one-shot por semana es
      suficiente para mantenimiento) a interval minutes (que sería caro
      con SELECT cross-table) sin justificación documentada → falla
      via inspección del CronTrigger.
    - UPDATE pierde el filtro `AND user_id = %s` (defense-in-depth I2)
      → falla.
    - UPDATE migra a full-overwrite (`SET plan_data = %s::jsonb`) sin
      advisory lock (invariante I7) → falla.

Whitelist:
    No prevista. El sweep es atómico via `||` jsonb merge y debe
    permanecer así para no requerir lock.

Tooltip-anchor: P2-NEXT-3-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"


def _read_function_body(source: str, fn_name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(r"^(def |class |@)", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Función _sweep_meal_plans_without_chunks existe
# ---------------------------------------------------------------------------
def test_sweep_function_defined(cron_source: str):
    """`_sweep_meal_plans_without_chunks` debe estar definida en cron_tasks.py."""
    assert re.search(
        r"^def\s+_sweep_meal_plans_without_chunks\s*\(",
        cron_source,
        re.MULTILINE,
    ), (
        "P2-NEXT-3 violation: `_sweep_meal_plans_without_chunks` no está "
        "definida en cron_tasks.py. Sin ella, planes huérfanos (status="
        "in_progress/generating_next sin chunks vivos) permanecen "
        "indefinidamente en mid-generation."
    )


# ---------------------------------------------------------------------------
# 2. SELECT filtra por status + age + NOT EXISTS chunks vivos
# ---------------------------------------------------------------------------
def test_sweep_selects_in_progress_and_aged_orphans(cron_source: str):
    """El SELECT debe filtrar:
      - generation_status IN ('in_progress', 'generating_next')
      - created_at < NOW() - INTERVAL '<N> days'
      - NOT EXISTS chunks vivos (pending/stale/processing/completed)
    """
    body = _read_function_body(cron_source, "_sweep_meal_plans_without_chunks")
    assert body, "Función ausente — cubre test #1."

    assert re.search(
        r"generation_status['\"]?\s*IN\s*\(\s*['\"]in_progress['\"]\s*,\s*['\"]generating_next['\"]",
        body,
    ), (
        "P2-NEXT-3 violation: SELECT no filtra `generation_status IN "
        "('in_progress', 'generating_next')`. Planes en `complete` o "
        "`abandoned` no deben ser sweep."
    )
    assert re.search(
        r"created_at\s*<\s*NOW\s*\(\s*\)\s*-\s*INTERVAL",
        body,
        re.IGNORECASE,
    ), (
        "P2-NEXT-3 violation: SELECT no aplica filtro de edad "
        "(`created_at < NOW() - INTERVAL '<N> days'`). Sin esto un "
        "plan recién creado (legítimamente in-progress) sería sweep."
    )
    assert re.search(r"NOT\s+EXISTS", body, re.IGNORECASE), (
        "P2-NEXT-3 violation: SELECT no usa `NOT EXISTS` para detectar "
        "planes sin chunks vivos. Sin esto se marcaría como abandoned "
        "un plan ACTIVAMENTE generándose."
    )


# ---------------------------------------------------------------------------
# 3. UPDATE usa jsonb merge (||) o jsonb_set, NO full-overwrite
# ---------------------------------------------------------------------------
def test_sweep_update_is_atomic_merge_not_full_overwrite(cron_source: str):
    """El UPDATE debe ser jsonb merge atómico (`||` o `jsonb_set`) — no
    full-overwrite. Razón: full-overwrite requiere advisory lock por I7;
    el sweep semanal no toma lock por diseño (filtro NOT EXISTS ya
    garantiza no race con T1/T2)."""
    body = _read_function_body(cron_source, "_sweep_meal_plans_without_chunks")
    # Debe contener jsonb merge:
    assert (
        re.search(r"plan_data\s*\|\|\s*jsonb_build_object", body)
        or re.search(r"jsonb_set\s*\(\s*[^,]*plan_data", body)
    ), (
        "P2-NEXT-3 violation: el UPDATE no usa `plan_data || jsonb_build_object(...)` "
        "ni `jsonb_set(plan_data, ...)`. Si migró a full-overwrite "
        "`SET plan_data = %s::jsonb`, requiere `acquire_meal_plan_advisory_lock` "
        "antes (invariante I7). Mantener el merge atómico es preferible para "
        "un sweep de mantenimiento."
    )
    # NO debe contener full-overwrite:
    fullovr = re.search(
        r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s",
        body,
        re.IGNORECASE,
    )
    assert not fullovr, (
        "P2-NEXT-3 violation: el UPDATE migró a full-overwrite. "
        "Sin advisory lock (`acquire_meal_plan_advisory_lock(... purpose='general')`) "
        "antes, viola invariante I7. Volver a `||` merge o añadir lock."
    )


# ---------------------------------------------------------------------------
# 4. UPDATE incluye AND user_id = %s (defense-in-depth I2)
# ---------------------------------------------------------------------------
def test_sweep_update_filters_user_id(cron_source: str):
    """Aunque el SELECT ya resolvió (id, user_id), el UPDATE debe
    repetir `AND user_id = %s` para no abrir IDOR si un refactor
    futuro separa el SELECT del UPDATE."""
    body = _read_function_body(cron_source, "_sweep_meal_plans_without_chunks")
    assert re.search(
        r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        body,
        re.IGNORECASE | re.DOTALL,
    ), (
        "P2-NEXT-3 violation: UPDATE no filtra `AND user_id = %s` "
        "(invariante I2 CLAUDE.md, defense-in-depth). Sin este filtro "
        "un refactor futuro del SELECT puede abrir IDOR silente."
    )


# ---------------------------------------------------------------------------
# 5. Marca `_abandoned_at` y `_abandoned_reason`
# ---------------------------------------------------------------------------
def test_sweep_writes_abandoned_metadata(cron_source: str):
    """El sweep debe escribir `_abandoned_at` (timestamp) y
    `_abandoned_reason` para forensics post-mortem y para que el
    cron diario `_shopping_coherence_alert_job` y otros consumers
    puedan filtrar planes abandoned."""
    body = _read_function_body(cron_source, "_sweep_meal_plans_without_chunks")
    assert "_abandoned_at" in body, (
        "P2-NEXT-3 violation: el sweep no escribe `_abandoned_at`. "
        "Sin timestamp, forensics post-mortem ('¿cuándo se abandonó?') "
        "es imposible."
    )
    assert re.search(r"_abandoned_reason.*orphan_chunks", body, re.DOTALL), (
        "P2-NEXT-3 violation: el sweep no escribe "
        "`_abandoned_reason = 'orphan_chunks'`. Sin razón, no podemos "
        "distinguir abandoned por sweep vs futuros otros mecanismos."
    )


# ---------------------------------------------------------------------------
# 6. Cron registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------
def test_sweep_cron_registered_in_scheduler(cron_source: str):
    """`register_plan_chunk_scheduler` debe registrar el cron con
    id='sweep_meal_plans_without_chunks'. Sin registro, la función
    nunca corre."""
    register_body = _read_function_body(cron_source, "register_plan_chunk_scheduler")
    assert register_body, "register_plan_chunk_scheduler no encontrada."
    assert "sweep_meal_plans_without_chunks" in register_body, (
        "P2-NEXT-3 violation: cron `sweep_meal_plans_without_chunks` "
        "no registrado en `register_plan_chunk_scheduler`. Sin registro, "
        "la función definida nunca corre."
    )


# ---------------------------------------------------------------------------
# 7. Knobs auto-registrados (age_days, batch)
# ---------------------------------------------------------------------------
def test_sweep_knobs_registered(cron_source: str):
    """Los knobs `MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS` y
    `MEALFIT_SWEEP_ORPHAN_PLANS_BATCH` deben leerse via `_env_int`
    para auto-registrarse en `_KNOBS_REGISTRY`."""
    body = _read_function_body(cron_source, "_sweep_meal_plans_without_chunks")
    assert re.search(
        r"_env_int\s*\(\s*['\"]MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS['\"]",
        body,
    ), (
        "P2-NEXT-3 violation: knob `MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS` "
        "no leído via `_env_int`. Sin auto-registro en `_KNOBS_REGISTRY`, "
        "SRE no puede tunearlo desde `/health/version`."
    )
    assert re.search(
        r"_env_int\s*\(\s*['\"]MEALFIT_SWEEP_ORPHAN_PLANS_BATCH['\"]",
        body,
    ), (
        "P2-NEXT-3 violation: knob `MEALFIT_SWEEP_ORPHAN_PLANS_BATCH` "
        "no leído via `_env_int`."
    )


# ---------------------------------------------------------------------------
# 8. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_next_3"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener slug `p2_next_3` para cross-link con "
        "test_p2_hist_audit_14."
    )
