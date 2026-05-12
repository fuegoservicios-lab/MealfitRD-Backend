"""[P2-NEXT-2 · 2026-05-11] El cron diario `_shopping_coherence_alert_job`
(cron_tasks.py:685+) DEBE invocar el helper SSOT
`run_shopping_coherence_guard_and_append_history` (en lugar del guard
pelado `run_shopping_coherence_guard`) y persistir el `plan_data`
mutado via `update_meal_plan_data` cuando el knob
`MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` está enabled (default True).

Cierra el gap legacy del audit 2026-05-11:
    Antes de P2-NEXT-2, el cron era pure read-only (mode='warn' +
    Counter agregado), por diseño para NO mutar planes ya entregados.
    Resultado: planes pre-P1-NEXT-2 (sin guard en write-time) nunca
    recibían `_shopping_coherence_block_history`, aunque el cron
    detectara divergencias 24h después.

    Plan 005c5a99/75be68b8/98d902e3 en prod (audit 2026-05-11): 3
    planes en 14 días, NINGUNO con history populated. P3-NEW-C
    observability story rota para chunked plans legacy.

Fix P2-NEXT-2:
    - Cron usa el helper SSOT en lugar del guard pelado.
    - Pasa `action_taken="warn_only_cron_daily"` para distinguir
      origen post-mortem (cron) vs write-time (T2/recalc/agent_tool).
    - Tras helper appendea entry al `_shopping_coherence_block_history`
      en memoria, persiste plan_data via `update_meal_plan_data(...,
      user_id=user_id)` que toma advisory lock interno (P1-NEXT-1).
    - Knob `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` (default True)
      kill switch para rollback sin redeploy.
    - SELECT añade `user_id` (necesario para defense-in-depth I2).

Drift detection:
    - Cron pierde el helper call → falla.
    - SELECT no incluye user_id → falla.
    - Persist call no pasa user_id → falla.
    - Knob desaparece → falla.

Tooltip-anchor: P2-NEXT-2-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"


def _read_function_body(source: str, fn_name: str) -> str:
    """Aísla el cuerpo de `def fn_name(...)` hasta el siguiente top-level
    `def `/`class ` (defensa contra refactor)."""
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    assert m, (
        f"No se encontró `def {fn_name}(...)` en cron_tasks.py. "
        "El test P2-NEXT-2 perdió su anchor — verifica que el cron "
        "no fue renombrado o movido."
    )
    body_start = m.start()
    next_def_pattern = re.compile(r"^(def |class )", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[body_start:next_def.start()]
    return source[body_start:]


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_body(cron_source: str) -> str:
    return _read_function_body(cron_source, "_shopping_coherence_alert_job")


# ---------------------------------------------------------------------------
# 1. Cron invoca el helper SSOT (no el guard pelado)
# ---------------------------------------------------------------------------
def test_cron_calls_helper_not_bare_guard(cron_body: str):
    """`_shopping_coherence_alert_job` DEBE usar el helper SSOT
    `run_shopping_coherence_guard_and_append_history` para popular
    history; el guard pelado `run_shopping_coherence_guard` no appendea.
    """
    helper_re = re.compile(
        r"run_shopping_coherence_guard_and_append_history",
    )
    assert helper_re.search(cron_body), (
        "P2-NEXT-2 violation: el cron `_shopping_coherence_alert_job` "
        "no invoca `run_shopping_coherence_guard_and_append_history`. "
        "Sin el helper, el cron sigue siendo read-only y los planes "
        "legacy (pre-P1-NEXT-2) nunca reciben `_shopping_coherence_"
        "block_history` populated.\n\n"
        "Fix: importar el helper y reemplazar la llamada al guard pelado:\n"
        "  from shopping_calculator import run_shopping_coherence_guard_and_append_history\n"
        "  divs, block_set = run_shopping_coherence_guard_and_append_history(\n"
        "      plan_data, multiplier=mult_cached, mode_override='warn',\n"
        "      action_taken='warn_only_cron_daily', plan_id_hint=plan_id,\n"
        "  )"
    )


# ---------------------------------------------------------------------------
# 2. Helper se invoca con action_taken="warn_only_cron_daily"
# ---------------------------------------------------------------------------
def test_helper_receives_cron_daily_action_taken(cron_body: str):
    """`action_taken='warn_only_cron_daily'` distingue entries originados
    en este cron de los write-time (T2/recalc/agent_tool)."""
    action_re = re.compile(
        r"action_taken\s*=\s*['\"]warn_only_cron_daily['\"]",
    )
    assert action_re.search(cron_body), (
        "P2-NEXT-2 violation: el cron invoca el helper pero NO pasa "
        "`action_taken='warn_only_cron_daily'`. Sin ese marker, no "
        "podemos distinguir post-mortem entries de cron vs write-time. "
        "El cron diario es la ÚLTIMA red de telemetría para planes que "
        "bypasearon el guard en write-time."
    )


# ---------------------------------------------------------------------------
# 3. SELECT incluye user_id (necesario para defense-in-depth I2 en persist)
# ---------------------------------------------------------------------------
def test_select_fetches_user_id(cron_body: str):
    """El SELECT de planes para evaluación DEBE incluir `user_id` para
    que `update_meal_plan_data(..., user_id=user_id)` aplique el filtro
    AND user_id = %s (invariante I2 CLAUDE.md)."""
    select_re = re.compile(
        r"\.select\s*\(\s*['\"][^'\"]*user_id[^'\"]*['\"]\s*\)",
    )
    assert select_re.search(cron_body), (
        "P2-NEXT-2 violation: el SELECT de `meal_plans` en el cron NO "
        "incluye `user_id`. Sin user_id, `update_meal_plan_data` cae al "
        "path legacy con warning `[I2-MISS]`, perdiendo defense-in-depth. "
        "Fix: `select('id,user_id,plan_data')`."
    )


# ---------------------------------------------------------------------------
# 4. Persist via update_meal_plan_data con user_id
# ---------------------------------------------------------------------------
def test_cron_persists_via_update_meal_plan_data_with_user_id(cron_body: str):
    """Tras helper appendea history en memoria, el cron persiste via
    `update_meal_plan_data(plan_id, plan_data, user_id=user_id)` (que
    adquiere advisory lock interno, P1-NEXT-1)."""
    persist_call_re = re.compile(
        r"update_meal_plan_data|_persist_plan_data\s*\(",
    )
    assert persist_call_re.search(cron_body), (
        "P2-NEXT-2 violation: el cron NO invoca `update_meal_plan_data` "
        "(directo o vía alias `_persist_plan_data`). Sin persist, el "
        "helper appendea history solo en memoria y el dict se descarta "
        "al final del loop → ningún beneficio sobre la versión pre-fix."
    )

    user_id_kwarg_re = re.compile(
        r"user_id\s*=\s*user_id",
    )
    assert user_id_kwarg_re.search(cron_body), (
        "P2-NEXT-2 violation: el call a `update_meal_plan_data` no "
        "pasa `user_id=user_id`. Aunque el helper acepta user_id=None "
        "(legacy back-compat con warning), el cron debe pasarlo para "
        "cumplir invariante I2 defense-in-depth."
    )


# ---------------------------------------------------------------------------
# 5. Knob kill switch MEALFIT_COHERENCE_CRON_PERSIST_HISTORY
# ---------------------------------------------------------------------------
def test_cron_reads_persist_kill_switch_knob(cron_body: str):
    """Knob `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` (default True) debe
    ser leído via `_env_bool` para registrarse en `_KNOBS_REGISTRY` y
    permitir rollback sin redeploy si persist genera contención."""
    knob_re = re.compile(
        r"_env_bool\s*\(\s*['\"]MEALFIT_COHERENCE_CRON_PERSIST_HISTORY['\"]",
    )
    assert knob_re.search(cron_body), (
        "P2-NEXT-2 violation: el cron no lee el knob "
        "`MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` vía `_env_bool`. Sin "
        "el knob, no hay kill switch para deshabilitar el persist sin "
        "redeploy si genera contención con write paths.\n\n"
        "Fix: `persist_history_enabled = _env_bool("
        "'MEALFIT_COHERENCE_CRON_PERSIST_HISTORY', True)`. El knob se "
        "auto-registra en `_KNOBS_REGISTRY` y aparece en `/health/version`."
    )


# ---------------------------------------------------------------------------
# 6. Persist está gated por el knob
# ---------------------------------------------------------------------------
def test_persist_is_gated_by_knob(cron_body: str):
    """El call a `update_meal_plan_data` debe estar dentro de un `if`
    que chequea el knob — sino el kill switch no surte efecto."""
    # Buscar el patrón `if persist_history_enabled` (o nombre similar)
    # antes del call a update_meal_plan_data.
    knob_var_re = re.compile(r"if\s+persist_history_enabled\b")
    assert knob_var_re.search(cron_body), (
        "P2-NEXT-2 violation: el call a `update_meal_plan_data` no está "
        "gated por el knob. Sin gate, el kill switch no surte efecto. "
        "Fix: envolver el persist en `if persist_history_enabled:`."
    )


# ---------------------------------------------------------------------------
# 7. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_next_2"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p2_next_1`) para cross-link con `test_p2_hist_audit_14`."
    )
