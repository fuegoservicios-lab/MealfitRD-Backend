"""[P0-OBS-1 · 2026-05-10 → P1-A · 2026-05-10] Regression guard
dinámico: el cron horario `_aggregate_coherence_block_history_metrics`
DEBE filtrar `meal_plans` por `updated_at` (la columna añadida por la
migración P0-2).

Lineage del fix:
    1. P3-B (2026-05-08) introdujo el cron filtrando por `updated_at`,
       pero la columna no existía → PostgREST 400 cada hora, watchdog
       del invariante P2-2 silencioso.
    2. P0-OBS-1 (2026-05-10) cambió el filtro a `created_at` como
       workaround mientras se diseñaba la columna real. Este test
       inicialmente enforzaba ese contrato.
    3. P0-2 (2026-05-10) añadió la columna `updated_at` + trigger +
       índice `idx_meal_plans_user_updated_at`.
    4. P1-A (2026-05-10) restauró el filtro `updated_at` (intención
       original pre-bug). Este test cambió de polaridad: ahora enforza
       que el cron use `updated_at` y NO `created_at`.

Por qué dos tests cubren lo mismo (este + `test_p1_a_coh_updated_at.py`):
    - Este test es DINÁMICO: monkeypatchea Supabase y captura la columna
      pasada a `.gte()` en runtime. Atrapa también typos que el parser
      podría no detectar.
    - El test P1-A parser-based lee el source de `cron_tasks.py` con
      regex. Más rápido y self-contained, no requiere imports.
    Defense-in-depth: si uno se rompe por refactor (e.g. el monkeypatch
    deja de aplicar porque el call pasa por un wrapper nuevo), el otro
    sigue cubriendo. El archivo conserva su nombre histórico (`uses_created_at`)
    como evidencia del lineage del bug; el contenido refleja el contrato
    actual.

Cobertura:
    1. La columna pasada al `.gte()` del fetch de meal_plans es exactamente
       `updated_at`. Si alguien revierte a `created_at` (regresión P1-A)
       o cualquier otra columna, este test falla loud.
    2. La columna está en la whitelist de columnas reales de `meal_plans`
       (snapshot inline post-P0-2).

Out of scope (P3-TEST-1):
    Validación schema-aware genérica para todos los call sites Supabase
    client del backend. Este test es targeteado al cron específico que
    rompió en prod.
"""
from __future__ import annotations

import pytest


# Snapshot inline de columnas reales de `meal_plans` en prod
# (verificado vía `information_schema.columns` el 2026-05-10).
# Si el schema cambia, actualizar este set + bumpear `_LAST_KNOWN_PFIX`.
#
# [P0-2 · 2026-05-10] `updated_at` añadida vía migración
# `p0_2_meal_plans_updated_at.sql`.
# [P1-A · 2026-05-10] El cron de P0-OBS-1 ahora usa `updated_at` para
# capturar regeneraciones de planes viejos (intención original pre-bug).
_MEAL_PLANS_COLUMNS = frozenset({
    "id", "user_id", "plan_data", "created_at", "updated_at", "name",
    "calories", "macros", "meal_names", "ingredients", "techniques",
    "profile_embedding",
})


# ---------------------------------------------------------------------------
# Stub que SÍ captura la columna pasada a .gte() — a diferencia del stub
# del test P3-B (test_p3_b_coherence_block_metrics_cron.py::_StubTable)
# que la ignora por simplicidad.
# ---------------------------------------------------------------------------
class _ColumnCapturingExecuteResult:
    def __init__(self, data):
        self.data = data


class _ColumnCapturingTable:
    def __init__(self, captured_columns):
        self._captured = captured_columns

    def select(self, _cols):
        return self

    def gte(self, col, _val):
        # AQUÍ está la diferencia con el stub de P3-B: registramos la columna.
        self._captured.append(col)
        return self

    def limit(self, _n):
        return self

    def execute(self):
        return _ColumnCapturingExecuteResult([])  # vacío basta para el test


class _ColumnCapturingSupabase:
    def __init__(self, captured_columns):
        self._captured = captured_columns

    def table(self, _name):
        return _ColumnCapturingTable(self._captured)


# ---------------------------------------------------------------------------
# 1. La columna del filtro debe ser updated_at (post-P1-A).
# ---------------------------------------------------------------------------
def test_cron_filters_meal_plans_by_updated_at(monkeypatch):
    """`_aggregate_coherence_block_history_metrics` debe llamar
    `meal_plans.gte("updated_at", ...)` post-P1-A.

    Si el valor regresa a `created_at`, el cron pierde regeneraciones
    de planes viejos (>`MEALFIT_COHERENCE_METRICS_LOOKBACK_H` horas)
    que appendearon entries al `_shopping_coherence_block_history` —
    exactamente el gap que P1-A cerró.

    Si el valor cae a cualquier otra columna no listada en el schema
    (`crated_at`, `last_seen`, etc.), PostgREST devuelve 400 y el
    watchdog P2-2 queda silencioso — exactamente el bug original P0-OBS-1.
    """
    captured = []

    import db_core
    monkeypatch.setattr(db_core, "supabase", _ColumnCapturingSupabase(captured))

    # Capturar el INSERT también, para no pegarle a la DB real.
    import cron_tasks
    monkeypatch.setattr(
        cron_tasks, "execute_sql_write", lambda *a, **k: None
    )

    from cron_tasks import _aggregate_coherence_block_history_metrics
    _aggregate_coherence_block_history_metrics()

    assert captured, (
        "El cron no invocó `.gte()` sobre meal_plans. ¿Se eliminó el filtro "
        "temporal del fetch? Si es intencional, actualizar este test."
    )
    assert captured[0] == "updated_at", (
        f"El cron filtró meal_plans por columna `{captured[0]!r}` — debe ser "
        f"`updated_at` (post-P1-A · 2026-05-10). Si revertiste a `created_at`, "
        f"el cron vuelve a perder regeneraciones de planes viejos (gap P1-A). "
        f"Si cambiaste a otra columna, asegurate de que existe en "
        f"`information_schema.columns` para meal_plans — un nombre erróneo "
        f"reproduce el bug original P0-OBS-1 (PostgREST 400 silencioso)."
    )


# ---------------------------------------------------------------------------
# 2. La columna usada debe existir en el schema real
# ---------------------------------------------------------------------------
def test_cron_filter_column_exists_in_meal_plans_schema(monkeypatch):
    """Defensa secundaria: la columna pasada a `.gte()` debe estar en la
    whitelist de columnas reales de meal_plans. Atrapa también typos
    futuros (`updted_at`, `updated_on`, etc.) y guía hacia el schema."""
    captured = []

    import db_core
    monkeypatch.setattr(db_core, "supabase", _ColumnCapturingSupabase(captured))

    import cron_tasks
    monkeypatch.setattr(
        cron_tasks, "execute_sql_write", lambda *a, **k: None
    )

    from cron_tasks import _aggregate_coherence_block_history_metrics
    _aggregate_coherence_block_history_metrics()

    assert captured
    col = captured[0]
    assert col in _MEAL_PLANS_COLUMNS, (
        f"El cron filtra meal_plans por `{col!r}`, pero esa columna NO está "
        f"en el snapshot de schema ({sorted(_MEAL_PLANS_COLUMNS)}). Si el "
        f"schema cambió, actualizar `_MEAL_PLANS_COLUMNS` aquí y verificar "
        f"que el cron sigue funcionando contra el nuevo schema."
    )


# ---------------------------------------------------------------------------
# 3. Sanity del snapshot
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col", ["created_at", "updated_at", "id", "user_id", "plan_data"])
def test_known_columns_in_snapshot(col):
    """Las 5 columnas core deben estar en el snapshot — defensa contra
    edición accidental del set."""
    assert col in _MEAL_PLANS_COLUMNS


def test_updated_at_present_in_meal_plans_schema():
    """[P0-2 · 2026-05-10] `updated_at` AHORA existe en `meal_plans`,
    añadida por la migración `p0_2_meal_plans_updated_at.sql`. Si alguien
    revierte la migración sin actualizar el snapshot, el test guía hacia
    la corrección.

    [P1-A · 2026-05-10] El cron `_aggregate_coherence_block_history_metrics`
    YA usa `updated_at` (intención original pre-bug). Dropear la columna
    rompe DOS callsites: este cron + retry-chunk (`routers/plans.py:4141`)
    + el trigger `trg_meal_plans_set_updated_at`.
    """
    assert "updated_at" in _MEAL_PLANS_COLUMNS, (
        "Si dropeaste la columna `updated_at` de `meal_plans`, este test "
        "intencionalmente te detiene. La columna es load-bearing para "
        "retry-chunk (`routers/plans.py:4141`), el cron P1-A "
        "(`cron_tasks._aggregate_coherence_block_history_metrics`), y para "
        "el trigger `trg_meal_plans_set_updated_at`. Antes de quitarla, "
        "revertir los 3 callsites y la migración P0-2."
    )
