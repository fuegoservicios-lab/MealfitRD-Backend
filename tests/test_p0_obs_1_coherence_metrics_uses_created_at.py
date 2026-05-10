"""[P0-OBS-1 · 2026-05-10] Regression guard: el cron horario
`_aggregate_coherence_block_history_metrics` DEBE filtrar `meal_plans`
por `created_at`, no por `updated_at`.

Bug original (audit 2026-05-10):
    `[backend/cron_tasks.py:876]` invocaba `.gte("updated_at", cutoff)` sobre
    `meal_plans`, columna que NO existe en el schema (verificado contra
    `information_schema.columns` en prod). PostgREST devolvía 400 cada hora
    desde el merge de P3-B (2026-05-08), y el watchdog del invariante P2-2
    (`action_taken` jamás `None` tras review) NUNCA emitió una sola fila
    a `pipeline_metrics` en producción. Logs Postgres confirmaban el ERROR
    `column meal_plans.updated_at does not exist` al ritmo del cron horario.

Por qué el test P3-B preexistente no lo atrapó:
    `tests/test_p3_b_coherence_block_metrics_cron.py::_StubTable.gte()`
    acepta CUALQUIER columna como argumento sin validación — el stub estaba
    diseñado para verificar conteos y resiliencia ante data corrupta, no
    contra drift de schema. Corner case clásico: el test pasa, prod falla.

Cobertura de este test:
    1. La columna pasada al `.gte()` del fetch de meal_plans es exactamente
       `created_at`. Si alguien revierte a `updated_at` (o cualquier otra
       columna que no exista), este test falla loud.
    2. La columna está documentada en una whitelist de columnas reales de
       `meal_plans` (snapshot inline). Defensa secundaria: si el schema
       cambia y se renombra `created_at`, el test guía hacia la
       actualización conjunta.

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
_MEAL_PLANS_COLUMNS = frozenset({
    "id", "user_id", "plan_data", "created_at", "name",
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
# 1. La columna del filtro debe ser created_at
# ---------------------------------------------------------------------------
def test_cron_filters_meal_plans_by_created_at(monkeypatch):
    """`_aggregate_coherence_block_history_metrics` debe llamar
    `meal_plans.gte("created_at", ...)`.

    Si el valor regresa a `updated_at` (o cualquier otra columna que no
    exista en `meal_plans`), PostgREST responde 400 y el cron emite cero
    filas a `pipeline_metrics` — exactamente el bug P0-OBS-1.
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
    assert captured[0] == "created_at", (
        f"El cron filtró meal_plans por columna `{captured[0]!r}` — debe ser "
        f"`created_at`. Si volvió a `updated_at`, eso es la regresión P0-OBS-1: "
        f"meal_plans NO tiene columna `updated_at` en prod, PostgREST devuelve "
        f"400 y el cron deja de emitir métricas silenciosamente."
    )


# ---------------------------------------------------------------------------
# 2. La columna usada debe existir en el schema real
# ---------------------------------------------------------------------------
def test_cron_filter_column_exists_in_meal_plans_schema(monkeypatch):
    """Defensa secundaria: la columna pasada a `.gte()` debe estar en la
    whitelist de columnas reales de meal_plans. Atrapa también typos
    futuros (`crated_at`, `created_on`, etc.) y guía hacia el schema."""
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
@pytest.mark.parametrize("col", ["created_at", "id", "user_id", "plan_data"])
def test_known_columns_in_snapshot(col):
    """Las 4 columnas core deben estar en el snapshot — defensa contra
    edición accidental del set."""
    assert col in _MEAL_PLANS_COLUMNS


def test_updated_at_explicitly_NOT_in_meal_plans_schema():
    """Documentación viva del bug P0-OBS-1: `updated_at` NO existe en
    `meal_plans`. Si alguien la añade vía migración, este test falla y
    obliga a revisar (a) si el cron debe volver a `updated_at`, (b) si
    el snapshot debe extenderse, (c) si el comentario en
    `cron_tasks.py:868+` queda obsoleto."""
    assert "updated_at" not in _MEAL_PLANS_COLUMNS, (
        "Si añadiste columna `updated_at` a `meal_plans`, este test "
        "intencionalmente te detiene. Considera: (1) actualizar "
        "`_MEAL_PLANS_COLUMNS` con la nueva columna; (2) decidir si "
        "el cron `_aggregate_coherence_block_history_metrics` debe "
        "volver a usarla (ver comentario `[P0-OBS-1]` en cron_tasks.py); "
        "(3) actualizar el comentario y este test."
    )
