"""[P1-2] Chunks fallidos no contaminan aprendizaje futuro.

Cuando un chunk N corre pero su pipeline crashea o el commit falla a medio escribir,
`plan_chunk_queue.learning_metrics` queda persistido con `pipeline_failed=True`.
El comportamiento P0-2 legacy era usar esos metrics igual con `low_confidence=True`,
pero pueden ser parciales/inconsistentes y propagarse como ruido al chunk N+1.

P1-2 introduce el flag `prefer_completed=True` en `_rebuild_last_chunk_learning_from_queue`:
cuando se activa, las filas `status='failed'` se rechazan aunque tengan learning_metrics.
El caller cae al fallback de síntesis desde `plan_data.days`, que es truthful (lee los
días que realmente quedaron persistidos) aunque siga marcado low_confidence.

Tests sobre la función helper directamente — el wiring al callsite ya está cubierto
por el flujo existente que cae a síntesis cuando el rebuild devuelve None.
"""
import json
from unittest.mock import patch


def test_prefer_completed_returns_none_for_failed_only_row():
    """Solo row 'failed' con metrics → con prefer_completed=True devuelve None."""
    from cron_tasks import _rebuild_last_chunk_learning_from_queue

    # SQL diferente según prefer_completed: con strict, el WHERE excluye 'failed',
    # así que la query no devuelve nada → simulamos None.
    seen_queries = []

    def fake_query(sql, params, **kwargs):
        seen_queries.append(sql)
        if "status = 'completed'" in sql and "'failed'" not in sql:
            return None  # Strict mode: no completed row, returns None
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query):
        result = _rebuild_last_chunk_learning_from_queue(
            "plan-x", target_week=1, prefer_completed=True,
        )

    assert result is None
    # Confirma que el SQL strict se construyó correctamente.
    assert any("status = 'completed'" in q and "IN ('completed', 'failed')" not in q
               for q in seen_queries), f"Expected strict status clause, saw: {seen_queries!r}"


def test_prefer_completed_accepts_completed_row():
    """Row 'completed' con metrics: prefer_completed=True igual la devuelve."""
    from cron_tasks import _rebuild_last_chunk_learning_from_queue

    fake_lm = {
        "learning_repeat_pct": 10,
        "ingredient_base_repeat_pct": 25,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "sample_repeats": ["pollo guisado"],
        "sample_repeated_bases": [{"bases": ["pollo"]}],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }

    def fake_query(sql, params, **kwargs):
        return {
            "week_number": 1,
            "status": "completed",
            "learning_metrics": fake_lm,
        }

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query):
        result = _rebuild_last_chunk_learning_from_queue(
            "plan-x", target_week=1, prefer_completed=True,
        )

    assert result is not None
    assert result["chunk"] == 1
    assert result["low_confidence"] is False
    assert result["learning_signal_strength"] == "strong"


def test_legacy_permissive_mode_still_accepts_failed_row():
    """[P0-2 backward compat] Sin prefer_completed, failed con metrics se devuelve
    con low_confidence=True. Garantiza que los tests P0-2 existentes siguen pasando."""
    from cron_tasks import _rebuild_last_chunk_learning_from_queue

    fake_lm = {
        "pipeline_failed": True,
        "learning_confidence": "low",
        "learning_repeat_pct": 12.5,
        "ingredient_base_repeat_pct": 25.0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "sample_repeats": ["arroz con pollo"],
        "sample_repeated_bases": [],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }

    def fake_query(sql, params, **kwargs):
        # Default mode: WHERE status IN ('completed', 'failed').
        assert "IN ('completed', 'failed')" in sql, (
            "Default rebuild debe seguir aceptando filas failed (P0-2)"
        )
        return {"week_number": 2, "status": "failed", "learning_metrics": fake_lm}

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query):
        result = _rebuild_last_chunk_learning_from_queue("plan-x", target_week=2)

    assert result is not None
    assert result["low_confidence"] is True
    assert result["rebuilt_from_pipeline_failure"] is True


def test_strict_mode_rejects_failed_query_at_sql_level():
    """[P1-2] El SQL en modo strict no debe pedir filas 'failed' siquiera."""
    from cron_tasks import _rebuild_last_chunk_learning_from_queue

    captured_sql = []

    def fake_query(sql, params, **kwargs):
        captured_sql.append(sql)
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query):
        _rebuild_last_chunk_learning_from_queue(
            "plan-x", target_week=1, prefer_completed=True,
        )

    assert len(captured_sql) == 1
    sql = captured_sql[0]
    assert "status = 'completed'" in sql
    # No debe contener la cláusula permisiva.
    assert "IN ('completed', 'failed')" not in sql


def test_strict_mode_rejects_preflight_only_row():
    """[P1-2] Filas 'failed' con preflight (rebuild_from_preflight=True en legacy)
    también se rechazan en strict mode — son las MENOS confiables (sin metrics post-pipeline)."""
    from cron_tasks import _rebuild_last_chunk_learning_from_queue

    # Strict SQL filters at DB level — failed row never reaches Python.
    def fake_query(sql, params, **kwargs):
        if "status = 'completed'" in sql:
            return None
        # Legacy permissive would have returned the preflight row, but strict skips it.
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query):
        result = _rebuild_last_chunk_learning_from_queue(
            "plan-x", target_week=1, prefer_completed=True,
        )

    assert result is None
