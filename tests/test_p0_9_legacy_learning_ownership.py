"""[P0-9] Tests para garantizar que `persist_legacy_learning_to_plan_data`
NO escribe lecciones cross-user al `meal_plans.plan_data` ajeno.

Bug original (audit P0-9):
  El UPDATE era `WHERE id = %s` con sólo `meal_plan_id`. La función se llama
  desde 4 contextos legacy (`P0_3_LEGACY_LEARNING_CONTEXTS`):
  `seed_chunk1_sync`, `seed_chunk1_sse`, `rebuild_from_queue`,
  `synthesis_from_days`. Si cualquier path upstream pasa un `meal_plan_id`
  que NO pertenece al usuario actual (bug, swap de variable, race con
  `save_new_meal_plan_atomic`), la lección de un usuario A sobrescribiría
  `_last_chunk_learning` / `_recent_chunk_lessons` del plan_data del usuario
  B sin guardia. El siguiente chunk del usuario B arrancaría con datos
  cross-user contaminando el prompt LLM.

  El test `test_p0_chunks_1_synthesized_lesson_isolation.py` cubre la mezcla
  synth/real DENTRO del agregador, NO la integridad cross-user a nivel
  persistencia.

Fix:
  - `user_id` ahora es kw-only requerido en `persist_legacy_learning_to_plan_data`.
  - El SQL incluye `AND user_id = %s` en el WHERE (ambas variantes con/sin
    `recent_chunk_lessons`).
  - `RETURNING id` permite detectar rowcount=0 → log ERROR
    `[P0.3/P0-9/CROSS-USER]` y devolver False sin escribir.

Cobertura:
  - test_helper_requires_user_id_keyword
  - test_helper_returns_false_when_user_id_empty
  - test_helper_includes_user_id_in_where_clause
  - test_helper_uses_returning_id_for_rowcount_detection
  - test_helper_returns_false_on_cross_user_mismatch_rowcount_zero
  - test_helper_logs_error_on_cross_user_mismatch
  - test_helper_with_recent_lessons_also_filters_by_user_id
  - test_persist_legacy_signature_user_id_is_required
"""
import inspect
import logging
from unittest.mock import patch, MagicMock

import pytest

import cron_tasks


# ---------------------------------------------------------------------------
# 1. Signature: user_id es kw-only requerido.
# ---------------------------------------------------------------------------
def test_persist_legacy_signature_user_id_is_required():
    """`user_id` debe ser parámetro requerido (kw-only) — un caller que no
    pase user_id genera TypeError en lugar de escribir cross-user."""
    sig = inspect.signature(cron_tasks.persist_legacy_learning_to_plan_data)
    assert "user_id" in sig.parameters
    user_id_param = sig.parameters["user_id"]
    # KEYWORD_ONLY: viene después de `*` separator → solo se acepta por nombre
    assert user_id_param.kind == inspect.Parameter.KEYWORD_ONLY, \
        f"user_id debe ser kw-only para evitar passing posicional accidental, got {user_id_param.kind}"
    # Sin default → required.
    assert user_id_param.default is inspect.Parameter.empty, \
        "user_id NO debe tener default — el caller debe proveerlo explícitamente"


def test_helper_requires_user_id_keyword():
    """Llamar el helper sin `user_id` debe lanzar TypeError, no caer al SQL."""
    with patch.object(cron_tasks, "execute_sql_write", MagicMock()) as wm:
        with pytest.raises(TypeError):
            cron_tasks.persist_legacy_learning_to_plan_data(
                "plan-1", {"chunk": 1}, context="rebuild_from_queue",
            )
        wm.assert_not_called()


# ---------------------------------------------------------------------------
# 2. user_id falsy aborta antes del SQL.
# ---------------------------------------------------------------------------
def test_helper_returns_false_when_user_id_empty():
    """Si user_id llega vacío (None/""/0), el helper aborta sin emitir SQL."""
    captured = []
    write_mock = MagicMock(side_effect=lambda *a, **kw: captured.append(a))
    with patch.object(cron_tasks, "execute_sql_write", write_mock):
        for bad in (None, "", 0):
            ok = cron_tasks.persist_legacy_learning_to_plan_data(
                "plan-1", {"chunk": 1}, context="rebuild_from_queue",
                user_id=bad,
            )
            assert ok is False, f"user_id={bad!r} debe abortar"
    assert captured == [], "no debe emitir SQL con user_id falsy"


# ---------------------------------------------------------------------------
# 3. WHERE incluye AND user_id = %s.
# ---------------------------------------------------------------------------
def test_helper_includes_user_id_in_where_clause():
    """El SQL emitido debe filtrar por user_id además de id, garantizando
    que un caller no pueda mutar plan_data de un plan ajeno aunque pase
    el meal_plan_id correcto del víctima."""
    captured = {}

    def _capture(query, params=None, returning=False):
        captured["query"] = query
        captured["params"] = params
        return [{"id": "plan-1"}] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_capture):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1}, context="rebuild_from_queue",
            user_id="user-a",
        )

    assert ok is True
    sql = captured["query"]
    assert "WHERE id = %s AND user_id = %s" in sql, \
        f"P0-9: WHERE debe filtrar por user_id, got: {sql!r}"
    # Y user_id está en params (último arg).
    assert "user-a" in captured["params"], \
        f"user_id no está en params: {captured['params']}"


def test_helper_with_recent_lessons_also_filters_by_user_id():
    """La variante con `recent_chunk_lessons` debe igualmente filtrar."""
    captured = {}

    def _capture(query, params=None, returning=False):
        captured["query"] = query
        captured["params"] = params
        return [{"id": "plan-1"}] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_capture):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1},
            recent_chunk_lessons=[{"chunk": 1}],
            context="seed_chunk1_sync",
            user_id="user-a",
        )

    assert ok is True
    sql = captured["query"]
    assert "WHERE id = %s AND user_id = %s" in sql
    assert "user-a" in captured["params"]


# ---------------------------------------------------------------------------
# 4. RETURNING id detecta rowcount=0.
# ---------------------------------------------------------------------------
def test_helper_uses_returning_id_for_rowcount_detection():
    """El SQL debe pedir `RETURNING id` para que el helper pueda distinguir
    UPDATE-con-match (rows!=[]) de UPDATE-sin-match (rows==[]) y log ERROR
    sobre el cross-user attempt."""
    captured = {}

    def _capture(query, params=None, returning=False):
        captured["query"] = query
        captured["returning"] = returning
        return [{"id": "plan-1"}] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_capture):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1}, context="rebuild_from_queue",
            user_id="user-a",
        )

    assert ok is True
    assert "RETURNING id" in captured["query"], "P0-9: debe pedir RETURNING id"
    assert captured["returning"] is True, \
        "P0-9: debe llamar execute_sql_write con returning=True"


def test_helper_returns_false_on_cross_user_mismatch_rowcount_zero():
    """Si el (id, user_id) no coincide (cross-user attempt o plan eliminado),
    `execute_sql_write` con `returning=True` retorna `[]`. El helper debe
    devolver False, NO escribir lección, y log ERROR."""

    def _no_match(query, params=None, returning=False):
        # Simulamos UPDATE que no afectó filas (filtro user_id falló).
        return [] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_no_match):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "victim-plan-id", {"chunk": 1, "evil": "lesson from user A"},
            context="rebuild_from_queue",
            user_id="attacker-user-id",
        )

    assert ok is False, \
        "P0-9: cross-user mismatch debe devolver False (no escribir)"


def test_helper_logs_error_on_cross_user_mismatch(caplog):
    """El log ERROR debe contener `[P0.3/P0-9/CROSS-USER]` para que SRE pueda
    alertar específicamente sobre intentos de cross-user write."""

    def _no_match(query, params=None, returning=False):
        return [] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_no_match):
        with caplog.at_level(logging.ERROR, logger=cron_tasks.logger.name):
            cron_tasks.persist_legacy_learning_to_plan_data(
                "victim-plan-id", {"chunk": 1},
                context="rebuild_from_queue",
                user_id="attacker-user-id",
            )

    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("CROSS-USER" in m for m in error_msgs), \
        f"P0-9: falta log ERROR con marker CROSS-USER, got: {error_msgs}"
    # El log debe incluir IDs para diagnostico.
    assert any("victim-plan-id" in m and "attacker-user-id" in m for m in error_msgs), \
        f"P0-9: log debe incluir meal_plan_id + user_id ofensores"


# ---------------------------------------------------------------------------
# 5. Defensa textual contra reintroducción del bug.
# ---------------------------------------------------------------------------
def test_source_no_longer_uses_naked_where_id_only_for_legacy_learning():
    """Defensa: el SQL del helper NO debe contener `WHERE id = %s` sin un
    AND user_id. Si alguien borra el filtro accidentalmente, este test rompe."""
    src = inspect.getsource(cron_tasks.persist_legacy_learning_to_plan_data)
    # Buscamos `WHERE id = %s` que NO esté seguido por `AND user_id`.
    import re
    bad = re.findall(r"WHERE\s+id\s*=\s*%s\s*\n(?!\s*AND\s+user_id)", src)
    assert not bad, (
        f"P0-9 regression: WHERE id = %s sin AND user_id reapareció: {bad}"
    )


def test_documentation_p0_9_present():
    """Comentario `[P0-9]` debe estar presente para documentar el contrato."""
    src = inspect.getsource(cron_tasks.persist_legacy_learning_to_plan_data)
    assert "[P0-9]" in src, "P0-9: falta documentación de auditoría"
