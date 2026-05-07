"""[P1-1] Tests para _filter_lessons_excluding_dead_lettered.

Cubre el escenario fantasma:
  - Chunk N termina como dead-lettered (recovery_exhausted) tras agotar reintentos.
  - plan_data queda con _last_chunk_learning calculado sobre días que nunca
    completaron validación final.
  - Chunk N+1 NO debe propagar esas lecciones al prompt del LLM.

El helper centraliza la decisión: lee `_recovery_exhausted_chunks` (escrito por
_escalate_unrecoverable_chunk) y filtra:
  - last_chunk_learning: vacío si su `chunk` está en dead_weeks o si el
    predecesor inmediato fue dead-lettered.
  - recent_chunk_lessons: excluye items cuyo `chunk` está en dead_weeks,
    preserva el resto.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_1_failed_chunk_no_lesson_propagation.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cron_tasks import _filter_lessons_excluding_dead_lettered


def _make_lesson(chunk_num: int, repeat_pct: float = 50.0) -> dict:
    return {
        "chunk": chunk_num,
        "repeat_pct": repeat_pct,
        "ingredient_base_repeat_pct": repeat_pct,
        "rejection_violations": 1,
        "repeated_bases": [f"base_chunk_{chunk_num}"],
        "rejected_meals_that_reappeared": [f"meal_{chunk_num}"],
    }


# ---------------------------------------------------------------------------
# Caso base: sin dead-lettered → no filtra nada
# ---------------------------------------------------------------------------
def test_no_dead_lettered_returns_inputs_intact():
    last = _make_lesson(2)
    recent = [_make_lesson(1), _make_lesson(2)]
    plan_data = {"days": []}

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=3
    )

    assert f_last == last
    assert f_recent == recent
    assert dead == []


def test_empty_recovery_exhausted_chunks_returns_inputs_intact():
    last = _make_lesson(2)
    recent = [_make_lesson(1), _make_lesson(2)]
    plan_data = {"_recovery_exhausted_chunks": []}

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=3
    )

    assert f_last == last
    assert f_recent == recent
    assert dead == []


# ---------------------------------------------------------------------------
# Predecesor inmediato dead-lettered → vacía last_chunk_learning
# ---------------------------------------------------------------------------
def test_immediate_predecessor_dead_lettered_clears_last_chunk_learning():
    """Chunk 3 está corriendo; chunk 2 fue dead-lettered. last_chunk_learning
    de chunk 2 NO se debe propagar a chunk 3."""
    last = _make_lesson(2)
    recent = []
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 2, "escalated_at": "2026-05-01T10:00:00Z", "recovery_attempts": 2}
        ]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=3
    )

    assert f_last == {}, "last_chunk_learning del predecesor dead-lettered debe limpiarse"
    assert dead == [2]


def test_last_chunk_learning_chunk_field_in_dead_weeks_clears_it():
    """Aunque el predecesor inmediato no esté dead-lettered, si la lección
    persistida apunta a un chunk dead-lettered (desincronización), se limpia."""
    last = _make_lesson(1)  # apunta a chunk 1 dead-lettered
    recent = []
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 1, "recovery_attempts": 2}
        ]
    }

    # Chunk actual es 5; predecesor inmediato (4) NO está dead-lettered, pero
    # last_chunk_learning apunta a chunk 1 que sí lo está.
    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=5
    )

    assert f_last == {}
    assert dead == [1]


# ---------------------------------------------------------------------------
# recent_chunk_lessons: filtra items dead-lettered, preserva el resto
# ---------------------------------------------------------------------------
def test_recent_lessons_excludes_dead_lettered_preserves_others():
    """Chunks 1, 2, 3 en la ventana rolling. Chunk 2 fue dead-lettered.
    Resultado: 1 y 3 se conservan, 2 se filtra."""
    last = _make_lesson(3)
    recent = [_make_lesson(1), _make_lesson(2), _make_lesson(3)]
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 2, "recovery_attempts": 2}
        ]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=4
    )

    # last (chunk 3) sobrevive porque chunk 3 != dead_week (2) y predecesor
    # inmediato (3) tampoco es dead.
    assert f_last == last
    # recent: chunk 2 filtrado, 1 y 3 sobreviven.
    assert len(f_recent) == 2
    assert all(l.get("chunk") != 2 for l in f_recent)
    assert {l.get("chunk") for l in f_recent} == {1, 3}
    assert dead == [2]


def test_multiple_dead_lettered_chunks_all_filtered():
    last = _make_lesson(4)
    recent = [_make_lesson(1), _make_lesson(2), _make_lesson(3), _make_lesson(4)]
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 1, "recovery_attempts": 2},
            {"week_number": 3, "recovery_attempts": 2},
        ]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=5
    )

    assert {l.get("chunk") for l in f_recent} == {2, 4}
    assert dead == [1, 3]


# ---------------------------------------------------------------------------
# Robustez frente a plan_data corrupto
# ---------------------------------------------------------------------------
def test_prior_plan_data_not_dict_returns_inputs():
    last = _make_lesson(2)
    recent = [_make_lesson(1)]

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, "not_a_dict", current_week_number=3
    )

    assert f_last == last
    assert f_recent == recent
    assert dead == []


def test_recovery_exhausted_not_a_list_returns_inputs():
    last = _make_lesson(2)
    recent = [_make_lesson(1)]
    plan_data = {"_recovery_exhausted_chunks": "corrupted_string"}

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=3
    )

    assert f_last == last
    assert f_recent == recent
    assert dead == []


def test_recovery_exhausted_entry_missing_week_number_skipped():
    """Entrada malformada (sin week_number) no rompe el helper."""
    last = _make_lesson(2)
    recent = [_make_lesson(1), _make_lesson(2)]
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"escalated_at": "2026-05-01T10:00:00Z"},  # sin week_number
            {"week_number": 2},
        ]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=3
    )

    # Solo week 2 cuenta como dead.
    assert dead == [2]
    assert f_last == {}  # predecesor inmediato (2) dead


def test_current_week_number_non_int_no_crash():
    last = _make_lesson(1)
    recent = []
    plan_data = {
        "_recovery_exhausted_chunks": [{"week_number": 1}]
    }

    # current_week_number como string no parseable → no crashea.
    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number="N/A"
    )

    # last apunta a chunk 1 (dead) → se limpia por match de chunk field,
    # no por predecesor (que no se puede calcular).
    assert f_last == {}
    assert dead == [1]


def test_lesson_without_chunk_field_preserved_in_recent():
    """Lecciones legacy sin `chunk` field se conservan en recent (no podemos
    saber si vienen de dead-lettered)."""
    last = {}
    legacy_lesson = {"repeat_pct": 30, "repeated_bases": ["arroz"]}  # sin chunk
    recent = [legacy_lesson, _make_lesson(2)]
    plan_data = {
        "_recovery_exhausted_chunks": [{"week_number": 2}]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=4
    )

    # Legacy preservada, chunk 2 filtrado.
    assert legacy_lesson in f_recent
    assert all(l.get("chunk") != 2 for l in f_recent if isinstance(l, dict))


# ---------------------------------------------------------------------------
# Caso: predecesor SÍ dead-lettered pero hay lecciones válidas previas
# ---------------------------------------------------------------------------
def test_predecessor_dead_but_earlier_lessons_kept():
    """Chunk 5 corre, chunk 4 fue dead-lettered, pero chunks 1-3 son válidos.
    Resultado: last se vacía (por predecesor 4 dead), recent conserva 1-3."""
    last = _make_lesson(4)  # del predecesor dead
    recent = [_make_lesson(1), _make_lesson(2), _make_lesson(3), _make_lesson(4)]
    plan_data = {
        "_recovery_exhausted_chunks": [{"week_number": 4, "recovery_attempts": 2}]
    }

    f_last, f_recent, dead = _filter_lessons_excluding_dead_lettered(
        last, recent, plan_data, current_week_number=5
    )

    assert f_last == {}
    assert {l.get("chunk") for l in f_recent} == {1, 2, 3}
    assert dead == [4]
