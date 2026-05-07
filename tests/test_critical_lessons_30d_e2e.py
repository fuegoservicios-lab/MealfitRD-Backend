"""
Test E2E simulado de 30 días (P0-7): verifica que lecciones críticas del chunk 1
(ej. rechazo fuerte de "pollo") persistan hasta el chunk 9 a pesar del rolling
window de _recent_chunk_lessons (cap 8).
"""
import pytest
import copy
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# ---------- helpers para simular la lógica de persistencia de lecciones ----------

def _is_lesson_critical(lesson: dict) -> bool:
    """Replica la lógica de cron_tasks.py para decidir si una lección es crítica."""
    return (
        int(lesson.get('rejection_violations') or 0) > 0
        or int(lesson.get('allergy_violations') or 0) > 0
        or float(lesson.get('ingredient_base_repeat_pct') or 0) >= 85.0
        or (
            lesson.get('learning_signal_strength') == 'strong'
            and not lesson.get('low_confidence')
            and not lesson.get('metrics_unavailable')
        )
    )


def _simulate_lesson_persistence(plan_data: dict, new_lesson: dict, week_number: int, win_size: int = 8, critical_max: int = 50):
    """Simula exactamente la lógica de persistencia de cron_tasks.py post-merge."""
    plan_data['_last_chunk_learning'] = new_lesson

    # Rolling window
    _recent = plan_data.get('_recent_chunk_lessons', [])
    if not isinstance(_recent, list):
        _recent = []
    _recent.append(new_lesson)
    plan_data['_recent_chunk_lessons'] = _recent[-win_size:]

    # Critical permanent
    _critical = plan_data.get('_critical_lessons_permanent', [])
    if not isinstance(_critical, list):
        _critical = []
    if _is_lesson_critical(new_lesson):
        new_lesson['_critical'] = True
        _critical.append(new_lesson)
        _critical = _critical[-critical_max:]
        plan_data['_critical_lessons_permanent'] = _critical


def _simulate_lesson_injection(plan_data: dict, week_number: int):
    """Simula la lógica de inyección en el LLM (P0-7 merge de critical + rolling)."""
    last_chunk_learning = plan_data.get('_last_chunk_learning', {})
    recent_chunk_lessons = plan_data.get('_recent_chunk_lessons', [])
    critical_permanent = plan_data.get('_critical_lessons_permanent', [])

    if not isinstance(critical_permanent, list):
        critical_permanent = []

    # Dedup por chunk number
    _recent_chunk_nums = set()
    for rcl in recent_chunk_lessons:
        if isinstance(rcl, dict) and rcl.get('chunk') is not None:
            _recent_chunk_nums.add(rcl['chunk'])
    if last_chunk_learning and last_chunk_learning.get('chunk') is not None:
        _recent_chunk_nums.add(last_chunk_learning['chunk'])

    critical_extras = [
        cl for cl in critical_permanent
        if isinstance(cl, dict) and cl.get('chunk') not in _recent_chunk_nums
    ]

    all_lessons = ([last_chunk_learning] if last_chunk_learning else []) + recent_chunk_lessons + critical_extras
    return all_lessons


# ---------- Tests ----------

class TestCriticalLessonsPermanent:
    """P0-7: Las lecciones críticas del chunk 1 persisten hasta el chunk 9+."""

    def test_strong_rejection_survives_full_30d_plan(self):
        """
        Simula un plan de 30 días (10 chunks).
        Chunk 1: marca "pollo" como rechazo fuerte (rejection_violations=1).
        Chunks 2-9: lecciones normales sin violaciones.
        Chunk 10: al inyectar, "pollo" debe seguir presente en las restricciones.
        """
        plan_data = {
            'total_days_requested': 30,
            '_recent_chunk_lessons': [],
            '_critical_lessons_permanent': [],
        }

        # Chunk 1: lección CRÍTICA - rechazo fuerte de "pollo"
        critical_lesson_chunk_1 = {
            'chunk': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rejection_violations': 1,
            'allergy_violations': 0,
            'ingredient_base_repeat_pct': 20.0,
            'repeat_pct': 10.0,
            'rejected_meals_that_reappeared': ['Pollo a la plancha'],
            'repeated_bases': [{'bases': ['pollo'], 'count': 3}],
            'allergy_hits': [],
            'metrics_unavailable': False,
            'low_confidence': False,
            'learning_signal_strength': 'strong',
        }
        _simulate_lesson_persistence(plan_data, critical_lesson_chunk_1, week_number=1)

        # Verificar que se guardó como crítica
        assert len(plan_data['_critical_lessons_permanent']) == 1
        assert plan_data['_critical_lessons_permanent'][0]['chunk'] == 1

        # Chunks 2-9: lecciones normales (sin violaciones, señal débil)
        for chunk_num in range(2, 10):
            normal_lesson = {
                'chunk': chunk_num,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'rejection_violations': 0,
                'allergy_violations': 0,
                'ingredient_base_repeat_pct': 15.0,
                'repeat_pct': 5.0,
                'rejected_meals_that_reappeared': [],
                'repeated_bases': [],
                'allergy_hits': [],
                'metrics_unavailable': False,
                'low_confidence': True,
                'learning_signal_strength': 'weak',
            }
            _simulate_lesson_persistence(plan_data, normal_lesson, week_number=chunk_num)

        # Tras 9 chunks, el rolling window (cap 8) ya borró el chunk 1
        recent_chunks = [l.get('chunk') for l in plan_data['_recent_chunk_lessons']]
        assert 1 not in recent_chunks, \
            f"Chunk 1 debería haber salido del rolling window, pero está en: {recent_chunks}"

        # Sin embargo, la lección crítica del chunk 1 DEBE seguir en permanent
        assert len(plan_data['_critical_lessons_permanent']) == 1
        assert plan_data['_critical_lessons_permanent'][0]['chunk'] == 1

        # Ahora simular la inyección para chunk 10
        all_lessons = _simulate_lesson_injection(plan_data, week_number=10)

        # Verificar que "pollo" sigue siendo visible en all_lessons
        all_rejected = []
        all_bases = []
        for lesson in all_lessons:
            if not isinstance(lesson, dict):
                continue
            all_rejected.extend(lesson.get('rejected_meals_that_reappeared', []))
            for rb in (lesson.get('repeated_bases') or []):
                if isinstance(rb, dict):
                    all_bases.extend(rb.get('bases', []))

        assert 'Pollo a la plancha' in all_rejected, \
            f"El rechazo de 'Pollo a la plancha' debería persistir hasta chunk 10. Rechazos visibles: {all_rejected}"
        assert 'pollo' in all_bases, \
            f"La base 'pollo' debería persistir hasta chunk 10. Bases visibles: {all_bases}"

    def test_allergy_violation_is_critical(self):
        """Verifica que violaciones de alergia se marquen como críticas."""
        plan_data = {
            'total_days_requested': 15,
            '_recent_chunk_lessons': [],
            '_critical_lessons_permanent': [],
        }

        allergy_lesson = {
            'chunk': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rejection_violations': 0,
            'allergy_violations': 2,
            'ingredient_base_repeat_pct': 10.0,
            'allergy_hits': ['gluten', 'maní'],
            'repeated_bases': [],
            'rejected_meals_that_reappeared': [],
            'metrics_unavailable': False,
            'low_confidence': False,
            'learning_signal_strength': 'strong',
        }
        _simulate_lesson_persistence(plan_data, allergy_lesson, week_number=1)

        assert len(plan_data['_critical_lessons_permanent']) == 1
        assert plan_data['_critical_lessons_permanent'][0].get('_critical') is True

    def test_high_base_repeat_is_critical(self):
        """Verifica que ingredient_base_repeat_pct >= 85% sea crítica."""
        plan_data = {'_recent_chunk_lessons': [], '_critical_lessons_permanent': []}
        lesson = {
            'chunk': 3,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rejection_violations': 0,
            'allergy_violations': 0,
            'ingredient_base_repeat_pct': 90.0,
            'repeated_bases': [{'bases': ['arroz'], 'count': 5}],
            'rejected_meals_that_reappeared': [],
            'allergy_hits': [],
            'metrics_unavailable': False,
            'low_confidence': False,
            'learning_signal_strength': 'strong',
        }
        _simulate_lesson_persistence(plan_data, lesson, week_number=3)

        assert len(plan_data['_critical_lessons_permanent']) == 1

    def test_normal_lesson_not_critical(self):
        """Verifica que lecciones normales de señal débil NO se almacenen."""
        plan_data = {'_recent_chunk_lessons': [], '_critical_lessons_permanent': []}
        lesson = {
            'chunk': 2,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rejection_violations': 0,
            'allergy_violations': 0,
            'ingredient_base_repeat_pct': 15.0,
            'repeated_bases': [],
            'rejected_meals_that_reappeared': [],
            'allergy_hits': [],
            'metrics_unavailable': False,
            'low_confidence': True,
            'learning_signal_strength': 'weak',
        }
        _simulate_lesson_persistence(plan_data, lesson, week_number=2)

        assert len(plan_data['_critical_lessons_permanent']) == 0

    def test_critical_dedup_in_injection(self):
        """Si la lección crítica aún está en el rolling window, no se duplica."""
        plan_data = {
            '_recent_chunk_lessons': [],
            '_critical_lessons_permanent': [],
        }

        # Chunk 1: crítico
        lesson = {
            'chunk': 1,
            'rejection_violations': 1,
            'allergy_violations': 0,
            'ingredient_base_repeat_pct': 10.0,
            'rejected_meals_that_reappeared': ['Arroz con pollo'],
            'repeated_bases': [],
            'allergy_hits': [],
            'metrics_unavailable': False,
            'low_confidence': False,
            'learning_signal_strength': 'strong',
        }
        _simulate_lesson_persistence(plan_data, lesson, week_number=1)

        # Chunk 2: normal
        lesson2 = {
            'chunk': 2,
            'rejection_violations': 0,
            'allergy_violations': 0,
            'ingredient_base_repeat_pct': 5.0,
            'rejected_meals_that_reappeared': [],
            'repeated_bases': [],
            'allergy_hits': [],
            'metrics_unavailable': False,
            'low_confidence': True,
            'learning_signal_strength': 'weak',
        }
        _simulate_lesson_persistence(plan_data, lesson2, week_number=2)

        # Chunk 1 todavía está en recent window
        all_lessons = _simulate_lesson_injection(plan_data, week_number=3)

        # Contar cuántas veces aparece chunk 1
        chunk_1_count = sum(1 for l in all_lessons if isinstance(l, dict) and l.get('chunk') == 1)
        assert chunk_1_count == 1, f"Chunk 1 debería aparecer solo 1 vez, no {chunk_1_count}"

    def test_cap_at_50(self):
        """Verifica que las critical lessons se capean a 50."""
        plan_data = {'_recent_chunk_lessons': [], '_critical_lessons_permanent': []}

        for i in range(60):
            lesson = {
                'chunk': i + 1,
                'rejection_violations': 1,
                'allergy_violations': 0,
                'ingredient_base_repeat_pct': 10.0,
                'rejected_meals_that_reappeared': [f'Receta_{i}'],
                'repeated_bases': [],
                'allergy_hits': [],
                'metrics_unavailable': False,
                'low_confidence': False,
                'learning_signal_strength': 'strong',
            }
            _simulate_lesson_persistence(plan_data, lesson, week_number=i + 1)

        assert len(plan_data['_critical_lessons_permanent']) == 50
        # Los más recientes deben ser los que sobreviven
        assert plan_data['_critical_lessons_permanent'][0]['chunk'] == 11
        assert plan_data['_critical_lessons_permanent'][-1]['chunk'] == 60
