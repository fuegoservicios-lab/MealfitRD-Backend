"""[P1-4] Decay temporal de lecciones en `_lifetime_lessons_summary`.

Antes del fix:
    El recompute del summary (worker `_chunk_worker` y db_plans
    `_inherit_lifetime_lessons_from_prior_plan`) trataba todas las lecciones
    igual:
      - `top_rejection_hits` y `top_repeated_bases` se construían como sets,
        sin ranking por recencia.
      - `top_repeated_meal_names` se ordenaba por count distinto de chunks,
        ignorando edad.

    En plan 30d con ~10 chunks (8 semanas de plan), una lección del chunk 1
    competía con una del chunk 9 por los caps (top 20, top 30) — un patrón
    superado hace 2 meses bloqueaba a uno relevante de la última semana.
    En cross-plan inheritance el problema era idéntico: el plan nuevo
    arrancaba con summary cuyo ranking era arbitrario.

Después del fix:
    1. Helper `compute_lifetime_lesson_weight(lesson, now)` retorna
       `LIFETIME_LESSON_WEEKLY_DECAY ** weeks_old` (default 0.9).
    2. El recompute aggrega weights en lugar de sets/counts. Items con
       weight < `LIFETIME_LESSON_MIN_WEIGHT` (default 0.10, ~22 semanas)
       se excluyen del ranking — preservados implícitamente por el cutoff
       de LIFETIME_LESSONS_WINDOW_DAYS pero forward-compat para ventanas amplias.
    3. Sort descendente por weight: items recientes ocupan los primeros slots
       de los caps.
"""
from datetime import datetime, timezone, timedelta

import pytest


def test_constants_default_values():
    """Defaults razonables: decay 0.9 → 1 semana = 0.9, 8 semanas = 0.43."""
    from constants import LIFETIME_LESSON_WEEKLY_DECAY, LIFETIME_LESSON_MIN_WEIGHT

    assert 0 < LIFETIME_LESSON_WEEKLY_DECAY <= 1.0, (
        "Decay > 1 invertiría el efecto (lecciones viejas pesarían más); "
        "decay <= 0 anularía el ranking."
    )
    assert LIFETIME_LESSON_WEEKLY_DECAY == 0.9, (
        "Default 0.9 es el balance recomendado: 1 semana = 0.9, 8 semanas = "
        "0.43, 22 semanas = 0.10. Si lo cambias, valida que el ranking se "
        "comporta como esperabas con tus datos reales."
    )
    assert 0.0 <= LIFETIME_LESSON_MIN_WEIGHT < 1.0, (
        "MIN_WEIGHT >= 1.0 filtraría TODO; MIN_WEIGHT < 0 no tiene sentido."
    )


def test_weight_no_timestamp_assumed_recent():
    """Una lesson sin `timestamp` se asume reciente (peso 1.0). Caso típico:
    stub puro persistido en P0-3 cuando learning_metrics no estaba disponible.
    Falla cerrada (descartar) sería peor: perderíamos la señal del chunk."""
    from cron_tasks import compute_lifetime_lesson_weight
    assert compute_lifetime_lesson_weight({}) == 1.0
    assert compute_lifetime_lesson_weight({"chunk": 1, "rejection_violations": 5}) == 1.0


def test_weight_decreases_with_age():
    """Lessons con timestamp viejo pesan menos. Verificación con valores
    exactos para los puntos de referencia del default."""
    from cron_tasks import compute_lifetime_lesson_weight

    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 1 semana: 0.9^1 = 0.9
    lesson_1w = {"timestamp": (now - timedelta(weeks=1)).isoformat()}
    assert abs(compute_lifetime_lesson_weight(lesson_1w, now=now) - 0.9) < 0.001

    # 4 semanas: 0.9^4 ≈ 0.6561
    lesson_4w = {"timestamp": (now - timedelta(weeks=4)).isoformat()}
    assert abs(compute_lifetime_lesson_weight(lesson_4w, now=now) - 0.6561) < 0.001

    # 8 semanas: 0.9^8 ≈ 0.4305
    lesson_8w = {"timestamp": (now - timedelta(weeks=8)).isoformat()}
    assert abs(compute_lifetime_lesson_weight(lesson_8w, now=now) - 0.4305) < 0.001


def test_weight_future_timestamp_clamped_to_one():
    """Clock skew: si el timestamp es en el futuro (e.g. snapshot inválido),
    no devolvemos un peso > 1 (sería una sobreponderación arbitraria).
    Tratamos como "ahora" → peso 1.0."""
    from cron_tasks import compute_lifetime_lesson_weight
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    future = {"timestamp": (now + timedelta(days=30)).isoformat()}
    assert compute_lifetime_lesson_weight(future, now=now) == 1.0


def test_weight_unparseable_timestamp_falls_back_to_one():
    """Timestamp corrupto (string mal formado, número aleatorio): fallamos
    open con peso 1.0. La primera línea de defensa es el cutoff por
    LIFETIME_LESSONS_WINDOW_DAYS, que ya filtra entries muy viejas."""
    from cron_tasks import compute_lifetime_lesson_weight
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_lifetime_lesson_weight({"timestamp": "garbage"}, now=now) == 1.0
    assert compute_lifetime_lesson_weight({"timestamp": "2026-13-99"}, now=now) == 1.0


def test_weight_returns_zero_for_non_dict():
    """Robustez: lessons malformadas (None, list, str) → peso 0 (no contribuye)."""
    from cron_tasks import compute_lifetime_lesson_weight
    assert compute_lifetime_lesson_weight(None) == 0.0
    assert compute_lifetime_lesson_weight([1, 2, 3]) == 0.0
    assert compute_lifetime_lesson_weight("not a lesson") == 0.0


def test_old_lesson_below_min_weight_is_skipped_in_summary():
    """Forward-compat: si LIFETIME_LESSONS_WINDOW_DAYS se amplía a 1 año,
    las lessons con weight < MIN_WEIGHT (peso 0.1 ≈ 22 semanas) se excluyen
    del summary aún cuando entren al historial filtrado."""
    from cron_tasks import compute_lifetime_lesson_weight
    from constants import LIFETIME_LESSON_MIN_WEIGHT

    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    very_old = {"timestamp": (now - timedelta(weeks=30)).isoformat()}
    weight = compute_lifetime_lesson_weight(very_old, now=now)
    assert weight < LIFETIME_LESSON_MIN_WEIGHT, (
        f"30 semanas → peso {weight:.4f}, debe ser < MIN_WEIGHT "
        f"({LIFETIME_LESSON_MIN_WEIGHT}) para validar el filtro."
    )


def test_db_plans_inheritance_ranks_recent_lessons_higher():
    """Cross-plan inheritance: el plan nuevo hereda summary del plan previo
    con ranking por recencia. Si plan previo tiene "pollo" rejected en chunk
    1 (8 semanas atrás) y "salmón" rejected en chunk 9 (1 semana atrás), el
    summary heredado debe rankear "salmón" antes que "pollo" — sin esto, el
    primer chunk del plan nuevo recibe un prompt con bias hacia patrones
    obsoletos del plan anterior.
    """
    from unittest.mock import patch
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    # Lesson vieja (8 semanas atrás) que reporta "pollo" rejected.
    old_lesson = {
        "chunk": 1,
        "timestamp": (now - timedelta(weeks=8)).isoformat(),
        "rejection_violations": 2,
        "rejected_meals_that_reappeared": ["pollo"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "allergy_violations": 0,
    }
    # Lesson reciente (1 semana atrás) que reporta "salmón" rejected.
    new_lesson = {
        "chunk": 9,
        "timestamp": (now - timedelta(weeks=1)).isoformat(),
        "rejection_violations": 1,
        "rejected_meals_that_reappeared": ["salmón"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "allergy_violations": 0,
    }
    fake_row = {
        "id": "plan-prev-uuid",
        "plan_data": {
            "_lifetime_lessons_history": [old_lesson, new_lesson],
            # Sin _lifetime_lessons_summary → forza recompute.
        },
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p14")

    assert result is not None
    summary = result["summary"]
    rej_hits = summary["top_rejection_hits"]
    assert "salmón" in rej_hits and "pollo" in rej_hits, (
        "Ambas rejections deben estar en el summary; sólo el ranking cambia."
    )
    assert rej_hits.index("salmón") < rej_hits.index("pollo"), (
        "Recency-weighted ranking: 'salmón' (1 semana, peso 0.9) debe ir "
        "antes que 'pollo' (8 semanas, peso 0.43). Si el orden es inverso, "
        "el decay no se está aplicando en el inheritance recompute."
    )
    # totals son brutos (no decayed) — telemetría histórica acumulada.
    assert summary["total_rejection_violations"] == 3  # 2 + 1


def test_db_plans_inheritance_filters_below_min_weight():
    """Lessons con peso < MIN_WEIGHT NO contribuyen al ranking heredado.
    Total counts SÍ siguen incluyéndolas (telemetría histórica bruta)."""
    from unittest.mock import patch
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    # 30 semanas atrás → peso ≈ 0.04, < MIN_WEIGHT (0.1) → filtrado del ranking.
    very_old = {
        "chunk": 1,
        "timestamp": (now - timedelta(weeks=30)).isoformat(),
        "rejection_violations": 5,
        "rejected_meals_that_reappeared": ["lechuga"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "allergy_violations": 0,
    }
    recent = {
        "chunk": 5,
        "timestamp": (now - timedelta(weeks=2)).isoformat(),
        "rejection_violations": 1,
        "rejected_meals_that_reappeared": ["pavo"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "allergy_violations": 0,
    }
    fake_row = {
        "id": "plan-prev-uuid",
        "plan_data": {"_lifetime_lessons_history": [very_old, recent]},
    }

    # Override LIFETIME_LESSONS_WINDOW_DAYS via constants module (db_plans.py
    # importa la constante localmente dentro de la función, así que el patch
    # debe ir sobre constants, no sobre db_plans).
    with patch("db_plans.execute_sql_query", return_value=fake_row), \
         patch("constants.LIFETIME_LESSONS_WINDOW_DAYS", 365):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p14b")

    assert result is not None
    rej_hits = result["summary"]["top_rejection_hits"]
    assert "pavo" in rej_hits, "Lesson reciente debe estar en el ranking."
    assert "lechuga" not in rej_hits, (
        "Lesson de 30 semanas (peso < MIN_WEIGHT) debe excluirse del ranking. "
        "Si aparece, el filtro `if _w < MIN_WEIGHT: continue` no se aplicó."
    )
    # totals brutos siguen acumulando ambas (no decayed).
    assert result["summary"]["total_rejection_violations"] == 6  # 5 + 1


def test_top_repeated_meal_names_ranked_by_recency_weight_not_just_chunk_count():
    """Si un meal aparece en 1 chunk reciente (peso 0.9) vs otro en 1 chunk
    viejo (peso 0.4), el reciente debe rankear primero. ANTES, el sort era
    por # chunks distintos (ambos = 1), tie-break arbitrario."""
    from unittest.mock import patch
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    old_meal_lesson = {
        "chunk": 1,
        "timestamp": (now - timedelta(weeks=7)).isoformat(),
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": [],
        "repeated_bases": [],
        "repeated_meal_names": ["Pollo Asado"],
    }
    recent_meal_lesson = {
        "chunk": 5,
        "timestamp": (now - timedelta(weeks=1)).isoformat(),
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": [],
        "repeated_bases": [],
        "repeated_meal_names": ["Salmón Plancha"],
    }
    fake_row = {
        "id": "plan-prev",
        "plan_data": {"_lifetime_lessons_history": [old_meal_lesson, recent_meal_lesson]},
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p14c")

    top_meals = result["summary"]["top_repeated_meal_names"]
    assert "Salmón Plancha" in top_meals
    assert "Pollo Asado" in top_meals
    assert top_meals.index("Salmón Plancha") < top_meals.index("Pollo Asado"), (
        "top_repeated_meal_names debe rankear por peso (recencia) primero, "
        "no por count de chunks. Ambos meals aparecen en 1 chunk; tie-break "
        "lo decide el peso."
    )


def test_total_rejections_unaffected_by_decay():
    """Los `total_rejection_violations` y `total_allergy_violations` son
    counters HISTÓRICOS BRUTOS, no rankeados. Sirven para telemetría
    operacional ("este usuario rechazó X comidas en los últimos 60 días").
    Aplicar decay a ellos confundiría dashboards."""
    from unittest.mock import patch
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    # 5 lessons antiguas con violations.
    history = [
        {
            "chunk": i,
            "timestamp": (now - timedelta(weeks=8 - i)).isoformat(),
            "rejection_violations": 2,
            "allergy_violations": 1,
            "rejected_meals_that_reappeared": [],
            "repeated_bases": [],
            "repeated_meal_names": [],
        }
        for i in range(1, 6)
    ]
    fake_row = {
        "id": "plan-prev",
        "plan_data": {"_lifetime_lessons_history": history},
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p14d")

    summary = result["summary"]
    assert summary["total_rejection_violations"] == 10  # 5 * 2 raw, NO decayed
    assert summary["total_allergy_violations"] == 5     # 5 * 1 raw, NO decayed
