"""[P1-7] Provenance de lecciones — distinguir user_logs vs proxy/synthesis.

Antes del fix:
    El recompute de `_lifetime_lessons_summary` (worker + db_plans inheritance)
    pesaba todas las lecciones por igual sin distinguir su origen:
      - user_logs: learning_metrics calculado desde consumed_meals reales (high signal).
      - inventory_proxy: ratio basado en mutaciones de inventario (low signal).
      - synthesis: derivado de plan_data.days sin logs ni proxy (low signal).
      - stub: metrics_unavailable, datos en cero (no signal).

    Resultado: usuarios que no logean comidas tenían su lifetime dominado por
    señales de baja confianza, bloqueando proteínas/platos que en realidad
    nunca tuvieron problema con — solo no los registraron.

Después del fix:
    1. `_derive_learning_provenance(lesson)` deriva provenance desde flags
       existentes (`metrics_unavailable`, `low_confidence`,
       `learning_signal_strength`, `rebuilt_from_*`) o lee `_learning_provenance`
       explícito si fue marcado en construcción.
    2. Worker y db_plans summary recompute multiplican el peso por
       `LIFETIME_LESSON_PROXY_WEIGHT_FACTOR` (0.5) cuando provenance != user_logs.
       Combinado con decay temporal de P1-4: final_weight = decay * provenance.
    3. Summary persiste `_lifetime_proxy_ratio`, `_lifetime_user_logs_count`,
       `_lifetime_proxy_count` para `/admin/metrics`.
    4. Cuando el ratio supera `CHUNK_MAX_LIFETIME_PROXY_RATIO` (default 0.6) Y
       hay >= `CHUNK_LIFETIME_PROXY_MIN_TOTAL` lessons, emite evento de
       telemetría `lifetime_proxy_ratio_exceeded`. NO bloquea (eso lo hace el
       gate vía `chronic_zero_logging`); es observabilidad complementaria.
    5. `/admin/metrics` expone sección `learning_provenance` con avg/max ratio
       y conteo de planes con > 50% proxy.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest


def test_provenances_constant_canonical():
    """Las 4 provenances son las únicas válidas. Si añades una nueva debes
    extender el helper Y los crons de alerta downstream."""
    from cron_tasks import P1_7_LEARNING_PROVENANCES
    assert set(P1_7_LEARNING_PROVENANCES) == {
        "user_logs", "inventory_proxy", "synthesis", "stub",
    }


def test_explicit_provenance_takes_precedence():
    """Si la lección lleva `_learning_provenance` explícito (post-P1-7), se
    respeta sin re-derivar de flags."""
    from cron_tasks import _derive_learning_provenance
    lesson = {
        "_learning_provenance": "synthesis",
        # Flags que sugerirían user_logs, pero el explícito gana.
        "low_confidence": False,
        "metrics_unavailable": False,
    }
    assert _derive_learning_provenance(lesson) == "synthesis"


def test_explicit_invalid_provenance_falls_back_to_derivation():
    """Si `_learning_provenance` viene con un valor desconocido (legacy o
    refactor erróneo), no se respeta; se deriva de flags."""
    from cron_tasks import _derive_learning_provenance
    lesson = {
        "_learning_provenance": "garbage_value",
        "metrics_unavailable": True,
    }
    assert _derive_learning_provenance(lesson) == "stub"


def test_derive_stub_from_metrics_unavailable():
    """Lección con metrics_unavailable=True → 'stub' (sin datos reales)."""
    from cron_tasks import _derive_learning_provenance
    assert _derive_learning_provenance({"metrics_unavailable": True}) == "stub"


def test_derive_synthesis_from_rebuilt_markers():
    """Lecciones marcadas con rebuilt_from_pipeline_failure /
    synthesized_from_days / rebuilt_from_preflight → 'synthesis'."""
    from cron_tasks import _derive_learning_provenance
    assert _derive_learning_provenance({"rebuilt_from_pipeline_failure": True}) == "synthesis"
    assert _derive_learning_provenance({"synthesized_from_days": True}) == "synthesis"
    assert _derive_learning_provenance({"rebuilt_from_preflight": True}) == "synthesis"


def test_derive_inventory_proxy_from_low_confidence():
    """low_confidence=True (sin marcadores de síntesis) → 'inventory_proxy'.
    Caso típico: learning_metrics tenía inventory_activity_proxy_used=True."""
    from cron_tasks import _derive_learning_provenance
    assert _derive_learning_provenance({"low_confidence": True}) == "inventory_proxy"
    assert _derive_learning_provenance({"learning_signal_strength": "weak"}) == "inventory_proxy"


def test_derive_user_logs_default_for_high_confidence():
    """Sin flags negativos, asumimos user_logs (datos reales del usuario)."""
    from cron_tasks import _derive_learning_provenance
    lesson = {
        "rejection_violations": 2,
        "low_confidence": False,
        "metrics_unavailable": False,
        "learning_signal_strength": "strong",
    }
    assert _derive_learning_provenance(lesson) == "user_logs"
    # Empty dict también default a user_logs (preferimos sobrestimar calidad).
    assert _derive_learning_provenance({}) == "user_logs"


def test_derive_returns_stub_for_non_dict():
    """Robustez: lessons malformadas → 'stub' (no contribuyen sin romper)."""
    from cron_tasks import _derive_learning_provenance
    assert _derive_learning_provenance(None) == "stub"
    assert _derive_learning_provenance([1, 2, 3]) == "stub"
    assert _derive_learning_provenance("not a lesson") == "stub"


def test_provenance_weight_factor_user_logs_full():
    """Lessons de logs reales NO se penalizan: factor 1.0."""
    from cron_tasks import _provenance_weight_factor
    user_lesson = {"rejection_violations": 1, "low_confidence": False}
    assert _provenance_weight_factor(user_lesson) == 1.0


def test_provenance_weight_factor_proxy_halved():
    """Lessons de proxy/synthesis/stub se reducen al 50% por default."""
    from cron_tasks import _provenance_weight_factor, LIFETIME_LESSON_PROXY_WEIGHT_FACTOR
    proxy_lesson = {"low_confidence": True}
    synth_lesson = {"synthesized_from_days": True}
    stub_lesson = {"metrics_unavailable": True}
    assert _provenance_weight_factor(proxy_lesson) == LIFETIME_LESSON_PROXY_WEIGHT_FACTOR
    assert _provenance_weight_factor(synth_lesson) == LIFETIME_LESSON_PROXY_WEIGHT_FACTOR
    assert _provenance_weight_factor(stub_lesson) == LIFETIME_LESSON_PROXY_WEIGHT_FACTOR
    # Default factor sensato: 0.5.
    assert LIFETIME_LESSON_PROXY_WEIGHT_FACTOR == 0.5


def test_db_plans_inheritance_persists_proxy_ratio():
    """db_plans `_inherit_lifetime_lessons_from_prior_plan` debe persistir
    `_lifetime_proxy_ratio` y counts en el summary heredado."""
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    history = [
        # 2 user_logs + 3 proxy → ratio 0.6.
        {
            "chunk": 1,
            "timestamp": (now - timedelta(weeks=1)).isoformat(),
            "rejection_violations": 0,
            "allergy_violations": 0,
            "rejected_meals_that_reappeared": [],
            "repeated_bases": [],
            "repeated_meal_names": [],
            "low_confidence": False,
        },
        {
            "chunk": 2,
            "timestamp": (now - timedelta(weeks=2)).isoformat(),
            "rejection_violations": 0,
            "allergy_violations": 0,
            "rejected_meals_that_reappeared": [],
            "repeated_bases": [],
            "repeated_meal_names": [],
            "low_confidence": False,
        },
        # Proxy lessons.
        *[
            {
                "chunk": i,
                "timestamp": (now - timedelta(weeks=i)).isoformat(),
                "rejection_violations": 0,
                "allergy_violations": 0,
                "rejected_meals_that_reappeared": [],
                "repeated_bases": [],
                "repeated_meal_names": [],
                "low_confidence": True,  # → inventory_proxy
            }
            for i in range(3, 6)
        ],
    ]
    fake_row = {
        "id": "plan-prev",
        "plan_data": {"_lifetime_lessons_history": history},
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p17")

    summary = result["summary"]
    assert "_lifetime_proxy_ratio" in summary, (
        "Inheritance recompute debe persistir _lifetime_proxy_ratio en el summary."
    )
    assert summary["_lifetime_proxy_ratio"] == round(3 / 5, 3)
    assert summary["_lifetime_user_logs_count"] == 2
    assert summary["_lifetime_proxy_count"] == 3


def test_db_plans_inheritance_proxy_lesson_weight_halved():
    """Una rejection que aparece SOLO en lecciones proxy debe rankear MÁS BAJO
    que una que aparece en lecciones user_logs, aunque ambas tengan misma edad.
    Sin esto, el ranking hereda señal de baja confianza con prioridad equivalente.
    """
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    proxy_lesson = {
        "chunk": 1,
        "timestamp": (now - timedelta(weeks=1)).isoformat(),
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": ["camarones"],  # solo en proxy
        "repeated_bases": [],
        "repeated_meal_names": [],
        "low_confidence": True,  # → proxy
    }
    user_lesson = {
        "chunk": 2,
        "timestamp": (now - timedelta(weeks=1)).isoformat(),  # misma edad
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": ["pollo"],  # solo en user_logs
        "repeated_bases": [],
        "repeated_meal_names": [],
        "low_confidence": False,  # → user_logs
    }
    fake_row = {
        "id": "plan-prev",
        "plan_data": {"_lifetime_lessons_history": [proxy_lesson, user_lesson]},
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p17b")

    rej_hits = result["summary"]["top_rejection_hits"]
    assert "pollo" in rej_hits and "camarones" in rej_hits
    assert rej_hits.index("pollo") < rej_hits.index("camarones"), (
        "user_logs rejection debe rankear ANTES que proxy con misma edad. "
        "Si están igualados o invertidos, el factor de provenance no se "
        "está aplicando en el inheritance recompute."
    )


def test_provenance_factor_combines_with_decay():
    """El peso final = decay * provenance_factor. Una lesson user_logs de 4
    semanas (decay 0.66) supera a una proxy de 1 semana (decay 0.9 * 0.5 = 0.45).
    """
    from db_plans import _inherit_lifetime_lessons_from_prior_plan

    now = datetime.now(timezone.utc)
    # Proxy de 1 semana atrás → peso = 0.9 * 0.5 = 0.45.
    fresh_proxy = {
        "chunk": 1,
        "timestamp": (now - timedelta(weeks=1)).isoformat(),
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": ["fresh_proxy_item"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "low_confidence": True,
    }
    # User_logs de 4 semanas atrás → peso = 0.6561 * 1.0 = 0.6561.
    older_user = {
        "chunk": 2,
        "timestamp": (now - timedelta(weeks=4)).isoformat(),
        "rejection_violations": 0,
        "allergy_violations": 0,
        "rejected_meals_that_reappeared": ["older_user_item"],
        "repeated_bases": [],
        "repeated_meal_names": [],
        "low_confidence": False,
    }
    fake_row = {
        "id": "plan-prev",
        "plan_data": {"_lifetime_lessons_history": [fresh_proxy, older_user]},
    }

    with patch("db_plans.execute_sql_query", return_value=fake_row):
        result = _inherit_lifetime_lessons_from_prior_plan(cursor=None, user_id="u-p17c")

    rej_hits = result["summary"]["top_rejection_hits"]
    assert rej_hits.index("older_user_item") < rej_hits.index("fresh_proxy_item"), (
        "user_logs de 4 semanas (peso 0.66) debe rankear ANTES que proxy de "
        "1 semana (peso 0.45). Si está al revés, decay y provenance no "
        "están multiplicándose correctamente — uno de los dos sobrescribe al otro."
    )


def test_admin_metrics_includes_learning_provenance_section():
    """Regression guard del endpoint: `/admin/metrics` debe incluir la
    sección `learning_provenance` para que dashboards puedan trackear el
    ratio."""
    from routers import plans as plans_router
    import inspect
    src = inspect.getsource(plans_router.api_admin_metrics)
    assert '"learning_provenance"' in src
    assert "_lifetime_proxy_ratio" in src
    assert "avg_proxy_ratio" in src
    assert "max_proxy_ratio" in src


def test_admin_metrics_query_uses_correct_field():
    """El query de `/admin/metrics` lee `_lifetime_proxy_ratio` desde
    `plan_data->_lifetime_lessons_summary`. Si por refactor cambias el path
    JSON, los dashboards reportarán zeros sin error."""
    from routers import plans as plans_router
    import inspect
    src = inspect.getsource(plans_router.api_admin_metrics)
    # El path JSON canónico (sin escapar comillas dobles que serían chaos).
    assert "_lifetime_lessons_summary" in src
    assert "_lifetime_proxy_ratio" in src
