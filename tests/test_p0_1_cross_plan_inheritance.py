"""[P0-1] Cross-plan lifetime lessons inheritance.

When a user creates a new meal_plan after completing/abandoning a prior one
(e.g. switching 7d → 15d → 30d), the new plan must inherit
`_lifetime_lessons_history` and `_lifetime_lessons_summary` from the most
recent prior plan, filtered by LIFETIME_LESSONS_WINDOW_DAYS.

Without this, every new plan started with empty learning despite the user
having weeks of meal-rejection / repetition history in their prior plans.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from db_core import execute_sql_query, execute_sql_write
from db_plans import save_new_meal_plan_atomic


pytestmark = pytest.mark.e2e


def _insert_prior_plan(user_id: str, plan_data: dict) -> str:
    plan_id = str(uuid.uuid4())
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, name, plan_data, calories, macros, created_at) "
        "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, NOW() - INTERVAL '1 hour')",
        (plan_id, user_id, "Plan previo", json.dumps(plan_data, ensure_ascii=False),
         2000, json.dumps({"protein": 150, "carbs": 200, "fat": 70})),
    )
    return plan_id


def _read_plan_data(plan_id: str) -> dict:
    row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True
    )
    return row["plan_data"] if row else {}


def _make_lesson(chunk: int, days_ago: int = 1, repeated_bases=None,
                 rejected_meals=None, rej_viol: int = 0, alg_viol: int = 0) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "chunk": chunk,
        "timestamp": ts,
        "repeated_bases": repeated_bases or [],
        "repeated_meal_names": [],
        "rejected_meals_that_reappeared": rejected_meals or [],
        "allergy_hits": [],
        "rejection_violations": rej_viol,
        "allergy_violations": alg_viol,
        "repeat_pct": 0.0,
        "ingredient_base_repeat_pct": 0.0,
    }


def test_new_plan_inherits_lifetime_lessons_from_prior_plan(seeded_user_profile):
    """Plan nuevo sin lifetime_lessons hereda del plan previo del mismo usuario."""
    user_id, _fixture_plan_id = seeded_user_profile

    prior_data = {
        "days": [],
        "_lifetime_lessons_history": [
            _make_lesson(1, days_ago=5, repeated_bases=[{"bases": ["pollo", "arroz"]}],
                         rejected_meals=["Pollo guisado"], rej_viol=1),
            _make_lesson(2, days_ago=3, repeated_bases=[{"bases": ["res"]}],
                         rejected_meals=["Bistec encebollado"]),
        ],
        "_lifetime_lessons_summary": {
            "total_rejection_violations": 1,
            "total_allergy_violations": 0,
            "top_rejection_hits": ["Pollo guisado", "Bistec encebollado"],
            "top_repeated_bases": ["pollo", "arroz", "res"],
            "_lifetime_window_days": 60,
        },
    }
    prior_plan_id = _insert_prior_plan(user_id, prior_data)

    new_plan_data = {"days": [], "generation_status": "partial"}
    insert_data = {
        "user_id": user_id,
        "name": "Plan nuevo (15d)",
        "plan_data": new_plan_data,
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fat": 70},
    }
    new_plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)
    assert new_plan_id and new_plan_id != prior_plan_id

    persisted = _read_plan_data(new_plan_id)
    history = persisted.get("_lifetime_lessons_history") or []
    summary = persisted.get("_lifetime_lessons_summary") or {}

    assert len(history) == 2, f"Esperaba 2 lecciones heredadas, obtuve {len(history)}"
    assert {l["chunk"] for l in history} == {1, 2}
    assert "pollo" in (summary.get("top_repeated_bases") or [])
    assert "Pollo guisado" in (summary.get("top_rejection_hits") or [])
    assert persisted.get("_lifetime_lessons_inherited_from") == prior_plan_id


def test_inheritance_filters_lessons_older_than_window(seeded_user_profile):
    """Lecciones >60d se descartan; el summary se recomputa desde lo restante."""
    user_id, _ = seeded_user_profile

    # Una lección reciente (5d) y una vieja (90d, fuera del window de 60d)
    fresh = _make_lesson(5, days_ago=5, repeated_bases=[{"bases": ["pollo"]}],
                         rejected_meals=["Pollo al curry"])
    stale = _make_lesson(1, days_ago=90, repeated_bases=[{"bases": ["lechuga"]}],
                         rejected_meals=["Ensalada antigua"])
    prior_data = {
        "days": [],
        "_lifetime_lessons_history": [stale, fresh],
        "_lifetime_lessons_summary": {
            "total_rejection_violations": 0,
            "total_allergy_violations": 0,
            "top_rejection_hits": ["Ensalada antigua", "Pollo al curry"],
            "top_repeated_bases": ["lechuga", "pollo"],
            "_lifetime_window_days": 60,
        },
    }
    _insert_prior_plan(user_id, prior_data)

    insert_data = {
        "user_id": user_id,
        "name": "Plan nuevo post-TTL",
        "plan_data": {"days": []},
        "calories": 2000,
        "macros": {},
    }
    new_plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)

    persisted = _read_plan_data(new_plan_id)
    history = persisted.get("_lifetime_lessons_history") or []
    summary = persisted.get("_lifetime_lessons_summary") or {}

    assert len(history) == 1, "La lección de 90d debió ser filtrada por TTL"
    assert history[0]["chunk"] == 5
    # Summary recomputado: solo refleja la lección fresca
    assert "pollo" in (summary.get("top_repeated_bases") or [])
    assert "lechuga" not in (summary.get("top_repeated_bases") or []), \
        "El summary no debe incluir bases de lecciones expiradas"
    assert "Pollo al curry" in (summary.get("top_rejection_hits") or [])
    assert "Ensalada antigua" not in (summary.get("top_rejection_hits") or [])


def test_inheritance_does_not_overwrite_explicit_values(seeded_user_profile):
    """Si plan_data ya trae lifetime lessons (renovación), no se sobreescribe."""
    user_id, _ = seeded_user_profile

    prior_data = {
        "days": [],
        "_lifetime_lessons_history": [
            _make_lesson(1, days_ago=2, rejected_meals=["Plato del plan previo"]),
        ],
        "_lifetime_lessons_summary": {
            "top_rejection_hits": ["Plato del plan previo"],
            "top_repeated_bases": [],
            "total_rejection_violations": 0,
            "total_allergy_violations": 0,
            "_lifetime_window_days": 60,
        },
    }
    _insert_prior_plan(user_id, prior_data)

    explicit_history = [_make_lesson(99, days_ago=1,
                                      rejected_meals=["Plato explícito"])]
    explicit_summary = {
        "top_rejection_hits": ["Plato explícito"],
        "top_repeated_bases": [],
        "total_rejection_violations": 0,
        "total_allergy_violations": 0,
        "_lifetime_window_days": 60,
    }
    insert_data = {
        "user_id": user_id,
        "name": "Plan con lessons explícitas",
        "plan_data": {
            "days": [],
            "_lifetime_lessons_history": explicit_history,
            "_lifetime_lessons_summary": explicit_summary,
        },
        "calories": 2000,
        "macros": {},
    }
    new_plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)

    persisted = _read_plan_data(new_plan_id)
    history = persisted.get("_lifetime_lessons_history") or []
    summary = persisted.get("_lifetime_lessons_summary") or {}

    assert len(history) == 1 and history[0]["chunk"] == 99
    assert summary.get("top_rejection_hits") == ["Plato explícito"]
    assert "_lifetime_lessons_inherited_from" not in persisted, \
        "No debe marcarse como heredado cuando el caller pasó valores explícitos"


def test_no_prior_plan_yields_no_inheritance(seeded_user_profile):
    """Usuario sin planes previos: el plan nuevo no recibe lifetime lessons."""
    user_id, _ = seeded_user_profile

    insert_data = {
        "user_id": user_id,
        "name": "Primer plan del usuario",
        "plan_data": {"days": []},
        "calories": 2000,
        "macros": {},
    }
    new_plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)

    persisted = _read_plan_data(new_plan_id)
    assert "_lifetime_lessons_history" not in persisted
    assert "_lifetime_lessons_summary" not in persisted
    assert "_lifetime_lessons_inherited_from" not in persisted


def test_build_insert_sql_falls_back_to_inheritance_when_caller_skipped(
    seeded_user_profile, caplog
):
    """[P0-1/CENTRAL] Si un caller futuro invoca _build_meal_plan_insert_sql
    directamente sin pasar por save_new_meal_plan_atomic/_robust, la herencia
    cross-plan se aplica igualmente (last-line-of-defense) y emite warning de
    telemetría apuntando al call site infractor.

    Sella el invariante de que la única vía de insertar meal_plans en
    producción (`_build_meal_plan_insert_sql`) NO puede saltarse la herencia,
    aunque el caller la haya olvidado.
    """
    import logging

    from db_plans import _build_meal_plan_insert_sql

    user_id, _ = seeded_user_profile

    prior_data = {
        "days": [],
        "_lifetime_lessons_history": [
            _make_lesson(
                1,
                days_ago=2,
                repeated_bases=[{"bases": ["pollo"]}],
                rejected_meals=["Pollo guisado"],
            ),
        ],
        "_lifetime_lessons_summary": {
            "top_rejection_hits": ["Pollo guisado"],
            "top_repeated_bases": ["pollo"],
            "total_rejection_violations": 0,
            "total_allergy_violations": 0,
            "_lifetime_window_days": 60,
        },
    }
    _insert_prior_plan(user_id, prior_data)

    # Simulamos el caller infractor: arma insert_data SIN _lifetime_lessons_*
    # y NO invoca _apply_inherited_lifetime_lessons antes.
    insert_data = {
        "user_id": user_id,
        "name": "Plan vía helper directo",
        "plan_data": {"days": []},
        "calories": 2000,
        "macros": {},
    }

    with caplog.at_level(logging.WARNING, logger="db_plans"):
        sql, vals = _build_meal_plan_insert_sql(insert_data, with_returning=False)

    # Mutación in-place: plan_data ahora trae las lecciones heredadas pese a
    # que el caller no las pidió.
    persisted_plan_data = insert_data["plan_data"]
    history = persisted_plan_data.get("_lifetime_lessons_history") or []
    summary = persisted_plan_data.get("_lifetime_lessons_summary") or {}

    assert len(history) == 1, (
        "Fallback debió heredar la lección del plan previo aunque el caller "
        "no invocara _apply_inherited_lifetime_lessons explícitamente"
    )
    assert "pollo" in (summary.get("top_repeated_bases") or [])
    assert "Pollo guisado" in (summary.get("top_rejection_hits") or [])

    # Telemetría: warning apuntando al call site sin herencia explícita.
    assert any(
        "[P0-1/CENTRAL]" in r.getMessage() for r in caplog.records
    ), "Se esperaba warning [P0-1/CENTRAL] señalando el call site infractor"


def test_build_insert_sql_does_not_double_apply_inheritance(seeded_user_profile, caplog):
    """[P0-1/CENTRAL] Si el caller YA invocó la herencia (ej. el path normal
    save_new_meal_plan_atomic), el fallback dentro de _build_meal_plan_insert_sql
    NO debe re-ejecutar la lectura ni emitir warning espurio.
    """
    import logging

    from db_plans import _apply_inherited_lifetime_lessons, _build_meal_plan_insert_sql

    user_id, _ = seeded_user_profile

    prior_data = {
        "days": [],
        "_lifetime_lessons_history": [
            _make_lesson(
                1,
                days_ago=2,
                repeated_bases=[{"bases": ["res"]}],
                rejected_meals=["Bistec"],
            ),
        ],
        "_lifetime_lessons_summary": {
            "top_rejection_hits": ["Bistec"],
            "top_repeated_bases": ["res"],
            "total_rejection_violations": 0,
            "total_allergy_violations": 0,
            "_lifetime_window_days": 60,
        },
    }
    _insert_prior_plan(user_id, prior_data)

    insert_data = {
        "user_id": user_id,
        "name": "Plan con herencia explícita previa",
        "plan_data": {"days": []},
        "calories": 2000,
        "macros": {},
    }

    # Caller ejemplar: aplica herencia ANTES de pasar al builder.
    _apply_inherited_lifetime_lessons(user_id, insert_data, cursor=None)
    snapshot_history = list(insert_data["plan_data"]["_lifetime_lessons_history"])
    snapshot_summary = dict(insert_data["plan_data"]["_lifetime_lessons_summary"])

    with caplog.at_level(logging.WARNING, logger="db_plans"):
        _sql, _vals = _build_meal_plan_insert_sql(insert_data, with_returning=False)

    # El builder no debió tocar nada: idempotencia.
    assert insert_data["plan_data"]["_lifetime_lessons_history"] == snapshot_history
    assert insert_data["plan_data"]["_lifetime_lessons_summary"] == snapshot_summary
    # Y NO debió loguear el warning de fallback (no hubo fallback real).
    assert not any(
        "[P0-1/CENTRAL]" in r.getMessage() for r in caplog.records
    ), "El builder no debe emitir warning cuando la herencia ya estaba aplicada"
