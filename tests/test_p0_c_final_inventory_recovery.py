"""[P0-C] Tests para auto-recovery de chunks pausados por validación final de inventario.

Cubre:
  1. Helper `_pantry_covers_missing`:
       - Cobertura completa por base normalizada → (True, []).
       - Items faltantes → (False, [...]).
       - Lista vacía de faltantes → (False, []) (caller decide).
       - Pantry como list[str] y como list[dict] (defensivo).
  2. `_pause_chunk_for_final_inventory_validation`:
       - Acepta missing_ingredients y los persiste deduplicados+saneados.
       - Sin missing_ingredients, no agrega la key.
  3. `_recover_pantry_paused_chunks` rama P0-C:
       - Chunk con reason='flexible_live_unreachable' + missing cubierto → resume pending.
       - Chunk con missing NO cubierto → mantiene status (cae al TTL existente).
       - Chunk dentro del grace period → no intenta recovery aún.
       - Chunk con recovery_attempts >= MAX → cae al TTL existente.
       - Live fetch falla → cae al TTL existente sin error.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_c_final_inventory_recovery.py -v
"""
import sys
import os
import types
import json
import copy
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos para importar cron_tasks aislado.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_kw: None,
    execute_sql_write=lambda *_a, **_kw: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    get_inventory_activity_since=lambda *_a, **_kw: [],
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: None,
    # [test fix] db_inventory.reserve_plan_ingredients returns int (count of items reserved).
    # Stubbing with a dict caused TypeError: '>=' not supported between dict and int in
    # cron_tasks.py:5239/16519. Mirror production: count ingredients with len>=3 in days[2].
    reserve_plan_ingredients=lambda *_a, **_kw: sum(
        1 for d in (_a[2] if len(_a) >= 3 else (_kw.get("days") or []))
        for m in ((d or {}).get("meals") or [])
        for i in (m.get("ingredients") or [])
        if i and len(str(i).strip()) >= 3
    ),
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_recent_plans=lambda *_a, **_kw: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_kw: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_kw: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_kw: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_kw: {})


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_shopping_list_delta=lambda *_a, **_kw: [],
        _parse_quantity=_stub_parse_quantity,
    )
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


from unittest.mock import patch
import cron_tasks


# ---------------------------------------------------------------------------
# 1. _pantry_covers_missing
# ---------------------------------------------------------------------------
def test_covers_missing_full_match():
    pantry = ["500g pechuga de pollo", "1kg arroz blanco"]
    missing = ["pollo", "arroz"]
    covered, still = cron_tasks._pantry_covers_missing(missing, pantry)
    assert covered is True
    assert still == []


def test_covers_missing_partial():
    pantry = ["pollo entero"]
    missing = ["pollo", "arroz"]
    covered, still = cron_tasks._pantry_covers_missing(missing, pantry)
    assert covered is False
    assert "arroz" in still


def test_covers_missing_empty_missing_list():
    """Lista vacía → (False, []) para que caller decida."""
    covered, still = cron_tasks._pantry_covers_missing([], ["pollo"])
    assert covered is False
    assert still == []


def test_covers_missing_pantry_as_dict_list():
    """Pantry puede venir como list[dict] con campo 'name'."""
    pantry = [{"name": "pechuga de pollo deshuesada"}, {"name": "huevos x12"}]
    missing = ["pollo", "huevos"]
    covered, _ = cron_tasks._pantry_covers_missing(missing, pantry)
    assert covered is True


def test_covers_missing_dedupes_and_strips():
    pantry = ["pollo asado"]
    missing = ["  pollo  ", "Pollo"]  # whitespace + case
    covered, _ = cron_tasks._pantry_covers_missing(missing, pantry)
    assert covered is True


# ---------------------------------------------------------------------------
# 2. _pause_chunk_for_final_inventory_validation persiste missing_ingredients
# ---------------------------------------------------------------------------
def test_pause_persists_missing_ingredients_deduplicated():
    captured = {}

    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        return {"pipeline_snapshot": {}}

    def fake_write(query, params=None):
        captured["params"] = params

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification", lambda *_a, **_kw: None):
        cron_tasks._pause_chunk_for_final_inventory_validation(
            task_id=1, user_id="u", week_number=2,
            reason="flexible_live_unreachable",
            missing_ingredients=["pollo", "POLLO", "  pollo ", "arroz", ""],
        )

    snapshot_json = captured["params"][0]
    snapshot = json.loads(snapshot_json)
    assert "_pantry_pause_missing_ingredients" in snapshot
    items = snapshot["_pantry_pause_missing_ingredients"]
    # Tras strip + lowercase dedup, debería quedar 2 (pollo, arroz).
    items_lc = [i.lower() for i in items]
    assert items_lc.count("pollo") == 1
    assert items_lc.count("arroz") == 1
    assert "" not in items
    assert snapshot.get("_pantry_pause_recovery_attempts") == 0


def test_pause_without_missing_does_not_set_key():
    captured = {}

    def fake_query(*_a, **_kw):
        return {"pipeline_snapshot": {}}

    def fake_write(query, params=None):
        captured["params"] = params

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification", lambda *_a, **_kw: None):
        cron_tasks._pause_chunk_for_final_inventory_validation(
            task_id=1, user_id="u", week_number=2,
            reason="final_inventory_unavailable",
            missing_ingredients=None,
        )

    snapshot = json.loads(captured["params"][0])
    assert "_pantry_pause_missing_ingredients" not in snapshot


# ---------------------------------------------------------------------------
# 3. _recover_pantry_paused_chunks rama P0-C
# ---------------------------------------------------------------------------
def _make_paused_row(reason="flexible_live_unreachable", missing=None,
                     paused_seconds=600, recovery_attempts=0):
    """Construye un row simulado de plan_chunk_queue paused."""
    snap = {
        "_pantry_pause_started_at": "2026-05-01T10:00:00+00:00",
        "_pantry_pause_reminders": 0,
        "_pantry_pause_reason": reason,
        "_pantry_pause_ttl_hours": 2,
        "_pantry_pause_reminder_hours": 2,
        "_pantry_pause_recovery_attempts": recovery_attempts,
    }
    if missing is not None:
        snap["_pantry_pause_missing_ingredients"] = missing
    return {
        "id": 42,
        "user_id": "user-123",
        "meal_plan_id": "plan-x",
        "week_number": 3,
        "pipeline_snapshot": snap,
        "paused_seconds": paused_seconds,
    }


def _drive_recover(paused_rows, live_inventory=None, live_raises=False):
    """Helper común: simula execute_sql_query / get_user_inventory_net y
    captura UPDATEs ejecutados durante el cron."""
    updates = []
    select_count = {"n": 0}

    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        q = (query or "").strip()
        if "FROM plan_chunk_queue" in q and "pending_user_action" in q:
            select_count["n"] += 1
            return paused_rows
        return None

    def fake_write(query, params=None):
        updates.append((query, params))
        return None

    def fake_live(_user_id):
        if live_raises:
            raise RuntimeError("live boom")
        return live_inventory

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks.get_user_inventory_net", side_effect=fake_live), \
         patch("cron_tasks._dispatch_push_notification", lambda *_a, **_kw: None), \
         patch("cron_tasks.get_inventory_activity_since",
               lambda *_a, **_kw: {"consumption_mutations_count": 0}):
        cron_tasks._recover_pantry_paused_chunks()

    return updates, select_count["n"]


def test_recovery_resumes_chunk_when_pantry_covers_missing():
    row = _make_paused_row(missing=["pollo", "arroz"])
    updates, _ = _drive_recover([row], live_inventory=["1kg pollo", "500g arroz blanco"])
    # Debe haber un UPDATE que cambia status a pending.
    resume_updates = [
        u for u in updates
        if u[1] is not None
        and "status = 'pending'" in (u[0] or "")
        and "execute_after = NOW()" in (u[0] or "")
    ]
    assert len(resume_updates) == 1, (
        f"esperaba 1 UPDATE de resume, hubo {len(resume_updates)}: {updates}"
    )
    # El snapshot persistido debe marcar resolution y bumpear attempts.
    snap_json = resume_updates[0][1][0]
    snap = json.loads(snap_json)
    assert snap["_pantry_pause_resolution"] == "missing_ingredients_covered"
    assert snap["_pantry_pause_recovery_attempts"] == 1


def test_recovery_does_not_resume_when_missing_not_covered():
    row = _make_paused_row(missing=["pollo", "arroz"])
    # Pantry solo tiene pollo; arroz sigue faltando.
    updates, _ = _drive_recover([row], live_inventory=["pollo entero"])
    # No debe haber UPDATE con status='pending' + execute_after=NOW().
    resume_updates = [
        u for u in updates
        if u[1] is not None
        and "status = 'pending'" in (u[0] or "")
        and "execute_after = NOW()" in (u[0] or "")
    ]
    assert resume_updates == [], (
        f"no debió reanudar con items aún faltantes; updates={updates}"
    )


def test_recovery_respects_grace_period():
    """Pausa muy reciente (< grace) → no intenta recovery aún."""
    # grace = 5min = 300s; usamos paused_seconds=60 (1 min).
    row = _make_paused_row(missing=["pollo"], paused_seconds=60)
    updates, _ = _drive_recover([row], live_inventory=["pollo asado"])
    # No debe disparar resume (grace activo).
    resume_updates = [
        u for u in updates
        if u[1] is not None
        and "status = 'pending'" in (u[0] or "")
        and "execute_after = NOW()" in (u[0] or "")
    ]
    assert resume_updates == []


def test_recovery_caps_attempts():
    """recovery_attempts >= MAX → cae al TTL existente."""
    from constants import CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS as _MAX
    row = _make_paused_row(missing=["pollo"], recovery_attempts=int(_MAX))
    updates, _ = _drive_recover([row], live_inventory=["pollo"])
    # No debe disparar resume P0-C (attempts agotados); el TTL existente decidirá.
    p0c_resume = [
        u for u in updates
        if u[1] is not None
        and "status = 'pending'" in (u[0] or "")
        and "missing_ingredients_covered" in (u[1][0] if u[1] else "")
    ]
    assert p0c_resume == []


def test_recovery_falls_through_when_live_fetch_fails():
    """Live fetch lanza excepción → P0-C cede al TTL existente sin propagar error."""
    row = _make_paused_row(missing=["pollo"])
    updates, _ = _drive_recover([row], live_raises=True)
    # No debe disparar resume P0-C.
    p0c_resume = [
        u for u in updates
        if u[1] is not None
        and "missing_ingredients_covered" in (u[1][0] if u[1] else "")
    ]
    assert p0c_resume == []


def test_recovery_resumes_when_no_missing_list_but_live_ok():
    """Si live original cayó sin saber qué faltaba, live OK ahora basta para reintentar."""
    row = _make_paused_row(missing=None)  # No persistimos missing_ingredients.
    updates, _ = _drive_recover([row], live_inventory=["pollo", "arroz"])
    resume_updates = [
        u for u in updates
        if u[1] is not None
        and "status = 'pending'" in (u[0] or "")
        and "missing_ingredients_covered" in (u[1][0] if u[1] else "")
    ]
    assert len(resume_updates) == 1


def test_other_pause_reasons_not_affected_by_p0c():
    """Reason = 'empty_pantry' (genérico) → P0-C no se activa, ruta TTL preservada."""
    snap = {
        "_pantry_pause_started_at": "2026-05-01T10:00:00+00:00",
        "_pantry_pause_reminders": 0,
        "_pantry_pause_reason": "empty_pantry",
        "_pantry_pause_ttl_hours": 12,
        "_pantry_pause_reminder_hours": 6,
    }
    row = {
        "id": 99,
        "user_id": "u",
        "meal_plan_id": "p",
        "week_number": 1,
        "pipeline_snapshot": snap,
        "paused_seconds": 600,
    }
    updates, _ = _drive_recover([row], live_inventory=["pollo"])
    # No debe haber resume P0-C; el push reminder/TTL de la ruta original puede
    # disparar pero NO con la marca missing_ingredients_covered.
    p0c_resume = [
        u for u in updates
        if u[1] is not None
        and "missing_ingredients_covered" in (u[1][0] if u[1] else "")
    ]
    assert p0c_resume == []
