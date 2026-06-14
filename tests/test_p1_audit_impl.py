"""[P1-AUDIT-IMPL · 2026-05-28] Tests de regresión parser-based del bundle de
implementación de los P1 del audit prod-readiness 2026-05-28.

Cubre 4 fixes (más P1-5 es operacional .env, sin código):
  - P1-3 P1-PUSH-TIMEOUT      : webpush(timeout=...) en utils_push.py + notifications.py
  - P1-4 P1-SCHEDULER-STAGGER : ancla determinístico de IntervalTriggers en _add_job_jittered
  - P1-2 P1-PROACTIVE-BUDGET  : cap usuarios/runtime/nudges por tick en run_proactive_checks
  - P1-1 P1-DB-STMT-TIMEOUT-ROLE : migración SSOT dual-dir ALTER ROLE service_role

Parser-based a propósito (lee el source, no importa módulos): el venv de test
tiene una colisión de import con `backend/migrations/` y los módulos prod requieren
langgraph/supabase ausentes. Correr con: py -3 -m pytest tests/test_p1_audit_impl.py --noconftest -q
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)


def _read(*parts):
    with open(os.path.join(*parts), "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# P1-3 · P1-PUSH-TIMEOUT
# --------------------------------------------------------------------------
def test_p1_push_timeout_knob_defined():
    src = _read(_BACKEND, "utils_push.py")
    assert "MEALFIT_PUSH_HTTP_TIMEOUT_S" in src, "falta el knob de timeout de push"
    assert "_PUSH_HTTP_TIMEOUT_S" in src
    assert "_env_float" in src, "el knob debe leerse vía _env_float (clamp/registro)"


def test_p1_push_timeout_applied_in_utils_push():
    src = _read(_BACKEND, "utils_push.py")
    # Ancla a la LLAMADA real (multilínea: `webpush(\n  args...\n)`), NO a la
    # mención `webpush()` del comentario del knob.
    m = re.search(r"webpush\(\s*\n(.*?)\)", src, re.DOTALL)
    assert m, "no se encontró la llamada multilínea webpush(...)"
    assert "timeout=_PUSH_HTTP_TIMEOUT_S" in m.group(1), \
        "webpush() en utils_push.py debe pasar timeout=_PUSH_HTTP_TIMEOUT_S"


def test_p1_push_timeout_applied_in_notifications_router():
    src = _read(_BACKEND, "routers", "notifications.py")
    m = re.search(r"webpush\(\s*\n(.*?)\)", src, re.DOTALL)
    assert m, "no se encontró la llamada multilínea webpush(...) en notifications.py"
    assert "timeout=" in m.group(1), \
        "webpush() en notifications.py debe pasar timeout="


# --------------------------------------------------------------------------
# P1-4 · P1-SCHEDULER-STAGGER
# --------------------------------------------------------------------------
def test_p1_scheduler_stagger_knobs_and_anchor():
    src = _read(_BACKEND, "cron_tasks.py")
    assert "MEALFIT_SCHEDULER_STAGGER_ENABLED" in src
    assert "MEALFIT_SCHEDULER_STAGGER_MAX_S" in src
    assert "P1-SCHEDULER-STAGGER" in src, "falta el tooltip-anchor del fix"
    # el wrapper debe setear un next_run_time desfasado para triggers interval
    body = src[src.index("def _add_job_jittered("):]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    assert "next_run_time" in body, "_add_job_jittered debe anclar next_run_time"
    assert "hashlib" in body, "el offset debe ser determinístico (hashlib del job_id)"
    assert "interval" in body


def test_p1_scheduler_stagger_has_kill_switch_fallback():
    """El stagger toca el SSOT por el que pasan TODOS los crons: debe ser
    reversible sin redeploy (kill switch) y no-fatal (try/except fallback)."""
    src = _read(_BACKEND, "cron_tasks.py")
    start = src.index("def _add_job_jittered(")
    body = src[start: start + 3000]
    assert "_SCHEDULER_STAGGER_ENABLED" in body
    assert "try:" in body and "except" in body, \
        "el stagger debe degradar al comportamiento previo si algo falla"


# --------------------------------------------------------------------------
# P1-2 · P1-PROACTIVE-BUDGET
# --------------------------------------------------------------------------
def test_p1_proactive_budget_knobs():
    src = _read(_BACKEND, "proactive_agent.py")
    for knob in (
        "MEALFIT_PROACTIVE_MAX_USERS_PER_TICK",
        "MEALFIT_PROACTIVE_MAX_RUNTIME_S",
        "MEALFIT_PROACTIVE_MAX_NUDGES_PER_TICK",
    ):
        assert knob in src, f"falta knob de budget {knob}"
    assert "P1-PROACTIVE-BUDGET" in src


def test_p1_proactive_budget_guards_in_loop():
    src = _read(_BACKEND, "proactive_agent.py")
    fn = src[src.index("def run_proactive_checks("):]
    # cap de usuarios por tick (slice) + guard de wall-clock + contador de nudges
    assert "sessions[:_max_users]" in fn, "falta el cap de usuarios por tick"
    assert "_nudges_sent" in fn, "falta el contador/cap de nudges (gasto LLM)"
    assert "_t_start" in fn and "total_seconds()" in fn, "falta el guard de wall-clock"
    assert fn.count("break") >= 2, "deben existir los breaks de runtime y de nudges"


# --------------------------------------------------------------------------
# P1-1 · P1-DB-STMT-TIMEOUT-ROLE (migración SSOT dual-dir)
# --------------------------------------------------------------------------
_MIGRATION = "p1_service_role_statement_timeout_2026_05_28.sql"


def test_p1_stmt_timeout_migration_exists_both_dirs():
    backend_mig = os.path.join(_BACKEND, "migrations", _MIGRATION)
    root_mig = os.path.join(_ROOT, "migrations", _MIGRATION)
    assert os.path.exists(backend_mig), "falta la migración en backend/migrations"
    assert os.path.exists(root_mig), "falta la migración en migrations (root)"


def test_p1_stmt_timeout_migration_content_and_dual_dir_identical():
    backend_mig = _read(_BACKEND, "migrations", _MIGRATION)
    root_mig = _read(_ROOT, "migrations", _MIGRATION)
    assert backend_mig == root_mig, "SSOT dual-dir: las migraciones deben ser idénticas"
    assert "ALTER ROLE service_role SET statement_timeout" in backend_mig
    assert "idle_in_transaction_session_timeout" in backend_mig
    assert "RAISE EXCEPTION" in backend_mig, "debe tener sanity check idempotente"


# --------------------------------------------------------------------------
# Cross-link: marker bumpeado
# --------------------------------------------------------------------------
def test_p1_marker_bumped():
    src = _read(_BACKEND, "app.py")
    # [supersede] El bundle P1-AUDIT-IMPL bumpeó el marker el 2026-05-28; un
    # bundle posterior (P2-AUDIT-IMPL, etc.) puede haberlo avanzado. Aceptamos
    # P1-AUDIT-IMPL o cualquier supersede con fecha >= 2026-05-28 (date-floor,
    # patrón del repo para exact-match de markers entre bundles del mismo día).
    import re
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "no se encontró _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    date_m = re.search(r'(\d{4}-\d{2}-\d{2})', marker)
    assert date_m and date_m.group(1) >= "2026-05-28", f"marker stale o sin supersede: {marker}"
