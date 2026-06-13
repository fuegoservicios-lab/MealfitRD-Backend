"""[P3-SELECT-STAR-AGENT-SESSIONS · 2026-05-15] Test bundle marker (último
alfabético de la Fase 3 audit 2026-05-15 noche). Cubre los 8 P3 fixes:

    - P3-CONSOLE-DEV-GUARDS: convención documentada en CLAUDE.md
    - P3-CRONS-STATUS-ADMIN: endpoint admin con auth Bearer
    - P3-DB-POOL-MAX-IDLE-KNOB: knob ya implementado (regression test)
    - P3-DIARY-LATE-IMPORT: import al top de diary.py
    - P3-MIGRATION-IDEMPOTENCE-DOC: convención documentada en CLAUDE.md
    - P3-READY-REASON-HASH: hash de reason en /ready response
    - P3-SCHEDULER-ALERT-DEDUP: cache TTL 5s en _scheduler_alert_listener
    - P3-SELECT-STAR-AGENT-SESSIONS: columnas explícitas en db_chat.py

Bundle marker = último alfabético: `P3-SELECT-STAR-AGENT-SESSIONS`.

Tooltip-anchor: P3-SELECT-STAR-AGENT-SESSIONS-BUNDLE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO_ROOT / "backend"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


# ===========================================================================
# P3-SELECT-STAR-AGENT-SESSIONS (marker bundle)
# ===========================================================================
class TestSelectStarAgentSessions:
    @pytest.fixture(scope="class")
    def db_chat_src(self) -> str:
        return (_BACKEND / "db_chat.py").read_text(encoding="utf-8")

    def test_no_select_star_in_get_or_create_session(self, db_chat_src: str):
        """Después del fix, `SELECT * FROM agent_sessions` y `RETURNING *` no
        deben aparecer en `get_or_create_session` (excluyendo comentarios
        explicativos del fix)."""
        fn_match = re.search(
            r"def get_or_create_session\([\s\S]+?(?=\ndef |\nclass )",
            db_chat_src,
        )
        assert fn_match is not None
        body = fn_match.group(0)
        # Strip comentarios (líneas que empiezan con `#`) para evitar matching
        # de strings explicativos. Los comentarios pueden mencionar `SELECT *`
        # como referencia al fix; las líneas SQL ejecutables no pueden.
        non_comment = "\n".join(
            ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
        )
        assert "SELECT * FROM agent_sessions" not in non_comment, (
            "P3-SELECT-STAR-AGENT-SESSIONS regresión: `SELECT *` reapareció en "
            "`get_or_create_session` (fuera de comentarios)."
        )
        assert "RETURNING *" not in non_comment, (
            "P3-SELECT-STAR-AGENT-SESSIONS regresión: `RETURNING *` reapareció "
            "(fuera de comentarios) — callers ahora pueden recibir más columnas "
            "de las esperadas."
        )

    def test_explicit_columns_constant(self, db_chat_src: str):
        """Constante local con columnas explícitas debe existir."""
        assert "_AGENT_SESSION_COLS" in db_chat_src, (
            "P3-SELECT-STAR-AGENT-SESSIONS: constante `_AGENT_SESSION_COLS` "
            "ausente — qué columnas se seleccionan es ambiguo en source."
        )

    def test_marker_present(self, db_chat_src: str):
        assert "P3-SELECT-STAR-AGENT-SESSIONS" in db_chat_src, (
            "Marker bundle ausente — actualizar cross-link test."
        )


# ===========================================================================
# P3-DIARY-LATE-IMPORT
# ===========================================================================
class TestDiaryLateImport:
    @pytest.fixture(scope="class")
    def diary_src(self) -> str:
        return (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")

    def test_trigger_imported_at_top(self, diary_src: str):
        """`from cron_tasks import trigger_incremental_learning` debe estar
        en los primeros 50 líneas (top imports), no dentro de un handler."""
        first_lines = "\n".join(diary_src.splitlines()[:50])
        assert "from cron_tasks import trigger_incremental_learning" in first_lines, (
            "P3-DIARY-LATE-IMPORT: el import no está en el top del archivo."
        )

    def test_no_inline_late_import(self, diary_src: str):
        """No debe quedar `from cron_tasks import trigger_incremental_learning`
        DENTRO de un handler (after line 50)."""
        rest = "\n".join(diary_src.splitlines()[50:])
        assert "from cron_tasks import trigger_incremental_learning" not in rest, (
            "P3-DIARY-LATE-IMPORT regresión: el late import reapareció dentro "
            "de un handler — frágil ante renames."
        )


# ===========================================================================
# P3-READY-REASON-HASH
# ===========================================================================
class TestReadyReasonHash:
    @pytest.fixture(scope="class")
    def app_src(self) -> str:
        return (_BACKEND / "app.py").read_text(encoding="utf-8")

    def test_reason_hash_computed(self, app_src: str):
        """El endpoint `/ready` debe computar `reason_hash` con SHA-256[:8]."""
        assert re.search(
            r"reason_hash\s*=\s*_hashlib_ready\.sha256\([\s\S]{0,80}\)\.hexdigest\(\)\[:8\]",
            app_src,
        ), (
            "P3-READY-REASON-HASH: SHA-256(reason)[:8] no se computa en el "
            "endpoint `/ready`."
        )

    def test_reason_hash_in_response(self, app_src: str):
        """`reason_hash` debe estar en el HTTPException 503 detail."""
        assert re.search(
            r'"reason_hash":\s*reason_hash',
            app_src,
        ), (
            "P3-READY-REASON-HASH: `reason_hash` no se incluye en el response."
        )


# ===========================================================================
# P3-SCHEDULER-ALERT-DEDUP
# ===========================================================================
class TestSchedulerAlertDedup:
    @pytest.fixture(scope="class")
    def app_src(self) -> str:
        return (_BACKEND / "app.py").read_text(encoding="utf-8")

    def test_dedup_cache_constants_defined(self, app_src: str):
        assert "_SCHEDULER_ALERT_LAST_EMIT" in app_src, (
            "P3-SCHEDULER-ALERT-DEDUP: cache `_SCHEDULER_ALERT_LAST_EMIT` no "
            "definido — listener no puede deduplicar."
        )
        assert "_SCHEDULER_ALERT_DEDUP_TTL_S" in app_src, (
            "P3-SCHEDULER-ALERT-DEDUP: TTL `_SCHEDULER_ALERT_DEDUP_TTL_S` no definido."
        )

    def test_dedup_check_before_upsert(self, app_src: str):
        """El listener debe consultar el cache ANTES del UPSERT y skip si
        dentro de TTL. [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado del
        builder PostgREST (`supabase.table("system_alerts").upsert(`) al SQL
        directo `INSERT INTO system_alerts` (UPSERT via ON CONFLICT)."""
        listener_match = re.search(
            r"def _scheduler_alert_listener\([\s\S]+?(?=\ndef |\nasync def )",
            app_src,
        )
        assert listener_match is not None
        body = listener_match.group(0)
        # Cache lookup debe aparecer ANTES del upsert.
        cache_pos = body.find("_SCHEDULER_ALERT_LAST_EMIT.get(")
        upsert_pos = body.find("INSERT INTO system_alerts")
        assert cache_pos != -1 and upsert_pos != -1, (
            "P3-SCHEDULER-ALERT-DEDUP: cache lookup o upsert (INSERT INTO "
            "system_alerts) no encontrados en listener."
        )
        assert cache_pos < upsert_pos, (
            "P3-SCHEDULER-ALERT-DEDUP: cache lookup debe ocurrir ANTES del UPSERT "
            "— de otro modo el dedup no aplica."
        )


# ===========================================================================
# P3-CRONS-STATUS-ADMIN
# ===========================================================================
class TestCronsStatusAdmin:
    @pytest.fixture(scope="class")
    def system_src(self) -> str:
        return (_BACKEND / "routers" / "system.py").read_text(encoding="utf-8")

    def test_endpoint_defined(self, system_src: str):
        assert re.search(
            r'@router\.get\(["\']\/admin\/crons-status["\']\)',
            system_src,
        ), (
            "P3-CRONS-STATUS-ADMIN: endpoint `/admin/crons-status` no definido."
        )
        assert "def admin_crons_status" in system_src, (
            "P3-CRONS-STATUS-ADMIN: función handler `admin_crons_status` no definida."
        )

    def test_auth_gated(self, system_src: str):
        """El endpoint debe llamar a `_verify_admin_token` para gateo Bearer."""
        # El admin endpoint es el último en system.py — no hay `@router` ni
        # `def` siguiente. Anchor al final del archivo o a la próxima func.
        fn_match = re.search(
            r"def admin_crons_status\([\s\S]+?\Z",
            system_src,
        )
        assert fn_match is not None
        body = fn_match.group(0)
        assert "_verify_admin_token" in body, (
            "P3-CRONS-STATUS-ADMIN: endpoint no gateado por `_verify_admin_token` — "
            "expondría listing de jobs sin auth."
        )

    def test_returns_jobs_and_knobs(self, system_src: str):
        """Response debe incluir `jobs` + `knobs_kill_switches` + `has_scheduler`."""
        fn_match = re.search(
            r"def admin_crons_status\([\s\S]+?\Z",
            system_src,
        )
        assert fn_match is not None
        body = fn_match.group(0)
        for key in ("jobs", "knobs_kill_switches", "has_scheduler"):
            assert f'"{key}"' in body, (
                f"P3-CRONS-STATUS-ADMIN: response no incluye `{key}` — operador "
                f"pierde la info clave del snapshot."
            )


# ===========================================================================
# P3-DB-POOL-MAX-IDLE-KNOB (regression test — already implemented)
# ===========================================================================
class TestDbPoolMaxIdleKnob:
    @pytest.fixture(scope="class")
    def db_core_src(self) -> str:
        return (_BACKEND / "db_core.py").read_text(encoding="utf-8")

    def test_knob_referenced(self, db_core_src: str):
        """Knob `MEALFIT_DB_POOL_MAX_IDLE_S` debe estar leído via env var."""
        assert "MEALFIT_DB_POOL_MAX_IDLE_S" in db_core_src, (
            "P3-DB-POOL-MAX-IDLE-KNOB: knob env var no se lee — `max_idle` "
            "habría vuelto a ser hardcoded."
        )

    def test_knob_clamped(self, db_core_src: str):
        """`_float_env` debe aplicar clamp [30, 1800] al knob para evitar
        valores patológicos (0 cierra todas las conexiones instantáneamente)."""
        assert re.search(
            r'_float_env\(\s*["\']MEALFIT_DB_POOL_MAX_IDLE_S["\'][\s\S]{0,100}30\.0[\s\S]{0,40}1800\.0',
            db_core_src,
        ), (
            "P3-DB-POOL-MAX-IDLE-KNOB: clamp [30, 1800] no aplicado al knob — "
            "valores fuera de rango podrían cerrar conexiones prematuramente."
        )


# ===========================================================================
# P3-CONSOLE-DEV-GUARDS + P3-MIGRATION-IDEMPOTENCE-DOC (doc-only)
# ===========================================================================
class TestConventionDocs:
    @pytest.fixture(scope="class")
    def claude_md(self) -> str:
        return _CLAUDE_MD.read_text(encoding="utf-8")

    def test_console_dev_guards_documented(self, claude_md: str):
        assert "P3-CONSOLE-DEV-GUARDS" in claude_md, (
            "P3-CONSOLE-DEV-GUARDS: convención no documentada en CLAUDE.md — "
            "futuros contributors agregarán DEV guards a console.error rompiendo "
            "la captura de Sentry."
        )

    def test_migration_idempotence_documented(self, claude_md: str):
        assert "P3-MIGRATION-IDEMPOTENCE-DOC" in claude_md, (
            "P3-MIGRATION-IDEMPOTENCE-DOC: convención no documentada en CLAUDE.md."
        )
        assert "IF NOT EXISTS" in claude_md, (
            "P3-MIGRATION-IDEMPOTENCE-DOC: la convención debe mencionar "
            "`IF NOT EXISTS` explícitamente."
        )
