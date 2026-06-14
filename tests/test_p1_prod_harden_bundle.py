"""[P1-PROD-HARDEN-BUNDLE · 2026-05-27] Tests del bundle de 2 P1 del audit
prod-readiness 2026-05-27 (post-P0-DEAD-LETTER-USER-NOTIFY).

Parser-based (lee el source de prod con regex) — NO importa `app`/`db_core`
para evitar (a) el arranque de scheduler/DB en `import app`, (b) la colisión
conocida `supabase` (pip) vs `backend/migrations/` (dir migrations) que rompe
el import en el venv de tests. Cada assertion ancla un tooltip-anchor en el
código fuente: si alguien renombra/borra la defensa, el test falla ANTES de
que el comportamiento de prod cambie.

P1-1 (P1-DB-STMT-TIMEOUT): `db_core.py` aplica `statement_timeout` +
`idle_in_transaction_session_timeout` a nivel de sesión sobre cada conexión
de los pools sync/async — cierra el agotamiento del pool por query atascada
que retiene su slot indefinidamente. El `chat_checkpoint_pool` sensible queda
SIN timeouts (configure propio).

P1-2 (P1-SCHEDULER-LEADER-LOCK): `app.py` adquiere un advisory lock de
Postgres a nivel de sesión (forma two-int → espacio separado de los locks
single-bigint de meal_plans) antes de arrancar el scheduler + hard-floor.
Sólo el worker leader corre crons. Fail-open. `Procfile` fija `--workers 1`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE = _BACKEND_ROOT / "db_core.py"
_APP_PY = _BACKEND_ROOT / "app.py"
_DB_PLANS = _BACKEND_ROOT / "db_plans.py"
_PROCFILE = _BACKEND_ROOT / "Procfile"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===========================================================================
# P1-1 · P1-DB-STMT-TIMEOUT (db_core.py)
# ===========================================================================
class TestP1StatementTimeout:
    def test_knob_statement_timeout_default_and_clamp(self):
        src = _read(_DB_CORE)
        m = re.search(
            r'DB_STATEMENT_TIMEOUT_MS\s*=\s*_knob_env_int\(\s*'
            r'"MEALFIT_DB_STATEMENT_TIMEOUT_MS"\s*,\s*30000\s*,\s*'
            r'validator=lambda v:\s*0\s*<=\s*v\s*<=\s*600000',
            src,
        )
        assert m is not None, (
            "MEALFIT_DB_STATEMENT_TIMEOUT_MS debe definirse con default 30000 "
            "y validator clamp [0, 600000] via _knob_env_int."
        )

    def test_knob_idle_in_txn_default_and_clamp(self):
        src = _read(_DB_CORE)
        m = re.search(
            r'DB_IDLE_IN_TXN_TIMEOUT_MS\s*=\s*_knob_env_int\(\s*'
            r'"MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS"\s*,\s*15000\s*,\s*'
            r'validator=lambda v:\s*0\s*<=\s*v\s*<=\s*600000',
            src,
        )
        assert m is not None, (
            "MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS debe definirse con default 15000 "
            "y validator clamp [0, 600000]."
        )

    def test_knob_uses_registered_helper(self):
        """Los knobs nuevos usan `knobs._env_int` (registrado en
        _KNOBS_REGISTRY → visible en /health/version), no el `_int_env`
        local no-registrado de los DB pool knobs siblings."""
        src = _read(_DB_CORE)
        assert "from knobs import _env_int as _knob_env_int" in src

    def test_session_timeout_helper_emits_both_sets(self):
        src = _read(_DB_CORE)
        assert "def _session_timeout_statements(" in src, (
            "Falta el helper SSOT `_session_timeout_statements`."
        )
        # El helper debe emitir AMBAS sentencias SET.
        assert 'SET statement_timeout = {int(DB_STATEMENT_TIMEOUT_MS)}' in src
        assert (
            'SET idle_in_transaction_session_timeout = {int(DB_IDLE_IN_TXN_TIMEOUT_MS)}'
            in src
        )

    def test_zero_disables_each_timeout(self):
        """`> 0` gating: knob=0 desactiva ese timeout (rollback sin redeploy)."""
        src = _read(_DB_CORE)
        helper = src.split("def _session_timeout_statements(")[1].split("\n\n")[0]
        assert "DB_STATEMENT_TIMEOUT_MS > 0" in helper
        assert "DB_IDLE_IN_TXN_TIMEOUT_MS > 0" in helper

    def test_sync_and_async_configure_apply_timeouts(self):
        src = _read(_DB_CORE)
        sync_body = src.split("def configure_sync_conn(")[1].split("async def configure_async_conn(")[0]
        async_body = src.split("async def configure_async_conn(")[1].split("def configure_checkpoint_conn(")[0]
        assert "_session_timeout_statements()" in sync_body, (
            "configure_sync_conn debe aplicar los session timeouts."
        )
        assert "_session_timeout_statements()" in async_body, (
            "configure_async_conn debe aplicar los session timeouts."
        )

    def test_checkpoint_pool_has_own_configure_without_timeouts(self):
        """El pool sensible del checkpointer (P1-CHAT-CHECKPOINT-*) NO debe
        recibir los session timeouts — configure propio."""
        src = _read(_DB_CORE)
        assert "def configure_checkpoint_conn(" in src
        ckpt_body = src.split("def configure_checkpoint_conn(")[1].split("connection_pool = ConnectionPool(")[0]
        assert "_session_timeout_statements()" not in ckpt_body, (
            "configure_checkpoint_conn NO debe aplicar timeouts (riesgo de "
            "regresar el bug SSL bad length / EOF del pool del checkpointer)."
        )
        # El chat_checkpoint_pool debe enlazar configure_checkpoint_conn.
        ckpt_ctor = src.split("chat_checkpoint_pool = ConnectionPool(")[1].split("open=False")[0]
        assert "configure=configure_checkpoint_conn" in ckpt_ctor, (
            "chat_checkpoint_pool debe usar configure=configure_checkpoint_conn."
        )

    def test_marker_anchor_present(self):
        src = _read(_DB_CORE)
        assert src.count("P1-DB-STMT-TIMEOUT") >= 3, (
            "Tooltip-anchor P1-DB-STMT-TIMEOUT debe aparecer en db_core.py."
        )


# ===========================================================================
# P1-2 · P1-SCHEDULER-LEADER-LOCK (app.py + Procfile)
# ===========================================================================
class TestP1SchedulerLeaderLock:
    def test_helper_functions_exist(self):
        src = _read(_APP_PY)
        assert "def _acquire_scheduler_leader_lock(" in src
        assert "def _build_session_mode_db_url(" in src

    def test_uses_two_int_session_advisory_lock(self):
        """Forma two-int `pg_try_advisory_lock(%s, %s)` (espacio de locks
        separado del single-bigint de meal_plans) y a nivel de SESIÓN (NO
        `_xact_`, que se liberaría al commit)."""
        src = _read(_APP_PY)
        assert "pg_try_advisory_lock(%s, %s)" in src, (
            "Debe usar la forma two-int de pg_try_advisory_lock para no "
            "colisionar con los locks single-bigint de db_plans.py."
        )
        leader_body = src.split("def _acquire_scheduler_leader_lock(")[1].split("\n\n\n")[0]
        assert "pg_advisory_xact_lock" not in leader_body, (
            "El leader lock debe ser de sesión (persiste con la conn), NO "
            "transaccional (`_xact_` se libera al terminar la sentencia)."
        )

    def test_collision_safety_against_db_plans_single_bigint(self):
        """Cross-file: db_plans usa single-bigint; app usa two-int. Postgres
        mantiene ambos espacios separados → cero colisión por diseño."""
        plans_src = _read(_DB_PLANS)
        assert "pg_advisory_xact_lock(hashtextextended(" in plans_src, (
            "db_plans debe seguir usando la forma single-bigint "
            "(hashtextextended) — si migra a two-int, revisar colisión."
        )

    def test_knob_enable_default_on(self):
        src = _read(_APP_PY)
        assert '_env_bool("MEALFIT_SCHEDULER_LEADER_LOCK", True)' in src, (
            "Knob MEALFIT_SCHEDULER_LEADER_LOCK debe defaultear a True (guard on)."
        )

    def test_knob_lock_key_clamped_to_int4(self):
        src = _read(_APP_PY)
        m = re.search(
            r'"MEALFIT_SCHEDULER_LEADER_LOCK_KEY"\s*,\s*424242\s*,\s*'
            r'validator=lambda v:\s*-2147483648\s*<=\s*v\s*<=\s*2147483647',
            src,
        )
        assert m is not None, (
            "MEALFIT_SCHEDULER_LEADER_LOCK_KEY debe clampear al rango int4."
        )

    def test_session_url_forces_5432(self):
        src = _read(_APP_PY)
        body = src.split("def _build_session_mode_db_url(")[1].split("def _acquire_scheduler_leader_lock(")[0]
        assert 'replace(":6543", ":5432")' in body, (
            "El leader lock requiere session mode (5432); el transaction "
            "pooler (6543) liberaría el advisory lock al terminar la sentencia."
        )

    def test_fail_open_acts_as_leader_on_error(self):
        """FAIL-OPEN: ante error de adquisición, el worker actúa como leader
        (prefiere crons duplicados a CERO crons)."""
        src = _read(_APP_PY)
        body = src.split("def _acquire_scheduler_leader_lock(")[1].split("\n\n\n")[0]
        # La rama except debe retornar (True, None).
        except_tail = body.split("except Exception as _lock_err:")[1]
        assert "return True, None" in except_tail, (
            "La rama except debe fail-open (return True, None)."
        )

    def test_scheduler_start_gated_on_leader(self):
        src = _read(_APP_PY)
        assert "if _is_scheduler_leader and HAS_SCHEDULER and scheduler:" in src, (
            "El bloque que arranca el scheduler debe estar gateado por "
            "_is_scheduler_leader."
        )

    def test_hardfloor_gated_on_leader(self):
        src = _read(_APP_PY)
        # Rama no-leader: hard-floor NO se crea.
        assert "if not _is_scheduler_leader:" in src
        assert "app.state.is_scheduler_leader = _is_scheduler_leader" in src

    def test_shutdown_closes_leader_conn(self):
        src = _read(_APP_PY)
        # Tras yield, en teardown, cerrar la conn que sostiene el lock.
        teardown = src.split("yield", 1)[1]
        assert 'getattr(app.state, "scheduler_leader_conn", None)' in teardown, (
            "El teardown del lifespan debe cerrar scheduler_leader_conn para "
            "liberar el advisory lock de sesión."
        )

    def test_marker_anchor_present(self):
        src = _read(_APP_PY)
        assert src.count("P1-SCHEDULER-LEADER-LOCK") >= 4, (
            "Tooltip-anchor P1-SCHEDULER-LEADER-LOCK debe aparecer en app.py."
        )

    def test_procfile_pins_single_worker(self):
        assert _PROCFILE.exists(), "Falta el Procfile (SSOT del start command)."
        content = _read(_PROCFILE)
        assert "--workers 1" in content, (
            "Procfile debe fijar --workers 1 (asunción single-worker del "
            "BackgroundScheduler in-process)."
        )
        assert "uvicorn app:app" in content


# ===========================================================================
# Marker bump (RELAJADO a date-floor tras supersede por P2-PROD-HARDEN-BUNDLE)
# ===========================================================================
def test_last_known_pfix_bumped():
    """Exact-match al cierre; RELAJADO a date-floor tras ser superseded por
    P2-PROD-HARDEN-BUNDLE (mismo día). Solo verifica no-regresión bajo el
    floor de este bundle (patrón emergente desde P1-PROD-FINAL-1)."""
    from datetime import datetime, date
    src = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m is not None
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 27), (
        f"Marker `{marker}` (fecha {marker_date}) regresó por debajo del "
        f"floor de P1-PROD-HARDEN-BUNDLE (2026-05-27)."
    )
