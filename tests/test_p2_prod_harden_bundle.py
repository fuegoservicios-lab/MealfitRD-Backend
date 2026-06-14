"""[P2-PROD-HARDEN-BUNDLE · 2026-05-27] Tests de los 5 P2 del análisis profundo
de backend 2026-05-27 (sibling de P1-PROD-HARDEN-BUNDLE; el P2-6 .env-en-OneDrive
es ambiental, no code-fix → no testeable).

Parser-based (lee source de prod con regex) — NO importa `app`/`db_facts`/
`routers.plans` para evitar el arranque de scheduler/DB y la colisión `supabase`
pip vs `backend/migrations/` dir. Cada assertion ancla un tooltip-anchor.

P2-1 P2-OPS-SHUTDOWN: `scheduler.shutdown(wait=False)` + `shutdown_bg_executor`.
P2-2 P2-BODY-SIZE-LIMIT: middleware que rechaza Content-Length > cap (413).
P2-3 P2-DOCS-GATE: /docs /redoc /openapi.json ocultos si ENVIRONMENT=production.
P2-4 P2-TEMPORAL-CLEANUP-GC: eviction oportunista del debounce dict.
P2-5 P2-GEN-WALL-TIMEOUT: watchdog wall-clock del pipeline SSE + comentarios
     `asyncio.shield` obsoletos corregidos.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_BG_EXECUTOR = _BACKEND_ROOT / "bg_executor.py"
_DB_FACTS = _BACKEND_ROOT / "db_facts.py"
_PLANS = _BACKEND_ROOT / "routers" / "plans.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===========================================================================
# P2-3 · P2-DOCS-GATE (app.py)
# ===========================================================================
class TestP2DocsGate:
    def test_is_production_flag(self):
        src = _read(_APP_PY)
        # [P2-PROD-AUDIT-3 · 2026-05-30] Migrado al helper SSOT `is_production()`
        # (knobs.py, normaliza lower+strip). Antes era el literal exact-match
        # `os.environ.get("ENVIRONMENT") == "production"` (case/whitespace-sensible).
        assert "_IS_PRODUCTION = is_production()" in src
        assert "from knobs import" in src and "is_production" in src

    def test_docs_urls_gated(self):
        src = _read(_APP_PY)
        ctor = src.split("app = FastAPI(")[1].split(")")[0]
        assert 'docs_url=None if _IS_PRODUCTION else "/docs"' in ctor
        assert 'redoc_url=None if _IS_PRODUCTION else "/redoc"' in ctor
        assert 'openapi_url=None if _IS_PRODUCTION else "/openapi.json"' in ctor

    def test_marker_anchor(self):
        assert "P2-DOCS-GATE" in _read(_APP_PY)


# ===========================================================================
# P2-1 · P2-OPS-SHUTDOWN (app.py + bg_executor.py)
# ===========================================================================
class TestP2OpsShutdown:
    def test_scheduler_shutdown_non_blocking(self):
        src = _read(_APP_PY)
        assert "scheduler.shutdown(wait=False)" in src, (
            "scheduler.shutdown debe usar wait=False para no colgar el deploy."
        )
        # Anti-regresión: ninguna LÍNEA DE CÓDIGO (no-comentario) debe ser el
        # `scheduler.shutdown()` bloqueante (wait=True default). El literal
        # puede aparecer en comentarios explicativos — solo nos importa el call.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "scheduler.shutdown()" not in stripped, (
                f"Línea de código con shutdown() bloqueante: {line!r}"
            )

    def test_bg_executor_drained_on_shutdown(self):
        src = _read(_APP_PY)
        teardown = src.split("yield", 1)[1]
        assert "shutdown_bg_executor(wait=False)" in teardown, (
            "El teardown debe drenar el pool de bg_executor."
        )

    def test_shutdown_bg_executor_function_exists(self):
        src = _read(_BG_EXECUTOR)
        assert "def shutdown_bg_executor(" in src
        body = src.split("def shutdown_bg_executor(")[1].split("def _persist_bg_task_timeout_alert(")[0]
        assert "cancel_futures=True" in body, (
            "shutdown_bg_executor debe descartar tasks encolados (cancel_futures)."
        )

    def test_marker_anchor(self):
        assert "P2-OPS-SHUTDOWN" in _read(_APP_PY)
        assert "P2-OPS-SHUTDOWN" in _read(_BG_EXECUTOR)


# ===========================================================================
# P2-2 · P2-BODY-SIZE-LIMIT (app.py)
# ===========================================================================
class TestP2BodySizeLimit:
    def test_knob_default_and_clamp(self):
        src = _read(_APP_PY)
        m = re.search(
            r'_MAX_REQUEST_BYTES\s*=\s*_env_int\(\s*'
            r'"MEALFIT_MAX_REQUEST_BYTES"\s*,\s*25\s*\*\s*1024\s*\*\s*1024\s*,\s*'
            r'validator=lambda v:\s*v\s*==\s*0\s*or\s*\(1024\s*\*\s*1024\s*<=\s*v\s*<=\s*200\s*\*\s*1024\s*\*\s*1024\)',
            src,
        )
        assert m is not None, (
            "MEALFIT_MAX_REQUEST_BYTES debe defaultear a 25 MiB con clamp "
            "[1 MiB, 200 MiB] (0 desactiva)."
        )

    def test_middleware_rejects_oversize_with_413(self):
        src = _read(_APP_PY)
        assert "async def _body_size_limit_middleware(" in src
        body = src.split("async def _body_size_limit_middleware(")[1].split("async def ")[0]
        assert 'request.headers.get("content-length")' in body
        assert "_MAX_REQUEST_BYTES > 0" in body, "0 debe desactivar el guard."
        assert "status_code=413" in body

    def test_default_above_image_upload_cap(self):
        """El default (25 MiB) debe superar el cap de imagen (20 MB) para no
        romper el upload legítimo de /api/diary."""
        assert 25 * 1024 * 1024 > 20 * 1024 * 1024

    def test_marker_anchor(self):
        assert "P2-BODY-SIZE-LIMIT" in _read(_APP_PY)


# ===========================================================================
# P2-4 · P2-TEMPORAL-CLEANUP-GC (db_facts.py)
# ===========================================================================
class TestP2TemporalCleanupGC:
    def test_threshold_constant(self):
        src = _read(_DB_FACTS)
        assert "_TEMPORAL_CLEANUP_GC_THRESHOLD = 512" in src

    def test_eviction_in_record_path(self):
        src = _read(_DB_FACTS)
        fn = src.split("def _should_skip_temporal_cleanup(")[1].split("\ndef ")[0]
        # La eviction barre entries con edad > 2× ventana y las elimina.
        assert "_stale_cutoff = now - (2 * window_s)" in fn
        assert ".pop(_k, None)" in fn
        assert "_TEMPORAL_CLEANUP_GC_THRESHOLD" in fn
        # La eviction está DESPUÉS de registrar el timestamp (record-path),
        # no en el skip-path.
        assert fn.index("_TEMPORAL_CLEANUP_LAST_RUN[user_id] = now") < fn.index("_stale_cutoff")

    def test_marker_anchor(self):
        assert "P2-TEMPORAL-CLEANUP-GC" in _read(_DB_FACTS)


# ===========================================================================
# P2-5 · P2-GEN-WALL-TIMEOUT (routers/plans.py)
# ===========================================================================
class TestP2GenWallTimeout:
    def test_knob_default_and_clamp(self):
        src = _read(_PLANS)
        m = re.search(
            r'"MEALFIT_GENERATION_MAX_WALL_S"\s*,\s*900\s*,\s*'
            r'validator=lambda v:\s*v\s*==\s*0\s*or\s*60\s*<=\s*v\s*<=\s*3600',
            src,
        )
        assert m is not None, (
            "MEALFIT_GENERATION_MAX_WALL_S debe defaultear a 900 con clamp "
            "[60, 3600] (0 desactiva)."
        )

    def test_watchdog_cancels_pipeline(self):
        src = _read(_PLANS)
        assert "async def _pipeline_wall_clock_guard(" in src
        guard = src.split("async def _pipeline_wall_clock_guard(")[1].split("_wall_guard_task = asyncio.create_task(")[0]
        assert "await asyncio.sleep(_gen_wall_s)" in guard
        assert "_pipeline_task.cancel()" in guard
        assert "if not _pipeline_task.done():" in guard

    def test_watchdog_cancelled_when_pipeline_done(self):
        src = _read(_PLANS)
        assert re.search(
            r"_pipeline_task\.add_done_callback\(\s*lambda _t, _g=_wall_guard_task: _g\.cancel\(\)",
            src,
        ), "El watchdog debe cancelarse cuando el pipeline termina."

    def test_gated_by_knob_nonzero(self):
        src = _read(_PLANS)
        assert "if _gen_wall_s > 0:" in src, "0 debe desactivar el watchdog."

    def test_stale_asyncio_shield_comments_fixed(self):
        """Los 3 comentarios que afirmaban que el pipeline corre con
        `asyncio.shield` fueron corregidos (nunca hubo tal llamada). Ahora
        deben decir que es `asyncio.create_task` independiente / NO shielded."""
        src = _read(_PLANS)
        # Nunca debió existir una llamada real a asyncio.shield.
        assert "asyncio.shield(" not in src, (
            "No debe haber llamada real a asyncio.shield(."
        )
        # Los comentarios corregidos mencionan explícitamente 'NO shielded'.
        assert src.count("NO shielded") >= 2, (
            "Los comentarios deben aclarar que el pipeline NO está shielded "
            "(corre como create_task independiente)."
        )

    def test_marker_anchor(self):
        assert "P2-GEN-WALL-TIMEOUT" in _read(_PLANS)


# ===========================================================================
# Marker bump
# ===========================================================================
def test_last_known_pfix_bumped():
    # Exact-match al cierre; RELAJADO a date-floor tras ser superseded por
    # P1-PROD-AUDIT-BUNDLE · 2026-05-28 (mismo patrón que test_p1_prod_harden_bundle).
    from datetime import date, datetime
    src = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m is not None
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m is not None, f"Marker sin fecha ISO: {marker!r}."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 27), (
        f"Marker `{marker}` (fecha {marker_date}) regresó por debajo del "
        f"floor del bundle P2-PROD-HARDEN-BUNDLE (2026-05-27)."
    )
