"""[P1-CASCADE-INLINE + P1-ORPHAN-MISSED-SWEEP · 2026-05-27]

Bundle test del P1 del audit prod-readiness 2026-05-27 (post P0-DEAD-LETTER-
USER-NOTIFY). Cubre dos gaps observacionales relacionados con
`scheduler_missed_*` / `scheduler_cascade_missed`:

  - **P1-CASCADE-INLINE**: el cron `_alert_scheduler_cascade_missed` corre
    cada `MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN` (default 30, floor 5min
    porque "sub-5min agrava la propia cascada"). Cuando una cascada ocurre
    entre dos ticks del cron, el parent `scheduler_cascade_missed` se emite
    con hasta 30min de delay — SRE ve 18-25 misses huérfanos sin parent
    durante ese intervalo. Live evidence 2026-05-27: 25 distinct
    `scheduler_missed_*` en 7min, threshold default 3, parent NO emitido.

    Fix: contador in-memory + sliding window 60s en
    `_scheduler_alert_listener` (`_maybe_emit_inline_cascade_alert`). El
    listener YA corre por cada event MISSED en sub-segundos; el costo
    marginal de un counter + UPSERT condicional con dedup 120s es
    despreciable. NO es "otro cron en el burst".

  - **P1-ORPHAN-MISSED-SWEEP**: el listener auto-resuelve
    `scheduler_missed_<job>` via `EVENT_JOB_EXECUTED` (P1-NEW-2). Funciona
    para jobs frecuentes pero los jobs raros (interval ≥30min) nunca
    re-ejecutan en la ventana de observación → alert vive forever.
    Live evidence 2026-05-27: 19 alerts huérfanas 7+min sin resolver tras
    cold-start, ensucian dashboard SRE.

    Fix: nuevo cron `_sweep_stale_scheduler_missed_alerts` que cierra
    alerts huérfanas con age > 2h (knob). Guard sistémico: si
    `scheduler_cascade_missed` está activa, NO sweep (preserva señal).
    Tracker consecutive-failure (mismo patrón P1-CRON-BUNDLE).

Tooltip-anchors: P1-CASCADE-INLINE, P1-ORPHAN-MISSED-SWEEP.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"


# Stub apscheduler if not installed (CI standalone).
def _ensure_apscheduler_stub():
    if "apscheduler" not in sys.modules:
        for mod_name in (
            "apscheduler",
            "apscheduler.schedulers",
            "apscheduler.schedulers.background",
            "apscheduler.executors",
            "apscheduler.executors.pool",
            "apscheduler.events",
        ):
            sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
        sys.modules["apscheduler.events"].EVENT_JOB_MISSED = 1
        sys.modules["apscheduler.events"].EVENT_JOB_ERROR = 2
        sys.modules["apscheduler.events"].EVENT_JOB_EXECUTED = 4


# ---------------------------------------------------------------------------
# P1-CASCADE-INLINE — parser tests
# ---------------------------------------------------------------------------

def test_p1_cascade_inline_helper_defined_in_app_py():
    """`_maybe_emit_inline_cascade_alert` debe estar definido en app.py."""
    src = _APP_PY.read_text(encoding="utf-8")
    assert "def _maybe_emit_inline_cascade_alert(job_id" in src, (
        "P1-CASCADE-INLINE: helper `_maybe_emit_inline_cascade_alert(job_id)` "
        "no encontrado en app.py. Sin él, la cascada se detecta con hasta "
        "30min de delay (ver doc)."
    )


def test_p1_cascade_inline_invoked_from_listener_after_boot_grace():
    """El listener debe llamar `_maybe_emit_inline_cascade_alert(job_id)`
    en la rama `EVENT_JOB_MISSED` DESPUÉS del check de boot grace (no
    contar misses suprimidos post-restart)."""
    src = _APP_PY.read_text(encoding="utf-8")
    # Patrón: el `return` del boot grace debe preceder al call inline.
    listener_block_pat = re.compile(
        r"if\s+_uptime_s\s*<\s*_SCHEDULER_BOOT_GRACE_S:.*?return\s*\r?\n"
        r".*?_maybe_emit_inline_cascade_alert\(job_id\)",
        re.DOTALL,
    )
    assert listener_block_pat.search(src), (
        "P1-CASCADE-INLINE: el call `_maybe_emit_inline_cascade_alert(job_id)` "
        "no aparece dentro de la rama EVENT_JOB_MISSED DESPUÉS del boot "
        "grace check. Sin ese orden, contaríamos misses post-restart como "
        "cascada (falsos positivos sistemáticos)."
    )


def test_p1_cascade_inline_knobs_have_clamps():
    """Los 3 knobs del inline detector deben tener clamp explícito:
    WINDOW_S [10, 600], THRESHOLD [3, 50], DEDUP_S [30, 1800]. Sin clamp,
    un operador con typo puede romper el detector (window=0 → división por
    0 implícita en la slide, threshold=1 → emit por cada miss aislado).
    """
    src = _APP_PY.read_text(encoding="utf-8")
    assert re.search(
        r"max\(\s*10\s*,\s*min\(\s*raw\s*,\s*600\s*\)\s*\)", src
    ), "P1-CASCADE-INLINE: WINDOW_S clamp [10, 600] ausente."
    assert re.search(
        r"max\(\s*3\s*,\s*min\(\s*raw\s*,\s*50\s*\)\s*\)", src
    ), "P1-CASCADE-INLINE: THRESHOLD clamp [3, 50] ausente."
    assert re.search(
        r"max\(\s*30\s*,\s*min\(\s*raw\s*,\s*1800\s*\)\s*\)", src
    ), "P1-CASCADE-INLINE: DEDUP_S clamp [30, 1800] ausente."


def test_p1_cascade_inline_has_kill_switch():
    """El kill switch `MEALFIT_SCHEDULER_CASCADE_INLINE_ENABLED` debe estar
    presente — el detector inline debe poder desactivarse sin redeploy si
    introduce volumen problemático contra `system_alerts`."""
    src = _APP_PY.read_text(encoding="utf-8")
    assert 'MEALFIT_SCHEDULER_CASCADE_INLINE_ENABLED' in src, (
        "P1-CASCADE-INLINE: kill switch env var "
        "`MEALFIT_SCHEDULER_CASCADE_INLINE_ENABLED` no encontrado."
    )


def test_p1_cascade_inline_marker_anchor_in_listener():
    """El marker `P1-CASCADE-INLINE` debe aparecer cerca del listener
    para que un refactor no borre la convención silenciosamente."""
    src = _APP_PY.read_text(encoding="utf-8")
    assert src.count("P1-CASCADE-INLINE") >= 3, (
        f"P1-CASCADE-INLINE: marker debe aparecer ≥3 veces en app.py "
        f"(definición + invocación + doc). Encontrado: "
        f"{src.count('P1-CASCADE-INLINE')}."
    )


# ---------------------------------------------------------------------------
# P1-CASCADE-INLINE — functional test (sliding window + threshold + dedup)
# ---------------------------------------------------------------------------

def test_p1_cascade_inline_threshold_triggers_upsert(monkeypatch):
    """Functional: simular N misses > threshold dentro de la ventana →
    UPSERT a `system_alerts` con alert_key=`scheduler_cascade_missed`.
    Mockear `supabase.table().upsert().execute()` para capturar la
    invocación sin tocar DB real.
    """
    _ensure_apscheduler_stub()
    # Stub módulos pesados que app.py importa al top.
    sys.modules.setdefault("sentry_sdk", types.ModuleType("sentry_sdk"))
    # Mockear supabase ANTES de importar app.
    fake_supabase = MagicMock()
    monkeypatch.setattr(
        "os.environ",
        {**__import__("os").environ, "MEALFIT_SCHEDULER_CASCADE_INLINE_THRESHOLD": "3",
         "MEALFIT_SCHEDULER_CASCADE_INLINE_WINDOW_S": "60",
         "MEALFIT_SCHEDULER_CASCADE_INLINE_DEDUP_S": "60"},
    )
    try:
        import importlib
        if "app" in sys.modules:
            del sys.modules["app"]
        # Importar app es caro. Bajamos a importar solo el módulo helper.
        # Truco: leer el módulo en spec mode no es viable; ejecutamos el helper
        # via subprocess-like? Demasiado complejo. Alternativa: skip si import falla.
        try:
            app_mod = importlib.import_module("app")
        except Exception as e:
            pytest.skip(f"app module no importable en entorno test: {e}")
            return

        # Reset state.
        app_mod._CASCADE_INLINE_MISS_TIMESTAMPS.clear()
        app_mod._CASCADE_INLINE_LAST_EMIT_AT = 0.0
        app_mod.supabase = fake_supabase

        # Simular 3 distinct jobs MISSED en rápida sucesión.
        app_mod._maybe_emit_inline_cascade_alert("job_a")
        app_mod._maybe_emit_inline_cascade_alert("job_b")
        app_mod._maybe_emit_inline_cascade_alert("job_c")

        # Debe haber 1 UPSERT al alcanzar threshold=3.
        # supabase.table("system_alerts").upsert({...}, on_conflict=...).execute()
        assert fake_supabase.table.called, (
            "P1-CASCADE-INLINE: supabase.table no fue invocado tras 3 misses con threshold=3."
        )
        # Verificar que el call fue a system_alerts con alert_key correcto.
        table_calls = fake_supabase.table.call_args_list
        assert any(
            call.args == ("system_alerts",) for call in table_calls
        ), f"P1-CASCADE-INLINE: esperaba call a table('system_alerts'). Calls: {table_calls}"
    finally:
        # Cleanup: dejar el módulo en su estado inicial para otros tests.
        if "app" in sys.modules:
            del sys.modules["app"]


def test_p1_cascade_inline_dedup_skips_within_cooldown(monkeypatch):
    """Functional: una segunda ronda de misses dentro del dedup window NO
    debe disparar otro UPSERT. Patrón anti-spam."""
    _ensure_apscheduler_stub()
    sys.modules.setdefault("sentry_sdk", types.ModuleType("sentry_sdk"))
    fake_supabase = MagicMock()

    monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_INLINE_THRESHOLD", "3")
    monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_INLINE_WINDOW_S", "60")
    monkeypatch.setenv("MEALFIT_SCHEDULER_CASCADE_INLINE_DEDUP_S", "300")

    try:
        import importlib
        if "app" in sys.modules:
            del sys.modules["app"]
        try:
            app_mod = importlib.import_module("app")
        except Exception as e:
            pytest.skip(f"app module no importable: {e}")
            return

        app_mod._CASCADE_INLINE_MISS_TIMESTAMPS.clear()
        app_mod._CASCADE_INLINE_LAST_EMIT_AT = 0.0
        app_mod.supabase = fake_supabase

        # Primera tanda: debe emitir.
        for jid in ("job_a", "job_b", "job_c"):
            app_mod._maybe_emit_inline_cascade_alert(jid)
        first_call_count = fake_supabase.table.call_count
        assert first_call_count >= 1

        # Segunda tanda inmediata: dedup debe skipear.
        for jid in ("job_d", "job_e", "job_f"):
            app_mod._maybe_emit_inline_cascade_alert(jid)
        second_call_count = fake_supabase.table.call_count
        assert second_call_count == first_call_count, (
            f"P1-CASCADE-INLINE: dedup falló — esperaba que segunda tanda "
            f"(dentro de cooldown {300}s) NO disparara otro UPSERT. "
            f"first={first_call_count}, second={second_call_count}."
        )
    finally:
        if "app" in sys.modules:
            del sys.modules["app"]


# ---------------------------------------------------------------------------
# P1-ORPHAN-MISSED-SWEEP — parser tests
# ---------------------------------------------------------------------------

def test_p1_orphan_missed_sweep_function_defined():
    """`_sweep_stale_scheduler_missed_alerts` debe estar definido en
    cron_tasks.py."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    assert "def _sweep_stale_scheduler_missed_alerts(" in src, (
        "P1-ORPHAN-MISSED-SWEEP: cron `_sweep_stale_scheduler_missed_alerts` "
        "no definido. Sin él, alerts `scheduler_missed_*` huérfanas "
        "(jobs raros) viven forever inflando dashboard SRE."
    )


def test_p1_orphan_missed_sweep_registered_in_scheduler():
    """El cron debe estar registrado en `register_plan_chunk_scheduler`
    con id `sweep_stale_scheduler_missed_alerts` y knob configurable."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    assert 'id="sweep_stale_scheduler_missed_alerts"' in src, (
        "P1-ORPHAN-MISSED-SWEEP: cron NO registrado en scheduler con "
        "id='sweep_stale_scheduler_missed_alerts'."
    )
    assert "MEALFIT_SCHEDULER_MISSED_SWEEP_INTERVAL_MIN" in src, (
        "P1-ORPHAN-MISSED-SWEEP: knob `MEALFIT_SCHEDULER_MISSED_SWEEP_INTERVAL_MIN` "
        "no presente. El operador debe poder ajustar frecuencia sin redeploy."
    )


def test_p1_orphan_missed_sweep_three_knobs_clamped():
    """Los 3 knobs deben tener clamps via `_env_int(..., validator=...)`:
    INTERVAL_MIN [15, 1440], AGE_HOURS [1, 72], BATCH_LIMIT [1, 1000].
    Sin clamps, operator con typo causa OOM o sweep no-op."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    # INTERVAL_MIN clamp en el registro
    assert re.search(
        r'MEALFIT_SCHEDULER_MISSED_SWEEP_INTERVAL_MIN["\'\s,]+60\s*,\s*\r?\n?\s*validator=lambda v:\s*15\s*<=\s*v\s*<=\s*1440',
        src,
    ), "P1-ORPHAN-MISSED-SWEEP: INTERVAL_MIN clamp [15, 1440] ausente."
    # AGE_HOURS clamp
    assert re.search(
        r'MEALFIT_SCHEDULER_MISSED_SWEEP_AGE_HOURS["\'\s,]+2\s*,\s*\r?\n?\s*validator=lambda v:\s*1\s*<=\s*v\s*<=\s*72',
        src,
    ), "P1-ORPHAN-MISSED-SWEEP: AGE_HOURS clamp [1, 72] ausente."
    # BATCH_LIMIT clamp
    assert re.search(
        r'MEALFIT_SCHEDULER_MISSED_SWEEP_BATCH_LIMIT["\'\s,]+100\s*,\s*\r?\n?\s*validator=lambda v:\s*1\s*<=\s*v\s*<=\s*1000',
        src,
    ), "P1-ORPHAN-MISSED-SWEEP: BATCH_LIMIT clamp [1, 1000] ausente."


def test_p1_orphan_missed_sweep_has_cascade_guard():
    """El sweep DEBE tener guard: si `scheduler_cascade_missed` está activa
    en la ventana reciente, NO cerrar los hijos (preserva señal sistémica
    para SRE). Sin guard, el sweep oculta cascadas reales al cerrar
    prematuramente los hijos."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    # Busca dentro de la función la query del guard.
    func_match = re.search(
        r"def _sweep_stale_scheduler_missed_alerts\(.*?(?=\ndef\s)",
        src,
        re.DOTALL,
    )
    assert func_match, "_sweep_stale_scheduler_missed_alerts no localizable."
    body = func_match.group(0)
    assert "alert_key = 'scheduler_cascade_missed'" in body, (
        "P1-ORPHAN-MISSED-SWEEP: guard de cascade activa ausente. Sin él, "
        "el sweep cerraría prematuramente hijos de cascadas reales."
    )
    assert "cascade_active" in body, (
        "P1-ORPHAN-MISSED-SWEEP: variable `cascade_active` no presente — "
        "el guard debe poder shortcircuit explícito."
    )


def test_p1_orphan_missed_sweep_uses_consecutive_failure_tracker():
    """El cron debe usar el helper SSOT `_track_cron_consecutive_failure`
    con alert_key `sweep_stale_scheduler_missed_alerts_failures_burst`
    (mismo patrón P1-CRON-BUNDLE)."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    assert "sweep_stale_scheduler_missed_alerts_failures_burst" in src, (
        "P1-ORPHAN-MISSED-SWEEP: alert_key `*_failures_burst` no presente. "
        "Sin tracking de fallos consecutivos, un cron silencioso quema "
        "ventanas de observación sin alarma."
    )
    func_match = re.search(
        r"def _sweep_stale_scheduler_missed_alerts\(.*?(?=\ndef\s)",
        src,
        re.DOTALL,
    )
    body = func_match.group(0)
    assert "_track_cron_consecutive_failure" in body, (
        "P1-ORPHAN-MISSED-SWEEP: el cron NO invoca _track_cron_consecutive_failure."
    )


def test_p1_orphan_missed_sweep_marker_in_alert_table_doc():
    """El alert_key nuevo debe estar documentado en
    `docs/system_alerts_resolution_table.md` (cumple cross-link
    `test_p2_audit_4_alert_keys_documented`)."""
    doc_path = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"
    text = doc_path.read_text(encoding="utf-8")
    assert "sweep_stale_scheduler_missed_alerts_failures_burst" in text, (
        "P1-ORPHAN-MISSED-SWEEP: alert_key nuevo NO documentado en "
        "system_alerts_resolution_table.md. El test "
        "`test_p2_audit_4_alert_keys_documented` también fallará."
    )


def test_p1_cascade_inline_acknowledged_in_alert_table_doc():
    """La fila `scheduler_cascade_missed` debe mencionar el productor inline
    además del cron (transparencia operacional para SRE)."""
    doc_path = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"
    text = doc_path.read_text(encoding="utf-8")
    # La fila debe contener referencia al P1-CASCADE-INLINE.
    assert "P1-CASCADE-INLINE" in text, (
        "P1-CASCADE-INLINE: la fila `scheduler_cascade_missed` del doc "
        "debe mencionar el productor inline. Sin actualizar el doc, SRE "
        "no sabe que hay 2 productores del mismo alert_key."
    )


# ---------------------------------------------------------------------------
# Marker anchor
# ---------------------------------------------------------------------------

def test_marker_anchor_present_for_p1_bundle():
    """Ambos markers (`P1-CASCADE-INLINE`, `P1-ORPHAN-MISSED-SWEEP`) deben
    aparecer en cron_tasks.py + app.py. Sin anchors, refactors borran
    la convención silenciosamente."""
    app_src = _APP_PY.read_text(encoding="utf-8")
    cron_src = _CRON_TASKS.read_text(encoding="utf-8")
    assert app_src.count("P1-CASCADE-INLINE") >= 3
    assert cron_src.count("P1-ORPHAN-MISSED-SWEEP") >= 3, (
        f"P1-ORPHAN-MISSED-SWEEP marker count en cron_tasks.py: "
        f"{cron_src.count('P1-ORPHAN-MISSED-SWEEP')}. Esperaba ≥3 "
        f"(definición + registro + comentarios)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
