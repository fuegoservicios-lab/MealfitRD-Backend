"""[P0-1-DEPLOY-LAG · 2026-05-10] Regression guard: detector de deriva
de despliegue (`_alert_deploy_lag_marker_stale`).

Bug original (audit 2026-05-10):
    El binario corriendo en EasyPanel quedó rezagado vs HEAD. Logs Postgres
    mostraron `column "updated_at" does not exist` cada hora (cron P3-B
    golpeando un schema que HEAD ya había corregido en P0-OBS-1) más
    `column "completed_at"`, `column "acquired_at"`, `column "generation_status"`
    — todos atribuibles a código pre-P0-OBS-1 que el binario seguía
    ejecutando. Marker `_LAST_KNOWN_PFIX` y `/health/version` ya existían
    pero la verificación era manual y nadie comparaba.

Fix:
    `cron_tasks._alert_deploy_lag_marker_stale` corre diario y emite
    `system_alerts` cuando:
      A) El marker del binario tiene fecha > umbral (default 168h).
      B) `app_kv_store["expected_last_known_pfix"]` no matchea el marker
         del binario (señal estricta cuando el operador publica el esperado).

Cobertura de este test:
    1. Parser del marker (formato Pn-X · YYYY-MM-DD).
    2. Señal A — marker_stale por edad (>umbral).
    3. Señal A — marker fresco no emite.
    4. Señal B — drift vs valor publicado en app_kv_store.
    5. Señal B — match exacto no emite.
    6. Marker malformado → alerta separada `deploy_lag_marker_malformed`.
    7. SELECT app_kv_store falla → no crash, marker_stale igual evalúa.
    8. Cron registrado en `register_plan_chunk_scheduler` con id correcto.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: instala un fake `app` module con `_LAST_KNOWN_PFIX` controlable.
# La función real hace `from app import _LAST_KNOWN_PFIX` dentro del body —
# no importamos el app real (dispararía sentry_sdk.init + DB + scheduler).
# ---------------------------------------------------------------------------
def _install_fake_app(marker: str | None):
    fake = types.ModuleType("app")
    if marker is not None:
        fake._LAST_KNOWN_PFIX = marker
    return fake


def _patch_app(marker: str | None):
    return patch.dict(sys.modules, {"app": _install_fake_app(marker)})


# ---------------------------------------------------------------------------
# 1. Parser del marker
# ---------------------------------------------------------------------------
class TestParsePfixMarker:
    """Cobertura del helper `_parse_pfix_marker` (mismo regex que P3-1)."""

    def test_valid_basic(self):
        from cron_tasks import _parse_pfix_marker

        out = _parse_pfix_marker("P0-OBS-1 · 2026-05-10")
        assert out is not None
        prefix, dt = out
        assert prefix == "P0-OBS-1"
        assert dt.date().isoformat() == "2026-05-10"
        assert dt.tzinfo is timezone.utc

    def test_valid_multi_segment_new(self):
        from cron_tasks import _parse_pfix_marker

        out = _parse_pfix_marker("P2-NEW-A · 2026-04-01")
        assert out is not None
        assert out[0] == "P2-NEW-A"

    def test_valid_long_segments(self):
        from cron_tasks import _parse_pfix_marker

        out = _parse_pfix_marker("P0-1-DEPLOY-LAG · 2026-05-10")
        assert out is not None
        assert out[0] == "P0-1-DEPLOY-LAG"

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "garbage",
            "P0 2026-05-10",                # falta separador
            "P0-X · 2026/05/10",            # formato fecha inválido
            "P0-X · 2026-13-40",            # fecha imposible
            "p0-x · 2026-05-10",            # lowercase prefix
        ],
    )
    def test_invalid(self, raw):
        from cron_tasks import _parse_pfix_marker

        assert _parse_pfix_marker(raw) is None


# ---------------------------------------------------------------------------
# 2. Señal A — marker stale por edad
# ---------------------------------------------------------------------------
class TestSignalAAge:
    def test_stale_marker_emits_alert(self, monkeypatch):
        """Marker con fecha > 168h: debe emitir `deploy_lag_marker_stale`."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        # 30 días atrás → claramente >168h.
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        marker = f"P0-OLD · {old_date}"

        write_mock = MagicMock()
        # SELECT app_kv_store devuelve None (no expected publicado) → solo señal A.
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert write_mock.called, "Debió emitir alerta para marker stale."
        called_sql = write_mock.call_args[0][0]
        called_args = write_mock.call_args[0][1]
        assert "system_alerts" in called_sql
        assert called_args[0] == "deploy_lag_marker_stale"
        assert "Deploy rezagado" in called_args[1]

    def test_fresh_marker_no_alert(self):
        """Marker de hoy: NO debe emitir alerta señal A."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        today = datetime.now(timezone.utc).date().isoformat()
        marker = f"P0-FRESH · {today}"

        write_mock = MagicMock()
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert not write_mock.called, "Marker fresco no debe emitir alerta."

    def test_threshold_knob_respected(self, monkeypatch):
        """Si MEALFIT_DEPLOY_LAG_ALERT_HOURS=1, marker de hace 2h debe alertar."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        monkeypatch.setenv("MEALFIT_DEPLOY_LAG_ALERT_HOURS", "1")
        # Fecha de hace 2 días para superar 1h con margen.
        old = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
        marker = f"P0-X · {old}"

        write_mock = MagicMock()
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert write_mock.called


# ---------------------------------------------------------------------------
# 3. Señal B — drift vs valor publicado
# ---------------------------------------------------------------------------
class TestSignalBExpected:
    def test_drift_emits_when_expected_mismatches(self):
        """Marker en proceso != app_kv_store.expected → emit drift."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        today = datetime.now(timezone.utc).date().isoformat()
        live = f"P0-OLD · {today}"
        expected = f"P0-NEW · {today}"

        write_mock = MagicMock()
        with _patch_app(live), \
             patch("cron_tasks.execute_sql_query", return_value={"value": expected}), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert write_mock.called, "Mismatch live↔expected debe emitir."
        keys = [c.args[1][0] for c in write_mock.call_args_list]
        assert "deploy_lag_drift_vs_expected" in keys

    def test_match_does_not_emit(self):
        """Marker en proceso == app_kv_store.expected → no emit señal B."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        today = datetime.now(timezone.utc).date().isoformat()
        marker = f"P0-OK · {today}"

        write_mock = MagicMock()
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", return_value={"value": marker}), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert not write_mock.called, "Match exacto no debe emitir."

    def test_expected_as_object_with_marker_key(self):
        """Soporta `value` como objeto `{"marker": "..."}` además de string puro."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        today = datetime.now(timezone.utc).date().isoformat()
        live = f"P0-OLD · {today}"
        expected_obj = {"marker": f"P0-NEW · {today}", "published_by": "ci"}

        write_mock = MagicMock()
        with _patch_app(live), \
             patch("cron_tasks.execute_sql_query", return_value={"value": expected_obj}), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert write_mock.called

    def test_no_expected_published_silent(self):
        """Si nadie publicó expected, señal B silenciosa (no falsos positivos)."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        today = datetime.now(timezone.utc).date().isoformat()
        marker = f"P0-OK · {today}"

        write_mock = MagicMock()
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert not write_mock.called


# ---------------------------------------------------------------------------
# 4. Robustez: marker malformado, DB caída
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_malformed_marker_emits_specific_alert(self):
        """Marker no parsea → alerta `deploy_lag_marker_malformed` (NO crash)."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        write_mock = MagicMock()
        with _patch_app("garbage_no_separator"), \
             patch("cron_tasks.execute_sql_query", return_value=None), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        assert write_mock.called
        assert write_mock.call_args[0][1][0] == "deploy_lag_marker_malformed"

    def test_select_failure_does_not_crash_signal_a(self):
        """SELECT app_kv_store falla → señal A igual debe haber evaluado."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        old = (datetime.now(timezone.utc) - timedelta(days=400)).date().isoformat()
        marker = f"P0-VERY-OLD · {old}"

        write_mock = MagicMock()
        with _patch_app(marker), \
             patch("cron_tasks.execute_sql_query", side_effect=RuntimeError("db down")), \
             patch("cron_tasks.execute_sql_write", write_mock):
            _alert_deploy_lag_marker_stale()

        # Señal A se evaluó ANTES del SELECT, así que la alerta stale debió emitirse.
        assert write_mock.called
        assert write_mock.call_args_list[0].args[1][0] == "deploy_lag_marker_stale"

    def test_app_import_failure_returns_silently(self):
        """Si `from app import _LAST_KNOWN_PFIX` falla, retorna sin escribir."""
        from cron_tasks import _alert_deploy_lag_marker_stale

        # Forzamos ImportError eliminando el módulo `app` y haciendo que
        # __getattr__ de un fake lance.
        broken = types.ModuleType("app")
        # No setear _LAST_KNOWN_PFIX → AttributeError en `from app import _LAST_KNOWN_PFIX`.

        write_mock = MagicMock()
        with patch.dict(sys.modules, {"app": broken}), \
             patch("cron_tasks.execute_sql_write", write_mock):
            # No debe levantar.
            _alert_deploy_lag_marker_stale()

        assert not write_mock.called


# ---------------------------------------------------------------------------
# 5. Cron registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_with_correct_id():
    """`register_plan_chunk_scheduler` debe añadir un job con id
    `alert_deploy_lag_marker_stale` invocando `_alert_deploy_lag_marker_stale`.
    Sin esto, la función vive pero nunca corre."""
    from cron_tasks import register_plan_chunk_scheduler, _alert_deploy_lag_marker_stale

    fake_scheduler = MagicMock()
    # `get_job(id)` debe devolver None para que el bloque registre.
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    job_ids = [
        call.kwargs.get("id") for call in fake_scheduler.add_job.call_args_list
    ]
    assert "alert_deploy_lag_marker_stale" in job_ids, (
        f"`alert_deploy_lag_marker_stale` no registrado. Jobs vistos: {job_ids}"
    )

    # Verificar que el callable es la función correcta.
    deploy_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_deploy_lag_marker_stale"
    )
    assert deploy_call.args[0] is _alert_deploy_lag_marker_stale


def test_cron_uses_hours_interval():
    """El cron debe registrarse con trigger=interval y `hours=` (no minutes).
    Diario por default; subscription al evento 'deploy' es raro, no necesita
    polling agresivo."""
    from cron_tasks import register_plan_chunk_scheduler

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None

    register_plan_chunk_scheduler(fake_scheduler)

    deploy_call = next(
        c for c in fake_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "alert_deploy_lag_marker_stale"
    )
    assert deploy_call.args[1] == "interval"
    assert "hours" in deploy_call.kwargs
    assert deploy_call.kwargs["hours"] >= 1
