"""[P3-2 + P3-5 · 2026-05-10] Regression guard: `/health/version` extendido
y nuevo endpoint `/admin/knobs`.

Bug que esto cubre:
    Operador necesita responder "¿está vivo lo último?" en 1 segundo sin
    abrir shell ni Supabase logs. P1-A (2026-05-08) instaló el endpoint
    base; P3-2 lo extiende con 3 señales operacionales: process_uptime_s,
    knobs_diff (env vars activas), cron_missed_1h_total. P3-5 expone el
    registry completo de knobs vía `/admin/knobs` para diagnóstico.

Cobertura (parser-estático sobre `app.py`):
    1. `/health/version` declarado y mantiene los 6 campos pre-existentes.
    2. Nuevos campos P3-2 presentes: `process_uptime_s`, `knobs_diff`,
       `cron_missed_1h_total`.
    3. `process_uptime_s` se calcula desde `_PROCESS_START_ISO` (no
       hardcoded ni 0).
    4. `knobs_diff` filtra solo knobs con `value != default`.
    5. `/admin/knobs` declarado y devuelve `{count, knobs}`.
    6. Ambos endpoints son GET sin auth dependency.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"


def _read_app() -> str:
    if not _APP_PY.exists():
        pytest.skip(f"app.py no encontrado: {_APP_PY}")
    return _APP_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. /health/version sigue declarado con todos sus campos
# ---------------------------------------------------------------------------
def test_health_version_endpoint_declared():
    src = _read_app()
    assert '@app.get("/health/version")' in src or "@app.get('/health/version')" in src
    assert re.search(r"def\s+health_version\s*\(", src)


def test_health_version_preserves_pre_existing_fields():
    """Los 6 campos base de P1-A no fueron removidos."""
    src = _read_app()
    for field in [
        '"git_sha"',
        '"git_short_sha"',
        '"deploy_timestamp"',
        '"process_started_at"',
        '"last_known_pfix"',
        '"knobs_count"',
    ]:
        assert field in src, (
            f"Campo {field} fue removido del response de /health/version. "
            f"Si es intencional, actualizar este test y la memoria de cierre."
        )


# ---------------------------------------------------------------------------
# 2. Nuevos campos P3-2
# ---------------------------------------------------------------------------
def test_health_version_exposes_process_uptime_s():
    src = _read_app()
    assert '"process_uptime_s"' in src, (
        "P3-2: /health/version debe exponer `process_uptime_s` para diagnóstico "
        "de restart frecuente."
    )


def test_health_version_exposes_knobs_overrides_count_not_values():
    """[P2-HEALTH-KNOBS-COUNT · 2026-05-28] El endpoint público ya NO serializa
    `knobs_diff` (nombres+valores de las defensas tuneadas) — solo el CONTEO de
    overrides. El detalle completo sigue gateado en `/admin/knobs`."""
    src = _read_app()
    assert '"knobs_overrides_count"' in src
    # El dict con valores ya NO se expone como key del response público.
    assert '"knobs_diff":' not in src


def test_health_version_exposes_cron_missed_1h_total():
    src = _read_app()
    assert '"cron_missed_1h_total"' in src


def test_process_uptime_computed_from_start_iso():
    """`_PROCESS_START_ISO` debe usarse para calcular el uptime, no hardcoded."""
    src = _read_app()
    # Anchor mínimo: la ruta `health_version` debe referenciar `_PROCESS_START_ISO`
    # en su body. Buscamos los dos en el mismo bloque (~50 líneas tras `def health_version`).
    m = re.search(r"def\s+health_version\s*\([^)]*\):\s*\n(?:.*\n){0,150}", src)
    assert m is not None
    body = m.group(0)
    assert "_PROCESS_START_ISO" in body, (
        "process_uptime_s debe derivarse del marker _PROCESS_START_ISO, "
        "no de un hardcoded."
    )


# ---------------------------------------------------------------------------
# 3. /admin/knobs (P3-5)
# ---------------------------------------------------------------------------
def test_admin_knobs_endpoint_declared():
    src = _read_app()
    assert '@app.get("/admin/knobs")' in src or "@app.get('/admin/knobs')" in src
    assert re.search(r"def\s+admin_knobs\s*\(", src)


def test_admin_knobs_returns_count_and_full_registry():
    """Response shape: `{count, knobs}` donde knobs es el snapshot completo."""
    src = _read_app()
    # Anchor: dentro de la función admin_knobs, debe haber `count` y `knobs` keys.
    m = re.search(r"def\s+admin_knobs\s*\(.*?\):\s*\n(?:.*\n){0,40}", src, re.DOTALL)
    assert m is not None
    body = m.group(0)
    assert '"count"' in body
    assert '"knobs"' in body
    assert "get_knobs_registry_snapshot" in body, (
        "admin_knobs debe usar `get_knobs_registry_snapshot()` (SSOT del registry)."
    )


# ---------------------------------------------------------------------------
# 4. Sin auth dependency (info de diagnóstico no sensible)
# ---------------------------------------------------------------------------
def test_health_and_admin_endpoints_have_no_auth_dep():
    """Ninguno de los dos endpoints debe requerir `verified_user_id` o `verify_api_quota`.

    Estos endpoints existen para diagnóstico operacional rápido — un operador
    necesita poder hacer `curl <prod>/health/version` desde cualquier sitio
    sin manejar tokens. Los valores expuestos NO son secretos."""
    src = _read_app()
    # Capturar el signature de cada función.
    for fn_name in ("health_version", "admin_knobs"):
        m = re.search(rf"def\s+{fn_name}\s*\(([^)]*)\):", src)
        assert m is not None, f"Función `{fn_name}` no encontrada."
        sig = m.group(1)
        for forbidden in ("verified_user_id", "verify_api_quota", "get_verified_user_id"):
            assert forbidden not in sig, (
                f"`{fn_name}` adquirió dependency `{forbidden}` — el endpoint "
                f"de diagnóstico debe ser auth-less por diseño."
            )
