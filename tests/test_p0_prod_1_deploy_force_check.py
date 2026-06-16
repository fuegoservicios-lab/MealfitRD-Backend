"""[P0-PROD-1-DEPLOY · 2026-05-12] Mejoras al detector de deploy lag tras
el incidente del audit 2026-05-11 (binary en prod rezagado vs marker en HEAD;
`expected_last_known_pfix` actualizado a las 03:05 UTC pero alerts
`deploy_lag_drift_vs_expected` NUNCA dispararon porque el cron corría 1×/día).

Cambios:
    1. `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` default 24h → 1h (clamp [1, 24]).
    2. Endpoint admin `POST /api/system/admin/deploy-lag/check` invoca el
       detector inline para validación inmediata post-deploy sin esperar al
       cron.

Razón: la frecuencia anterior (24h) significaba que el operador podía
publicar el marker esperado en KV y esperar hasta un día completo para que
el detector emita la alert. En la práctica, el incidente del audit pasó
desapercibido durante toda esa ventana — los errores `is_guest` en logs
postgres se acumularon sin que nadie supiera que el deploy estaba rezagado.

Lo que este test enforza:
    A) `cron_tasks.py` registra el cron con default `1` (no `24`).
    B) Clamp explícito a `[1, 24]` para evitar que un operador setee
       `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS=0` (loop ridículo) o
       `=720` (medio mes, lo que reintroduce el bug).
    C) `routers/system.py` declara `POST /admin/deploy-lag/check`.
    D) El endpoint usa `_verify_admin_token(...)` (no público) antes de
       invocar el detector.
    E) El endpoint retorna `live_marker`, `expected_marker`, `drift` para
       feedback inmediato al operador.
    F) Anchor `P0-PROD-1-DEPLOY` permanece.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"
_SYSTEM_ROUTER = _BACKEND_ROOT / "routers" / "system.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_src() -> str:
    return _SYSTEM_ROUTER.read_text(encoding="utf-8")


def test_a_cron_default_interval_is_1h(cron_src: str):
    """`MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` default ahora 1, no 24."""
    pattern = re.compile(
        r'_env_int\(\s*[\'"]MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS[\'"]\s*,\s*(\d+)\s*\)'
    )
    matches = pattern.findall(cron_src)
    assert matches, (
        "P0-PROD-1-DEPLOY: lookup de "
        "`MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` desapareció. Restaurar."
    )
    for default_str in matches:
        default_v = int(default_str)
        assert default_v == 1, (
            f"P0-PROD-1-DEPLOY: default de "
            f"MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS es {default_v}, "
            f"esperado 1 (bajado desde 24 para feedback rápido)."
        )


def test_b_cron_clamps_interval_to_safe_range(cron_src: str):
    """El cron debe clampear el knob a [1, 24]. Una asignación tipo
    `_DEPLOY_LAG_INT_H = 24` (downgrade hard) o `_DEPLOY_LAG_INT_H = 1`
    (upgrade hard) tras un comparador debe estar presente."""
    # Aislar el bloque del cron de deploy-lag.
    block_match = re.search(
        r'_DEPLOY_LAG_INT_H\s*=\s*_env_int.*?'
        r'(?:_add_job_jittered|scheduler\.add_job)',
        cron_src,
        re.DOTALL,
    )
    assert block_match is not None, "Bloque de registro del cron no aislable."
    block = block_match.group(0)
    has_low_clamp = bool(re.search(r"_DEPLOY_LAG_INT_H\s*<\s*1", block))
    has_high_clamp = bool(re.search(r"_DEPLOY_LAG_INT_H\s*>\s*24", block))
    assert has_low_clamp and has_high_clamp, (
        "P0-PROD-1-DEPLOY: clamp ausente. El bloque debe contener "
        "`if _DEPLOY_LAG_INT_H < 1: _DEPLOY_LAG_INT_H = 1` y "
        "`if _DEPLOY_LAG_INT_H > 24: _DEPLOY_LAG_INT_H = 24` (o equivalente). "
        "Sin clamp un operador puede setear 0 (loop infinito) o 720 (medio mes)."
    )


def test_c_admin_endpoint_declared(system_src: str):
    """El endpoint `POST /admin/deploy-lag/check` está declarado en
    `routers/system.py`.
    """
    assert '@router.post("/admin/deploy-lag/check")' in system_src, (
        "P0-PROD-1-DEPLOY: endpoint `POST /admin/deploy-lag/check` ausente. "
        "Restaurarlo en `backend/routers/system.py`."
    )


def test_d_admin_endpoint_uses_verify_admin_token(system_src: str):
    """El handler `admin_force_deploy_lag_check` invoca `_verify_admin_token`
    ANTES de cualquier import del detector — gate de defense-in-depth."""
    # Aislar el handler.
    # [stale-parser fix 2026-06-16] P2-DEPLOY-LAG-AUTO-BUMP (2026-05-25)
    # añadió un body Pydantic `body: Optional[_DeployLagCheckBody] =
    # Body(default=None)` a la signature, que ahora abarca varias líneas y
    # contiene parens anidados (`Body(default=None)`). El regex previo usaba
    # `\([^)]*\)` que se detenía en el primer `)` interno y dejaba de matchear
    # el `):` de cierre real. Capturamos hasta el `:` que cierra la signature
    # (tolerando `def`/`async def` + parens anidados con [\s\S]) y luego el
    # cuerpo hasta el próximo `@router.`. El handler sigue siendo sync `def`
    # con `_verify_admin_token` como primera línea ejecutable.
    handler_match = re.search(
        r'@router\.post\("/admin/deploy-lag/check"\)\s*\n'
        r'(?:async\s+)?def\s+(\w+)\([\s\S]*?\)\s*:(.*?)(?=@router\.|\Z)',
        system_src,
        re.DOTALL,
    )
    assert handler_match is not None, (
        "P0-PROD-1-DEPLOY: handler del endpoint no encontrado."
    )
    body = handler_match.group(2)
    # `_verify_admin_token(` debe aparecer ANTES del primer `_alert_deploy_lag` call.
    verify_idx = body.find("_verify_admin_token(")
    detector_idx = body.find("_alert_deploy_lag_marker_stale(")
    assert verify_idx >= 0, (
        "P0-PROD-1-DEPLOY: `_verify_admin_token` no invocado. Endpoint "
        "admin SIN auth — cualquiera podría disparar el detector."
    )
    assert detector_idx >= 0, (
        "P0-PROD-1-DEPLOY: el endpoint no invoca el detector "
        "`_alert_deploy_lag_marker_stale`. Mejor delegado al cron, "
        "pero entonces este endpoint es no-op."
    )
    assert verify_idx < detector_idx, (
        "P0-PROD-1-DEPLOY: `_verify_admin_token` debe llamarse ANTES de "
        "invocar el detector. Sino, el detector ya hizo I/O antes del "
        "rechazo de auth."
    )


def test_e_endpoint_returns_drift_snapshot(system_src: str):
    """El endpoint retorna `live_marker`, `expected_marker`, `drift` en el
    JSON de respuesta — feedback inmediato sin segundo round-trip."""
    handler_match = re.search(
        r'@router\.post\("/admin/deploy-lag/check"\)(.*?)(?=@router\.|\Z)',
        system_src,
        re.DOTALL,
    )
    assert handler_match is not None
    body = handler_match.group(1)
    for key in ('"live_marker"', '"expected_marker"', '"drift"'):
        assert key in body, (
            f"P0-PROD-1-DEPLOY: response del endpoint no contiene la key "
            f"{key}. Snapshot incompleto."
        )


def test_f_anchor_present(cron_src: str, system_src: str):
    assert "P0-PROD-1-DEPLOY" in cron_src or "P0-PROD-1" in cron_src, (
        "P0-PROD-1-DEPLOY: anchor desapareció de cron_tasks.py."
    )
    assert "P0-PROD-1-DEPLOY" in system_src, (
        "P0-PROD-1-DEPLOY: anchor desapareció de routers/system.py."
    )
