"""[P3-PROFILE-METRICS-QUOTA-GATE · 2026-05-20] Test anti-regresión del
pre-check de quota en `handleUpdatePlanWithMetrics` (Settings.jsx).

Bug pre-fix:
    El flow guardaba body metrics (RPC `update_health_profile_merge`) y
    DESPUÉS invocaba `regeneratePlan`, que internamente verifica quota.
    Si quota=0, el regenerate abortaba con toast "Límite alcanzado" PERO
    el health_profile ya quedaba mutado con los nuevos valores → user
    se quedaba hasta el próximo ciclo de billing con weight/height nuevos
    en su perfil + plan vigente con macros stale.

Fix:
    Pre-check del quota ANTES de la persistencia. Si `_freshCount >=
    userPlanLimit`, abortar con toast claro sin mutar nada. Reusa el
    cache `window.__cachedQuota` (TTL 5s) que `useRegeneratePlan`
    también consulta — cero roundtrip duplicado si Dashboard ya
    consultó hace <5s.

Fail-open en network error: si `checkPlanLimit` lanza (red caída),
dejar pasar al flow normal — `regeneratePlan` hará su propio check
downstream y abortará si efectivamente está al tope. Peor caso = igual
que el comportamiento pre-fix (state inconsistente), no peor.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_handler_body(src: str) -> str:
    """Extrae el cuerpo de handleUpdatePlanWithMetrics — el flow específico
    que requiere el gate. Esto evita falsos positivos si el patrón aparece
    en otro handler (e.g. handleSaveProfile que solo guarda nombre)."""
    start = src.find("const handleUpdatePlanWithMetrics")
    assert start != -1, (
        "Handler `handleUpdatePlanWithMetrics` no encontrado en Settings.jsx — "
        "renombre rompió el contrato de P3-PROFILE-METRICS-QUOTA-GATE."
    )
    # Buscar el cierre del arrow function (rough): primer `};` en columna 4
    # tras el start. Es heurística pero suficiente para este test parser-based.
    body_start = src.index("=>", start)
    # Contar braces para encontrar el cierre.
    depth = 0
    i = body_start
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[body_start:i + 1]
        i += 1
    raise AssertionError("No se pudo extraer body de handleUpdatePlanWithMetrics")


def test_quota_gate_appears_in_handler():
    """[P3-PROFILE-METRICS-QUOTA-GATE] El handler invoca `checkPlanLimit`
    antes de persistir. Tooltip-anchor del marker debe estar presente
    para que un renombre futuro falle el test antes de remover el gate."""
    src = _read(_SETTINGS_JSX)
    body = _extract_handler_body(src)
    assert "P3-PROFILE-METRICS-QUOTA-GATE" in body, (
        "Marker `P3-PROFILE-METRICS-QUOTA-GATE` ausente del handler — "
        "tooltip-anchor removido. Restaurar para que renombres rompan este test."
    )
    assert "checkPlanLimit" in body, (
        "`checkPlanLimit` no se invoca en handleUpdatePlanWithMetrics — el gate "
        "fue removido. Restaurar el pre-check antes de safeUpdateHealthProfile."
    )


def test_quota_gate_runs_before_persistence():
    """[P3-PROFILE-METRICS-QUOTA-GATE] El check de quota DEBE aparecer
    antes de `safeUpdateHealthProfile` en el flujo lineal del handler. Si
    el gate corre DESPUÉS del persist, el bug original vuelve."""
    src = _read(_SETTINGS_JSX)
    body = _extract_handler_body(src)
    idx_check = body.find("checkPlanLimit")
    idx_persist = body.find("safeUpdateHealthProfile")
    assert idx_check != -1, "checkPlanLimit no encontrado en handler."
    assert idx_persist != -1, "safeUpdateHealthProfile no encontrado en handler."
    assert idx_check < idx_persist, (
        "`checkPlanLimit` aparece DESPUÉS de `safeUpdateHealthProfile` en "
        "handleUpdatePlanWithMetrics. El gate debe correr ANTES para no dejar "
        "health_profile mutado cuando el quota está agotado. Ver "
        "P3-PROFILE-METRICS-QUOTA-GATE · 2026-05-20."
    )


def test_quota_gate_returns_when_limit_reached():
    """[P3-PROFILE-METRICS-QUOTA-GATE] Si el gate detecta quota >= limit,
    DEBE retornar antes de mutar (early return). Verificamos que existe
    un `return;` dentro del bloque del gate (entre el check `>=
    userPlanLimit` y el setIsRegeneratingFromMetrics)."""
    src = _read(_SETTINGS_JSX)
    body = _extract_handler_body(src)
    # Buscar el bloque del gate: desde `_freshCount` o `>= userPlanLimit`
    # hasta `setIsRegeneratingFromMetrics(true)`.
    gate_start = body.find(">= userPlanLimit")
    persist_start = body.find("setIsRegeneratingFromMetrics(true)")
    assert gate_start != -1, "Patrón `>= userPlanLimit` no encontrado en gate."
    assert persist_start != -1, "`setIsRegeneratingFromMetrics(true)` ausente."
    assert gate_start < persist_start, (
        "El check `>= userPlanLimit` aparece después de "
        "`setIsRegeneratingFromMetrics(true)` — gate mal ubicado."
    )
    gate_block = body[gate_start:persist_start]
    assert "return" in gate_block, (
        "El gate no tiene `return` early-exit entre el check y la persistencia. "
        "Sin return, el flow continúa y muta health_profile pese al límite."
    )


def test_quota_gate_uses_shared_cache_pattern():
    """[P3-PROFILE-METRICS-QUOTA-GATE] El gate reusa el cache
    `window.__cachedQuota` (TTL 5s) compartido con
    `useRegeneratePlan.regeneratePlan` para evitar duplicar el roundtrip.
    Si alguien refactoriza el gate sin reusar la cache, este test alerta."""
    src = _read(_SETTINGS_JSX)
    body = _extract_handler_body(src)
    assert "__cachedQuota" in body, (
        "Gate no usa `window.__cachedQuota`. Refactor rompió el cache "
        "compartido — un click en 'Actualizar Plan' ahora hace 2 roundtrips "
        "(uno acá, uno en regeneratePlan). Ver patrón en useRegeneratePlan.js."
    )
    assert "__lastQuotaCheckTime" in body, (
        "Gate no actualiza `window.__lastQuotaCheckTime` — el cache nunca "
        "se invalida desde acá y queda desincronizado con regeneratePlan."
    )


def test_quota_gate_failopen_in_network_error():
    """[P3-PROFILE-METRICS-QUOTA-GATE] El gate envuelve `checkPlanLimit`
    en try/catch para fail-open en network error. Sin el try, una caída
    de red bloquea TODA edición de body metrics aunque el quota esté ok."""
    src = _read(_SETTINGS_JSX)
    body = _extract_handler_body(src)
    # Heurística: el primer `try` después del comentario del gate y antes
    # de `setIsRegeneratingFromMetrics(true)` envuelve checkPlanLimit.
    marker_idx = body.find("P3-PROFILE-METRICS-QUOTA-GATE")
    assert marker_idx != -1
    persist_idx = body.find("setIsRegeneratingFromMetrics(true)")
    gate_window = body[marker_idx:persist_idx]
    assert "try {" in gate_window, (
        "Gate sin `try {` — `checkPlanLimit` puede lanzar y matar el handler "
        "antes del check (fail-closed indeseado en network error)."
    )
    assert "catch" in gate_window, (
        "Gate sin `catch` — fail-open no implementado. Restaurar try/catch."
    )


def test_last_known_pfix_bumped():
    """[P3-PROFILE-METRICS-QUOTA-GATE] El marker debe estar bumped en
    backend/app.py para que `/health/version` exponga el nuevo slug y
    el cron `_alert_deploy_lag_marker_stale` detecte el deploy. Sin
    bump, un operador no puede confirmar que el fix está vivo en prod.

    `_LAST_KNOWN_PFIX` es un marker GLOBAL único: cada P-fix posterior lo
    re-bumpea (ver test_p3_1_last_known_pfix_freshness). Tras P3-PROFILE-
    METRICS-QUOTA-GATE (2026-05-20) varios P-fixes más avanzaron el marker
    (hoy `P2-TRIAGE-REALBUGS · 2026-06-16`). Asertar el slug literal de
    este fix iría stale por diseño; lo invariante es que el marker nunca
    regrese antes de la fecha en que este fix aterrizó."""
    from datetime import datetime

    app_py = _REPO_ROOT / "backend" / "app.py"
    src = app_py.read_text(encoding="utf-8")
    match = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert match, "_LAST_KNOWN_PFIX no encontrado en backend/app.py"
    marker = match.group(1)
    m = re.match(r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+(\d{4}-\d{2}-\d{2})$", marker)
    assert m is not None, (
        f"_LAST_KNOWN_PFIX={marker!r} no sigue el formato `Pn-X · YYYY-MM-DD`."
    )
    marker_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    assert marker_date >= datetime(2026, 5, 20).date(), (
        f"_LAST_KNOWN_PFIX={marker!r} tiene fecha anterior a la de este "
        f"P-fix (2026-05-20). El marker regresó — investigar."
    )
