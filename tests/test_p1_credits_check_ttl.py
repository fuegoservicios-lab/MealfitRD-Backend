"""[P1-CREDITS-CHECK-TTL · 2026-05-20] Test anti-regresión del TTL del
cache `validateCreditsAsync` en Dashboard.jsx.

Bug observado:
    Cada click del botón "Actualizar platos" disparaba un fetch a
    `/api/user/credits/<id>` (~200-500ms) antes de abrir el modal. El
    cache de 5s era demasiado corto: cualquier interacción tras 5s
    pagaba el delay otra vez. Reportado 2026-05-20: "el botón de
    actualizar platos tiene delay cuando lo presiono".

Fix:
    - TTL cache `validateCreditsAsync` subido `5s → 120s`. El `planCount`
      solo cambia al regenerar plan (mutación que invalida cache
      explícito) o month rollover (1/mes). 120s captura clicks rápidos
      sin perder correctness.
    - Fast path: si `userPlanLimit` es ilimitado (∞), retornar true
      sin fetch.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_credits_ttl_at_least_60s():
    """[P1-CREDITS-CHECK-TTL] El TTL del gate de créditos del Dashboard debe ser >=60s.
    Pre-fix era 5s, perceptible como delay en cada click.

    [reapuntado 2026-07-27] La constante `_CACHE_TTL_MS` murió el 2026-07-09 (P2-3): el cache
    migró a TanStack (`utils/quotaCache.js::getFreshPlanCount`) y el TTL se expresa por callsite
    vía `ttlMs`. La INVARIANTE es la misma — el Dashboard no debe re-fetchear en cada click — y
    hoy vive en el `ttlMs` que el Dashboard pasa (120s). Este test estuvo rojo desde la migración
    sin proteger nada: la constante que exigía ya no podía existir.
    """
    src = _read(_DASHBOARD_JSX)
    call = re.search(
        r"getFreshPlanCount\(\s*[^)]*?\{\s*ttlMs\s*:\s*([0-9*\s]+?)\s*\}",
        src,
        re.DOTALL,
    )
    assert call, (
        "El Dashboard debe invocar getFreshPlanCount con `ttlMs` explícito — sin él cae al "
        "default de 5s de quotaCache.js y vuelve el delay por click pre-fix."
    )
    # Sin eval: la clase de captura del regex es [0-9*\s], así que la expresión solo puede ser
    # un producto de enteros ("120 * 1000"). Se multiplica a mano.
    try:
        ttl_ms = 1
        for _factor in call.group(1).split("*"):
            ttl_ms *= int(_factor.strip())
    except Exception:
        ttl_ms = 0
    assert ttl_ms >= 60 * 1000, (
        f"ttlMs del Dashboard = {ttl_ms}ms < 60s. El gate de créditos del Dashboard tolera "
        f"staleness (el quota real se enforza server-side con 402); re-fetchear por click solo "
        f"añade latencia. Ver P1-CREDITS-CHECK-TTL · 2026-05-20 y P2-3 · 2026-07-09."
    )


def test_credits_fast_path_for_unlimited():
    """[P1-CREDITS-CHECK-TTL] Si `userPlanLimit` es ilimitado (∞/Ilimitado/
    no-number), la función debe retornar true sin fetch."""
    src = _read(_DASHBOARD_JSX)
    fn_match = re.search(
        r"const\s+validateCreditsAsync\s*=\s*async\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*;",
        src,
        re.DOTALL,
    )
    assert fn_match
    body = fn_match.group(1)
    # Anchor: check de userPlanLimit por ilimitado antes del fetch.
    has_fast_path = bool(
        re.search(
            r"userPlanLimit\s*===?\s*['\"]∞['\"]"
            r"|userPlanLimit\s*===?\s*['\"]Ilimitado['\"]"
            r"|typeof\s+userPlanLimit\s*!==\s*['\"]number['\"]",
            body,
        )
    )
    assert has_fast_path, (
        "Fast path para usuarios ilimitados ausente. Sin esto, plan ULTRA/admin "
        "paga fetch innecesario en cada click. Ver P1-CREDITS-CHECK-TTL."
    )
