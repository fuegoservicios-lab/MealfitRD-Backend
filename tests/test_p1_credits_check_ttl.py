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
    """[P1-CREDITS-CHECK-TTL] TTL debe ser >=60s. Pre-fix era 5s,
    perceptible como delay en cada click."""
    src = _read(_DASHBOARD_JSX)
    # Buscar la constante o literal del TTL dentro de validateCreditsAsync.
    fn_match = re.search(
        r"const\s+validateCreditsAsync\s*=\s*async\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*;",
        src,
        re.DOTALL,
    )
    assert fn_match, "validateCreditsAsync no encontrada"
    body = fn_match.group(1)
    # Buscar _CACHE_TTL_MS o el literal numérico.
    ttl_match = re.search(
        r"(_CACHE_TTL_MS|__lastQuotaCheckTime\s*\|\|\s*0\)\s*>\s*)(\d+)",
        body,
    )
    # Mejor: extraer cualquier `const _CACHE_TTL_MS = N * 1000;`
    const_match = re.search(r"_CACHE_TTL_MS\s*=\s*(.+?);", body)
    assert const_match, (
        "Constante `_CACHE_TTL_MS` ausente — TTL puede estar hardcoded. "
        "Usar constante named para claridad. Ver P1-CREDITS-CHECK-TTL."
    )
    expr = const_match.group(1).strip()
    try:
        ttl_ms = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        ttl_ms = 0
    assert ttl_ms >= 60 * 1000, (
        f"_CACHE_TTL_MS = {ttl_ms}ms < 60s. Pre-fix era 5000ms causando "
        f"delay perceptible en cada click. Subir a >=60s. Ver "
        f"P1-CREDITS-CHECK-TTL · 2026-05-20."
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
