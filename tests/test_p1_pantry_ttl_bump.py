"""[P1-PANTRY-TTL-BUMP · 2026-05-20] Test anti-regresión del TTL del cache
de inventory en `pantryCache.js`.

Bug observado:
    `_INVENTORY_TTL_MS = 30 * 1000` (30s) era demasiado corto. Si el
    user pasaba >30s en otra tab (chat, plan, etc.) y volvía a Nevera,
    el cache expiraba → fetch fresh a Supabase ~500-1500ms con spinner
    visible. Reportado 2026-05-20: "el apartado de Nevera cada cierto
    tiempo dura un poquito más de lo normal".

Fix:
    TTL aumentado a 10 min. El realtime channel `pantry-realtime`
    suscrito en `Pantry.jsx` empuja UPDATE/INSERT/DELETE al state
    sin importar el TTL. Mutaciones del propio user via UI llaman
    `invalidateInventoryCache()` explícito. El cache es buffer para
    el primer paint, no la única fuente de truth.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_CACHE_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "pantryCache.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_inventory_ttl_at_least_5_minutes():
    """[P1-PANTRY-TTL-BUMP] El TTL del inventory cache debe ser >= 5 min
    (300_000ms). Pre-fix era 30s que disparaba spinner cada vez que el
    user navegaba entre tabs por más de 30s."""
    src = _read(_PANTRY_CACHE_JS)
    # Buscar la constante _INVENTORY_TTL_MS.
    match = re.search(
        r"_INVENTORY_TTL_MS\s*=\s*(.+?);",
        src,
    )
    assert match, "_INVENTORY_TTL_MS no encontrada en pantryCache.js"
    expr = match.group(1).strip()
    # Evaluar la expresión (números + operadores básicos).
    # Acepta formatos: `300 * 1000`, `5 * 60 * 1000`, `300000`, etc.
    try:
        # Strip cualquier comentario inline.
        expr_clean = re.sub(r"//.*$", "", expr).strip()
        ttl_ms = eval(expr_clean, {"__builtins__": {}}, {})
    except Exception as e:
        raise AssertionError(f"No pude evaluar _INVENTORY_TTL_MS expression `{expr}`: {e}")

    MIN_TTL = 5 * 60 * 1000  # 5 min
    assert ttl_ms >= MIN_TTL, (
        f"_INVENTORY_TTL_MS = {ttl_ms}ms ({ttl_ms / 1000}s) < {MIN_TTL}ms "
        f"({MIN_TTL / 60000} min). Pre-fix era 30s causando spinner visible "
        f"cuando el user navegaba >30s. Subir a >=5min (idealmente 10min). "
        f"Ver P1-PANTRY-TTL-BUMP · 2026-05-20."
    )
