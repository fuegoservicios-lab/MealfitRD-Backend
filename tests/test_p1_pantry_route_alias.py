"""[P1-PANTRY-ROUTE-ALIAS · 2026-07-11] Ruta canónica de la Nevera + aliases de redirect.

Bug vivo (screenshot del owner, 09:59): el desvío del modo constructor navegaba a
'/pantry' → catch-all 404 "Esta página no existe". La ruta REAL de la Nevera es
'/dashboard/pantry'; además el deep-link de pushes del backend usaba '/mi-nevera'
(ruta que NUNCA existió) y el default del canal de nudges usaba '/pantry'. Clase
entera de deep-links rotos: nadie tenía la ruta canónica.

Contrato:
1. App.jsx registra aliases `/pantry` y `/mi-nevera` → <Navigate to="/dashboard/pantry">
   (cubre pushes YA entregadas con URL rota y cualquier callsite futuro que se equivoque).
2. Blanket frontend: ningún navigate()/Link apunta a '/pantry' a secas.
3. Backend: `_dispatch_pantry_nudge` default url y `CHUNK_STALE_PANTRY_DEEPLINK` default
   apuntan a '/dashboard/pantry'.

tooltip-anchor: P1-PANTRY-ROUTE-ALIAS
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_FRONT = _BACKEND.parent / "frontend" / "src"

_APP_SRC = (_FRONT / "App.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Aliases de redirect en el router
# ---------------------------------------------------------------------------

def test_router_has_pantry_aliases():
    for alias in ("/pantry", "/mi-nevera"):
        m = re.search(
            rf'<Route path="{re.escape(alias)}" element={{<Navigate to="/dashboard/pantry" replace />}} />',
            _APP_SRC,
        )
        assert m, (
            f"alias {alias} → /dashboard/pantry desapareció del router — pushes viejas "
            "y deep-links históricos vuelven al catch-all 404"
        )


def test_canonical_route_still_exists():
    assert '<Route path="/dashboard/pantry"' in _APP_SRC, (
        "la ruta canónica /dashboard/pantry desapareció — si se renombra, actualizar "
        "aliases + _dispatch_pantry_nudge + CHUNK_STALE_PANTRY_DEEPLINK en el mismo commit"
    )


# ---------------------------------------------------------------------------
# 2. Blanket frontend: nadie navega a '/pantry' a secas
# ---------------------------------------------------------------------------

def test_no_frontend_navigation_to_bare_pantry():
    bad = []
    for f in _FRONT.rglob("*.jsx"):
        src = f.read_text(encoding="utf-8")
        for pat in (r"navigate\(\s*['\"]/pantry['\"]", r'to=["\']\/pantry["\']'):
            if re.search(pat, src):
                bad.append(f"{f.name}: {pat}")
    for f in _FRONT.rglob("*.js"):
        src = f.read_text(encoding="utf-8")
        if re.search(r"navigate\(\s*['\"]/pantry['\"]", src):
            bad.append(f.name)
    assert not bad, (
        f"navegación a '/pantry' a secas (404 sin el alias; usar /dashboard/pantry): {bad}"
    )


# ---------------------------------------------------------------------------
# 3. Backend: defaults canónicos
# ---------------------------------------------------------------------------

def test_nudge_default_url_is_canonical():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    m = re.search(r"def _dispatch_pantry_nudge\([^)]*url: str = \"([^\"]+)\"", src)
    assert m, "firma de _dispatch_pantry_nudge cambió — re-anclar"
    assert m.group(1) == "/dashboard/pantry", (
        f"default del nudge = {m.group(1)!r} — la push abriría 404 al tocarla"
    )


def test_stale_pantry_deeplink_default_is_canonical():
    import importlib
    import constants
    importlib.reload(constants)
    assert constants.CHUNK_STALE_PANTRY_DEEPLINK == "/dashboard/pantry" or \
        not constants.CHUNK_STALE_PANTRY_DEEPLINK.startswith("/mi-nevera"), (
        f"CHUNK_STALE_PANTRY_DEEPLINK={constants.CHUNK_STALE_PANTRY_DEEPLINK!r}: "
        "'/mi-nevera' nunca existió en el SPA"
    )
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert 'os.environ.get("CHUNK_STALE_PANTRY_DEEPLINK", "/dashboard/pantry")' in src


def test_marker_anchored_in_source():
    app_jsx = _APP_SRC
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    consts = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert app_jsx.count("P1-PANTRY-ROUTE-ALIAS") >= 1
    assert cron.count("P1-PANTRY-ROUTE-ALIAS") >= 1
    assert consts.count("P1-PANTRY-ROUTE-ALIAS") >= 1
