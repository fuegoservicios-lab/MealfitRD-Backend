"""[P2-PRICING-CTA-NO-CHECKOUT · 2026-08-16] `/upgrade` no existía: era un 404 con el layout viejo.

Reporte del dueño (screenshot, bioboros.com/precios): «cuando le doy para elegir un
plan me redirige a una página de un diseño que ya ni existe». Los tres botones de pago
del landing estático apuntaban a `https://app.bioboros.com/upgrade`, y esa ruta NUNCA
existió en el router: la real es `/dashboard/upgrade` (P3-UPGRADE-PAGE). El path caía
al catch-all `*` → `<Layout><NotFound /></Layout>`, es decir el Header/Footer de
marketing que el apex ya no usa — de ahí «un diseño que ya ni existe».

Dos mitades, y solo UNA es testeable desde aquí:

1. El landing estático (`Software/bioboros-cinematic/content/precios.html`) vive FUERA
   de este repo y ya no ofrece checkout: sus CTA de pago apuntan a la raíz del app y
   el plan se elige dentro de la cuenta. Sin fichero en el árbol no hay nada que
   anclar — no se finge una aserción sobre un fichero que este test no puede leer.
2. El alias `/upgrade` → `/dashboard/upgrade` en el router. Esto es lo que ancla este
   test: enlaces ya compartidos, marcadores y cualquier callsite futuro que se
   equivoque de path aterrizan donde SÍ se paga, en vez del 404. Mismo patrón que
   P1-PANTRY-ROUTE-ALIAS y P2-LANDING-MANIFEST-SHORTCUT.

tooltip-anchor: P2-PRICING-CTA-NO-CHECKOUT
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_FRONT = _BACKEND.parent / "frontend" / "src"

_APP_SRC = (_FRONT / "App.jsx").read_text(encoding="utf-8")


def test_router_has_upgrade_alias():
    """`/upgrade` redirige a la página real de planes, no al catch-all."""
    m = re.search(
        r'<Route\s+path="/upgrade"\s+element=\{<Navigate\s+to="/dashboard/upgrade"\s+replace\s*/>\}\s*/>',
        _APP_SRC,
    )
    assert m, (
        "el alias /upgrade → /dashboard/upgrade desapareció del router: los enlaces "
        "ya compartidos del landing y cualquier marcador vuelven al 404 con el layout viejo"
    )


def test_canonical_upgrade_route_still_exists():
    """Si alguien renombra la ruta real, el alias apuntaría al vacío."""
    assert '<Route path="/dashboard/upgrade"' in _APP_SRC, (
        "la ruta canónica /dashboard/upgrade desapareció — al renombrarla hay que "
        "actualizar el alias /upgrade en el mismo commit"
    )


def test_alias_precedes_catch_all():
    """El alias debe estar declarado ANTES del wildcard, o nunca matchea."""
    alias = _APP_SRC.index('<Route path="/upgrade"')
    catch_all = _APP_SRC.index('<Route path="*"')
    assert alias < catch_all, (
        "el alias /upgrade quedó después del catch-all '*' — react-router resuelve por "
        "ranking, pero declararlo después es una señal de que el bloque se movió sin revisar"
    )


def test_marker_anchored_in_source():
    assert _APP_SRC.count("P2-PRICING-CTA-NO-CHECKOUT") >= 1, (
        "el marcador desapareció de App.jsx — sin él, el próximo que lea el alias no "
        "sabe que existe por un 404 reportado en producción"
    )
