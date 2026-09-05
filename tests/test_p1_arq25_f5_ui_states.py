"""[P1-ARQ25-F5-UI-STATES · 2026-09-05] Fase 5: los estados de la proyección de compras (none/pending/
ready/failed/stale) llegan al Dashboard en una línea discreta que lee `GET /api/plans/{plan_id}/projections`
y sondea solo mientras está `pending`. Ancla parser-based del cableado frontend.
"""
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()


def _frontend(*parts):
    for base in (_HERE.parents[2], _HERE.parents[1].parent):
        p = base.joinpath("frontend", "src", *parts)
        if p.exists():
            return p.read_text(encoding="utf-8").replace(chr(13), "")
    pytest.skip("frontend hermano no disponible")


def test_a_componente_lee_el_endpoint_y_sondea_solo_pending():
    src = _frontend("components", "dashboard", "ShoppingProjectionStatus.jsx")
    assert "fetchWithAuth(`/api/plans/${encodeURIComponent(planId)}/projections`)" in src
    # [P2-PROJECTION-LINE-NO-FLICKER · 2026-09-05] El `j &&` de esta línea se mudó ARRIBA, a un
    # `if (!j || typeof j !== 'object') return;`: la línea conserva el último estado conocido cuando la
    # respuesta viene rara, en vez de parpadear. La guarda es más fuerte que antes (rechaza también
    # respuestas que no son objeto), y lo que este test vigila —que solo se sondee en `pending` y bajo el
    # tope— sigue intacto, así que se ancla eso y no el `j &&` que ya vive en otro sitio.
    assert "if (!j || typeof j !== 'object') return;" in src, "respuesta rara: conservar lo último conocido"
    assert "if (j.status === 'pending' && pollsRef.current < POLL_MAX) {" in src
    assert "document.visibilityState === 'hidden'" in src, "oculta no sondea (P2-PLAN-POLL-HIDDEN-NO-CLOCK)"
    # la línea la decide un helper puro fuera del .jsx (react-refresh solo quiere componentes ahí)
    line = _frontend("utils", "projectionLine.js")
    for status in ("'ready'", "'stale'", "'pending'", "'failed'"):
        assert status in line
    assert "formatNumber" in line and "Intl.NumberFormat('es-DO'" not in line, "el formateador lee el idioma activo (i18n:check)"


def test_b_dashboard_lo_monta_bajo_las_acciones_de_la_lista():
    src = _frontend("pages", "Dashboard.jsx")
    assert "import ShoppingProjectionStatus from '../components/dashboard/ShoppingProjectionStatus';" in src
    i = src.index("<ShoppingProjectionStatus")
    block = src[i:i + 400]
    assert "planId={planData?.id}" in block
    assert "enabled={!isGuest && !isPlanCorrupted && !isPlanExpired && !planFinished}" in block


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_c_copias_traducidas(locale):
    import json
    cat = json.loads(_frontend("i18n", "locales", f"{locale}.json"))
    for k in ("Calculando la proyección de tu ciclo de compras…", "No se pudo calcular la proyección de compras.",
              "Proyección del ciclo de {dias} días: {items} artículos · ≈{costo}"):
        assert cat.get(k), (locale, k)
