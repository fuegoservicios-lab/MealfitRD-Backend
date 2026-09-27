# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-469 · 2026-09-27] ¿La otra lista ya tiene este alimento? — por ALIMENTO del catálogo, no por palabra.

Los dos reconciliadores display↔raw (`_reconcile_display_missing_in_raw` y su espejo
`_reconcile_raw_missing_in_display`) deciden «falta» por el PRIMER token de ≥4 letras de la línea:

- el raw de la IA «155 g de carne de Filete de pescado blanco cocida» buscaba «carne» en un display que decía
  «1 filete de pescado» → la línea se AÑADÍA y el plato llevaba el pescado dos veces (≈305 g);
- el display «2 tortas pequeñas de casabe» buscaba «tortas» en un raw con «40 g de casabe» → casabe doble.

En la compra única cada copia se sustituía después por un duradero DISTINTO (la rueda evita repetir en el día) y el
paso quedaba «escurre las sardinas de atún», «los garbanzos de sardinas» (batería real del 27-sep, días 8-9 de 30).
El resolvedor del catálogo (`_resolve_line_food_grams`, el mismo que usa `_reconcile_display_raw_lines`) sabe que
son el mismo alimento. Aquí sólo se CONSULTA cuando la heurística de siempre ya dijo «falta»: donde el catálogo no
resuelve, la heurística sigue decidiendo como antes. tooltip-anchor: P1-PLAN-LOTE-469-RECONCILIA-POR-ALIMENTO
"""
from __future__ import annotations

import sys


def _alimento(linea):
    """El alimento del catálogo de la línea, o None (grafo no cargado, línea sin resolver o error)."""
    go = sys.modules.get("graph_orchestrator")
    if go is None or not hasattr(go, "_resolve_line_food_grams"):
        return None
    try:
        return go._resolve_line_food_grams(str(linea))[0] or None
    except Exception:
        return None


def ya_esta(linea, otras) -> bool:
    """True si alguna línea de `otras` resuelve al MISMO alimento del catálogo que `linea`. Sin resolución → False."""
    f = _alimento(linea)
    if not f:
        return False
    for o in otras or ():
        if isinstance(o, str) and o.strip() and _alimento(o) == f:
            return True
    return False
