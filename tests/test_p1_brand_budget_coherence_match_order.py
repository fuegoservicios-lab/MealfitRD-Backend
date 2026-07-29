"""[P1-BRAND-BUDGET-COHERENCE · 2026-07-29] Orden del SELECT de /api/supermarket/match.

Bug real hallado en el análisis brand↔budget: para ítems sin `package_grams`
(presentación no parseable, ~21% del catálogo) el frontend no puede aplicar su
propio sort por precio (`sizeFilteredVariants`/`stableSortedVariants` exigen
`targetG`) y consume el array crudo tal como llega del backend
(`g.variants.slice(0, MAX_VARIANTS_SHOWN)`, SupermarketBrands.jsx). Con
`ORDER BY ..., (brand IS NOT NULL), price_rd` el tiebreak de marca corría ANTES
que el precio: TODO producto "Genérico" (brand IS NULL) ordenaba antes que
CUALQUIER producto de marca, sin importar cuál era más barato — el "N opciones
· desde RD$X" del picker podía anunciar el genérico como más barato aunque una
marca costara menos.

Fix: `price_rd` pasa a ser el criterio de orden inmediatamente después del
nombre del alimento; `(brand IS NOT NULL)` queda como tiebreak de último orden
(solo desempata precios exactamente iguales).

Test parser-based (no requiere DB): ancla la posición relativa de las 3
claves de ORDER BY en el SELECT de `_match()`. Si alguien reordena el ORDER BY
sin revisar esta razón, el test falla ANTES de que el regreso llegue a prod.
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROUTER = BACKEND / "routers" / "supermarket.py"
SRC = ROUTER.read_text(encoding="utf-8")


def _match_body() -> str:
    m = re.search(r"def _match\(\).*?return \{", SRC, re.S)
    assert m, "no se encontró el cuerpo de _match() en routers/supermarket.py"
    return m.group(0)


def _order_by_clause(body: str) -> str:
    m = re.search(r"ORDER BY\s+(.+)", body)
    assert m, "no se encontró ORDER BY en el SELECT de _match()"
    # una sola línea de SQL: cortar en el primer salto de línea / triple-quote.
    return m.group(1).splitlines()[0]


def test_match_query_orders_by_price_before_brand_tiebreak():
    """price_rd debe aparecer ANTES que `(brand IS NOT NULL)` en el ORDER BY —
    si no, el genérico gana el orden de exhibición sin importar el precio real."""
    order_by = _order_by_clause(_match_body())
    price_pos = order_by.find("price_rd")
    brand_tiebreak_pos = order_by.find("brand IS NOT NULL")
    assert price_pos != -1, f"price_rd desapareció del ORDER BY: {order_by!r}"
    assert brand_tiebreak_pos != -1, f"(brand IS NOT NULL) desapareció del ORDER BY: {order_by!r}"
    assert price_pos < brand_tiebreak_pos, (
        "price_rd debe ordenar ANTES que el tiebreak de marca — con el orden "
        "invertido, todo producto Genérico se exhibe antes que todo producto "
        "de marca sin importar cuál es más barato (P1-BRAND-BUDGET-COHERENCE). "
        f"ORDER BY actual: {order_by!r}"
    )


def test_match_query_still_groups_by_food_name_first():
    """`lower(food_name)` sigue siendo el criterio PRIMARIO — el fix no debe
    mezclar variantes de alimentos distintos entre sí."""
    order_by = _order_by_clause(_match_body())
    food_pos = order_by.find("lower(food_name)")
    price_pos = order_by.find("price_rd")
    assert food_pos != -1, f"lower(food_name) desapareció del ORDER BY: {order_by!r}"
    assert food_pos < price_pos, (
        "lower(food_name) debe seguir siendo el criterio primario del ORDER BY "
        f"(antes que price_rd). ORDER BY actual: {order_by!r}"
    )


# [P1-BRAND-BUDGET-COHERENCE] Nota deliberada: NO se añaden tests "sintéticos"
# que ordenen filas de ejemplo con una réplica en Python de la fórmula SQL —
# ese patrón prueba la réplica, no el SQL real, y pasaría igual con el bug
# original todavía presente en routers/supermarket.py (verificado manualmente
# durante el desarrollo de este test: la réplica sigue "correcta" aunque el
# SELECT real esté saboteado). Los dos tests de arriba, que sí parsean el
# ORDER BY real, son los que efectivamente protegen contra la regresión.
