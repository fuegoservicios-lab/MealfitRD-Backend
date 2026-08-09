"""[P1-PANTRY-LOW-IS-A-LIE · 2026-08-09] La Nevera acusaba «7 por reponer» con
las cantidades que la propia app te dijo que compraras.

El umbral era `LOW_THRESHOLD = 0.5` comparado contra `item.quantity` — un
número cuyo significado depende ENTERAMENTE de `item.unit`. FORENSE sobre las
63 filas reales del owner (2026-08-09):

  · Solo puede dispararse sobre unidades de PESO. Las 50 filas discretas
    (paquete, pote, unidad, lata, mazo, cabeza…) tienen cantidad mínima 1, así
    que el umbral de 0,5 es estructuralmente inalcanzable para ellas — incluidas
    44 filas con EXACTAMENTE 1 unidad, que sí se agotan al usarse.
  · De las 13 filas de peso marcaba 9, partiendo por la mitad el rango normal
    de compra (0,2167 .. 1,75 lb). Yuca 0,87 lb pasaba; Cebolla 0,5 lb no.
  · 6 de las 9 marcadas traían `source='shopping_list'`: eran literalmente las
    cantidades que la app indicó comprar.

Y el argumento que lo cierra: justo después de «exportar la lista de compras a
la nevera», la nevera contiene EXACTAMENTE lo que el plan pide. Por
construcción «por reponer» tiene que valer 0 en ese instante — cualquier número
ahí es falso por definición.

LA LECCIÓN: «poco» no es una propiedad de un número, es una RELACIÓN entre lo
que tienes y lo que necesitas. Ningún umbral constante puede expresarla, y
menos sobre una columna cuya unidad varía fila a fila.

QUÉ SE PUSO EN SU LUGAR: la única señal que la Nevera puede sostener con los
datos que ya tiene — la caducidad (`utils/shelfLife.js`), que compara
`created_at + shelf_life_days` contra hoy y, sin datos, NO dice nada. El estado
de atención de la fila y el contador del sidebar cuelgan de ahí.

Tooltip-anchor: P1-PANTRY-LOW-IS-A-LIE
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
_SHELF = _REPO_ROOT / "frontend" / "src" / "utils" / "shelfLife.js"


def _src() -> str:
    return _PANTRY.read_text(encoding="utf-8")


def test_the_unit_blind_threshold_is_gone():
    """`LOW_THRESHOLD` comparaba una cantidad contra una constante sin unidad.
    0,5 «lb», 0,5 «paquete» y 0,5 «mazo» no son cosas comparables."""
    src = _src()
    assert "LOW_THRESHOLD" not in src, (
        "P1-PANTRY-LOW-IS-A-LIE: volvió `LOW_THRESHOLD`. Un umbral constante "
        "sobre `quantity` es ciego a `unit`: solo alcanza a las filas de peso "
        "(las discretas empiezan en 1) y dentro de ellas parte por la mitad el "
        "rango normal de compra. Medido: 9 de 13 filas de peso marcadas, 6 de "
        "ellas escritas por el propio botón de exportar la lista."
    )


def test_no_bare_quantity_comparison_decides_low_stock():
    """Ancla la CLASE, no el nombre: renombrar la constante y dejar la misma
    comparación reintroduce exactamente el mismo defecto."""
    src = _src()
    sospechosas = re.findall(r"^.*Number\(\s*it(?:em)?\.quantity\s*\)\s*<=?.*$", src, re.MULTILINE)
    vivas = [ln.strip() for ln in sospechosas if not ln.lstrip().startswith(("*", "//", "/*"))]
    assert not vivas, (
        "P1-PANTRY-LOW-IS-A-LIE: hay una comparación desnuda de `quantity` "
        "contra un número decidiendo estado de stock:\n  " + "\n  ".join(vivas) +
        "\nEse número no significa nada sin `unit`. Si necesitas un estado de "
        "stock real, tiene que ser una RELACIÓN contra lo que el plan pide, y "
        "el emparejamiento de nombres vive en `constants.pantry_names_match` "
        "(backend), no en el cliente."
    )


def test_the_row_state_hangs_from_the_signal_that_is_true():
    """La caducidad es la única afirmación que la Nevera puede sostener con lo
    que ya tiene cargado — y su fallback es callarse."""
    src = _src()
    assert "getShelfLifeBadge" in src, (
        "P1-PANTRY-LOW-IS-A-LIE: Pantry.jsx dejó de usar `getShelfLifeBadge`."
    )
    # Las dos superficies (escritorio y móvil) deben derivar su estado de
    # atención del badge, no de la cantidad.
    assert src.count("needsAttention") >= 3, (
        "P1-PANTRY-LOW-IS-A-LIE: se esperan al menos 3 usos de `needsAttention` "
        "(fila de escritorio, tarjeta móvil y contador del sidebar). Si una "
        "superficie se quedó con la lógica vieja, la Nevera acusa distinto según "
        "el ancho de pantalla."
    )


def test_the_vague_chip_was_replaced_by_the_precise_one():
    """«Queda poco» no dice cuánto ni respecto a qué. El chip de caducidad que
    ya se pintaba al lado dice «Caduca en 2 días» — misma tinta, afirmación
    verificable."""
    src = _src()
    assert "Queda poco" not in src, (
        "P1-PANTRY-LOW-IS-A-LIE: volvió el chip «Queda poco». Es la misma "
        "afirmación no sostenible, ahora en forma de etiqueta."
    )


def test_the_shelf_life_helper_stays_conservative():
    """El reemplazo solo vale si NO inventa: sin `shelf_life_days` o sin
    `created_at` debe devolver null y no pintar nada. Si alguien le añade una
    inferencia por categoría, la Nevera vuelve a acusar sin base — que es
    exactamente el defecto que este P-fix cerró."""
    shelf = _SHELF.read_text(encoding="utf-8")
    assert "if (typeof shelfLifeDays !== 'number' || shelfLifeDays <= 0) return null;" in shelf, (
        "P1-PANTRY-LOW-IS-A-LIE: `getShelfLifeBadge` perdió su guarda de "
        "«sin dato, sin chip»."
    )
    assert "if (!createdAt || typeof createdAt !== 'string') return null;" in shelf, (
        "P1-PANTRY-LOW-IS-A-LIE: `getShelfLifeBadge` perdió la guarda de "
        "`created_at` ausente."
    )
