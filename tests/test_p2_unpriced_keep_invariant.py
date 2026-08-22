"""[P2-UNPRICED-KEEP-INVARIANT · 2026-08-21] Nada defendía «toda fila sin precio está cubierta por
un keep».

El agregador dropea del carrito lo que no tiene precio RD, **en silencio**. Las 141 filas beta
nacieron sin precio A PROPÓSITO (su país no tiene mercado RD que cotizar), así que lo único que las
salva de desaparecer de la lista de la compra es que algún keep las reclame —hoy
`is_country_catalog_unpriced_item`—. Esa dependencia no la vigilaba ningún test: dar de alta una
fila beta y olvidarse de su token la borra del carrito sin log, sin alerta y sin que nadie lo note
hasta que un usuario diga «no me aparece el jamón».

POR QUÉ AHORA, Y NO ANTES. `P1-COUNTRY-CATALOG-BY-COUNTRY` partió la lista plana de tokens en un
diccionario por país. La partición se verificó conjunto a conjunto contra el histórico, pero eso
protege de perder un token; NO protege de que una fila NUEVA no entre en ninguna lista. El riesgo
de huérfanas subió justo cuando la estructura se volvió más rica, así que la invariante pasa de
«conveniente» a «necesaria».

MEDIDO HOY: 141 filas sin precio, **0 huérfanas** con el sistema de países encendido. Con el knob
apagado sale 1, «Tortilla de maíz», y es DELIBERADA: el propio predicado la excluye ahí porque
antes de Fase 2 esa fila no existía para efectos de República Dominicana, y conservarla habría
roto la byte-identidad. El test lo distingue en vez de tapar los dos casos con un número.

LA FORMA DEL TEST IMPORTA. No enumera nombres esperados: recorre el catálogo VIVO y pregunta por
cada fila sin precio. Una lista a mano volvería a quedarse atrás en la siguiente alta — que es
exactamente el fallo que esto existe para impedir.

Cubre:
  A. Cero filas sin precio huérfanas.
  B. Cada fila la reclama ALGÚN país (tras la partición).
  C. La excepción de «Tortilla de maíz» con el knob apagado es deliberada, no un descuido.
  D. El keep sigue siendo el que decide, no el precio.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(scope="module")
def sin_precio(sc):
    filas = sc.get_master_ingredients() or []
    if not filas:
        pytest.skip("catálogo no disponible (sin DB)")
    return [r for r in filas
            if not ((r.get("price_per_lb") or 0) > 0 or (r.get("price_per_unit") or 0) > 0)]


_PAISES = ("ES", "MX", "CO", "PR", "US", "DO")


# ── A. Cero huérfanas ───────────────────────────────────────────────────────────────────────────

def test_ninguna_fila_sin_precio_queda_sin_keep(sc, sin_precio, monkeypatch):
    """La invariante. Una fila sin precio que ningún keep reclama se cae del carrito EN SILENCIO:
    sin log, sin alerta, sin fila en ninguna tabla de fallos. El usuario sólo se entera al no
    encontrar el jamón en su lista."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    huerfanas = [r["name"] for r in sin_precio
                 if not sc.is_country_catalog_unpriced_item(r["name"])]
    assert not huerfanas, (
        f"{len(huerfanas)} fila(s) sin precio que ningún keep reclama — se caerán del carrito sin "
        f"aviso: {huerfanas}"
    )


def test_hay_filas_sin_precio_que_defender(sin_precio):
    """Guard del guard: si el catálogo dejara de tener filas sin precio, el test de arriba pasaría
    por vacío y dejaría de informar. Un veredicto que no puede fallar no es una defensa."""
    assert len(sin_precio) >= 100, (
        f"sólo {len(sin_precio)} filas sin precio; el catálogo beta tenía 141. Si el número cayó "
        f"de verdad, revisa que no se hayan perdido altas antes de relajar esta cota"
    )


# ── B. Cada fila tiene país ─────────────────────────────────────────────────────────────────────

def test_cada_fila_sin_precio_la_reclama_algun_pais(sc, sin_precio, monkeypatch):
    """Tras `P1-COUNTRY-CATALOG-BY-COUNTRY` el keep se pregunta POR PAÍS en el catálogo del
    generador. Una fila reclamada por la unión pero por ningún país concreto sería invisible para
    todos a la vez — el modo de fallo que la partición introdujo y que hay que vigilar."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    sin_pais = [r["name"] for r in sin_precio
                if sc.is_country_catalog_unpriced_item(r["name"])
                and not any(sc.is_country_catalog_unpriced_item(r["name"], country=cc)
                            for cc in _PAISES)]
    assert not sin_pais, f"filas que la unión reclama pero ningún país: {sin_pais}"


# ── C. La única excepción es deliberada ─────────────────────────────────────────────────────────

def test_la_tortilla_de_maiz_es_la_unica_excepcion_y_solo_con_el_knob_apagado(sc, sin_precio,
                                                                              monkeypatch):
    """Con el sistema de países APAGADO el predicado excluye «tortilla de maiz» a propósito: antes
    de Fase 2 esa fila no existía para efectos de RD y conservarla habría roto la byte-identidad
    (ver el docstring de `is_country_catalog_unpriced_item`). Se ancla que la excepción es UNA y
    que desaparece al encender el knob — si mañana aparece una segunda, alguien la metió sin
    documentarla."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    huerfanas = {r["name"] for r in sin_precio
                 if not sc.is_country_catalog_unpriced_item(r["name"])}
    assert huerfanas == {"Tortilla de maíz"}, (
        f"con el knob apagado las huérfanas deberían ser exactamente {{'Tortilla de maíz'}}: "
        f"{sorted(huerfanas)}"
    )


# ── D. El keep decide, no el precio ─────────────────────────────────────────────────────────────

def test_el_keep_no_inventa_precio(sc, sin_precio, monkeypatch):
    """El otro lado del contrato: conservar la fila NO puede significar darle un costo. Un precio
    inventado contamina `shopping_cost_summary` y el banner de presupuesto — el mismo daño que
    P1-SHOPLIST-SANITY-CAP midió con los frascos de orégano."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e")
    db_core.connection_pool.open()
    nombre = next(r["name"] for r in sin_precio
                  if sc.is_country_catalog_unpriced_item(r["name"]))
    res = sc.aggregate_and_deduct_shopping_list([f"30 g de {nombre}"], structured=True)
    items = res if isinstance(res, list) else (res.get("items") or [])
    item = next((it for it in items if it.get("name") == nombre), None)
    assert item is not None, f"{nombre!r} se cayó del agregador pese a tener keep"
    assert item.get("estimated_cost_rd") is None, (
        f"{nombre!r} sobrevivió CON un costo RD inventado"
    )
