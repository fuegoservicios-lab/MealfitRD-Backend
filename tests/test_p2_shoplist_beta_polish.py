"""[P2-SHOPLIST-BETA-POLISH · 2026-08-18] Regresión del pulido de la lista de compras
para países beta (pedido del dueño tras ver su primera lista ES).

Dos mitades:
  - Backend (este archivo): los ítems unpriced-keep salen con el pasillo REAL del súper
    (Vegetales/Frutas/...) resuelto del master, no con el label interno
    'CATÁLOGO SIN PRECIO' que se filtraba al PDF del usuario. Fallback al label
    histórico si el lookup no resuelve (display fail-open).
  - Frontend (test_p1_country_system_f1_t7 + Dashboard): banner beta minimalista con el
    nombre del país + remap display-only del label viejo para planes YA persistidos.

Los asserts e2e de conducta viva (Jamón serrano→'Proteínas', Hummus→'Despensa',
Tortilla de maíz→'Despensa', sweep de 8 filas) viven en test_p1_country_system_f2.py —
este archivo ancla la ESTRUCTURA para que el gate (que no corre e2e) la vigile.
"""
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def test_branch_unpriced_usa_categoria_real_con_fallback():
    src = _read("shopping_calculator.py")
    assert "tooltip-anchor: P2-SHOPLIST-BETA-POLISH" in src
    m = re.search(
        r'display_cat = _master_category_for_unpriced_item\(name\) or "CATÁLOGO SIN PRECIO"', src
    )
    assert m, (
        "el branch unpriced-keep debe resolver el pasillo real del master con fallback al "
        "label histórico — quitar el fallback rompe display si el lookup no resuelve; quitar "
        "el lookup devuelve el label interno al PDF del usuario"
    )


def test_helper_no_corre_en_el_camino_con_precio():
    """El lookup vive SOLO dentro del branch unpriced-keep (raro), jamás en el camino
    caliente con precio — un call site nuevo fuera del branch merece revisión de perf."""
    src = _read("shopping_calculator.py")
    calls = [m.start() for m in re.finditer(r"_master_category_for_unpriced_item\(", src)]
    # 1 def + 1 call site en el branch
    assert len(calls) == 2, (
        f"esperados exactamente 2 usos (def + branch unpriced-keep), hay {len(calls)} — "
        "si añadiste otro call site, verifica que no esté en el camino caliente del aggregator"
    )


@pytest.mark.e2e
def test_categoria_real_resuelve_para_filas_vivas():
    """Lookup vivo contra el catálogo (e2e, no corre en el gate): las filas del incidente
    resuelven a su pasillo real.

    [reconvertido · P2-COUNTRY-HOUSEKEEPING · 2026-08-21] Antes esperaba la categoría CRUDA de la
    base («Vegetales»), y eso era media solución: la rama CON precio emite el label de DISPLAY
    («VEGETALES»), el Dashboard agrupa por la cadena literal, y el usuario acababa viendo DOS
    secciones del mismo pasillo del súper — una con doce ítems y otra con las Acelgas solas. El
    helper devuelve ahora el label de display, así que ambas ramas caen en el mismo sitio.

    Se ancla contra `_get_display_category`, no contra literales, para que un cambio del mapa de
    pasillos no obligue a re-teclear este test."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e")
    db_core.connection_pool.open()
    import shopping_calculator as sc
    for nombre, cruda in (("Acelgas", "Vegetales"), ("Membrillo", "Frutas"),
                          ("Jamón serrano", "Proteínas"), ("Hummus", "Despensa")):
        got = sc._master_category_for_unpriced_item(nombre)
        assert got == sc._get_display_category(cruda, nombre), (
            f"{nombre!r} → {got!r}: no es el MISMO pasillo que emite la rama con precio, así que "
            f"el Dashboard volvería a pintar dos secciones para el mismo estante"
        )
        assert got != cruda, f"{nombre!r} devolvió la categoría cruda de la DB, no el label"
    assert sc._master_category_for_unpriced_item("nombre-que-no-existe-xyz") is None
