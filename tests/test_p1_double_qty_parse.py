"""[P1-DOUBLE-QTY-PARSE · 2026-07-27] Un pote de mantequilla de maní por cada cucharada.

## Lo que veía el owner en su lista de 30 días

    Mantequilla de maní   14 potes (16 oz c/u)   RD$1,638

La merienda usa **39 g**. En el ciclo de 30 días son ~390 g: **un pote**. Se compraron catorce, y
eso solo era la mitad de los RD$3,313 en que el ciclo se pasaba del presupuesto.

## La causa

La receta traía dos cantidades pegadas al frente:

    1½ 1/2 cdas de mantequilla de maní (39 g)

y el parser degradaba **en silencio**:

    '1½ 1/2 cdas de mantequilla de maní (39 g)' -> (1.5, 'unidad', '1/2 cdas de mantequilla de maní')
    '1½ cdas de mantequilla de maní (39 g)'     -> (1.5, 'cda',    'Mantequilla de maní')

Dos daños: la unidad cae de `cda` a **`unidad`** —y "una unidad" de mantequilla de maní es un
POTE— y el nombre sale hecho jirones.

## Por qué se arregla en el consumidor y no en el productor

⚠️ El productor de esa forma **no se pudo reproducir**: `_prettify_quantity_display` deja los
mixtos intactos, `_collapse_double_fraction` no la genera, y el modelo escribe el mixto legítimo
`1 1/2 cdas` —que este parser YA resolvía bien—. Dos hipótesis cayeron al medirlas. La forma
corrupta aparece en **1 línea de 1318** (12 planes vivos).

Endurecer el consumidor es lo correcto igualmente: ahí es donde se pierde el dinero, y devolver un
nombre basura ante una entrada rara es un defecto por sí mismo venga de donde venga.

tooltip-anchor: P1-DOUBLE-QTY-PARSE
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


# ───────────── 1. el caso del owner ─────────────

def test_el_caso_del_owner():
    qty, unit, name = sc._parse_quantity("1½ 1/2 cdas de mantequilla de maní (39 g)")
    assert unit == "cda", (
        f"unidad={unit!r}: con 'unidad' la lista compra UN POTE POR CUCHARADA (fueron 14, RD$1,638)"
    )
    assert abs(qty - 1.5) < 1e-6
    assert "1/2" not in name, f"el nombre salió con la cantidad dentro: {name!r}"
    assert "maní" in name.lower()


@pytest.mark.parametrize("linea", [
    "1½ 1/2 cdas de mantequilla de maní (39 g)",
    "2½ 1/4 tazas de arroz",
    "½ 1/2 cda de aceite de oliva",
])
def test_doble_cantidad_lider_no_degrada_la_unidad(linea):
    _, unit, name = sc._parse_quantity(linea)
    assert unit != "unidad", f"{linea!r} → unidad genérica: se comprará por envases"
    assert "/" not in name, f"{linea!r} → nombre contaminado: {name!r}"


# ───────────── 2. lo que YA funcionaba no se toca ─────────────

@pytest.mark.parametrize("linea,esperado", [
    ("1 1/2 cdas de mantequilla de maní", (1.5, "cda")),   # mixto LEGÍTIMO
    ("2 1/4 tazas de arroz", (2.25, "taza")),
    ("1/2 taza de leche", (0.5, "taza")),
    ("1½ cdas de aceite", (1.5, "cda")),
    ("½ taza de agua", (0.5, "taza")),
    ("100 g de atún", (100.0, "g")),
])
def test_no_rompe_las_formas_sanas(linea, esperado):
    qty, unit, _ = sc._parse_quantity(linea)
    assert (round(qty, 4), unit) == esperado


def test_el_mixto_legitimo_sobrevive():
    """`1 1/2 cdas` significa cucharada y media y el parser SIEMPRE lo entendió. Si la
    normalización se lo comiera, rompería el caso común para arreglar el raro."""
    assert sc._parse_quantity("1 1/2 cdas de mantequilla de maní")[:2] == (1.5, "cda")


def test_la_fraccion_dentro_del_nombre_no_se_toca():
    """'≈ 1/8 de la fruta' es parte del NOMBRE, no una segunda cantidad líder."""
    qty, unit, _ = sc._parse_quantity("1 lechosa (192g) ≈ 1/8 de la fruta")
    assert (qty, unit) == (1.0, "unidad")


def test_parentesis_con_fraccion_intacto():
    qty, unit, _ = sc._parse_quantity("¼ taza de arroz integral crudo (1/3 taza)")
    assert (qty, unit) == (0.25, "taza")


# ───────────── 3. fail-safe ─────────────

@pytest.mark.parametrize("basura", ["", "   ", "sin cantidad de nada"])
def test_no_revienta(basura):
    sc._parse_quantity(basura)


def test_dict_sigue_funcionando():
    """La normalización es solo para str; el guard de dict (P3-PARSE-QTY-DICT-GUARD) no se toca."""
    qty, unit, _ = sc._parse_quantity({"quantity": 2, "unit": "cda", "name": "Aceite"})
    assert (qty, unit) == (2.0, "cda")
