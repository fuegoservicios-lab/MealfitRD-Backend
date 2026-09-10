# -*- coding: utf-8 -*-
"""[P1-CATALOGO-DICE-QUE-PRODUCTO · 2026-09-10] El catálogo tenía números de productos concretos y no decía cuáles.

Nueve de las 19 notas de la tercera ronda pedían «especificar el porcentaje de grasa» y «verificar
según la etiqueta». Verificado contra USDA FoodData Central y contra el supermercado del dueño:

- **Queso cottage** llevaba el 1 % (fdc 173417) y su supermercado sólo vende **2 % y 4 %**: los
  números eran de un producto que no puede comprar ⇒ fila entera a 2 % (fdc 172182).
- **Tofu firme**: valores de tofu FIRME con el puntero del tofu NORMAL (172476) ⇒ 172475.
- **Salsa de soya**: números de soya REDUCIDA en sodio ⇒ se glosa, y su super la vende.
- **Yogurt griego sin azúcar**: números de griego natural 0 % ⇒ se glosa.

Estos tests leen el FICHERO de la migración, no la base: la migración es el contrato que se revisa,
y un test que dependa de Neon no corre en el gate. La base se verifica al aplicarla (su bloque
`DO $$` aborta si las cuatro filas no quedan como dice).
"""
import pathlib
import re

import pytest

BACK = pathlib.Path(__file__).resolve().parents[1]
NOMBRE = "p1_catalogo_dice_que_producto_2026_09_10.sql"


@pytest.fixture(scope="module")
def sql():
    return (BACK / "migrations" / NOMBRE).read_text(encoding="utf-8")


def test_la_migracion_vive_en_los_DOS_directorios():
    """P3-MIGRATIONS-SSOT: una migración que falta en un directorio no viaja con su repo."""
    raiz = BACK.parent / "migrations" / NOMBRE
    assert (BACK / "migrations" / NOMBRE).exists()
    assert raiz.exists(), "falta la copia del workspace-root"
    assert raiz.read_text(encoding="utf-8") == (BACK / "migrations" / NOMBRE).read_text(encoding="utf-8")


def test_el_cottage_pasa_ENTERO_al_2_por_ciento(sql):
    """Una fila mitad 1 % y mitad 2 % sería peor que cualquiera de las dos."""
    bloque = sql.split("WHERE name = 'Queso cottage'")[0].rsplit("UPDATE public.master_ingredients SET", 1)[1]
    assert "fdc_id                      = 172182" in bloque
    for col in ("kcal", "protein_g", "carbs_g", "fats_g", "sodium_mg", "calcium_mg", "phosphorus_mg",
                "saturated_fat_g", "cholesterol_mg", "selenium_mcg", "vitamin_a_mcg_rae", "omega3_ala_g"):
        assert re.search(rf"{col}_per_100g\s*=", bloque), f"el cottage no reescribe `{col}`: quedaría mezclado"
    assert "gloss_es                    = '2 % de grasa'" in bloque


def test_el_tofu_solo_cambia_el_PUNTERO(sql):
    """Los valores ya eran de tofu firme: tocar un número aquí sería inventar un defecto."""
    bloque = sql.split("WHERE name = 'Tofu firme'")[0].rsplit("UPDATE", 1)[1]
    assert "fdc_id = 172475" in bloque
    assert "_per_100g" not in bloque, "la corrección del tofu tocó valores que estaban bien"


@pytest.mark.parametrize("fila,gloss", [("Salsa de soya", "reducida en sodio"),
                                        ("Yogurt griego sin azúcar", "natural, 0 % de grasa")])
def test_las_glosas_dicen_que_producto_es(sql, fila, gloss):
    assert f"SET gloss_es = '{gloss}'\nWHERE name = '{fila}'" in sql


def test_el_nombre_canonico_no_se_toca(sql):
    """El nombre es la identidad del motor: Nevera, alergias y coherencia resuelven por él."""
    assert not re.search(r"SET\s+name\s*=|,\s*name\s*=", sql), "la migración renombra una fila"
    assert "aliases" not in sql.split("-- Idempotente")[1], "la glosa se coló en los aliases"


def test_es_idempotente_y_verifica_lo_que_escribe(sql):
    assert "IS DISTINCT FROM" in sql
    assert "RAISE EXCEPTION" in sql, "sin sanity, una migración que no casó ninguna fila pasa en silencio"


def test_el_queso_de_hoja_queda_FUERA_y_dicho(sql):
    """Sin etiqueta no hay referencia; inventarla sería la afirmación falsa que esto viene a quitar."""
    assert "Queso de hoja" in sql and "no se inventa" in sql
    assert "WHERE name = 'Queso de hoja'" not in sql


def test_los_platos_de_cottage_siguen_siendo_de_proteina():
    """Pasar el cottage a 2 % bajó «Queso cottage con casabe, tomate y aguacate» a 4,79 g/100 kcal,
    por debajo del 5,0 con el que entró al catálogo. Se reequilibró (casabe 55 -> 40 g, cottage
    170 -> 200 g) en vez de esconderlo: un plato de proteína que deja de serlo por un arreglo
    correcto sigue siendo un defecto, y es de quien hizo el arreglo."""
    import json
    reg = json.loads((BACK / "data" / "registry" / "dish_registry_do_v1.json").read_text(encoding="utf-8"))
    bajos = []
    for t in reg["templates"]:
        if any(c["canonical"] == "Queso cottage" for c in t["constituents"]):
            n = t["nutrition_per_serving"]
            dens = 100.0 * float(n["protein_g"]) / float(n["kcal"])
            if dens < 5.0:
                bajos.append((t["name"], round(dens, 2)))
    assert not bajos, f"platos de cottage por debajo de 5,0 g/100 kcal: {bajos}"
