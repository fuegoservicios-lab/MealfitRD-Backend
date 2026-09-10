# -*- coding: utf-8 -*-
"""[P1-QUESO-DE-PAPA-A-GOUDA · 2026-09-10] El «queso de papa» era edam con nombre puertorriqueño.

## De dónde sale

El dueño preguntó: *«el queso de papa, ¿se puede usar cualquier queso? mozzarella, gouda…»*. Medido:

```
                        kcal  prot  grasa  sodio   ¿lo vende su súper?
queso de papa (Edam)     357  25,0  28,6    973    no, con ningún nombre
gouda                    356  24,9  27,4    819    17 productos, RD$320/lb
mozzarella               297  22,2  22,1    486    18 productos, RD$214/lb
```

La fila «Queso de papa» es **Edam** (USDA 173419) y el término es **puertorriqueño**: el catálogo la
tiene en el léxico de PR, sin precio dominicano. Los 8 platos dominicanos que la usaban no se podían
costear — y lo sin precio se sirve casi el doble (`P1-DO-PLATO-SIN-MERCADO`). El gouda es el mismo
queso a efectos de nutrición y está en su súper.

Se renombra el PLATO además del ingrediente: con el nombre en «queso de papa» y la lista en
«Queso gouda», el dueño iría a buscar un queso que no existe en su mercado.

## Lo que salió al lado

1. La fila del **gouda** tenía valores de gouda con el puntero USDA del **feta** (173420). Mismo defecto
   que el tofu: el valor bien, el puntero mintiendo ⇒ se corrige el puntero a 171241.
2. **3 de 193 recetas estaban en voseo** («Rallá», «rompé», «añadí», «Comélo»). En RD se tutea.
"""
import json
import pathlib
import re

import pytest

import dish_registry as dr

BACK = pathlib.Path(__file__).resolve().parents[1]
REG = pathlib.Path(dr.REGISTRY_DIR)
MIG = "p1_queso_de_papa_a_gouda_2026_09_10.sql"


def _json(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reg_do():
    return _json(REG / "dish_registry_do_v1.json")


def test_ningun_plato_dominicano_usa_queso_de_papa(reg_do):
    usan = [t["name"] for t in reg_do["templates"]
            if any(c["canonical"] == "Queso de papa" for c in t["constituents"])]
    assert not usan, f"platos dominicanos con un queso que su mercado no tiene: {usan}"


def test_ni_lo_nombra(reg_do):
    """El nombre y la lista de compras tienen que hablar del MISMO queso."""
    assert not [t["name"] for t in reg_do["templates"] if "queso de papa" in t["name"].lower()]


def test_los_8_existen_con_gouda_y_con_receta(reg_do):
    rec = _json(REG / "recipe_library_do_v1.json")["por_id"]
    con_gouda = [t for t in reg_do["templates"]
                 if "queso gouda" in t["name"].lower()
                 and any(c["canonical"] == "Queso gouda" for c in t["constituents"])]
    assert len(con_gouda) >= 8, f"sólo {len(con_gouda)} platos quedaron con gouda"
    sin_receta = [t["name"] for t in con_gouda if t["template_id"] not in rec]
    assert not sin_receta, f"platos renombrados que perdieron su receta: {sin_receta}"


def test_ninguna_receta_quedo_huerfana(reg_do):
    vivos = {t["template_id"] for t in reg_do["templates"]}
    huerfanas = [k for k in _json(REG / "recipe_library_do_v1.json")["por_id"] if k not in vivos]
    assert not huerfanas, f"recetas que apuntan a plantillas que ya no existen: {huerfanas}"


def test_el_queso_de_papa_sigue_siendo_de_puerto_rico():
    """Se retira de la cocina dominicana, no del mundo: en Puerto Rico es suyo."""
    pr = _json(REG / "dish_registry_pr_v1.json")
    assert any(any(c["canonical"] == "Queso de papa" for c in t["constituents"]) for t in pr["templates"]), (
        "el queso de papa desapareció también de la biblioteca de Puerto Rico")


VOSEO = re.compile(r"(?<![\wáéíóúñ])(rallá|rompé|añadí|seguí|comélo|calentá|echá|serví|poné|pelá|cortá)(?![\wáéíóúñ])", re.I)


def test_el_recetario_dominicano_tutea():
    """En RD se tutea; «Rallá el queso» es de Buenos Aires."""
    rec = _json(REG / "recipe_library_do_v1.json")["por_id"]
    con_vos = {k: sorted({m.group(1) for p in r.get("pasos") or [] for m in VOSEO.finditer(p)})
               for k, r in rec.items()}
    con_vos = {k: v for k, v in con_vos.items() if v}
    assert not con_vos, f"recetas en voseo: {con_vos}"


def test_la_migracion_del_gouda_solo_cambia_el_PUNTERO():
    sql = (BACK / "migrations" / MIG).read_text(encoding="utf-8")
    assert (BACK.parent / "migrations" / MIG).read_text(encoding="utf-8") == sql, "las dos copias difieren"
    cuerpo = sql.split("-- Idempotente")[1]
    assert "SET fdc_id = 171241" in cuerpo and "WHERE name = 'Queso gouda'" in cuerpo
    assert not re.search(r"SET[^;]*_per_100g\s*=", cuerpo), "la migración tocó valores que estaban bien"
    assert "RAISE EXCEPTION" in cuerpo
