# -*- coding: utf-8 -*-
"""[P1-DO-PLATO-SIN-MERCADO · 2026-09-10] Un plato dominicano cuyo ingrediente no tiene mercado dominicano.

## De dónde sale

De la revisión de 35 platos quedó abierta una decisión del dueño: si los platos no dominicanos
—frijol cargamanto, habas, guisantes secos— se quedan. Delegó: «haz lo que mejor consideres».
Medido antes de decidir:

```
                     precio RD/lb   ¿se puede costear?   papel en la biblioteca
Frijol cargamanto    —  (fila vacía: precio 0, sin peso)   NO   duplica «Habichuelas rojas guisadas…»
Habas                205                                   sí   el único guiso de habas
Guisantes secos      133                                   sí   el único guiso de guisantes
Habichuelas rojas     53  (la criolla)                      sí
```

Sale el **cargamanto** y se quedan los otros dos. Tres razones que apuntan igual: es colombiano (vive
ya en la biblioteca de Colombia), su fila del catálogo está VACÍA para RD —así que ni la lista de
compras ni el filtro de presupuesto lo ven— y su papel lo cubre ya un plato criollo con precio. Habas
y guisantes son caros, pero eso ya lo resuelve el filtro de presupuesto sin borrar nada.

## Lo que la medición encontró al lado

Los platos que no se pueden costear son el **10,6 % de la biblioteca y el 19,8 % de lo que se le
sirve al dueño**: la regla «lo sin precio sobrevive siempre» (`P1-CANDIDATO-CON-PRECIO`) es correcta
—subestimar asciende lo que no se puede pagar— pero, al podar el 40 % caro de lo que SÍ tiene precio,
lo que no lo tiene gana cuota. La causa no es la regla: son **8 productos dominicanos que no existen
en `supermarket_products`** (queso de papa en 8 platos, azúcar morena en 4…). Eso lo rellena el dueño
en `/supermercado`; aquí no se inventa un precio.
"""
import json
import pathlib

import pytest

import dish_registry as dr

REG = pathlib.Path(dr.REGISTRY_DIR)
DATA = pathlib.Path(dr.DATA_DIR)
NOMBRE = "Frijol cargamanto guisado con arroz y auyama"


def _json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def test_el_cargamanto_ya_no_es_un_plato_dominicano():
    reg = _json(REG / "dish_registry_do_v1.json")
    assert not [t for t in reg["templates"] if "cargamanto" in t["name"].lower()], (
        "volvió un plato de frijol cargamanto a la biblioteca dominicana")


def test_salio_de_los_TRES_sitios_donde_vivia():
    """Dejar uno atrás es como nacen las recetas huérfanas."""
    tpl = _json(DATA / "dish_templates.json")
    lista = tpl["templates"] if isinstance(tpl, dict) else tpl
    assert NOMBRE not in [t.get("name") for t in lista], "sigue en la plantilla"
    assert NOMBRE not in _json(DATA / "dish_constituents_do.json")["templates"], "siguen sus constituyentes"
    assert "tpl_3fed1d443898" not in _json(REG / "recipe_library_do_v1.json")["por_id"], "sigue su receta"


def test_el_ingrediente_sigue_siendo_de_colombia():
    """Se retira el PLATO dominicano, no el alimento: en su cocina es de todos los días."""
    co = _json(REG / "dish_registry_co_v1.json")
    usos = [t["name"] for t in co["templates"]
            if any("cargamanto" in c["canonical"].lower() for c in t["constituents"])]
    assert usos, "el frijol cargamanto desapareció también de la biblioteca colombiana"


@pytest.mark.parametrize("nombre", ["Guiso de habas con ñame y ajo",
                                    "Guisantes secos guisados con papa y zanahoria"])
def test_habas_y_guisantes_se_quedan(nombre):
    """Caros, pero con precio: el filtro de presupuesto los gestiona sin que haga falta borrarlos."""
    reg = _json(REG / "dish_registry_do_v1.json")
    assert nombre in [t["name"] for t in reg["templates"]], f"se perdió «{nombre}»"


def test_su_papel_sigue_cubierto_por_un_plato_criollo():
    reg = _json(REG / "dish_registry_do_v1.json")
    criollo = [t for t in reg["templates"]
               if t["name"].startswith("Habichuelas rojas guisadas con auyama, arroz")]
    assert criollo and "almuerzo" in criollo[0]["slots"], (
        "se retiró el cargamanto porque su papel lo cubría este plato: si él tampoco está, "
        "el almuerzo de legumbre con arroz quedó sin sustituto")


def test_ninguna_receta_quedo_huerfana():
    reg = _json(REG / "dish_registry_do_v1.json")
    vivos = {t["template_id"] for t in reg["templates"]}
    huerfanas = [k for k in _json(REG / "recipe_library_do_v1.json")["por_id"] if k not in vivos]
    assert not huerfanas, f"recetas que apuntan a plantillas que ya no existen: {huerfanas}"
