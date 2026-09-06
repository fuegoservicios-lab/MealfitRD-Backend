# -*- coding: utf-8 -*-
"""[P1-UNKNOWN-UNIT-NOT-WHOLE · 2026-09-06] Una unidad que no conozco no es «una unidad entera».

`to_grams` hacía `canonicalize_unit(unit) or "unidad"`. Ese `or` convertía **cualquier** palabra
que el sistema no reconociera en *una unidad entera del alimento* — y para una hierba, la unidad
entera del catálogo es el MAZO de compra (`density_g_per_unit = 50` para Cilantro, Cebollín y
Perejil). De ahí:

    «2 tallos de cebollín»  → 100 g        «5 tallos de cebollín» → 250 g

con lo que un plan pedía 415 g de cebollín y 568 g de cilantro, y el `P3-HERB-CAP` tenía que
recortarlos. Medido el 06-sep en el journal: **2.043 recortes en 24 h**, dejando de media el 58 %
de lo que pedía el agregador, 799 de ellos cortando más de la mitad.

La distinción que faltaba: **«sin unidad» y «unidad que no conozco» no son lo mismo.**
`_split_qty_unit_name` devuelve literalmente `'unidad'` cuando la línea no trae unidad («2
huevos») y ahí la unidad entera **sí** es lo correcto. Cuando trae una palabra ajena al mapa,
adivinar es lo que produjo el kilo de cilantro.

Radio medido sobre 10.953 líneas de 200 planes vivos: **24 cambian (0,22 %)**, −778 g en total,
21 de ellas tallos de apio pasando de 40 a 37 g. Contenido y en la dirección correcta.

⚠️ Nota de método: la primera medición del radio dijo que tres líneas dejaban de resolver
(«130 g de vegetales al vapor» → None). Era **artefacto de la sonda**: corría con un directorio
`shim` que no tenía la librería de platos compuestos, así que `_compound_dish_lookup` fallaba.
En el árbol completo esas tres resuelven igual que antes. *Una sonda que corre en un árbol
incompleto mide el árbol, no el cambio.*
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from canonical_units import canonicalize_unit  # noqa: E402
from nutrition_db import IngredientNutritionDB, NutritionInfo  # noqa: E402


@pytest.fixture(scope="module")
def db():
    """Instancia SIN catálogo: `to_grams` recibe el `NutritionInfo` ya resuelto, así que no hace
    falta la base — y así el test no depende de la red."""
    return IngredientNutritionDB.__new__(IngredientNutritionDB)


def _info(nombre, unidad=None, taza=None, envase=None):
    return NutritionInfo(name=nombre, kcal=20, protein=1, carbs=3, fats=0,
                         density_g_per_cup=taza, density_g_per_unit=unidad,
                         container_weight_g=envase)


# ── el tallo deja de ser «una unidad» ────────────────────────────────────────────────────────
def test_el_tallo_es_una_unidad_conocida():
    assert canonicalize_unit("tallo") == "tallo"
    assert canonicalize_unit("tallos") == "tallo"


@pytest.mark.parametrize("q, esperado", [(1, 15.0), (2, 30.0), (5, 75.0), (0.5, 7.5)])
def test_el_tallo_de_cebollin_no_pesa_un_mazo(db, q, esperado):
    """El catálogo dice 50 g «por unidad» para el cebollín: eso es el mazo con el que se compra,
    no un tallo suelto. Antes del arreglo, 5 tallos pesaban 250 g."""
    assert db.to_grams(q, "tallos", _info("Cebollín", unidad=50.0, taza=100.0)) == esperado


def test_el_tallo_de_apio_usa_la_cifra_del_propio_sistema(db):
    """`P1-APIO-STALK-CAP` da ~4 tallos por 150 g. 4 × 37 = 148: la tabla no inventa un número,
    reusa el que el sistema ya defendía en otra capa."""
    assert db.to_grams(4, "tallos", _info("Apio", unidad=90.0)) == 148.0


def test_un_alimento_sin_fila_en_la_tabla_usa_el_default_pequeno(db):
    """20 g a propósito: equivocarse por abajo en un aromático cuesta una hierba de menos en la
    lista; por arriba, el kilo de cilantro que el cap tiene que recortar."""
    assert db.to_grams(2, "tallos", _info("Hierba rara", unidad=50.0)) == 40.0


# ── el mazo pasa a resolver ──────────────────────────────────────────────────────────────────
def test_el_mazo_resuelve_y_prefiere_el_catalogo(db):
    assert db.to_grams(1, "mazo", _info("Cilantro", unidad=50.0)) == 50.0
    assert db.to_grams(2, "mazos", _info("Perejil", unidad=50.0, envase=80.0)) == 160.0


# ── el corazón del fix ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("unidad", ["pellizco", "trocito", "vaso", "copa", "porcion", "zzz"])
def test_una_unidad_desconocida_no_se_adivina(db, unidad):
    """Devolver None es el contrato documentado de `to_grams`: «el caller deja el ingrediente tal
    cual». Inventar un peso es lo que llenaba la lista de compras de hierbas fantasma.

    Las seis son palabras que `canonicalize_unit` NO reconoce — se comprueba abajo, porque un
    caso que resulta ser una unidad conocida daría None por otra vía y este test no probaría
    nada (un veredicto que no puede fallar no informa)."""
    assert canonicalize_unit(unidad) is None, (
        f"{unidad!r} sí es una unidad conocida: este caso no ejercita la rama del arreglo")
    assert db.to_grams(3, unidad, _info("Cilantro", unidad=50.0)) is None


@pytest.mark.parametrize("unidad", ["", "unidad", "unidades", "UNIDAD"])
def test_sin_unidad_sigue_siendo_la_unidad_entera(db, unidad):
    """«2 huevos» no trae unidad y ahí la pieza entera SÍ es lo correcto. Si este caso se hubiera
    ido con el resto, el arreglo habría dejado sin gramos media lista de la compra."""
    assert db.to_grams(2, unidad, _info("Huevo", unidad=50.0)) == 100.0


def test_el_knob_devuelve_la_conducta_anterior(db, monkeypatch):
    """Rollback sin redeploy: un cambio en el corazón de la resolución de gramos tiene que poder
    revertirse desde el entorno."""
    monkeypatch.setenv("MEALFIT_UNKNOWN_UNIT_AS_WHOLE", "true")
    assert db.to_grams(3, "trocito", _info("Aceite", unidad=14.0)) == 42.0
    monkeypatch.setenv("MEALFIT_UNKNOWN_UNIT_AS_WHOLE", "false")
    assert db.to_grams(3, "trocito", _info("Aceite", unidad=14.0)) is None


# ── lo que NO debe cambiar ───────────────────────────────────────────────────────────────────
def test_las_unidades_de_peso_y_volumen_pasan_antes_del_cambio(db):
    """`to_base_amount` resuelve g/ml mucho antes de llegar a esta rama: el arreglo no puede
    tocar la vía por la que pasa la mayoría de las líneas."""
    assert db.to_grams(130, "g", _info("Lo que sea")) == 130.0
    assert db.to_grams(2, "tazas", _info("Leche", taza=240.0)) == 480.0


@pytest.mark.parametrize("unidad, campo, esperado", [
    ("rebanada", "unidad", 60.0), ("hoja", "unidad", 60.0), ("diente", "unidad", 60.0),
])
def test_las_discretas_conocidas_siguen_igual(db, unidad, campo, esperado):
    assert db.to_grams(2, unidad, _info("Pan", unidad=30.0)) == esperado


def test_una_discreta_sin_densidad_sigue_sin_resolver(db):
    """Contrato previo intacto: sin `density_g_per_unit` no se inventa el peso de una pieza."""
    assert db.to_grams(2, "unidad", _info("Cosa sin densidad")) is None
