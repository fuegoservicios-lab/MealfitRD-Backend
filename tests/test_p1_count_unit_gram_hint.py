"""[P1-COUNT-UNIT-GRAM-HINT · 2026-07-27] "½ pedazo de yautía" no dice cuánto usar.

## Lo que veía el owner

    ½ pedazo de yautía
    ½ pechuga de pollo (porción)
    ½ pedazo mediano de yuca

Un "pedazo" no tiene tamaño natural. El usuario compró una cantidad concreta y la receta no le
dice qué fracción de esa compra va al plato.

## La medición que definió el alcance

Sobre 1299 líneas de ingrediente de 12 planes vivos:

    con unidad de bulto vaga:            19  (1.5%)  — y las 19 SIN ningún peso
    pareadas por ALIMENTO contra el raw: 17 de 18 venían YA sin gramos del modelo

⚠️ El primer pareo se hizo por ÍNDICE y dio "½ pechuga de pollo" contra "0.5 cebolla": las dos
listas no están alineadas. Es la misma trampa de 2026-07-24 — parear por alimento, NUNCA por
índice. Con el pareo bueno el diagnóstico cambió: NO es un bug de display que tire los gramos.

## Por qué se puede responder sin tocar el modelo

El peso existe: es el mismo `weight` de `DOMINICAN_HOUSEHOLD_MEASURES` con el que el display
convirtió los gramos a bultos. Se anexa desde esa tabla, así que es imposible que contradiga a la
lista de compras — y de hecho "125 g de yautía" vuelve como "½ pedazo de yautía (≈125 g)".

⚠️ Hay DOS tablas de pesos (`DOMINICAN_HOUSEHOLD_MEASURES` para display, `constants.UNIT_WEIGHTS`
para la lista) y nadie fuera de `humanize_ingredients` importa la primera. Hoy sus **36 claves
comunes concuerdan exactamente**; el test de abajo ancla esa concordancia porque si divergen, este
anexo empezaría a mentirle al usuario respecto de lo que compró.

## Por qué SOLO las unidades vagas

Aplicado a toda la tabla el anexo tocaba **176 líneas de 1299 (13.5%)**: "1 diente de ajo (≈5 g)",
"1 huevo (≈50 g)". Un huevo es un objeto de tamaño conocido — anotarlo es ruido que enseña al
usuario a ignorar el paréntesis. Restringido a las vagas: 18 líneas (1.4%), todas ambiguas.

tooltip-anchor: P1-COUNT-UNIT-GRAM-HINT
"""
from __future__ import annotations

import copy

import pytest

import humanize_ingredients as H
from humanize_ingredients import append_gram_hint as A


# ───────────── 1. los casos reales del owner ─────────────

@pytest.mark.parametrize("linea,esperado", [
    ("½ pedazo de yautía", "½ pedazo de yautía (≈125 g)"),
    ("½ pedazo mediano de yuca", "½ pedazo mediano de yuca (≈200 g)"),
    ("1 pedazo de ñame", "1 pedazo de ñame (≈300 g)"),
    ("1½ pechuga de pollo (porción)", "1½ pechuga de pollo (≈300 g)"),
])
def test_las_unidades_vagas_reciben_su_peso(linea, esperado):
    assert A(linea) == esperado


def test_porcion_se_SUSTITUYE_no_se_encadena():
    """'(porción)' ES la marca de vaguedad que se está respondiendo. Dejarla produce el feo
    '… (porción) (≈100 g)' — dos paréntesis seguidos diciendo lo mismo."""
    out = A("½ pechuga de pollo (porción)")
    assert out == "½ pechuga de pollo (≈100 g)"
    assert "porción" not in out


# ───────────── 2. lo que NO debe tocar ─────────────

@pytest.mark.parametrize("linea", [
    "1 diente de ajo",      # objeto de tamaño conocido → anotarlo es ruido
    "1 huevo",
    "½ limón",
    "1 tomate",
    "½ cebolla",
])
def test_las_unidades_NO_vagas_quedan_intactas(linea):
    """El recorte que llevó de 176 líneas tocadas a 18. Sin él, el 13.5% del listado se llena de
    paréntesis que no resuelven ninguna duda."""
    assert A(linea) == linea


@pytest.mark.parametrize("linea", [
    "150 g de yautía",             # ya trae peso
    "1 lb de yuca",
    "½ pedazo de yautía (≈125 g)",  # ya anotado → idempotente
    "pedazo de yautía",             # sin cantidad líder: no hay nada que multiplicar
    "2 pedazos de kryptonita",      # fuera de la tabla: NO se inventa equivalencia
])
def test_no_anota_cuando_no_toca(linea):
    assert A(linea) == linea


@pytest.mark.parametrize("basura", [None, 123, "", "   ", [], {}])
def test_fail_safe(basura):
    assert A(basura) == basura


def test_idempotente():
    una = A("½ pedazo de yautía")
    assert A(una) == una


# ───────────── 3. el contrato que hace segura la operación ─────────────

def test_las_dos_tablas_de_pesos_concuerdan():
    """Ancla de la CLASE. El anexo toma el peso de la tabla del DISPLAY, pero el usuario compró
    según `constants.UNIT_WEIGHTS`. Mientras concuerden, el paréntesis dice la verdad respecto de
    la compra; si alguien edita una sola de las dos, este test cae ANTES que el usuario lo vea."""
    from constants import UNIT_WEIGHTS
    norm = {H.strip_accents(str(k).lower()): v for k, v in UNIT_WEIGHTS.items()}
    comunes = discrepantes = 0
    for k, v in H.DOMINICAN_HOUSEHOLD_MEASURES.items():
        otro = norm.get(H.strip_accents(str(k).lower()))
        if otro is None:
            continue
        comunes += 1
        if abs(float(otro) - float(v.get("weight") or 0)) > 0.51:
            discrepantes += 1
    assert comunes >= 30, f"solo {comunes} claves comunes: ¿se renombró una de las tablas?"
    assert discrepantes == 0, (
        f"{discrepantes} alimentos pesan distinto en display vs lista de compras — el peso anexado "
        f"contradiría lo que el usuario compró"
    )


def test_el_gramo_anexado_es_el_que_entro():
    """La prueba más fuerte de que se usa la MISMA tabla de la conversión: los gramos originales
    vuelven intactos al display."""
    plan = {"days": [{"day": 1, "meals": [{
        "name": "Yautía con pollo", "meal": "Almuerzo",
        "ingredients": ["125 g de yautia", "200 g de pechuga de pollo"],
        "recipe": ["Mise en place: pela la yautía.", "Montaje: sirve."],
    }]}]}
    out = H.humanize_plan_ingredients(copy.deepcopy(plan))
    ings = out["days"][0]["meals"][0]["ingredients"]
    assert any("≈125 g" in s for s in ings), ings
    assert any("≈200 g" in s for s in ings), ings


# ───────────── 4. no contamina lo que alimenta a la compra ni a los pasos ─────────────

def test_ingredients_raw_intacto():
    """`ingredients_raw` es lo que re-procesa el aggregator de la lista. Si el anexo llegara ahí,
    `_parse_quantity` vería '(≈125 g)' como parte del nombre."""
    plan = {"days": [{"day": 1, "meals": [{
        "name": "Yautía", "meal": "Almuerzo",
        "ingredients": ["125 g de yautia"],
        "recipe": ["Montaje: sirve."],
    }]}]}
    original = list(plan["days"][0]["meals"][0]["ingredients"])
    out = H.humanize_plan_ingredients(copy.deepcopy(plan))
    assert out["days"][0]["meals"][0]["ingredients_raw"] == original
    assert not any("≈" in s for s in out["days"][0]["meals"][0]["ingredients_raw"])


def test_los_pasos_no_arrastran_el_peso():
    """El anexo corre DESPUÉS de `sync_recipe_steps_to_household` justamente para esto: la prosa de
    la receta debe leer la medida casera limpia."""
    plan = {"days": [{"day": 1, "meals": [{
        "name": "Yautía", "meal": "Almuerzo",
        "ingredients": ["125 g de yautia"],
        "recipe": ["Mise en place: pela la yautía.", "Montaje: sirve."],
    }]}]}
    out = H.humanize_plan_ingredients(copy.deepcopy(plan))
    assert not any("≈" in str(s) for s in out["days"][0]["meals"][0]["recipe"])


def test_el_anexo_esta_conectado_al_pipeline():
    """Ancla 'código presente, efecto ausente': el helper podría existir y no invocarse nunca."""
    import inspect
    src = inspect.getsource(H.humanize_plan_ingredients)
    assert "append_gram_hint(" in src, "el helper existe pero el pipeline no lo llama"
