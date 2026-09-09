# -*- coding: utf-8 -*-
"""[P1-RAW-LINEA-MUERTA · 2026-09-09] El pavo que se compraba dos veces.

## El defecto, medido en un plan real

Plan `c4098931`, primer plan del dueño con cuenta limpia. Cena del día 1, «Guiso criollo de
Pavo»: `ingredients` traía **una** línea de pavo («185 g de pechuga de pavo») e
`ingredients_raw` traía **las dos** — la superada («135 g de pechuga de pavo en lonjas/tiras»)
y la vigente. La lista de compras lee `ingredients_raw` PRIMERO, así que compró:

    Pechuga de pavo  1 lb    RD$285,00
    Jamón de pavo    ¾ lb    RD$191,25   ← no aparece en NINGUNA de las 12 comidas

RD$476 de pavo para el único plato con pavo del plan. Segundo caso el mismo día: desayuno del
día 3, «1 huevo» superada por «1 clara de huevo». **2 de 12 comidas.**

## Por qué ninguna de las defensas existentes lo vio

Las tres se comprobaron ejecutándolas, no leyéndolas:

* `_reconcile_display_raw_lines` corrido sobre esa cena devuelve **0 filas y no cambia nada**:
  usa el resolvedor NUTRICIONAL, para el que ambas líneas son `Pechuga de pavo` — un alimento,
  presente en los dos lados, nada que reconciliar.
* `_reconcile_raw_missing_in_display` va en la dirección contraria y es solo-append por
  diseño: «el display solo GANA la línea que ya está en raw/compras (**la verdad**)». Una línea
  superada rompe justo esa premisa — raw deja de ser la verdad y pasa a ser la verdad más su
  historia.
* El guard de coherencia vio las 3 divergencias y las llamó `recipe_unquantified`, con
  `action_taken: not_applicable`. Tiene una hipótesis para «la receta no cuantifica» y ninguna
  para «sobra una línea».

## El fondo

**La misma cadena resuelve a dos alimentos distintos según quién pregunte.**
`macros_from_ingredient_string('…pechuga de pavo en lonjas/tiras')` → `Pechuga de pavo`;
`_parse_quantity(…)` → `Jamón de pavo`. El discriminante es el token «lonjas»: «en tiras» a
secas resuelve bien. Todo lo que reconcilia usa el nutricional, así que **del lado que MIRA hay
un solo pavo y del lado que GASTA hay dos**. Por eso el barrido pregunta por la identidad de
COMPRA: es la que paga.

*Dos resolvedores que no coinciden hacen el defecto invisible para todo el que use el otro.*
"""
import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

# Identidades de COMPRA falsas: el test no depende del catálogo ni de la base.
_COMPRA = {
    "185 g de pechuga de pavo": "Pechuga de pavo",
    "135 g de pechuga de pavo en lonjas/tiras": "Jamón de pavo",
    "1 clara de huevo": "Clara de huevo",
    "1 huevo": "Huevo",
    "1 cebolla": "Cebolla",
    "0.5 cebolla picada": "Cebolla",
    "1 diente de ajo": "Ajo",
    "Sal al gusto": "Sal",
    "225 g de maíz dulce en granos": "Maíz dulce en granos",
    "1 ají morrón": "Ají morrón",
}


@pytest.fixture
def barrido(monkeypatch):
    """Sustituye el resolvedor de compras por una tabla fija (sin catálogo, sin DB)."""
    import shopping_calculator

    def _falso(linea, **_kw):
        n = _COMPRA.get(linea)
        return (1.0, "unidad", n) if n else (1.0, "unidad", None)

    monkeypatch.setattr(shopping_calculator, "_parse_quantity", _falso)
    import graph_orchestrator
    return graph_orchestrator._barrer_lineas_muertas_de_raw


def _dia(ingredients, ingredients_raw, nombre="Guiso criollo de Pavo"):
    return [{"day": 1, "meals": [{"meal": "Cena", "name": nombre,
                                  "ingredients": list(ingredients),
                                  "ingredients_raw": list(ingredients_raw)}]}]


# ------------------------------------------------------------------ el caso real
def test_borra_la_linea_que_compra_lo_que_la_receta_ya_no_menciona(barrido):
    """El caso vivo: raw conserva la línea superada y la lista compra pavo dos veces."""
    dias = _dia(["185 g de pechuga de pavo", "1 cebolla"],
                ["135 g de pechuga de pavo en lonjas/tiras", "185 g de pechuga de pavo",
                 "1 cebolla"])
    assert barrido(dias) == 1
    raw = dias[0]["meals"][0]["ingredients_raw"]
    assert "135 g de pechuga de pavo en lonjas/tiras" not in raw, (
        "la línea superada sigue en raw: la lista volvería a comprar `Jamón de pavo`")
    assert "185 g de pechuga de pavo" in raw and "1 cebolla" in raw, (
        "se llevó por delante líneas vivas")


def test_el_segundo_caso_del_mismo_dia_huevo_vs_clara(barrido):
    """`1 huevo` superada por `1 clara de huevo`: identidades de compra DISTINTAS, y por eso
    un barrido que solo mirara «duplicados del mismo alimento» no lo vería."""
    dias = _dia(["1 clara de huevo", "1 cebolla"], ["1 huevo", "1 clara de huevo", "1 cebolla"],
                nombre="Revoltillo criollo")
    assert barrido(dias) == 1
    assert "1 huevo" not in dias[0]["meals"][0]["ingredients_raw"]


def test_NO_borra_la_misma_identidad_escrita_distinto(barrido):
    """`0.5 cebolla picada` y `1 cebolla` compran ambas `Cebolla`.

    Es la razón de comparar IDENTIDADES y no cadenas: raw existe precisamente para llevar la
    redacción pre-humanización, y borrar por texto distinto vaciaría media lista.
    """
    dias = _dia(["1 cebolla", "1 diente de ajo"],
                ["0.5 cebolla picada", "1 diente de ajo"])
    assert barrido(dias) == 0
    assert len(dias[0]["meals"][0]["ingredients_raw"]) == 2


def test_si_una_linea_del_DISPLAY_no_resuelve_la_comida_entera_se_salta(barrido):
    """Sin el conjunto completo de «vivos» no hay veredicto, solo un conjunto incompleto —
    y contra un conjunto incompleto TODO parece muerto. Fail-open explícito."""
    dias = _dia(["ingrediente que no está en el catálogo", "1 cebolla"],
                ["135 g de pechuga de pavo en lonjas/tiras", "1 cebolla"])
    assert barrido(dias) == 0, (
        "borró con el conjunto de vivos incompleto: así se vacía una lista entera")


def test_el_tope_por_comida_acusa_al_RESOLVEDOR_y_no_toca_nada(barrido, monkeypatch):
    """Si «muchas» líneas parecen muertas, lo que falló es el resolvedor.

    Un barrido sin tope convierte una regresión del resolvedor en una lista de compras vacía —
    y el usuario no tendría cómo saberlo. Se registra y no se toca nada.
    """
    import graph_orchestrator
    monkeypatch.setattr(graph_orchestrator, "RAW_DEAD_LINE_SWEEP_MAX_PER_MEAL", 1)
    dias = _dia(["1 cebolla"],
                ["135 g de pechuga de pavo en lonjas/tiras", "1 huevo", "1 cebolla"])
    assert barrido(dias) == 0
    assert len(dias[0]["meals"][0]["ingredients_raw"]) == 3, "tocó pese a superar el tope"


def test_jamas_vacia_raw(barrido):
    """Una lista de compras vacía es peor que una con una línea de más."""
    dias = _dia(["1 cebolla"], ["135 g de pechuga de pavo en lonjas/tiras"])
    barrido(dias)
    assert dias[0]["meals"][0]["ingredients_raw"], "vació ingredients_raw"


def test_es_idempotente(barrido):
    dias = _dia(["185 g de pechuga de pavo"],
                ["135 g de pechuga de pavo en lonjas/tiras", "185 g de pechuga de pavo"])
    assert barrido(dias) == 1
    assert barrido(dias) == 0, "la segunda pasada vuelve a tocar: no es idempotente"


def test_el_knob_lo_apaga(barrido, monkeypatch):
    import graph_orchestrator
    monkeypatch.setattr(graph_orchestrator, "RAW_DEAD_LINE_SWEEP_ENABLED", False)
    dias = _dia(["185 g de pechuga de pavo"],
                ["135 g de pechuga de pavo en lonjas/tiras", "185 g de pechuga de pavo"])
    assert barrido(dias) == 0
    assert len(dias[0]["meals"][0]["ingredients_raw"]) == 2


# ------------------------------------------------------- contrato estructural
def test_pregunta_al_resolvedor_de_COMPRAS_no_al_nutricional():
    """Es el punto entero del fix.

    El resolvedor nutricional ve UN pavo en las dos líneas — con él, este barrido no borraría
    nada. La identidad que importa es la que gasta dinero: `_parse_quantity`.
    """
    fn = next((n for n in ast.walk(ast.parse(_SRC))
               if isinstance(n, ast.FunctionDef) and n.name == "_barrer_lineas_muertas_de_raw"),
              None)
    assert fn is not None, "_barrer_lineas_muertas_de_raw desapareció"
    importados = {a.name for i in ast.walk(fn) if isinstance(i, ast.ImportFrom) for a in i.names}
    assert "_parse_quantity" in importados, (
        "no usa el resolvedor de compras; con el nutricional el barrido es un no-op")
    assert "macros_from_ingredient_string" not in ast.unparse(fn), (
        "usa el resolvedor nutricional, que para las dos líneas del pavo ve un solo alimento")


def test_corre_DESPUES_de_la_reciproca_en_los_DOS_sitios():
    """El orden es lo que impide que las dos guardas oscilen.

    `_reconcile_raw_missing_in_display` AÑADE al display la línea de raw que le falta; este
    barrido QUITA de raw la que el display no respalda. Invertidas, se pelean por la misma
    línea — dos guardas sobre la misma condición oscilan.
    """
    i_recip = [i for i in range(len(_SRC))
               if _SRC.startswith("_reconcile_raw_missing_in_display(days)", i)]
    i_barr = [i for i in range(len(_SRC))
              if _SRC.startswith("_barrer_lineas_muertas_de_raw(days)", i)]
    assert len(i_recip) >= 2, f"call sites de la recíproca: {len(i_recip)}"
    assert len(i_barr) >= 2, (
        f"el barrido está en {len(i_barr)} call site(s); la recíproca corre en "
        f"{len(i_recip)} — el que quede sin barrido sigue comprando de más")
    for a in i_barr:
        assert any(b < a for b in i_recip), (
            "hay un barrido que corre ANTES de toda invocación de la recíproca")
