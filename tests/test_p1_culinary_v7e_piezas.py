# -*- coding: utf-8 -*-
"""[P1-CULINARY-V7E-PIEZAS · 2026-09-07] «½ guineo» en la lista y «2 guineos» en el paso.

`V6` ya cubre esta dirección —el paso pide MÁS de lo que la lista compra, la que hace que el
usuario compre de menos— pero exige una palabra de UNIDAD. Medido:

    _v6_cuentas('½ diente de ajo')   -> {'Ajo': {('diente', 0.5)}}
    _v6_cuentas('½ guineo verde')    -> {}          <- conteo DESNUDO: nada que comparar
    _v6_cuentas('2 guineos verdes')  -> {}

Con los dos lados vacíos V6 no compara y el defecto pasa entero. `_v7_piezas` sí lee la pieza
desnuda, y ya rechaza «2 tazas» como medida — que era justo lo que hacía peligrosa esta
comparación.

Es el hueco más ancho medido en la jornada: **127 de 1.194 comidas vivas (10,6 %)**, trece veces
la prevalencia de V7d. Sobre las 140 comidas etiquetadas por el dueño dispara 16 veces, **las 16
en comidas que él marcó con defecto y cero en las que llamó correctas**, y sus notas describen
esta acusación palabra por palabra:

    «¾ de unidad en ingredientes frente a 2 unidades en la preparación»
    «Corregir pera: se declara ½ y se utiliza 1»
    «Unificar ajíes —1 frente a 2—, cebolla —1 frente a ½— y limones —2 frente a 1—»

Eso importa: la etiqueta del dueño es por COMIDA, no por acusación, así que una capa puede
acertar la comida por la razón equivocada. Aquí las notas confirman el motivo, no sólo el
veredicto.
"""
import pytest

import culinary_coherence as cc


def _fila(name, aliases=None):
    return {"name": name, "aliases": list(aliases or []), "prep_methods": ["ninguno"],
            "ready_to_eat": False}


CATALOGO = [
    _fila("Guineo verde", ["platano verde pequeno"]),
    _fila("Pera"),
    _fila("Kiwi"),
    _fila("Tomate"),
    _fila("Ajo"),
    _fila("Espinacas"),
    _fila("Tortilla integral"),
]


@pytest.fixture(scope="module")
def idx():
    return cc.build_culinary_index(CATALOGO)


def _meal(ings, pasos):
    return {"meal": "Desayuno", "name": "Sonda", "ingredients": ings, "recipe": pasos}


# ─────────────────────────────────────────────────────────────────────────────────────────
# Lo que DEBE ver.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_el_caso_del_mangu(idx):
    """«La preparación duplica el plátano» — la nota del dueño sobre una comida real.

    Es el caso que destapó el hueco: V6 callaba porque «½ guineo verde» no lleva unidad."""
    viols = cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["½ guineo verde", "1 huevo"],
        ["Mise en place: pela y aplasta 2 guineos verdes cocidos."]), idx)
    assert [v["food"] for v in viols] == ["Guineo verde"]
    assert "2" in viols[0]["detail"] and "0.5" in viols[0]["detail"]


@pytest.mark.parametrize("lista,paso", [
    ("½ pera mediana", "corta 1 pera en cubos"),
    ("¾ kiwi", "rebana 2 kiwis"),
    ("1 tomate", "pica 3 tomates"),
])
def test_las_formas_que_el_dueno_marco(lista, paso, idx):
    assert cc._v7e_paso_pide_mas_piezas({}, _meal([lista], [paso]), idx)


# ─────────────────────────────────────────────────────────────────────────────────────────
# Lo que NO debe ver — cada uno cierra una vía distinta de falso positivo.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_mencionar_el_mismo_alimento_en_dos_pasos_no_es_pedir_el_doble(idx):
    """LA decisión de diseño de esta capa: se compara POR PASO, nunca la suma.

    En la dirección contraria (V7a: la lista compra de más) sumar los pasos es seguro, porque
    más menciones sólo reducen el hueco. Aquí sumar sería un falso positivo garantizado:
    «corta 1 tomate» y luego «añade 1 tomate» daría 2 contra 1 en una receta impecable."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["1 tomate"],
        ["Mise en place: corta 1 tomate en cubos.",
         "Montaje: añade 1 tomate por encima."]), idx) == []


def test_una_medida_no_es_una_pieza(idx):
    """«2 tazas de espinacas» son tazas, no dos espinacas.

    `_v7_piezas` descarta la cuenta cuando la medida va DELANTE del alimento. Sin ese guard,
    cada especia medida en cucharadas sería un disparo."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["½ taza de espinacas"], ["Añade 2 tazas de espinacas."]), idx) == []


def test_no_pisa_el_territorio_de_v6(idx):
    """Cuando el texto trae unidad («1 diente de ajo»), `_v7_piezas` ve la medida delante del
    alimento y no cuenta la pieza — así que ese caso lo sigue reportando V6 y sólo V6.

    Dos capas sobre la misma condición oscilan; ya pasó en este repo con dos guardas de nevera."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["½ diente de ajo"], ["Pica 1 diente de ajo."]), idx) == []


def test_el_paso_que_usa_lo_mismo_o_menos_calla(idx):
    """Usar menos es legítimo (y si sobra mucho, ya lo dice V7a)."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["3 tomates"], ["Pica 2 tomates."]), idx) == []
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["2 tomates"], ["Pica 2 tomates."]), idx) == []


def test_un_alimento_ausente_de_la_lista_es_territorio_de_v5(idx):
    """Si el paso lo usa y la lista no lo tiene, el defecto es el FANTASMA, no la cantidad."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["1 huevo"], ["Pica 2 tomates."]), idx) == []


def test_la_tolerancia_absorbe_el_redondeo(idx):
    """«⅓» y «0.33» son la misma cantidad. Misma tolerancia que V6."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(
        ["⅓ pera"], ["Usa 0.34 pera."]), idx) == []


def test_sin_lista_o_sin_pasos_no_lanza(idx):
    """Fail-open total: esta capa jamás puede tumbar una generación."""
    assert cc._v7e_paso_pide_mas_piezas({}, _meal([], ["Pica 2 tomates."]), idx) == []
    assert cc._v7e_paso_pide_mas_piezas({}, _meal(["1 tomate"], []), idx) == []
    assert cc._v7e_paso_pide_mas_piezas({}, {"ingredients": None, "recipe": None}, idx) == []


# ─────────────────────────────────────────────────────────────────────────────────────────
# Anclas.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_v7e_esta_encadenada_en_el_escaner():
    """Una capa que existe y nadie invoca es el modo de fallo de esta misma jornada."""
    import inspect

    assert "_v7e_paso_pide_mas_piezas" in inspect.getsource(cc.culinary_contract_scan)


def test_v7e_compara_por_paso_y_no_la_suma():
    """Ancla textual de la decisión que sostiene la capa: el bucle recorre `pasos` y compara
    dentro, en vez de agregar los pasos antes de comparar."""
    import inspect

    src = inspect.getsource(cc._v7e_paso_pide_mas_piezas)
    assert "for paso in pasos:" in src, "V7e dejó de comparar paso a paso"
