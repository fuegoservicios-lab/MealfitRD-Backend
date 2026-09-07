# -*- coding: utf-8 -*-
"""[P1-CULINARY-V7D-MASA · 2026-09-07] El espejo en MASA que ninguna capa cubría.

`V6` cubre «el paso pide MÁS que la lista». `V7a` cubre lo contrario —la lista compra de más—
pero SÓLO en piezas contables, porque `_v7_piezas` descarta a propósito todo lo que lleve unidad
de medida: «2 cucharadas de cilantro» son cucharadas, no piezas, y contarlas convertiría cada
especia en un falso positivo.

El resultado era un hueco de forma exacta: **«420 ml de leche» en la lista y «250 ml de leche»
en el paso no lo veía nadie** — ni V4, ni V6, ni V7a. Lo destapó una nota del dueño sobre una
avena cremosa; verificado después contra las tres capas, las tres callaban.

Medido sobre la flota ANTES de escribir la capa: 10 de 1.194 comidas (0,8 %), y **siete son la
misma forma** —leche de avena comprada a 340-545 ml y usada a 200-250—, lo que apunta a un sesgo
del generador y no a ruido. Prevalencia comparable a la de `V7b` (7 de 1.194), que ya está
desplegado. Sobre las 140 comidas etiquetadas por el dueño: +2 disparos, los dos sobre comidas
que él marcó con defecto, **cero falsos positivos nuevos**.
"""
import pytest

import culinary_coherence as cc


def _fila(name, aliases=None):
    return {"name": name, "aliases": list(aliases or []), "prep_methods": ["ninguno"],
            "ready_to_eat": False}


CATALOGO = [
    _fila("Leche descremada", ["leche descremada"]),
    _fila("Avena", ["avena en hojuelas"]),
    _fila("Granola"),
    _fila("Auyama", ["calabaza criolla"]),
    _fila("Cilantro"),
    _fila("Queso blanco", ["queso blanco fresco"]),
]


@pytest.fixture(scope="module")
def idx():
    return cc.build_culinary_index(CATALOGO)


def _meal(ings, pasos):
    return {"meal": "Desayuno", "name": "Sonda", "ingredients": ings, "recipe": pasos}


# ─────────────────────────────────────────────────────────────────────────────────────────
# Lo que DEBE ver.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_el_caso_que_lo_motivo(idx):
    """«Leche declarada en 420 ml y utilizada en 250 ml, sin destino para el resto» — la nota
    del dueño, palabra por palabra, sobre una comida real."""
    viols = cc._v7d_masa_sobrante({}, _meal(
        ["45 g de avena", "420 ml de leche descremada", "30 g de queso blanco fresco"],
        ["Mise en place: mide 45 g de avena y 250 ml de leche descremada.",
         "Cocina la avena con la leche descremada removiendo hasta que espese."]), idx)
    assert [v["food"] for v in viols] == ["Leche descremada"]
    assert "420" in viols[0]["detail"] and "250" in viols[0]["detail"]


@pytest.mark.parametrize("unidad,factor", [("ml", 1), ("g", 1), ("gr", 1)])
def test_reconoce_las_unidades_de_masa_y_volumen(unidad, factor, idx):
    viols = cc._v7d_masa_sobrante({}, _meal(
        [f"400 {unidad} de avena"], [f"Usa 100 {unidad} de avena."]), idx)
    assert len(viols) == 1


def test_litros_y_kilos_se_convierten_antes_de_comparar(idx):
    """1 l de leche contra 250 ml usados: sin convertir, «1» parecería MENOR que «250»."""
    viols = cc._v7d_masa_sobrante({}, _meal(
        ["1 litro de leche descremada"], ["Anade 250 ml de leche descremada."]), idx)
    assert len(viols) == 1
    assert "1000" in viols[0]["detail"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Lo que NO debe ver — cada uno es una vía documentada al falso positivo.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_un_paso_que_no_cuantifica_no_contradice_a_la_lista(idx):
    """«Cocina la avena con la leche descremada» es una instrucción normal, no una declaración
    de cantidad. Exigir la cifra en los DOS lados es lo que separa esta capa de V3, que ya
    cubre el ingrediente que ningún paso menciona.

    Sin este guard, V7d dispararía en casi toda receta bien escrita: la prosa de cocina no
    repite gramos, y eso es una virtud del texto, no un defecto del plato."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["420 ml de leche descremada"],
        ["Cocina la avena con la leche descremada hasta que espese."]), idx) == []


def test_repartir_un_ingrediente_entre_dos_pasos_es_legitimo(idx):
    """«250 ml ahora, 170 ml al final» suma 420: no sobra nada.

    Por eso `_v7d_masas` suma a lo largo de TODO el texto en vez de mirar paso a paso."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["420 ml de leche descremada"],
        ["Anade 250 ml de leche descremada y remueve.",
         "Termina con 170 ml de leche descremada."]), idx) == []


def test_una_diferencia_pequena_en_absoluto_no_vale_una_alerta(idx):
    """100 g -> 75 g falla la tolerancia del 25 % pero son 25 gramos: no cambia una compra.

    Los dos umbrales son necesarios y ninguno sobra. Sólo el proporcional marcaría esto;
    sólo el absoluto dejaría pasar 400 g -> 320 g."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["100 g de avena"], ["Usa 75 g de avena."]), idx) == []


def test_dentro_de_la_tolerancia_proporcional_calla(idx):
    """400 g -> 320 g son 80 gramos de diferencia, pero sólo un 20 %: redacción, no sobrante."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["400 g de avena"], ["Usa 320 g de avena."]), idx) == []


def test_si_el_paso_usa_MAS_que_la_lista_esto_calla(idx):
    """Ese es el territorio de V6. Dos capas sobre la misma condición oscilan — ya pasó en este
    repo con dos guardas de nevera."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["200 ml de leche descremada"], ["Anade 400 ml de leche descremada."]), idx) == []


def test_un_alimento_ambiguo_no_se_cuenta(idx):
    """Si el texto no identifica UN alimento del catálogo, no hay nada que comparar.

    Contar sobre una identidad adivinada es peor que no contar: la alarma sería sobre un
    alimento que el usuario no compró."""
    assert cc._v7d_masa_sobrante({}, _meal(
        ["420 ml de zumo de dragon"], ["Usa 100 ml de zumo de dragon."]), idx) == []


def test_sin_lista_o_sin_pasos_no_lanza(idx):
    """Fail-open total: esta capa jamás puede tumbar una generación."""
    assert cc._v7d_masa_sobrante({}, _meal([], ["Usa 100 g de avena."]), idx) == []
    assert cc._v7d_masa_sobrante({}, _meal(["100 g de avena"], []), idx) == []
    assert cc._v7d_masa_sobrante({}, {"ingredients": None, "recipe": None}, idx) == []


# ─────────────────────────────────────────────────────────────────────────────────────────
# Anclas: que un renombre o un desconectado falle aquí, no en producción.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_v7d_esta_encadenada_en_el_escaner():
    """Una capa que existe y nadie invoca es el modo de fallo de esta misma jornada: el sello de
    PlanPolicy tenía 25 tests en verde y la cadena muerta en la única vía que genera planes."""
    import inspect

    src = inspect.getsource(cc.culinary_contract_scan)
    assert "_v7d_masa_sobrante" in src, "V7d no está encadenada en culinary_contract_scan"


def test_los_dos_umbrales_siguen_existiendo():
    """Anclaje textual: quitar cualquiera de los dos reabre una vía de falso positivo distinta."""
    assert 0 < cc._V7D_TOLERANCIA < 1
    assert cc._V7D_MIN_GRAMOS > 0
