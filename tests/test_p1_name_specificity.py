# -*- coding: utf-8 -*-
"""[P1-NAME-SPECIFICITY · 2026-09-06] El nombre dice el alimento específico y la lista el genérico.

    «Queso de hoja a la plancha con arroz cítrico»   →   lista: «30 g de queso»
    «Pastelitos… rellenos de chuleta de cerdo»       →   lista: «½ chuleta»

Es la clase de `nombre_no_corresponde` que quedaba tras materializar los cuatro fantasmas de
`P1-NAME-GHOST-GAPS`, y la que el juez describe como «el nombre declara 'gratinada con parmesano'
pero la lista solo contiene queso genérico». Espejo de `P1-RECIPE-POLISH-5`, que hace lo mismo en
la otra dirección.

**Solo se renombra el texto del alimento.** Cantidad, unidad y hint quedan byte a byte: no mueve
macros ni lista de compras, y el backstop de alérgenos lee el texto crudo, que solo gana precisión.

La medición que hizo falta para llegar aquí fue en tres rondas, y las tres correcciones están en
los negativos de este test:

  ronda 1 · 7 reescrituras, 3 malas: el calificador se tragaba el resto del nombre
            («60 g de queso de hoja **con vainitas**»).
  ronda 2 · el calificador se corta en el primer conector, pero «Wrap **integral** de ropa vieja»
            renombraba «1 tortilla integral» → la cabeza era un ADJETIVO.
  ronda 3 · la cabeza debe ABRIR un sintagma. Y la puntuación pegada no descalifica: sin ese
            `strip`, «…revoltillo, queso de hoja…» perdía un acierto legítimo por una coma.

Resultado contra 8 días de planes vivos: **6 de 6, cero falsos positivos.**
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _correr(nombre, ingredientes):
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "name": nombre,
                                  "ingredients": list(ingredientes),
                                  "recipe": ["Montaje: sirve."]}]}]
    cambios = go._specify_generic_lines_from_dish_name(days)
    return cambios, days[0]["meals"][0]["ingredients"]


# ── los casos reales de producción ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("nombre, ings, esperado", [
    ("Queso de hoja a la plancha con arroz cítrico y ensalada morada",
     ["30 g de queso", "50 g de arroz blanco crudo"], "30 g de queso de hoja"),
    ("Pastelitos de yuca horneados rellenos de chuleta de cerdo con ensalada fresca",
     ["½ chuleta", "2 huevos batido"], "½ chuleta de cerdo"),
    ("Wrap mañanero de tortilla con revoltillo, queso de hoja y Aguacate",
     ["15 g de queso", "2 huevos"], "15 g de queso de hoja"),
])
def test_la_linea_generica_hereda_el_nombre_del_plato(nombre, ings, esperado):
    cambios, salida = _correr(nombre, ings)
    assert len(cambios) == 1, f"no se precisó: {nombre[:60]}"
    assert salida[0] == esperado


def test_la_cantidad_y_el_hint_quedan_intactos():
    """Todo el permiso para tocar esto viene de que no mueve un solo número."""
    _, salida = _correr("Queso de hoja gratinado", ["30 g de queso (≈30 g)"])
    assert salida[0] == "30 g de queso de hoja (≈30 g)"


# ── ronda 1: el calificador se corta en el conector ──────────────────────────────────────────
def test_el_calificador_no_se_traga_el_resto_del_nombre():
    """«Queso de hoja con vainitas» dejaba «60 g de queso de hoja con vainitas»: el nombre del
    plato entero metido en una línea de la lista."""
    _, salida = _correr("Queso de hoja con vainitas al vapor", ["60 g de queso"])
    assert salida[0] == "60 g de queso de hoja"


# ── ronda 2: la cabeza no puede ser un adjetivo ──────────────────────────────────────────────
def test_un_adjetivo_no_es_cabeza_de_nada():
    """«Wrap **integral** de ropa vieja de res» renombraba «1 tortilla integral» a «tortilla
    integral de ropa vieja». `integral` modifica al wrap; no abre sintagma."""
    cambios, salida = _correr("Wrap integral de ropa vieja de res con vegetales criollos",
                              ["130 g de carne de res cocida", "1 tortilla integral"])
    assert cambios == [] and salida[1] == "1 tortilla integral"


# ── ronda 3: la puntuación abre sintagma ─────────────────────────────────────────────────────
def test_la_coma_no_descalifica():
    cambios, _ = _correr("Bowl criollo con revoltillo, queso de hoja y aguacate", ["15 g de queso"])
    assert len(cambios) == 1


# ── lo que NO se toca ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("nombre, ings", [
    # el calificador ya está en la lista
    ("Ensalada de aguacate con queso blanco", ["30 g de queso blanco", "1 aguacate"]),
    ("Bowl de yogurt griego con fresas", ["1 taza de yogurt griego", "50 g de fresas"]),
    # cabeza de formato, no de alimento
    ("Sopa de pollo con fideos", ["150 g de pollo", "40 g de fideos"]),
    ("Ensalada de atún con lechuga", ["100 g de atún", "2 tazas de lechuga"]),
    ("Crema de auyama", ["200 g de auyama"]),
    # la línea ya lleva su propio calificador
    ("Queso de hoja a la plancha", ["30 g de queso cottage"]),
])
def test_no_toca_lo_que_no_debe(nombre, ings):
    cambios, salida = _correr(nombre, ings)
    assert cambios == [] and salida == list(ings)


def test_dos_lineas_con_la_misma_cabeza_no_se_tocan():
    """Con dos quesos no se sabe cuál prometía el nombre. Renombrar el que toque por orden de
    aparición sería inventar."""
    cambios, _ = _correr("Queso de hoja gratinado", ["30 g de queso", "20 g de queso"])
    assert cambios == []


# ── contrato ─────────────────────────────────────────────────────────────────────────────────
def test_devuelve_el_contenido_no_un_contador():
    cambios, _ = _correr("Queso de hoja gratinado", ["30 g de queso"])
    assert len(cambios) == 1 and cambios[0][0] == "30 g de queso"


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setattr(go, "NAME_SPECIFICITY_ENABLED", False)
    cambios, salida = _correr("Queso de hoja gratinado", ["30 g de queso"])
    assert cambios == [] and salida == ["30 g de queso"]


def test_display_se_invalida():
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Queso de hoja gratinado",
                                  "ingredients": ["30 g de queso"], "recipe": ["Montaje."],
                                  "_display": {"en-US": {"ingredients": ["30 g of cheese"]}}}]}]
    go._specify_generic_lines_from_dish_name(days)
    assert "_display" not in days[0]["meals"][0]


def test_entrada_corrupta_es_no_op():
    assert go._specify_generic_lines_from_dish_name(None) == []
    assert go._specify_generic_lines_from_dish_name([{"meals": [{"ingredients": "no soy lista"}]}]) == []


def test_corre_donde_corre_el_pase_de_fantasmas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_specify_generic_lines_from_dish_name(") >= 4
