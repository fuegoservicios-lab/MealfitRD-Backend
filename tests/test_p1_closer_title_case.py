# -*- coding: utf-8 -*-
"""[P1-CLOSER-TITLE-CASE · 2026-09-06] El añadido del cerrador iba SIEMPRE en Title Case.

    Arepita de maíz rellena de ricotta, tomate y espinacas con Soya Texturizada
    ─────────────── el modelo, en frase normal ───────────────  ── el cerrador ──

Medido sobre los 84 planes vivos con ≥3 días: **el 80,4 % de los títulos que escribe el modelo van
en frase normal** («Batata dominicana con huevo pochado y melón fresco») y solo el 19,6 % en Title
Case («Carne de Res y Arroz»). El cerrador capitalizaba siempre, así que la costura se leía a simple
vista en **129 títulos de 57 planes**.

Un solo sitio produce las tres formas que se ven en producción —«con Soya Texturizada» al final,
«con Maní,» al medio, «mango, Lechosa y» dentro de la enumeración—: cada pase del cerrador añade un
alimento capitalizado y la reescritura de enumeración (P1-CLOSER-LIGHT-SLOT-NO-MEAT) mueve los
anteriores hacia el medio. Arreglar `_reflect_added_protein_in_name` las arregla las tres.

**Por qué no basta con minusculizar.** `capitalize()` destruía la caja del catálogo en las DOS
direcciones. Los 5 nombres propios del catálogo de 347 filas —«Nuez de Castilla», «Coles de
Bruselas», «Galletas Graham», «Flor de Jamaica», «Harina de Negrito»— la necesitan. La regla es
copiar la convención del título anfitrión y, cuando va en frase normal, minusculizar **solo la
primera palabra**: el catálogo es el SSOT de cómo se escribe un alimento.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from graph_orchestrator import (  # noqa: E402
    _food_display_for_title as display,
    _titulo_en_title_case as es_title,
)

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
# [P1-CLOSER-TITLE-CASE] la morfologia vive en su propio modulo desde que el techo de lineas del
# orquestador pidio extraerlo; el call site sigue en graph_orchestrator.
_NAMING = (_BACKEND / "dish_naming.py").read_text(encoding="utf-8")


# ── leer la convención del anfitrión ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("titulo", [
    "Batata dominicana con huevo pochado y melón fresco",
    "Pasta integral criolla con lentejas, tomate y espinacas",
    "Arepita de maíz rellena de ricotta, tomate y espinacas",
    "Batido-bowl cítrico de lechosa, mandarina, avena y mantequilla de maní",
    "Vaso instantáneo de yogur, lechosa y mantequilla de maní",
])
def test_reconoce_la_frase_normal(titulo):
    """Títulos vivos del modelo: son el 80,4 % de la base."""
    assert not es_title(titulo)


@pytest.mark.parametrize("titulo", [
    "Bowl Power de Batata Crujiente, Huevo y Mango",
    "Arroz Criollo de Pollo con Gandules, Apio y Espinaca",
    "Batata Rellena de Queso Fresco y Espinaca al Airfryer",
    "Avena con Frutas y Semillas",
    "Carne de Res y Arroz",
])
def test_reconoce_el_title_case(titulo):
    """También vivos: el 19,6 % restante. Los conectores en minúscula NO cuentan en contra."""
    assert es_title(titulo)


@pytest.mark.parametrize("titulo", ["", "Avena", "Bowl", None, "   "])
def test_ante_la_duda_se_queda_como_estaba(titulo):
    """Sin palabra significativa tras la primera no hay evidencia. El cambio actúa solo cuando la
    hay: un título corto conserva la conducta previa (capitalizar)."""
    assert es_title(titulo)


# ── la caja del añadido ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("host,food,esperado", [
    # el caso vivo del plan c5880d96
    ("Arepita de maíz rellena de ricotta, tomate y espinacas", "Soya texturizada", "soya texturizada"),
    ("Batata dominicana con huevo pochado", "Yogurt griego entero", "yogurt griego entero"),
    ("Pasta integral criolla con lentejas", "Queso ricotta", "queso ricotta"),
    # anfitrión en Title Case: se capitaliza como antes, conectores abajo
    ("Carne de Res y Arroz", "Soya texturizada", "Soya Texturizada"),
    ("Bowl Power de Batata Crujiente, Huevo y Mango", "carne de res", "Carne de Res"),
])
def test_el_anadido_copia_la_caja_del_anfitrion(host, food, esperado):
    assert display(food, host) == esperado


@pytest.mark.parametrize("propio,esperado", [
    ("Nuez de Castilla", "nuez de Castilla"),
    ("Coles de Bruselas", "coles de Bruselas"),
    ("Galletas Graham", "galletas Graham"),
    ("Flor de Jamaica", "flor de Jamaica"),
    ("Harina de Negrito", "harina de Negrito"),
])
def test_los_5_nombres_propios_del_catalogo_conservan_su_mayuscula(propio, esperado):
    """Las 347 filas del catálogo tienen exactamente 5 nombres con mayúscula interna, y las 5 son
    nombres propios. Minusculizar palabra por palabra los habría roto; `capitalize()` los rompía en
    la otra dirección («Nuez De Castilla»). Solo se toca la PRIMERA palabra."""
    assert display(propio, "Ensalada fresca de tomate y lechuga") == esperado
    # …y dentro de un título en Title Case tampoco se degradan
    assert display(propio, "Arroz Criollo de Pollo con Gandules") == propio


@pytest.mark.parametrize("basura", ["", "   ", None])
def test_fail_safe_ante_basura(basura):
    assert display(basura, "Lo que sea") == ""


def test_el_conector_sigue_en_minuscula():
    """`de/con/y` no se capitalizan en ninguna de las dos ramas — la regla que ya defendía
    P2-DISH-COHERENCE-NAMEFIX («Carne de Res», no «Carne De Res»)."""
    assert display("carne de res", "Bowl Power de Batata Crujiente y Mango") == "Carne de Res"
    assert display("Carne de res", "Batata dominicana con huevo") == "carne de res"


# ── anclaje en el fuente ──────────────────────────────────────────────────────────────────────
def test_el_call_site_ya_no_capitaliza_a_ciegas():
    """Si alguien reintroduce el `capitalize()` incondicional, la costura vuelve."""
    i = _SRC.index("def _reflect_added_protein_in_name(")
    cuerpo = _SRC[i:i + 6000]
    assert "_food_display_for_title(pname, name)" in cuerpo
    assert "else _NAME_STOPWORDS" not in cuerpo
    assert 'w.capitalize()\n                          for w in pname.split()' not in cuerpo


def test_la_caja_la_decide_el_anfitrion_no_el_bloque():
    """El helper recibe el título: sin él, la decisión no puede depender del anfitrión."""
    i = _NAMING.index("def _food_display_for_title(")
    cuerpo = _NAMING[i:i + 1500]
    assert "_titulo_en_title_case(host)" in cuerpo
