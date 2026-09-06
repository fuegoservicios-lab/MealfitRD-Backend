# -*- coding: utf-8 -*-
"""[P1-COLD-DAIRY-TECHNIQUE · 2026-09-06] «Cuece el yogur griego en agua hirviendo 8 minutos.»

Es la receta del **huevo duro** con la palabra cambiada. Una sustitución reemplazó el alimento
dentro de un paso cuya *técnica* pertenecía al original, y el resultado llega tal cual al usuario:

    «cuece las 3 yogurt griego entero en agua hirviendo durante 8 minutos, escúrrelas, pásalas
     por agua fría, pélalas y córtalas en cuartos»

El juez lo reporta como `tecnica_impropia` — 15 violaciones en 7 días, su tercera bolsa — y es el
texto más absurdo que llega a producción.

**La detección es todo el fix.** La primera sonda solo pedía que el verbo y el lácteo compartieran
cláusula: marcó 29 de 93 planes y casi todo era inocente («pela y separa 1 guineo, y mide… el
yogurt»). La descarté en vez de fiarme de ella. Esta exige que el lácteo sea el **objeto** del
verbo —verbo + artículo/cantidad opcional + hasta dos palabras + lácteo— y da **4 aciertos de 4**
sobre 400 planes vivos.

Se borra desde el verbo hasta el final de la frase, no solo el verbo: en los cuatro casos el
resto de la frase sigue hablando de la técnica imposible («escúrrelas… pélalas…») y dejarlo sería
cambiar una instrucción absurda por media. La sustitución **no** se revierte — puede venir de una
alergia, y arreglar el texto es barato mientras que devolver el alimento no lo es.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _correr(*pasos, ingredientes=("1 taza de yogurt griego",)):
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "prueba",
                                  "ingredients": list(ingredientes), "recipe": list(pasos)}]}]
    cambios = go._neutralize_cold_dairy_technique(days)
    return cambios, days[0]["meals"][0]["recipe"]


# ── los cuatro casos reales de producción ────────────────────────────────────────────────────
@pytest.mark.parametrize("paso", [
    "Mise en place: cuece las 3 yogurt griego entero en agua hirviendo durante 8 minutos, "
    "escúrrelas, pásalas por agua fría, pélalas y córtalas en cuartos.",
    "Mise en place: mide la quinoa cocida, pela y corta la lechosa en cubos, y cocina yogurt "
    "griego entero duro en agua hirviendo durante 10 minutos.",
    "Mise en place: corta 280 g de lechosa, mide la leche descremada y la mantequilla de maní, "
    "y pela el yogur griego previamente cocido.",
    "El Toque de Fuego: cocina el yogur griego en una sartén antiadherente a fuego medio durante "
    "3-4 minutos, removiendo hasta que estén cuajadas.",
])
def test_los_cuatro_casos_vivos_se_neutralizan(paso):
    cambios, rec = _correr(paso)
    assert len(cambios) == 1, f"no se detectó: {paso[:70]}"
    assert "incorpora el yogur frío" in rec[0]
    for imposible in ("hirviendo", "pélalas", "sartén", "previamente cocido"):
        assert imposible not in rec[0], f"quedó la técnica imposible {imposible!r}: {rec[0]}"


def test_lo_que_sobrevive_de_la_frase_es_lo_legitimo():
    """El fix borra desde el verbo, no la frase entera: lo que venía antes es una instrucción
    buena y perderla sería cambiar un defecto por otro."""
    _, rec = _correr("Mise en place: mide la quinoa cocida, pela y corta la lechosa en cubos, "
                     "y cocina yogurt griego entero duro en agua hirviendo durante 10 minutos.")
    assert "mide la quinoa cocida" in rec[0] and "corta la lechosa en cubos" in rec[0]


# ── la precisión, que es donde falló la primera versión ──────────────────────────────────────
@pytest.mark.parametrize("paso", [
    # el verbo y el lácteo comparten cláusula pero el lácteo NO es el objeto
    "Mise en place: pela y separa 1 guineo, y mide el yogurt griego entero y la canela.",
    "El Toque de Fuego: cocina el huevo en la sartén 3 minutos y sirve con el yogur griego aparte.",
    "Mise en place: tuesta el pan y unta el queso cottage encima.",
    # un lácteo que SÍ admite calor no entra en la lista fría
    "El Toque de Fuego: gratina el queso mozzarella hasta dorar.",
    # sin lácteo
    "El Toque de Fuego: cuece los huevos en agua hirviendo durante 8 minutos y pélalos.",
])
def test_no_toca_lo_inocente(paso):
    """La primera sonda marcó 29 de 93 planes con esta clase de frases. Si este test empieza a
    fallar, el detector volvió a mirar la cláusula en vez del objeto."""
    cambios, rec = _correr(paso)
    assert cambios == [] and rec == [paso]


# ── contrato ─────────────────────────────────────────────────────────────────────────────────
def test_devuelve_el_contenido_no_un_contador():
    cambios, _ = _correr("El Toque de Fuego: hornea el yogur griego 10 minutos.")
    assert len(cambios) == 1
    antes, despues = cambios[0]
    assert "hornea" in antes and "hornea" not in despues


def test_la_nota_de_seguridad_no_se_toca():
    nota = "⚠️ Seguridad alimentaria: cuece el yogur griego… (nota determinista)"
    cambios, rec = _correr(nota)
    assert cambios == [] and rec == [nota]


def test_el_knob_lo_apaga(monkeypatch):
    paso = "El Toque de Fuego: hornea el yogur griego 10 minutos."
    monkeypatch.setattr(go, "COLD_DAIRY_TECHNIQUE_ENABLED", False)
    cambios, rec = _correr(paso)
    assert cambios == [] and rec == [paso]


def test_display_se_invalida():
    """`_display[locale].recipe` espeja los pasos por índice: si el paso cambia y el espejo se
    queda, el usuario en inglés sigue leyendo que hierva el yogur."""
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "p",
                                  "ingredients": ["1 taza de yogurt griego"],
                                  "recipe": ["El Toque de Fuego: hornea el yogur griego 10 min."],
                                  "_display": {"en-US": {"recipe": ["Bake the greek yogurt."]}}}]}]
    go._neutralize_cold_dairy_technique(days)
    assert "_display" not in days[0]["meals"][0]


def test_entrada_corrupta_es_no_op():
    assert go._neutralize_cold_dairy_technique(None) == []
    assert go._neutralize_cold_dairy_technique([]) == []
    assert go._neutralize_cold_dairy_technique([{"meals": [{"recipe": "no soy lista"}]}]) == []


def test_corre_donde_corre_el_pase_de_fantasmas():
    """Los tres sitios donde ya se materializan los ghosts son los tres que ven el plan completo.
    Colgarlo de otro sitio dejaría fuera alguna superficie — que es como el guard de coherencia
    acabó cubriendo un camino de dos."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_neutralize_cold_dairy_technique(") >= 4, (
        "el neutralizador dejó de correr en alguno de los sitios del pase de fantasmas")
