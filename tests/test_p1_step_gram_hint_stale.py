# -*- coding: utf-8 -*-
"""[P1-STEP-GRAM-HINT-STALE · 2026-09-06] El hint de báscula de un PASO envejece.

El LLM escribe «mide ¾ taza de pasta integral seca (130 g)» una sola vez, contra la lista de
ingredientes de ese instante. Después el motor reescala esas líneas —`rescale_ingredient_string`
escala el hint DE LA LÍNEA, nadie escala el del PASO—, las sustituye, las cuantiza y las recorta.
El paréntesis del paso se queda con el número viejo y acaba contradiciendo a la línea, a veces al
número que tiene tres palabras a la izquierda: «mide 85 g de queso (135 g)».

Medido sobre los 93 planes vivos con días (06-sep): 45 menciones en 23 planes con el hint en
contra de su propia línea, 10 pasos con DOS hints acumulados. Simulando el pase sobre esos mismos
planes: 34 de 93 (37 %) tenían al menos un hint que corregir.

El hint es SIEMPRE decorativo — macros y lista de compras salen de `ingredients[]` — así que
corregirlo o quitarlo no mueve ningún número del plan. Ese es justamente el permiso para tocarlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _plato(ingredientes, *pasos):
    return {"name": "plato de prueba", "ingredients": list(ingredientes), "recipe": list(pasos)}


# ── (a) el líder ya está en gramos: el paréntesis sobra ──────────────────────────────────────
@pytest.mark.parametrize("paso, esperado", [
    ("Mise en place: bate 3 huevos, mide 85 g de queso (135 g), porciona el resto.",
     "Mise en place: bate 3 huevos, mide 85 g de queso, porciona el resto."),
    ("Mise en place: corta 30 g de queso blanco (65 g) en tiras.",
     "Mise en place: corta 30 g de queso blanco en tiras."),
])
def test_lider_en_gramos_pierde_el_hint(paso, esperado):
    """Un peso seguido de otro peso distinto es una contradicción a tres palabras de distancia.
    Aunque coincidiera, repetirlo no informa: se quita."""
    m = _plato(["85 g de queso cottage", "30 g de queso blanco", "3 huevos"], paso)
    cambios = go._sync_recipe_step_gram_hints(m)
    assert m["recipe"][0] == esperado
    assert len(cambios) == 1 and cambios[0][0] == paso


# ── (b) líder en volumen: el hint se corrige contra la línea ─────────────────────────────────
def test_lider_en_volumen_el_hint_se_corrige():
    m = _plato(["¾ taza de pasta integral seca (130 g)", "55 g de queso blanco"],
               "Mise en place: mide ¾ taza de pasta integral seca (200 g), pica la cebolla.")
    go._sync_recipe_step_gram_hints(m)
    assert m["recipe"][0] == "Mise en place: mide ¾ taza de pasta integral seca (130 g), pica la cebolla."


def test_hint_ya_correcto_no_se_toca():
    """Dentro de la tolerancia del 15 % el texto queda BYTE a byte igual: el pase no reformatea."""
    paso = "Mise en place: mide ¾ taza de pasta integral seca (128 g), pica la cebolla."
    m = _plato(["¾ taza de pasta integral seca (130 g)"], paso)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"][0] == paso


# ── (c) hints acumulados por reescalados sucesivos ───────────────────────────────────────────
@pytest.mark.parametrize("paso, esperado", [
    ("Mise en place: desmenuza ¼ cucharadita de comino (60 g) (55 g) y reserva.",
     "Mise en place: desmenuza ¼ cucharadita de comino y reserva."),
    ("Mise en place: mide 1 g de semillas de calabaza (30 g) (115 g), pesa el resto.",
     "Mise en place: mide 1 g de semillas de calabaza, pesa el resto."),
])
def test_dos_hints_seguidos_se_colapsan(paso, esperado):
    """Un cuarto de cucharadita no pesa 55 g. Sin línea que respalde ninguno de los dos números,
    quedarse con uno al azar sería inventar: se van los dos."""
    m = _plato(["¼ cucharadita de comino", "0.89 g de semillas de calabaza"], paso)
    go._sync_recipe_step_gram_hints(m)
    assert m["recipe"][0] == esperado


# ── (d) lo que NO se puede desmentir se respeta ──────────────────────────────────────────────
def test_alimento_sin_gramos_en_la_lista_queda_intacto():
    """«⅓ taza de avena» no declara gramos en ninguna parte: el hint del paso es la ÚNICA
    referencia de báscula que tiene el usuario. Borrarlo por sistema empobrecería media receta."""
    paso = "Mise en place: mide 0.33 taza de avena (50 g) y 1.5 tazas de leche descremada."
    m = _plato(["⅓ taza de avena", "1.5 tazas de leche descremada"], paso)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"][0] == paso


def test_token_ambiguo_no_se_corrige():
    """Dos líneas comparten el token principal con pesos distintos → no hay forma de saber a cuál
    se refiere el paso. Misma regla anti-ambigüedad que `_sync_recipe_step_quantities`."""
    paso = "Mise en place: mide 1 taza de queso (300 g) y reserva."
    m = _plato(["85 g de queso cottage", "30 g de queso blanco"], paso)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"][0] == paso


# ── las notas deterministas nunca se tocan ───────────────────────────────────────────────────
def test_nota_de_seguridad_intacta():
    paso = "⚠️ Seguridad alimentaria: usa 40 g de mango (999 g) bien lavado."
    m = _plato(["40 g de mango en cubos"], paso)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"][0] == paso


# ── contrato del pase ────────────────────────────────────────────────────────────────────────
def test_devuelve_el_contenido_no_un_contador():
    """La lección de P1-SLOT-FLOOR-UNCLOSED: un guard que reporta cuántas veces actuó, sin decir
    sobre qué, no se puede auditar. El pase devuelve (antes, después) y el log los imprime."""
    m = _plato(["85 g de queso cottage"], "Mise en place: mide 85 g de queso (135 g).")
    cambios = go._sync_recipe_step_gram_hints(m)
    assert len(cambios) == 1
    antes, despues = cambios[0]
    assert "(135 g)" in antes and "(135 g)" not in despues


def test_el_knob_apaga_el_pase(monkeypatch):
    paso = "Mise en place: mide 85 g de queso (135 g)."
    monkeypatch.setattr(go, "RECIPE_STEP_GRAM_HINT_SYNC", False)
    m = _plato(["85 g de queso cottage"], paso)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"][0] == paso


def test_display_se_invalida_al_reescribir():
    """`_display[locale].recipe` espeja los pasos POR ÍNDICE: si el paso cambia y el espejo se
    queda, el usuario en inglés sigue leyendo el hint viejo. DELETE-on-write."""
    m = _plato(["85 g de queso cottage"], "Mise en place: mide 85 g de queso (135 g).")
    m["_display"] = {"en-US": {"recipe": ["Mise en place: measure 85 g of cheese (135 g)."]}}
    go._sync_recipe_step_gram_hints(m)
    assert "_display" not in m


def test_no_toca_nada_cuando_no_hay_hints():
    pasos = ["Mise en place: pica la cebolla y el ajo.",
             "Montaje: sirve caliente con el cilantro por encima."]
    m = _plato(["1 cebolla", "3 dientes de ajo"], *pasos)
    assert go._sync_recipe_step_gram_hints(m) == []
    assert m["recipe"] == pasos


def test_entrada_corrupta_es_no_op():
    """Fail-safe: el pase jamás puede tumbar una generación por un plato con forma rara."""
    for basura in ({}, {"recipe": "no soy una lista"}, {"recipe": [None, 42]},
                   {"recipe": ["Mise en place: pica la cebolla."], "ingredients": "no soy lista"}):
        antes = repr(basura)
        assert go._sync_recipe_step_gram_hints(dict(basura)) == []
        assert repr(basura) == antes, "el pase mutó una entrada que no supo leer"


def test_la_regla_a_no_depende_de_la_lista():
    """«85 g de queso (135 g)» se desmiente SOLO, sin consultar `ingredients[]` — por eso sigue
    corrigiéndose aunque la lista venga inservible. Es deliberado: la contradicción está dentro
    de la propia frase."""
    m = {"recipe": ["mide 85 g de queso (135 g)."], "ingredients": "no soy una lista"}
    assert go._sync_recipe_step_gram_hints(m) == [("mide 85 g de queso (135 g).",
                                                  "mide 85 g de queso.")]


# ── el enganche: corre desde el qty-sync, también cuando ese no reescribe nada ───────────────
def test_corre_desde_el_qtysync_con_cero_reescrituras():
    """El hint envejece por reescalados que ni siquiera tocan el texto del paso — «85 g de queso
    (135 g)» tiene el líder YA correcto. Si el sub-pase colgara de `if fixed:`, ese caso (el más
    frecuente de los medidos) no se arreglaría nunca."""
    m = _plato(["85 g de queso cottage"], "Mise en place: mide 85 g de queso (135 g).")
    n = go._sync_recipe_step_quantities(m)
    assert n >= 1
    assert m["recipe"][0] == "Mise en place: mide 85 g de queso."


def test_el_qtysync_no_pierde_sus_propias_reescrituras():
    """Regresión del enganche: `new_steps` se reasigna tras el sub-pase. Si se copiara mal, la
    corrección de cantidades del qty-sync (su trabajo original) se perdería al escribir."""
    m = _plato(["120 g de arroz blanco crudo"],
               "El Toque de Fuego: cocina 50 g de arroz durante 15 minutos.")
    go._sync_recipe_step_quantities(m)
    assert "120 g de arroz" in m["recipe"][0]
