"""[P1-BATIDO-ADJETIVO · 2026-07-26] "batido" es sustantivo Y adjetivo.

`_NO_COOK_BLENDED` se matchea por SUBCADENA sobre el nombre del plato para decidir si la
preparación es un licuado. Pero en es-DO *batido* es las dos cosas:

    "Batido de Guineo"            → sustantivo: SÍ es un licuado
    "Queso Crema **Batido**"      → adjetivo: queso crema MONTADO, plato de cuchara

Medido en el plan vivo `0afa0ed5`, 2 de 12 comidas quedaron con una receta que se contradice
a sí misma porque el closer insertó el paso de licuadora:

    "Tostadas de Pan Integral con Crema de Queso Crema Batido y Mango"
        paso 2: "Tuesta las rebanadas… Agrega yogurt a la licuadora y licúa hasta integrar."
        ← no hay licuadora en toda la receta; es una tostada

    "Maní y Lechosa Fresco con Queso Crema Batido y Yogurt"
        paso 2: "Agrega yogurt a la licuadora y licúa hasta integrar."
        paso 3: "Sirve el queso crema batido en un bowl, coloca lechosa encima…"
        ← el paso 2 licúa y el 3 sirve en bowl

Es la misma familia que `"sal"`⊂`"salsa"`, `"pollo"`⊂`"repollo"` y `"ajo"`⊂`"abajo"`: una
palabra de comida que también es palabra común. Se descuentan las colocaciones adjetivales
antes de buscar el sustantivo, igual que hace `_SLOT_RICE_EXCLUDE` con "harina de arroz".

⚠️ El cálculo estaba DUPLICADO en dos sitios (paso del closer y paso del complemento) y
ninguno descontaba el uso adjetival. Ahora ambos llaman a `_name_suggests_blended` — dos
copias del mismo predicado divergen, que es la lección de P1-PANTRY-GATE-SSOT.

tooltip-anchor: P1-BATIDO-ADJETIVO
"""
from __future__ import annotations

import inspect

import pytest

import graph_orchestrator as g


# ───────────── 1. los casos vivos ─────────────

@pytest.mark.parametrize("nombre", [
    "Tostadas de Pan Integral con Crema de Queso Crema Batido y Mango",
    "Maní y Lechosa Fresco con Queso Crema Batido y Yogurt",
])
def test_queso_crema_batido_no_es_un_licuado(nombre):
    assert g._name_suggests_blended(nombre) is False


@pytest.mark.parametrize("nombre", [
    "Merengue con Clara Batida",
    "Panqueques con Huevos Batidos",
    "Postre de Nata Batida",
])
def test_otras_colocaciones_adjetivales(nombre):
    assert g._name_suggests_blended(nombre) is False


# ───────────── 2. lo que SÍ es un licuado ─────────────

@pytest.mark.parametrize("nombre", [
    "Batido Refrescante de Guineo y Chinola con Linaza y Yogurt",
    "Batido de Mamey",
    "Smoothie Verde de Espinaca",
    "Jugo de Chinola Natural",
    "Malteada de Lechosa",
    "Licuado Tropical",
])
def test_los_licuados_de_verdad_siguen_detectandose(nombre):
    """Si esto se rompe, los batidos reales pierden su wording natural y vuelven a leer
    'Incorpora X a la preparación', que fue el bug que P1-BLENDER-STEP-COHERENCE cerró."""
    assert g._name_suggests_blended(nombre) is True


def test_un_batido_que_ademas_lleva_queso_crema_batido_sigue_siendo_batido():
    """El descuento es de la COLOCACIÓN, no de la palabra: si el plato es un batido Y encima
    lleva queso crema batido, sigue siendo un batido."""
    assert g._name_suggests_blended("Batido de Guineo con Queso Crema Batido") is True


# ───────────── 3. bordes y fail-open ─────────────

@pytest.mark.parametrize("valor", ["", None, 12345])
def test_entradas_raras_no_rompen(valor):
    assert isinstance(g._name_suggests_blended(valor), bool)


def test_acentos_y_mayusculas():
    assert g._name_suggests_blended("BATIDO DE LECHOSA") is True
    assert g._name_suggests_blended("Queso Crema BATIDO") is False


# ───────────── 4. SSOT: las dos copias consultan el mismo predicado ─────────────

def test_las_otras_lecturas_de_la_tupla_son_de_no_cook_no_de_wording():
    """⚠️ Hay MÁS sitios que leen `_NO_COOK_BLENDED`, y está bien: responden otra pregunta.

    Dos decisiones distintas comparten la tupla:
      - "¿es un LICUADO?" (wording del paso) → necesita descontar el adjetivo, porque un
        queso crema batido no se licúa.
      - "¿es un plato SIN COCCIÓN?" (`no_cook`) → ahí el match es CORRECTO: un queso crema
        batido tampoco se cocina.

    Mi primera versión de este test exigía cero lecturas directas y fallaba señalando código
    sano. Se ancla lo que importa: que las lecturas restantes asignen a `no_cook`, no que no
    existan.
    """
    from pathlib import Path
    src = Path(g.__file__).resolve().read_text(encoding="utf-8").split("\n")
    sueltas = [l.strip() for l in src if "for b in _NO_COOK_BLENDED" in l]
    dentro_helper = inspect.getsource(g._name_suggests_blended)
    externas = [l for l in sueltas if l not in dentro_helper]
    assert externas, "si desaparecen todas, este test ya no protege nada"
    for l in externas:
        assert "no_cook" in l or "name_low" in l or "in text" in l, (
            f"lectura directa que NO es de no-cook: {l!r} — si decide wording de licuado "
            "debe pasar por _name_suggests_blended"
        )


def test_ambos_callsites_usan_el_helper():
    from pathlib import Path
    src = Path(g.__file__).resolve().read_text(encoding="utf-8")
    assert src.count("_name_suggests_blended(meal.get(\"name\", \"\")") == 2, \
        "los dos sitios que deciden el wording de licuado deben llamar al helper"
