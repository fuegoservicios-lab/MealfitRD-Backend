"""[P1-VEG-VOLUME-TOKENS · 2026-07-26] Una cabeza entera de coliflor para una persona.

El cap de realismo de volumen vegetal (250 g/línea) existía pero su lista de alimentos tenía
**cinco**: pepino, berro, rábano, rabanito y apio. Cualquier otra hortaliza pasaba por debajo
del techo genérico de 600 g. Plan vivo `0afa0ed5`, cena:

    "1 cabeza de coliflor mediana (499g, en floretes)"   ← para 1 persona

## Lo que dice la medición (y corrige mi propia descripción)

Yo hablé de "porciones inverosímiles" en plural. Medido sobre 60 planes, contando los gramos
**del paréntesis** —porque estas líneas vienen en medidas caseras ("1 cabeza", "3¾ tazas") y un
parser de gramos no las ve, que es como se me escapó la primera vez—:

    líneas ≥ 450 g en 60 planes:  DOS
        560 g  "3¾ tazas de auyama en cubos (560 g)"
        499 g  "1 cabeza de coliflor mediana (499g, en floretes)"

**No es sistémico.** Pero el cap ya existía y solo le faltaban los alimentos, así que el
arreglo es de una línea y sin riesgo.

## Por qué la auyama se queda fuera

Es un víver almidonado que hace de **base de carbohidrato** del plato (puré de auyama), no
relleno de volumen. Recortarla de 560 a 250 g le quita los carbos a la cena y obliga al motor
de macros a recomponer. Eso es un cambio con consecuencias clínicas que merece su propia
medición, no un token más en una tupla.

tooltip-anchor: P1-VEG-VOLUME-TOKENS
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g


def _cubierto(nombre: str) -> bool:
    import re
    return any(re.search(r"\b" + t, nombre.lower()) for t in g._REALISM_VOLUME_VEG_TOKENS)


# ───────────── 1. el caso vivo ─────────────

def test_la_coliflor_ahora_esta_cubierta():
    assert _cubierto("1 cabeza de coliflor mediana en floretes")


def test_el_cap_recorta_499_a_250():
    """Aritmética del recorte, con el valor real del plan."""
    cur_g = 499.0
    assert cur_g > float(g.REALISM_VEG_VOLUME_CAP_G)
    factor = float(g.REALISM_VEG_VOLUME_CAP_G) / cur_g
    assert round(cur_g * factor) == g.REALISM_VEG_VOLUME_CAP_G == 250


# ───────────── 2. las que se añadieron ─────────────

@pytest.mark.parametrize("linea", [
    "300 g de coliflor en floretes",
    "280 g de brocoli al vapor",
    "400 g de repollo picado",
    "280 g de calabacin en medias lunas",
])
def test_hortalizas_acuosas_cubiertas(linea):
    assert _cubierto(linea)


# ───────────── 3. las que NO se tocaron, a propósito ─────────────

def test_la_auyama_sigue_fuera():
    """Decisión explícita: es base de carbo, no relleno. Si alguien la añade sin medir el
    impacto en macros, este test se lo recuerda."""
    assert not _cubierto("3¾ tazas de auyama en cubos"), (
        "la auyama es la base de carbohidrato del plato; capearla mueve los macros de la cena "
        "— requiere su propia medición, no un token más"
    )


@pytest.mark.parametrize("viver", ["200 g de yuca", "150 g de batata", "1 platano maduro",
                                   "200 g de name rallado", "papa mediana"])
def test_los_viveres_no_entran_en_el_cap_de_volumen(viver):
    """Todos son bases de carbohidrato es-DO. El cap de volumen vegetal no es su sitio."""
    assert not _cubierto(viver), viver


# ───────────── 4. lo que ya estaba sigue ─────────────

@pytest.mark.parametrize("orig", ["pepino", "berro", "rabano", "rabanito", "apio"])
def test_los_cinco_originales_siguen(orig):
    assert orig in g._REALISM_VOLUME_VEG_TOKENS


def test_el_knob_y_su_clamp():
    assert g.REALISM_VEG_VOLUME_CAP_G == 250
    from pathlib import Path
    src = Path(g.__file__).resolve().read_text(encoding="utf-8")
    assert "100 <= v <= 500" in src, "el clamp del knob debe seguir vivo"


def test_el_techo_generico_sigue_siendo_el_backstop():
    """El cap por clase no sustituye al techo duro: si la clasificación falla, 600 g manda."""
    assert g.LINE_GRAM_HARD_CAP == 600
