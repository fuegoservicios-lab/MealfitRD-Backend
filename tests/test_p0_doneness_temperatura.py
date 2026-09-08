# -*- coding: utf-8 -*-
"""[P0-DONENESS-TEMPERATURA · 2026-09-08] El color no es un criterio de seguridad alimentaria.

La §21 del prompt del día, escrita el 07-sep, decía:

    «POLLO, CERDO o PAVO: hasta que el interior no tenga partes rosadas.
     Aquí no es textura, es SEGURIDAD.»

Nombra la seguridad y da acto seguido un criterio que no sirve para ella. **La carne se dora antes
de alcanzar temperatura segura y puede seguir rosada después** (mioglobina, nitritos, aves jóvenes).

## Cómo apareció, y por qué ninguna defensa automática lo vio

El dueño juzgó a ciegas 20 almuerzos y cenas: **3 «la serviría» / 16 «dudoso» / 1 «no»**, contra
**17/3/0** en desayunos. **13 de los 16 «dudoso» pedían exactamente esto.**

  · El escáner de coherencia culinaria daba **140/140 limpias**: mide que no se pida lo que no hay,
    que lo crudo se cocine, que lo compartido se reparta. No mide si un criterio de cocción es
    seguro.
  · Los desayunos sacaron 17/20 porque **casi no llevan carne**: la franja donde la regla
    defectuosa no se ejerce.
  · Y el test del 07-sep **exigía la palabra «rosadas»** — anclaba la regla equivocada con la misma
    firmeza con que habría anclado la correcta.

*Lo que lo delató no fue ninguna defensa automática: fue un humano leyendo el producto.*

Este fichero existe aparte del de `P1-DAYGEN-DONENESS-INSIDE` a propósito: aquel ancla que HAYA una
señal de interior; éste ancla que para la carne esa señal sea la TEMPERATURA. Son dos contratos, y
el segundo es de seguridad.
"""
import re

import pytest


@pytest.fixture(scope="module")
def bloque() -> str:
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as P

    i = P.find("21. CÓMO SE SABE QUE ESTÁ LISTO POR DENTRO")
    if i < 0:
        pytest.skip("§21 no está en el prompt (knob apagado)")
    return P[i:]


def test_las_tres_temperaturas_estan(bloque):
    """74 aves · 71 molida · 63 + reposo en pieza entera. Las tres, porque las tres son distintas."""
    for t in ("74 °C", "71 °C", "63 °C"):
        assert t in bloque, f"falta {t}: sin ella esa familia de carne se queda sin criterio real"
    assert re.search(r"3\s*minutos de\s*\n?\s*reposo|reposo", bloque), (
        "la pieza entera necesita el reposo: la temperatura sigue subiendo fuera del fuego")


def test_dice_EXPLICITAMENTE_que_el_color_no_sirve(bloque):
    """No basta con añadir la temperatura y dejar el color al lado: el modelo elegiría el más
    fácil de escribir. Hay que desautorizarlo."""
    assert "COLOR NO SIRVE" in bloque
    assert "se dora" in bloque and "rosada después" in bloque, (
        "falta el PORQUÉ; una prohibición sin razón se ignora en cuanto estorba")


def test_la_temperatura_va_ANTES_que_la_senal_de_respaldo(bloque):
    """Quien no tiene termómetro necesita una alternativa, pero si se ofrece primero se vuelve la
    norma. El orden es el mensaje."""
    i_temp = bloque.find("74 °C")
    i_resp = bloque.find("termómetro, di además")
    assert i_temp >= 0 and i_resp > i_temp, (
        "la señal de respaldo quedó antes que la temperatura: entonces la temperatura es opcional")


def test_no_volvio_el_color_como_criterio_unico(bloque):
    """El guard del regreso: la redacción vieja decía «hasta que el interior no tenga partes
    rosadas» como criterio ÚNICO. Si vuelve, este test cae."""
    assert "no tenga partes rosadas" not in bloque


def test_sigue_sin_pedirselo_al_salteado(bloque):
    """Lo que P1-DAYGEN-DONENESS-INSIDE ya había decidido y no se toca: en un sofrito no hay centro
    crudo, y pedir señal ahí es el ruido que hace que se ignore la regla entera."""
    assert "salteado de vegetales sí basta" in bloque
