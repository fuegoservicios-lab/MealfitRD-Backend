"""[P1-CHAT-SLOT-NOT-YET-TODAY · 2026-07-31] La cena que todavia no ha llegado.

A las 10:23 de la manana el usuario escribio "cene dos panes con queso" y el
coach lo anoto como la CENA DE HOY: 300 kcal en un slot del FUTURO.

Lo causo la regla 8 ("el dia por defecto es HOY"), que cerro un hueco real y
abrio su simetrico. Entre "se le olvido registrar la cena de anoche" y "ceno a
las 10 AM", la primera es la lectura normal.

Verificado A/B contra el modelo EN VIVO antes de escribir estos tests: sin la
regla 8-bis, flash devolvia days_ago=0 (el bug exacto); con ella, days_ago=1. El
control ("me desayune" a las 10:23 AM) sigue en days_ago=0 en ambos casos — la
regla no se dispara de mas. *Una regla de prompt que no mueve al modelo no vale
nada; medirlo es parte del fix.*
"""

import re

import pytest


@pytest.fixture(scope="module")
def rules_block() -> str:
    """El bloque de reglas EVALUADO, no el texto crudo del fichero: así un
    comentario de codigo nunca satisface un assert."""
    from prompts.chat_agent import _CHAT_BREVITY_RULES
    return _CHAT_BREVITY_RULES


# =====================================================================
# [P1-CHAT-SLOT-NOT-YET-TODAY · 2026-07-31] La cena que aún no ha llegado
# =====================================================================
# La regla 8 de arriba ("el día por defecto es HOY") cerró un hueco real, pero
# abrió su simétrico: a las 10:23 de la mañana el usuario escribió "cené dos
# panes con queso" y quedó anotado como la CENA DE HOY — una cena que todavía
# no había ocurrido. El panel mostró 300 kcal en un slot del futuro.
#
# Entre "se le olvidó registrar la cena de anoche" y "cenó a las 10 AM", la
# primera es la lectura normal. Verificado A/B contra el modelo en vivo antes
# de escribir estos tests: SIN la regla, flash devolvía `days_ago=0` (el bug
# exacto); CON ella, `days_ago=1`. El control ("me desayuné" a las 10:23 AM)
# sigue en `days_ago=0` con y sin la regla — la regla no se dispara de más.

def test_regla_8bis_existe_y_manda_al_dia_anterior(rules_block: str):
    assert "8-bis." in rules_block, (
        "P1-CHAT-SLOT-NOT-YET-TODAY: falta la regla 8-bis. Sin ella, una cena "
        "nombrada por la mañana se registra como la cena de HOY — un slot que "
        "todavía no ha ocurrido."
    )
    assert re.search(r"days_ago\s*=\s*1", rules_block), (
        "la regla no dice explícitamente `days_ago=1`; sin el valor concreto "
        "el modelo tiene que inferir el parámetro además de la intención."
    )


def test_regla_8bis_decide_por_la_FRANJA_no_por_el_horario_del_usuario(rules_block: str):
    """El bug nació de razonar sobre si el usuario tiene horario raro.

    Lo que decide es la franja nombrada contra la hora actual. Si la regla
    dejara la puerta abierta a "quizás trabaja de noche", vuelve la
    deliberación que produjo el registro incorrecto.
    """
    bloque = rules_block.split("8-bis.")[1]
    assert "FRANJA" in bloque or "franja" in bloque, (
        "la regla no ancla la decisión a la FRANJA nombrada vs la hora actual"
    )
    # El ejemplo inverso evita que la regla se lea como "todo es de ayer".
    assert re.search(r"desayuno.{0,40}23:00|23:00.{0,60}desayuno", bloque, re.IGNORECASE), (
        "falta el ejemplo simétrico (desayuno nombrado de noche = de esa "
        "misma mañana). Sin él la regla se puede leer como 'ante la duda, "
        "ayer', que rompería el caso normal."
    )


def test_regla_8bis_no_invalida_que_mande_el_usuario(rules_block: str):
    """Si él nombra el día, gana él. La 8-bis solo deriva el día NO dicho."""
    bloque = rules_block.split("8-bis.")[1]
    assert re.search(r"si él nombra el día", bloque, re.IGNORECASE), (
        "la regla 8-bis no reserva la última palabra al usuario: derivar el "
        "día es un default, no una corrección de lo que él afirmó."
    )


def test_regla_8bis_lleva_el_incidente_que_la_motiva(rules_block: str):
    """Sin el caso real al lado, la próxima persona la lee como una sutileza
    y la borra al simplificar el prompt."""
    bloque = rules_block.split("8-bis.")[1]
    assert "10:23" in bloque, "falta la hora del incidente real"
