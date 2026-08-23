"""[P1-I18N-DISPLAY-INSIGHTS-SIN-DEFENSA-DE-ECO · 2026-08-23] Los `insights` eran el único
de los tres hermanos sin defensa contra el ECO.

La MISMA llamada al LLM produce tres cosas: el `_display` de cada comida, el nombre del plan
y los `insights`. Las dos primeras ganaron defensa contra el eco —que el modelo devuelva el
español tal cual— el 21 y el 23 de agosto:

    P2-DISPLAY-ECO-NOMBRE      -> _validate_plan_name
    P1-DISPLAY-ECO-PERSISTIDO  -> _plan_name_already_translated
    P1-DISPLAY-ECO-CONTENIDO   -> _validate_and_build_display

Los `insights` —el TERCER campo del mismo contrato JSON, escrito por el MISMO
`_persist_batch`— no la tuvieron nunca: ni `_validate_insights` ni
`_insights_already_translated` comparan contra el original. `_validate_insights` recibe
`original` y lo usa SÓLO para la longitud.

Consecuencia exacta, y es la peor forma de fallar: un array de insights devuelto sin traducir
se persiste COMO SI FUERA la traducción, `_insights_already_translated` pasa a decir «ya
está», y nadie lo reintenta jamás. El panel «Diagnóstico / Plan de Acción / Tip del Chef» se
queda en español PARA SIEMPRE, en un dashboard por lo demás traducido.

Y no es teórico: el 23-ago se midió un eco VIVO en producción para el nombre del plan
(`fr-FR -> "Sazón Fuerte, Vida en Equilibrio"`, el español tal cual). El mecanismo que lo
produce es el mismo para los tres campos.

EL CRITERIO ES DE DOS SEÑALES, no de una. Una sola línea puede coincidir legítimamente con
su original —un tecnicismo, un nombre propio, «Mise en place»— y descartar el lote por eso
sería tirar traducciones buenas. Se descarta cuando TODAS las líneas son eco, que es la
firma de «el modelo devolvió el original» y no la de «una línea coincide». Es el mismo
criterio que `_validate_and_build_display` ya usa para el contenido de la comida.

tooltip-anchor: P1-I18N-DISPLAY-INSIGHTS-SIN-DEFENSA-DE-ECO
"""
from __future__ import annotations

import pytest

from plan_display_i18n import (
    _insights_already_translated,
    _validate_insights,
)

_MARKER = "P1-I18N-DISPLAY-INSIGHTS-SIN-DEFENSA-DE-ECO"

_ORIGINAL = [
    "Tu plan cubre el hierro con habichuelas y espinaca.",
    "Sube la proteína del desayuno a 25 g.",
    "Saltea el ajo antes de añadir el tomate.",
]
_TRADUCIDO = [
    "Votre plan couvre le fer avec des haricots et des épinards.",
    "Augmentez les protéines du petit-déjeuner à 25 g.",
    "Faites revenir l'ail avant d'ajouter la tomate.",
]


def test_un_lote_de_insights_que_es_eco_del_original_se_descarta() -> None:
    """El modelo devolvió el español. Persistirlo es peor que no traducir: cierra la puerta."""
    assert _validate_insights(list(_ORIGINAL), _ORIGINAL) is None, (
        f"un array de insights IDÉNTICO al original se aceptó como traducción. Se persiste, "
        f"el gate de «ya traducido» pasa a decir que sí, y el panel se queda en español para "
        f"siempre porque nadie lo reintenta. [{_MARKER}]"
    )


def test_el_eco_con_otra_caja_o_acentos_tambien_se_descarta() -> None:
    """Mismo criterio tolerante que `_eco_del_original` usa para el nombre: «HABICHUELAS
    guisadas» tampoco es una traducción."""
    gritado = [s.upper() for s in _ORIGINAL]
    assert _validate_insights(gritado, _ORIGINAL) is None, (
        f"el eco en otra caja pasó como traducción. [{_MARKER}]"
    )


def test_una_sola_linea_coincidente_NO_descarta_el_lote() -> None:
    """La otra dirección, que es la que hace que el guard sea útil y no un estorbo: una línea
    puede coincidir legítimamente (un tecnicismo, un nombre propio). Descartar por eso tiraría
    traducciones buenas, y el fallback silencioso al español es justo lo que se quiere evitar."""
    casi = [_ORIGINAL[0], _TRADUCIDO[1], _TRADUCIDO[2]]
    assert _validate_insights(casi, _ORIGINAL) == casi, (
        f"UNA línea coincidente tumbó el lote entero. El criterio es de dos señales: se "
        f"descarta cuando TODAS son eco, no cuando una lo es. [{_MARKER}]"
    )


def test_una_traduccion_de_verdad_pasa() -> None:
    assert _validate_insights(list(_TRADUCIDO), _ORIGINAL) == _TRADUCIDO, (
        f"una traducción legítima se descartó. [{_MARKER}]"
    )


def test_el_gate_de_ya_traducido_no_da_por_bueno_un_eco_persistido() -> None:
    """La segunda mitad: aunque el eco se haya persistido antes de este fix, el gate no puede
    seguir diciendo «ya está» — si lo dice, el plan no se repara nunca."""
    plan = {
        "insights": list(_ORIGINAL),
        "_display": {"fr-FR": {"insights": list(_ORIGINAL)}},
    }
    assert not _insights_already_translated(plan, "fr-FR"), (
        f"el gate da por traducido un `_display` que es el ESPAÑOL tal cual. Los planes que "
        f"ya tienen el eco persistido no se reintentarían jamás. [{_MARKER}]"
    )


def test_el_gate_sigue_diciendo_que_si_cuando_de_verdad_esta_traducido() -> None:
    plan = {
        "insights": list(_ORIGINAL),
        "_display": {"fr-FR": {"insights": list(_TRADUCIDO)}},
    }
    assert _insights_already_translated(plan, "fr-FR"), (
        f"el gate dejó de reconocer una traducción buena: se pagaría el enriquecimiento en "
        f"cada ciclo. [{_MARKER}]"
    )


@pytest.mark.parametrize("valor", [None, [], "texto", [""], ["a", "b"]])
def test_fail_open_ante_formas_inesperadas(valor) -> None:
    """Fail-open deliberado: cualquier forma rara devuelve None y el panel se queda en
    español, que es correcto aunque no sea lo pedido."""
    assert _validate_insights(valor, _ORIGINAL) is None
