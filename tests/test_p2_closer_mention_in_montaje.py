"""[P2-CLOSER-MENTION-IN-MONTAJE · 2026-07-31] El complemento del cerrador nunca llega al emplatado.

Medido sobre los 3 planes de la base: 9 de 32 comidas (28%) incluyen un alimento de 40 g
(`CLOSER_COOKABLE_MIN_G`) que el cerrador añade con su propio paso, y que el paso de MONTAJE —el que
le dice al usuario cómo servir— jamás menciona. Ejemplos reales:

  · "Canoa de Pan Integral…": ING "40g de soya texturizada"; el TdF termina con "Cocina soya
    texturizada en agua…"; Montaje: "pan tostado con el queso derretido… molondrones encima".
  · "Tostadas de Pan Integral con Ricotta y Mango y Yogurt": ING "40g de yogurt"; TdF: "Sirve yogurt
    al lado"; Montaje: ricotta + mango + canela + miel.

El usuario compra y paga un alimento que la receta no le dice cómo servir.

⚠️ POR QUÉ NO VALE FORZAR LA FUSIÓN EXISTENTE. `_merge_complement_into_montaje` MUEVE el ingrediente
al emplatado y por eso exige que NINGÚN paso cocine — su propio docstring cuenta que apoyarse en el
clasificador dejaba pasar 7 de 18 platos cocinados, y mover un alimento al emplatado de un plato que
se cocina puede saltarse su cocción. Ese gate es correcto y no se toca.

Lo que falta es distinto: en un plato COCINADO el complemento YA tiene su paso de cocción, así que al
montaje solo le falta decir que se sirva. Mencionar ≠ mover. La condición de seguridad que separa
ambos casos es que el alimento aparezca YA en otro paso: si no está en ninguno, mencionarlo en el
emplatado sería mandar servir algo que nadie preparó — exactamente lo que el gate de la fusión evita.

Anchor de producción: P2-CLOSER-MENTION-IN-MONTAJE.
"""
import pytest


TDF_COCINA = ("El Toque de Fuego: Calienta el aceite en una sartén a fuego medio, saltea la cebolla "
              "y el ajo 3-4 minutos. Tuesta las rebanadas de pan integral 2-3 minutos. "
              "Cocina soya texturizada en agua hasta que ablanden e incorpórala al plato.")
MONTAJE = ("Montaje: Coloca las rebanadas de pan tostado con el queso derretido en un plato, "
           "distribuye los molondrones salteados encima y sirve de inmediato.")


def _pasos():
    return ["Mise en place: Pica la cebolla y el ajo.", TDF_COCINA, MONTAJE]


# --------------------------------------------------------------- el caso de producción

def test_el_complemento_cocinado_se_menciona_al_servir():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = _pasos()
    assert _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"]) is True
    montaje = pasos[-1]
    assert "soya" in montaje.lower(), (
        f"el montaje sigue sin decir qué hacer con la soya: {montaje!r}"
    )


def test_no_toca_los_pasos_de_coccion():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = _pasos()
    _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"])
    assert pasos[1] == TDF_COCINA, "el paso de cocción no puede alterarse"
    assert pasos[0].startswith("Mise en place"), "el mise en place no puede alterarse"


# --------------------------------------------------------------- la condición de seguridad

def test_no_menciona_un_alimento_que_ningun_paso_prepara():
    """LA guarda: si no aparece en ningún paso, mencionarlo al servir manda emplatar algo crudo.

    Es la misma preocupación por la que `_merge_complement_into_montaje` exige que nada se cocine.
    """
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = _pasos()
    antes = list(pasos)
    assert _mention_cooked_complement_in_montaje(pasos, ["filete de pescado"]) is False
    assert pasos == antes, "no debe tocar nada si el alimento no está preparado en ningún paso"


def test_no_duplica_si_el_montaje_ya_lo_nombra():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = ["Mise en place: Pica.", TDF_COCINA,
             "Montaje: Sirve el pan con la soya texturizada encima."]
    antes = list(pasos)
    assert _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"]) is False
    assert pasos == antes


def test_sin_montaje_no_inventa_uno():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = ["Mise en place: Pica.", TDF_COCINA]
    antes = list(pasos)
    assert _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"]) is False
    assert pasos == antes


def test_es_idempotente():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    pasos = _pasos()
    _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"])
    primera = list(pasos)
    _mention_cooked_complement_in_montaje(pasos, ["soya texturizada"])
    assert pasos == primera, "correrlo dos veces no puede seguir añadiendo texto"


def test_tolera_basura():
    from graph_orchestrator import _mention_cooked_complement_in_montaje

    assert _mention_cooked_complement_in_montaje([], ["x"]) is False
    assert _mention_cooked_complement_in_montaje(_pasos(), []) is False
    assert _mention_cooked_complement_in_montaje(None, ["x"]) is False


# --------------------------------------------------------------- no pisa a la fusión existente

def test_la_fusion_de_platos_frios_sigue_teniendo_prioridad():
    """En un plato SIN cocción la fusión mueve el ingrediente ('Termina con X'); eso no cambia."""
    from graph_orchestrator import _merge_complement_into_montaje

    pasos = ["Mise en place: Mide el yogurt.",
             "Montaje: Coloca el yogurt en un bowl y añade la fruta."]
    assert _merge_complement_into_montaje(pasos, ["granola"]) is True
    assert "Termina con granola" in pasos[-1], pasos[-1]
