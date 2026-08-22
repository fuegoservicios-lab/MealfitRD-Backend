"""[P0-I18N-ALERGENOS-FR-IT-PT · 2026-08-22] El backstop clinico de alergias solo entendia
espanol e ingles, y la interfaz se vende en CINCO idiomas.

MEDIDO antes del arreglo, ejecutando `clinical_backstop_for_meal` de produccion sobre platos
que SI contienen el alergeno:

    declara 'Cacahuete' (es-ES) -> 1 violacion       declara 'Arachide' (fr) -> 0
    declara 'Peanut'    (en)    -> 1 violacion       declara 'Arachidi' (it) -> 0
                                                     declara 'Amendoim' (pt) -> 0
    'Fruits de mer' / 'Crostacei' / 'Frutos do mar'                        -> 0
    'Poisson' / 'Pesce' / 'Peixe'                                          -> 0
    'Gluten' / 'Gluten' -> 1                         'Glutine'  (it)       -> 0

10 de 21 combinaciones NO bloqueaban. El italiano «Glutine» fallaba mientras «Gluten» y
«Gluten» pasaban: por coincidencia ortografica con el espanol, no por diseno — que es la peor
forma de estar cubierto, porque parece que funciona hasta que cambias de palabra.

NO ES UNA REGRESION. `P0-ALLERGEN-VOCAB-I18N` (2026-08-21) cerro espanol e ingles porque ese
era el alcance del sistema de PAISES. El sistema de IDIOMAS anadio despues fr-FR, it-IT y
pt-BR. Son dos proyectos cuyos alcances no llegaron a tocarse, y el hueco vive justo en medio.

POR DONDE ENTRA. Los CHIPS estan bien y eso acota el dano: `QAllergies.jsx` manda `val`
canonico espanol («Mani», «Mariscos», «Pescado») y traduce solo el `label`. La exposicion es el
campo libre «Otra…» y `update_form_field(field='allergies')` desde el coach. Es la vieja
observacion de `P2-ALLERGEN-CHIPS-REACH-ENGINE` («media seguridad de un chip es que el usuario
no tenga que acordarse») invertida: aqui, si te acuerdas y lo escribes en tu idioma, tampoco
funciona. Y falla EN SILENCIO: ni violacion, ni aviso, ni fila de telemetria.

POR QUE ES P0. El backstop es la UNICA defensa determinista de swap, regenerate-day,
chat-modify y del camino degradado — superficies que no pasan por el grafo de generacion y no
tienen reviewer que compense. El repo ya califico este mismo defecto como P0 en agosto.

LO QUE ESTE GUARD MIDE: la CONDUCTA de punta a punta (`clinical_backstop_for_meal` sobre un
plato real con el alergeno dentro), no la presencia de cadenas en un diccionario. Un guard que
mirase `_ALLERGEN_DECLARATION_ALIASES` seguiria verde si manana alguien cambia como se resuelve
la clase y deja el vocabulario intacto.

tooltip-anchor: P0-I18N-ALERGENOS-FR-IT-PT
"""
from __future__ import annotations

import pytest

from graph_orchestrator import clinical_backstop_for_meal, _expand_allergy_declarations

_MARKER = "P0-I18N-ALERGENOS-FR-IT-PT"

_PLATOS = {
    "mani": {
        "name": "Pollo en salsa de mani",
        "ingredients": ["120 g de Pechuga de pollo", "30 g de Maní tostado", "100 g de Arroz blanco"],
        "recipe": ["Sofríe el pollo.", "Añade la salsa de maní."],
    },
    "mariscos": {
        "name": "Arroz con camarones",
        "ingredients": ["150 g de Camarón", "120 g de Arroz blanco"],
        "recipe": ["Saltea los camarones."],
    },
    "pescado": {
        "name": "Filete de tilapia",
        "ingredients": ["180 g de Tilapia", "150 g de Yuca"],
        "recipe": ["Asa la tilapia."],
    },
    "gluten": {
        "name": "Pasta con pollo",
        "ingredients": ["100 g de Espagueti", "120 g de Pechuga de pollo"],
        "recipe": ["Hierve el espagueti."],
    },
    "lacteos": {
        "name": "Arroz con leche",
        "ingredients": ["200 ml de Leche entera", "60 g de Arroz blanco"],
        "recipe": ["Hierve la leche."],
    },
    "huevo": {
        "name": "Tortilla de huevo",
        "ingredients": ["2 unidades de Huevo", "50 g de Cebolla"],
        "recipe": ["Bate los huevos."],
    },
}

# (clase de plato, idioma, termino tal y como lo escribiria el usuario en el campo libre).
# Espanol e ingles van incluidos A PROPOSITO: son la NO-regresion de P0-ALLERGEN-VOCAB-I18N,
# y sin ellos este guard no distinguiria «anadi tres idiomas» de «rompi los dos que habia».
_CASOS = [
    ("mani", "es-DO", "Maní"),
    ("mani", "es-ES", "Cacahuete"),
    ("mani", "en-US", "Peanut"),
    ("mani", "fr-FR", "Arachide"),
    ("mani", "it-IT", "Arachidi"),
    ("mani", "pt-BR", "Amendoim"),
    ("mariscos", "es-DO", "Mariscos"),
    ("mariscos", "en-US", "Shellfish"),
    ("mariscos", "fr-FR", "Fruits de mer"),
    ("mariscos", "it-IT", "Crostacei"),
    ("mariscos", "pt-BR", "Frutos do mar"),
    ("pescado", "es-DO", "Pescado"),
    ("pescado", "en-US", "Fish"),
    ("pescado", "fr-FR", "Poisson"),
    ("pescado", "it-IT", "Pesce"),
    ("pescado", "pt-BR", "Peixe"),
    ("gluten", "es-DO", "Gluten"),
    ("gluten", "en-US", "Gluten"),
    ("gluten", "fr-FR", "Gluten"),
    ("gluten", "it-IT", "Glutine"),
    ("gluten", "pt-BR", "Glúten"),
    ("lacteos", "fr-FR", "Produits laitiers"),
    ("lacteos", "it-IT", "Latticini"),
    ("lacteos", "pt-BR", "Laticínios"),
    ("huevo", "fr-FR", "Oeuf"),
    ("huevo", "it-IT", "Uova"),
    ("huevo", "pt-BR", "Ovos"),
]


@pytest.mark.parametrize("clase,idioma,termino", _CASOS,
                         ids=[f"{c}-{i}-{t}" for c, i, t in _CASOS])
def test_la_alergia_declarada_bloquea_el_plato_en_los_cinco_idiomas(clase, idioma, termino) -> None:
    """La invariante entera del gap, medida como la vive el usuario."""
    violaciones = clinical_backstop_for_meal(_PLATOS[clase], allergies=[termino])
    assert violaciones, (
        f"declarar «{termino}» ({idioma}) NO bloquea «{_PLATOS[clase]['name']}», que contiene "
        f"el alergeno. El backstop es la UNICA defensa de swap/regenerate-day/chat-modify y "
        f"del camino degradado: sin el, el plato se sirve. [{_MARKER}]"
    )


@pytest.mark.parametrize("termino", ["Arachide", "Amendoim", "Glutine", "Poisson",
                                     "Fruits de mer", "Latticini", "Uova"])
def test_la_declaracion_resuelve_a_una_clase_y_no_cae_a_literal(termino) -> None:
    """Sin clase, `_expand_allergy_declarations` devuelve el termino LITERAL.

    Es el modo de fallo exacto: la expansion tiene un solo elemento —la palabra que escribio
    el usuario— y buscar «arachide» dentro de una receta escrita en espanol canonico no
    encuentra nada jamas. Un solo termino de salida es la firma de «no lo reconoci».
    """
    expandido = _expand_allergy_declarations([termino])
    assert len(expandido) > 1, (
        f"«{termino}» no resolvio a ninguna clase: la expansion es {expandido!r}, o sea el "
        f"literal. Buscar esa palabra en una receta en espanol no encuentra nada. [{_MARKER}]"
    )


def test_lo_que_el_sistema_no_modela_sigue_cayendo_a_literal() -> None:
    """El contrato que NO se toca: quien declara algo fuera del modelo (fresa, kiwi) sigue
    recibiendo su termino tal cual. Si esto se rompiera, el arreglo habria convertido el
    vocabulario en un embudo que traga cualquier cosa."""
    for termino in ("fresa", "kiwi", "melocoton"):
        assert _expand_allergy_declarations([termino]) == {termino}, (
            f"«{termino}» dejo de caer a literal: alguna clase lo esta capturando por "
            f"subcadena. [{_MARKER}]"
        )


def test_un_plato_sin_el_alergeno_no_se_bloquea() -> None:
    """La direccion contraria. Sin esto, un guard que devolviera «hay violacion» siempre
    pasaria los 27 casos de arriba sin significar nada."""
    limpio = {
        "name": "Ensalada de aguacate",
        "ingredients": ["100 g de Aguacate", "80 g de Lechuga"],
        "recipe": ["Corta el aguacate."],
    }
    for termino in ("Arachide", "Glutine", "Poisson", "Latticini"):
        assert not clinical_backstop_for_meal(limpio, allergies=[termino]), (
            f"«{termino}» bloquea un plato que no contiene el alergeno: el guard no puede "
            f"fallar, asi que no informa. [{_MARKER}]"
        )
