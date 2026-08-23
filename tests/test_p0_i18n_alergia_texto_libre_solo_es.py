"""[P0-I18N-ALERGIA-TEXTO-LIBRE-SOLO-ES · 2026-08-23] El campo libre de alergias entendía
la CATEGORÍA en los cinco idiomas y el ALIMENTO sólo en español.

Los dos cierres P0 anteriores arreglaron el nombre de la CLASE, y ese trabajo está bien:
medido, 24 de 24 términos de categoría bloquean en los cinco idiomas («fruits de mer»,
«crostacei», «latticini», «poisson», «frutta a guscio»…). Lo que ninguno tocó es el nombre
del ALIMENTO CONCRETO, que es como escribe la gente cuando le preguntas «¿alguna otra?».

MEDIDO antes de este fix, ejecutando `clinical_backstop_for_meal` de producción sobre platos
que SÍ contienen el alérgeno:

    mariscos  fr crevettes 0   it gamberetti 0   pt camarao 0   en shrimp  0   *** NO BLOQUEA ***
    pescado   fr saumon    0   it tonno      0   pt atum    0   en tuna    0   *** NO BLOQUEA ***
    lacteos   fr fromage   0   it formaggio  0   pt queijo  0   en cheese  0   *** NO BLOQUEA ***
    frutos s. fr amandes   0   it mandorle   0   pt amendoas 0  en almonds 0   *** NO BLOQUEA ***
    gluten    fr ble       0   it grano      0

46 de 62 combinaciones no bloqueaban, y NUEVE de ellas eran en INGLÉS — o sea que no es sólo
un hueco de i18n, es estructural: en español funciona porque la tabla del INGREDIENTE
(`_ALLERGEN_SYNONYMS`) está en español y la resolución es por subcadena literal. Cada idioma
que se añadió heredó la mitad declarativa y ninguna de la mitad alimentaria.

LA LIGADURA, medida aparte y peor: `strip_accents` usa NFKD, que NO descompone `œ`.

    strip_accents('œufs') -> 'œufs'
    declara 'oeufs' -> 1 violación   bloquea
    declara 'œufs'  -> 0 violaciones *** NO BLOQUEA ***

«œufs» es la grafía CORRECTA del francés y la que el teclado francés produce. La que protege
es la que se teclea con dos letras.

POR DÓNDE ENTRA. Los chips están bien y eso acota el daño: mandan `val` canónico español. La
exposición es el campo libre «Otra…» (`QAllergies.jsx`) y `update_form_field(field='allergies')`
del coach. Es `P2-ALLERGEN-CHIPS-REACH-ENGINE` invertido: si te acuerdas y lo escribes en tu
idioma, tampoco funciona.

POR QUÉ LA FRONTERA DE PALABRA, Y SÓLO PARA LOS CORTOS. Añadir «ble» (fr) o «cod» (en) a un
matcher de subcadena bidireccional dispara con «roble», «problema» o «codorniz». Pero cambiar
la subcadena por frontera de palabra en TODO el vocabulario rompe la mitad alimentaria, donde
la subcadena es lo que hace que «camaron» case con «camarones». Así que la frontera se aplica
sólo a los alias DECLARATIVOS y sólo a los términos cortos (<6), que es exactamente donde la
subcadena miente. Medido: 0 regresiones sobre 47 declaraciones vigentes, 0 falsos positivos.

LO QUE ESTE GUARD MIDE: la CONDUCTA de punta a punta sobre un plato real con el alérgeno
dentro, no la presencia de cadenas en un diccionario — un guard que mirase la tabla seguiría
verde si mañana alguien cambia cómo se resuelve la clase.

tooltip-anchor: P0-I18N-ALERGIA-TEXTO-LIBRE-SOLO-ES
"""
from __future__ import annotations

import pytest

from graph_orchestrator import clinical_backstop_for_meal

_MARKER = "P0-I18N-ALERGIA-TEXTO-LIBRE-SOLO-ES"

# Platos REALES con el alérgeno dentro, en español canónico (que es como los escribe el motor).
_PLATOS = {
    "mariscos": {
        "name": "Arroz con camarones",
        "ingredients": ["150 g de Camarón", "120 g de Arroz blanco"],
        "recipe": ["Saltea los camarones."],
    },
    "pescado": {
        "name": "Filete de tilapia",
        "ingredients": ["150 g de Tilapia", "100 g de Arroz blanco"],
        "recipe": ["Asa la tilapia."],
    },
    "lacteos": {
        "name": "Pasta con queso",
        "ingredients": ["50 g de Queso cheddar", "100 g de Pasta"],
        "recipe": ["Funde el queso."],
    },
    "frutos secos": {
        "name": "Ensalada con nueces",
        "ingredients": ["30 g de Nueces", "100 g de Lechuga"],
        "recipe": ["Tuesta las nueces."],
    },
    "gluten": {
        "name": "Pasta al pesto",
        "ingredients": ["100 g de Pasta de trigo", "20 g de Pesto"],
        "recipe": ["Hierve la pasta de trigo."],
    },
    "huevo": {
        "name": "Tortilla de huevo",
        "ingredients": ["2 unidades de Huevo", "50 g de Cebolla"],
        "recipe": ["Bate los huevos."],
    },
}

# El ALIMENTO, no la categoría — que es lo que la gente escribe en un campo libre.
_ALIMENTOS = [
    ("mariscos", "fr", "crevettes"), ("mariscos", "fr", "coquillages"),
    ("mariscos", "fr", "homard"), ("mariscos", "fr", "moules"), ("mariscos", "fr", "huitres"),
    ("mariscos", "it", "gamberetti"), ("mariscos", "it", "gamberi"), ("mariscos", "it", "cozze"),
    ("mariscos", "it", "aragosta"), ("mariscos", "it", "vongole"),
    ("mariscos", "pt", "camarao"), ("mariscos", "pt", "camaroes"), ("mariscos", "pt", "lagosta"),
    ("mariscos", "en", "shrimp"), ("mariscos", "en", "prawns"), ("mariscos", "en", "lobster"),
    ("mariscos", "en", "crab"),
    ("pescado", "fr", "saumon"), ("pescado", "fr", "thon"), ("pescado", "it", "tonno"),
    ("pescado", "pt", "salmao"), ("pescado", "pt", "atum"), ("pescado", "en", "tuna"),
    ("frutos secos", "fr", "amandes"), ("frutos secos", "fr", "noisettes"),
    ("frutos secos", "fr", "cajou"), ("frutos secos", "it", "mandorle"),
    ("frutos secos", "it", "nocciole"), ("frutos secos", "it", "anacardi"),
    ("frutos secos", "pt", "amendoas"), ("frutos secos", "pt", "avelas"),
    ("frutos secos", "en", "almonds"), ("frutos secos", "en", "cashews"),
    ("lacteos", "fr", "fromage"), ("lacteos", "fr", "beurre"), ("lacteos", "it", "formaggio"),
    ("lacteos", "it", "burro"), ("lacteos", "pt", "queijo"), ("lacteos", "pt", "manteiga"),
    ("lacteos", "en", "cheese"), ("lacteos", "en", "butter"),
    ("gluten", "fr", "ble"), ("gluten", "fr", "farine de ble"), ("gluten", "it", "grano"),
]


@pytest.mark.parametrize("clase,idioma,termino", _ALIMENTOS)
def test_el_alimento_declarado_en_su_idioma_bloquea_el_plato(clase, idioma, termino) -> None:
    """La invariante entera, medida como la vive el usuario: escribo el alimento en MI idioma."""
    violaciones = clinical_backstop_for_meal(_PLATOS[clase], allergies=[termino])
    assert violaciones, (
        f"declarar «{termino}» ({idioma}) NO bloquea «{_PLATOS[clase]['name']}», que contiene el "
        f"alérgeno. El backstop es la ÚNICA defensa de swap/regenerate-day/chat-modify y del "
        f"camino degradado: sin él, el plato se sirve. [{_MARKER}]"
    )


@pytest.mark.parametrize("termino", ["œufs", "œuf", "Œufs", "ŒUFS"])
def test_la_ligadura_francesa_del_huevo_bloquea(termino) -> None:
    """`strip_accents` es NFKD y NFKD no descompone `œ`. La grafía correcta del francés era
    justo la que no protegía; la que funcionaba es la que se teclea con dos letras."""
    violaciones = clinical_backstop_for_meal(_PLATOS["huevo"], allergies=[termino])
    assert violaciones, (
        f"«{termino}» NO bloquea un plato con huevo. Es la grafía que produce un teclado "
        f"francés, y era la única que fallaba. [{_MARKER}]"
    )


@pytest.mark.parametrize("clase,termino", [
    ("mariscos", "les crevettes"),
    ("mariscos", "allergie aux crevettes"),
    ("frutos secos", "allergia alle mandorle"),
    ("lacteos", "alergia a queijo"),
    ("pescado", "I am allergic to tuna"),
])
def test_el_alimento_dentro_de_una_frase_tambien_bloquea(clase, termino) -> None:
    """Nadie escribe una palabra suelta en un campo libre."""
    assert clinical_backstop_for_meal(_PLATOS[clase], allergies=[termino]), (
        f"«{termino}» NO bloquea, y es la forma en que una persona escribe de verdad. [{_MARKER}]"
    )


# --------------------------------------------------------------------------------------
# NO-REGRESIÓN: lo que ya funcionaba tiene que seguir funcionando.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("clase,termino", [
    ("mariscos", "Mariscos"), ("mariscos", "camarones"), ("mariscos", "shellfish"),
    ("mariscos", "fruits de mer"), ("mariscos", "crostacei"), ("mariscos", "frutos do mar"),
    ("pescado", "Pescado"), ("pescado", "fish"), ("pescado", "poisson"),
    ("pescado", "pesce"), ("pescado", "peixe"), ("pescado", "tilapia"),
    ("lacteos", "Lacteos"), ("lacteos", "queso"), ("lacteos", "dairy"),
    ("lacteos", "produits laitiers"), ("lacteos", "latticini"), ("lacteos", "laticinios"),
    ("frutos secos", "Frutos Secos"), ("frutos secos", "nueces"), ("frutos secos", "tree nuts"),
    ("frutos secos", "fruits a coque"), ("frutos secos", "frutta a guscio"),
    ("gluten", "Gluten"), ("gluten", "Glutine"), ("gluten", "celiaco"),
    ("huevo", "Huevo"), ("huevo", "egg"), ("huevo", "oeuf"), ("huevo", "uovo"), ("huevo", "ovo"),
])
def test_no_regresion_lo_que_ya_bloqueaba_sigue_bloqueando(clase, termino) -> None:
    assert clinical_backstop_for_meal(_PLATOS[clase], allergies=[termino]), (
        f"REGRESIÓN: «{termino}» bloqueaba antes de este fix y ya no. [{_MARKER}]"
    )


@pytest.mark.parametrize("clase,termino", [
    # La frontera de palabra existe para que estos NO disparen. Sin ella, «ble» casa con
    # «roble» y «problema», y «cod» con «codorniz».
    ("gluten", "roble"),
    ("gluten", "problema con la cena"),
    ("pescado", "codorniz"),
])
def test_la_frontera_de_palabra_evita_el_falso_positivo_del_termino_corto(clase, termino) -> None:
    """Un guard que no puede fallar no informa: si estos disparasen, el vocabulario nuevo
    estaría comprando cobertura con ruido, y el ruido en una alerta clínica es lo que hace
    que se deje de leer."""
    assert not clinical_backstop_for_meal(_PLATOS[clase], allergies=[termino]), (
        f"«{termino}» dispara {clase} por subcadena: la frontera de palabra no está puesta "
        f"o no se aplica a los términos cortos. [{_MARKER}]"
    )
