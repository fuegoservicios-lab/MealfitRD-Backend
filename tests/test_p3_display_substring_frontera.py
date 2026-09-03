"""[P3-DISPLAY-SUBSTRING-SIN-FRONTERA · 2026-08-21] El check del canónico validaba por
accidente.

Cada línea de `ingredients` traducida tiene que llevar el nombre canónico español
literal —es el identificador del motor— y el validador lo comprobaba así:

    if strip_accents(canonical).lower() in strip_accents(translated_line).lower():

Substring, sin frontera de palabra. Es la clase de defecto que este repo ya ha pagado
TRES veces: «sal» dentro de Salami, «pollo» dentro de repollo, «res» dentro de fResco.

MEDIDO sobre los 347 nombres del catálogo (2026-08-21):

  · 17 nombres son subcadena de otro sin ser palabra: «Ajo» en «Ajonjolí», «Piña» en
    «Espinacas», «Uva» en «Uchuva», «Piñones» en «Champiñones», y «Sal» en NUEVE
    (Salami, Salmón, Salchichas, Salsa de soya, Salsa de tomate…).
  · 4 caen dentro de palabras inglesas corrientes: «Sal» en *salad*, *salmon* y *salt*;
    «Piña» en *spinach*.

Lo que eso significa en concreto: el LLM traduce «1 cdta de Sal» como «1 tsp salt» —sin
el paréntesis con el canónico, que es justo lo que el validador existe para exigir— y el
check dice que sí, porque «sal» está dentro de «salt». La línea se persiste sin
identificador y la Nevera deja de descontar esa fila, en silencio.

EL PRECIO, aceptado: con la frontera, un gloss que PLURALICE el canónico —«(Huevos)» por
«(Huevo)»— deja de validar y esa línea cae al español. Es lo correcto: la directiva pide
el canónico «literalmente, sin traducir, exactamente como en el original», y caer al
español es degradación, no corrupción. La alternativa —tolerar variaciones— es inventar
una regla de morfología por idioma que nadie puede validar.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MARKER = "P3-DISPLAY-SUBSTRING-SIN-FRONTERA"


@pytest.fixture()
def mod():
    import plan_display_i18n

    return importlib.reload(plan_display_i18n)


def _par(canonico: str, gloss: str, cifras: str = "") -> tuple[dict, dict]:
    """El original lleva LAS MISMAS cifras que el gloss, y eso no es cosmetica.

    `_conserva_las_cifras` (P2-DISPLAY-VALIDADOR-SIN-CIFRAS) corre ANTES que este check y
    cae al español por su cuenta si los números no cuadran. Con un original «5 g de Sal»
    contra un gloss «1 tsp salt», el que rechazaba era ESE guard — y la mitad de este
    fichero pasaba sin llegar a ejecutar la frontera que dice medir. Lo cazó el bloque de
    control, que es exactamente para lo que está.
    """
    original = {
        "day_idx": 0, "meal_idx": 0,
        "name": "Plato", "description": "Desc.",
        "recipe": ["Cocina 10 min."],
        "ingredients": [f"{cifras} de {canonico}".strip() if cifras else f"de {canonico}"],
    }
    traducido = {
        "i": 0, "name": "Dish", "description": "Desc.",
        "recipe": ["Cook 10 min."],
        "ingredients": [gloss],
    }
    return original, traducido


# ============================================================
# Los casos medidos
# ============================================================

@pytest.mark.parametrize(
    "canonico,gloss,cifras",
    [
        # El canónico DESAPARECIÓ del gloss; lo que valida es una subcadena accidental.
        ("Sal", "1 tsp salt", "1"),
        ("Sal", "120 g smoked salmon", "120"),
        ("Sal", "1 portion of green salad", "1"),
        ("Piña", "80 g fresh spinach", "80"),
        ("Ajo", "10 g sesame (Ajonjolí)", "10"),
        ("Uva", "50 g cape gooseberry (Uchuva)", "50"),
        ("Sal", "60 g italian sausage (Salchicha italiana)", "60"),
    ],
)
def test_un_canonico_que_solo_aparece_dentro_de_otra_palabra_no_vale(
    mod, canonico: str, gloss: str, cifras: str
) -> None:
    original, traducido = _par(canonico, gloss, cifras)
    d = mod._validate_and_build_display(original, traducido)
    assert d is not None, "el fallback es POR LÍNEA, no descarta el meal"
    assert d["ingredients"] == original["ingredients"], (
        f"«{canonico}» se dio por presente en «{gloss}» porque es SUBCADENA de otra "
        f"palabra. La línea se persiste sin el identificador del motor y la Nevera deja "
        f"de descontar esa fila, en silencio. [{_MARKER}]"
    )


@pytest.mark.parametrize(
    "canonico,gloss,cifras",
    [
        ("Sal", "1 tsp table salt (Sal)", "1"),
        ("Sal", "1 c. à café de sel (Sal)", "1"),
        ("Piña", "80 g fresh pineapple (Piña)", "80"),
        ("Ajo", "2 cloves of garlic (Ajo)", "2"),
        ("Habichuelas rojas", "30 g dried red beans (Habichuelas rojas)", "30"),
        # Con puntuación pegada a los lados: la frontera de palabra tiene que verla.
        ("Ajo", "2 cloves garlic [Ajo]", "2"),
        ("Ajo", "garlic —Ajo— minced", ""),
    ],
)
def test_el_canonico_presente_de_verdad_sigue_valiendo(
    mod, canonico: str, gloss: str, cifras: str
) -> None:
    """MUTACIÓN DE CONTROL, y la mitad que se rompe fácil: una frontera mal puesta
    convierte el arreglo en «todo cae al español», que es peor que el bug."""
    original, traducido = _par(canonico, gloss, cifras)
    d = mod._validate_and_build_display(original, traducido)
    assert d["ingredients"] == [gloss], (
        f"«{canonico}» SÍ está como palabra en «{gloss}» y aun así cayó al español. "
        f"La frontera está rechazando glosses buenos. [{_MARKER}]"
    )


def test_el_canonico_con_acentos_se_compara_sin_ellos(mod) -> None:
    """La comparación normaliza acentos en las DOS puntas; eso no cambia."""
    original, traducido = _par("Piña", "80 g pineapple (Pina)", "80")
    d = mod._validate_and_build_display(original, traducido)
    assert d["ingredients"] == ["80 g pineapple (Pina)"], (
        f"la normalización de acentos se perdió al añadir la frontera. [{_MARKER}]"
    )


def test_el_precio_declarado_un_canonico_pluralizado_cae_al_espanol(mod) -> None:
    """EL TRADE-OFF, anclado en positivo para que sea una decisión y no una sorpresa.

    Con la frontera, «(Huevos)» ya no vale por «Huevo». Es lo correcto —la directiva
    pide el canónico literal— y caer al español es degradación, no corrupción. Si algún
    día se decide tolerarlo, este test es el que hay que cambiar a sabiendas.
    """
    original, traducido = _par("Huevo", "2 units of egg (Huevos)", "2")
    d = mod._validate_and_build_display(original, traducido)
    assert d["ingredients"] == original["ingredients"], (
        f"un canónico pluralizado pasó. Si es un cambio deliberado, actualiza este test "
        f"y la nota del validador. [{_MARKER}]"
    )
