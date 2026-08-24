"""[P3-DISHLIB-TRANSFORM-DO-EXAMPLES · 2026-08-23] El cierre del bloque de inspiración pedía platos
transformados poniendo de ejemplo arepitas y bollitos, en los seis países.

Medido antes del arreglo (knob de países encendido): los renders de ES, MX, US, PR y CO terminaban
TODOS con la misma frase

    «🎯 Incluye HOY al menos 1 plato(s) TRANSFORMADO(s) (panqueques/arepitas/bollitos/guiso u
      horneado con nombre propio) siempre que encaje…»

y 'arepitas' y 'bollitos' son entradas de `constants._DO_LEXICON_NEUTRAL` — es decir, léxico
dominicano que el resto del stack beta ya neutraliza. Además es la ÚLTIMA frase del bloque más
concreto del prompt, la posición donde `P1-DIET-BLIND-DIRECTIVES` midió que el modelo obedece al
ejemplo antes que a la directiva de cabecera.

El arreglo describe la FORMA (masas, tortitas, croquetas, horneados) en vez de recetas concretas:
una categoría no pierde nada. DO conserva su literal byte a byte.

Lo que este P-fix NO hace, a propósito: aplicar `neutralize_do_lexicon` al bloque entero. Ese mapa
manda `casabe → «pan tostado integral»` y **`Casabe` es una fila VIVA del catálogo** — un nombre de
alimento es un identificador de punta a punta (`pantry_names_match`, guard de coherencia, backstop
de alergias resuelven por él).
"""
from __future__ import annotations

import pytest

import dish_library
from constants import COUNTRY_PROFILES

_BETA = ["ES", "MX", "US", "PR", "CO"]

# El literal histórico de DO. Vive AQUÍ porque su byte-identidad es la invariante que se protege:
# si el trailer de DO cambia, este test debe ponerse rojo y obligar a decidirlo a propósito.
_TRAILER_DO = ("plato(s) TRANSFORMADO(s) (panqueques/arepitas/bollitos/guiso u horneado con "
               "nombre propio) siempre que encaje con los macros, el horario y las reglas "
               "clínicas del día.")

_ESQUELETO = {
    "protein_pool": ["pollo", "huevo", "queso"],
    "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"],
}


@pytest.fixture(autouse=True)
def _knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _render(country=None) -> str:
    out = dish_library.build_dish_library_context(_ESQUELETO, 1, country=country)
    if not out:
        pytest.skip("biblioteca de platos deshabilitada en este entorno")
    return out


def test_do_conserva_su_trailer_byte_a_byte():
    assert _TRAILER_DO in _render(None), (
        "el trailer dominicano cambió: la byte-identidad de DO es la invariante que sostiene el "
        "diseño F1, así que esto se decide a propósito, no de rebote."
    )


@pytest.mark.parametrize("cc", _BETA)
def test_el_trailer_beta_no_nombra_platos_dominicanos(cc):
    render = _render(cc)
    cola = render[render.rfind("TRANSFORMADO(s)"):]
    assert cola, "el trailer de transformados desapareció del render beta"
    for token in ("arepitas", "bollitos", "mangú", "mofongo", "casabe"):
        assert token not in cola.lower(), (
            f"[{cc}] el último renglón del bloque más concreto del prompt sigue pidiendo "
            f"'{token}': {cola!r}"
        )


@pytest.mark.parametrize("cc", _BETA)
def test_el_trailer_beta_sigue_pidiendo_transformar(cc):
    """PERDER LA INSTRUCCIÓN sería peor que tenerla en dominicano: vuelven los staples sueltos."""
    render = _render(cc)
    assert "TRANSFORMADO(s)" in render
    cola = render[render.rfind("TRANSFORMADO(s)"):]
    assert "nombre propio" in cola, f"[{cc}] el trailer perdió su criterio: {cola!r}"
    assert any(t in cola.lower() for t in ("masas", "tortitas", "croquetas", "horneados")), (
        f"[{cc}] el trailer beta no describe ninguna FORMA de preparación: {cola!r}"
    )


@pytest.mark.parametrize("cc", _BETA)
def test_el_encabezado_y_el_trailer_deciden_el_pais_igual(cc):
    """Las dos decisiones de país del módulo salen de la MISMA derivación (`_canon_country_or_do`).

    Dos copias de la misma pregunta es como nacen los espejos que driftan: el encabezado ya decía
    «INSPIRACIÓN DE ESPAÑA» mientras el trailer seguía pidiendo arepitas.
    """
    render = _render(cc)
    assert f"INSPIRACIÓN DE {COUNTRY_PROFILES[cc]['name_es'].upper()}" in render
    assert dish_library._canon_country_or_do(cc) == cc
    assert dish_library._canon_country_or_do(None) == "DO"
    assert dish_library._canon_country_or_do("marte") == "DO", "fail-safe a DO"
