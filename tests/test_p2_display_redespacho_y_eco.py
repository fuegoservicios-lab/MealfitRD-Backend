"""[P2-DISPLAY-REDESPACHO-SIN-FILTRO + P2-DISPLAY-ECO-NOMBRE · 2026-08-21]

Dos defectos de la capa `_display` que salieron de la misma lectura: uno hace que se
pague dos veces por lo ya traducido, y el otro deja pasar por traducción algo que no
lo es.

── El re-despacho ───────────────────────────────────────────────────────────────
`_collect_targets` no tenía NI UNA referencia a `_display`. Reunía todas las comidas
de los días del lote y las mandaba al LLM, estuvieran ya traducidas o no. Como el
enriquecimiento se re-dispara —al cambiar de idioma, al llegar un chunk nuevo, y desde
el recovery— cada disparo volvía a pagar por lo mismo.

Filtrar ahí convierte además la reanudación en barata: un enriquecimiento que se cortó
a la mitad retoma sólo lo que falta, en vez de rehacerlo entero.

── El eco ───────────────────────────────────────────────────────────────────────
`_validate_plan_name` sólo comprobaba `isinstance(str)` y strip no vacío. Un LLM que
devuelve el nombre SIN traducir —lo más común cuando el nombre es un plato criollo que
no sabe traducir— pasaba la validación, se persistía como `_display[locale].name`, y a
partir de ahí el gate de «¿ya está traducido?» daba SÍ. Resultado: el nombre se queda
en español para siempre y nadie lo vuelve a intentar.

Un eco no es una traducción. La comparación es tolerante a mayúsculas, acentos y
espacio sobrante, porque «HABICHUELAS  guisadas» tampoco lo es.

CUIDADO, y por eso el test lo fija: hay nombres que legítimamente NO cambian entre
idiomas (una marca, un nombre propio). Devolver `None` para ellos es correcto: significa
«no hay traducción que aportar», el motor cae al español y el usuario ve lo mismo que
vería con el eco. Lo que se gana es que el gate deja de mentir.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pdi = importlib.import_module("plan_display_i18n")

_MARKER = "P2-DISPLAY-REDESPACHO-SIN-FILTRO"
_MARKER_ECO = "P2-DISPLAY-ECO-NOMBRE"


def _meal(nombre="Habichuelas guisadas", display=None) -> dict:
    m = {
        "name": nombre,
        "desc": "Guiso tradicional dominicano con habichuelas rojas.",
        "recipe": ["Sofreir el sazon.", "Cocinar 20 minutos."],
        "ingredients": ["30 g Habichuelas rojas", "1 unidad Cebolla"],
    }
    if display is not None:
        m["_display"] = display
    return m


def _display_completo(locale="en-US") -> dict:
    return {
        locale: {
            "name": "Stewed red beans",
            "description": "Traditional Dominican red bean stew.",
            "recipe": ["Sauté the sofrito.", "Cook for 20 minutes."],
            "ingredients": ["30 g Red beans", "1 unit Onion"],
        }
    }


# ───────────────────────── el re-despacho ─────────────────────────

def test_una_comida_ya_traducida_no_se_reenvia() -> None:
    days = [{"meals": [_meal(display=_display_completo("en-US"))]}]
    targets = pdi._collect_targets(days, [0], locale="en-US")
    assert targets == [], (
        "Una comida con `_display['en-US']` completo se vuelve a mandar al LLM. Cada "
        f"re-disparo del enriquecimiento paga otra vez por lo mismo. [{_MARKER}]"
    )


def test_una_comida_traducida_a_OTRO_idioma_si_se_reenvia() -> None:
    """MUTACIÓN DE CONTROL del test de arriba. Si el filtro ignorase el locale,
    aquel pasaría igual y no probaría nada."""
    days = [{"meals": [_meal(display=_display_completo("en-US"))]}]
    targets = pdi._collect_targets(days, [0], locale="fr-FR")
    assert len(targets) == 1, (
        f"Traducido a inglés no es traducido a francés. [{_MARKER}]"
    )


def test_una_comida_sin_display_se_reenvia() -> None:
    days = [{"meals": [_meal()]}]
    assert len(pdi._collect_targets(days, [0], locale="en-US")) == 1


@pytest.mark.parametrize(
    "entrada,etiqueta",
    [
        ({"en-US": {"name": "", "recipe": ["a", "b"], "ingredients": ["x", "y"]}}, "nombre vacío"),
        ({"en-US": {"name": "X", "recipe": ["a"], "ingredients": ["x", "y"]}}, "recipe corto"),
        ({"en-US": {"name": "X", "recipe": ["a", "b"], "ingredients": ["x"]}}, "ingredients corto"),
        ({"en-US": "no soy un dict"}, "entrada no-dict"),
    ],
)
def test_un_display_incompleto_se_reenvia(entrada, etiqueta) -> None:
    """El filtro tiene que exigir un display USABLE, no su mera presencia.

    Es la misma lección que P1-I18N-GATE-VALOR dejó en el validador de catálogos:
    medir que la clave existe no es medir que sirve. Un display a medias que se dé
    por bueno deja esa comida en español para siempre, porque nadie la reintenta.
    """
    days = [{"meals": [_meal(display=entrada)]}]
    assert len(pdi._collect_targets(days, [0], locale="en-US")) == 1, (
        f"Un `_display` con {etiqueta} se dio por bueno y la comida no se reintenta. "
        f"[{_MARKER}]"
    )


def test_sin_locale_se_comporta_como_antes() -> None:
    """NO REGRESIÓN. La firma gana un parámetro; los llamadores que no lo pasen
    tienen que seguir viendo exactamente la conducta anterior."""
    days = [{"meals": [_meal(display=_display_completo("en-US"))]}]
    assert len(pdi._collect_targets(days, [0])) == 1


# ───────────────────────── el eco ─────────────────────────

@pytest.mark.parametrize(
    "valor,etiqueta",
    [
        ("Habichuelas guisadas", "idéntico"),
        ("  Habichuelas guisadas  ", "con espacio sobrante"),
        ("HABICHUELAS GUISADAS", "en mayúsculas"),
        ("Habichuelas  guisadas", "con espacio doble"),
    ],
)
def test_un_eco_del_nombre_no_cuenta_como_traduccion(valor, etiqueta) -> None:
    assert pdi._validate_plan_name(valor, original="Habichuelas guisadas") is None, (
        f"El nombre devuelto {etiqueta} respecto al original se aceptó como "
        "traducción. A partir de ahí el gate de «¿ya está traducido?» dice SÍ y el "
        f"nombre se queda en español para siempre. [{_MARKER_ECO}]"
    )


def test_una_traduccion_de_verdad_pasa() -> None:
    """MUTACIÓN DE CONTROL. Sin esto, una función que devolviera SIEMPRE `None`
    pasaría todos los casos de arriba."""
    assert pdi._validate_plan_name(
        "Stewed red beans", original="Habichuelas guisadas"
    ) == "Stewed red beans"


def test_sin_original_se_comporta_como_antes() -> None:
    """NO REGRESIÓN: los llamadores que no pasen el original conservan la conducta
    vieja (sólo strip + no vacío)."""
    assert pdi._validate_plan_name("Habichuelas guisadas") == "Habichuelas guisadas"
    assert pdi._validate_plan_name("   ") is None
    assert pdi._validate_plan_name(None) is None
