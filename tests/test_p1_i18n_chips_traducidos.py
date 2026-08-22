"""[P1-I18N-CHIPS-SIN-TRADUCIR · 2026-08-21] La rejilla de alergias se leía en español.

El diseño YA era correcto y está escrito en el propio código
(`QDislikes.jsx`: «el `val` es el nombre de alimento que consume el motor: NO se
traduce nunca. Lo que se traduce es el `label` del chip, que es lo único que el usuario
lee»). Lo que faltaba era rellenar la traducción: 14 de las 15 etiquetas eran IDÉNTICAS
al español en los cuatro catálogos.

Así que un anglófono abría la pantalla de ALERGIAS —la más sensible del producto— y
leía «Mariscos», «Hongos», «Lácteos», «Nueces». El único chip que entendía acababa
siendo el centinela «None».

LAS DOS MITADES, y la segunda es la que importa de verdad:

  1. La ETIQUETA se traduce. Si vuelve a quedarse igual que el español, este test lo
     dice.
  2. El VALOR no, jamás. `val` es el identificador con el que `pantry_names_match`, el
     guard de coherencia recetas↔lista y el backstop clínico de alergias resuelven.
     Traducir «Mariscos» ahí rompe las tres, y dos de ellas EN SILENCIO: la nevera deja
     de descontar y una alergia deja de detectarse sin que nada avise.

El riesgo real de este P-fix no es olvidarse de traducir una etiqueta: es que alguien,
viendo que «la etiqueta se traduce», arrastre el `val` con ella. Por eso la segunda
mitad está anclada con nombres concretos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT = _ROOT / "frontend" / "src"
_PREGUNTAS = _FRONT / "components" / "assessment" / "questions"
_LOCALES = _FRONT / "i18n" / "locales"

_MARKER = "P1-I18N-CHIPS-SIN-TRADUCIR"

_FICHEROS = ("QAllergies.jsx", "QDislikes.jsx", "QSupplements.jsx")

# Etiquetas que legítimamente NO cambian entre idiomas. Cada entrada necesita su razón:
# una whitelist sin motivo es indistinguible de un olvido.
_IGUALES_A_PROPOSITO = {
    "Gluten": "misma palabra en en/pt/fr/it",
    # «Cilantro» ES la palabra inglesa: en EE. UU. la hoja se llama así (el «coriander»
    # británico designa la semilla). Traducirla sería el error, no dejarla.
    "Cilantro": "en-US usa «cilantro» para la hoja; «coriander» es la semilla",
}


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def _etiquetas() -> set[str]:
    claves = set()
    for nombre in _FICHEROS:
        for m in re.finditer(r"label:\s*t\('([^']+)'\)", _leer(_PREGUNTAS / nombre)):
            claves.add(m.group(1))
    return claves


def _valores() -> set[str]:
    vals = set()
    for nombre in _FICHEROS:
        for m in re.finditer(r'val:\s*"([^"]+)"', _leer(_PREGUNTAS / nombre)):
            vals.add(m.group(1))
    return vals


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_las_etiquetas_de_chip_estan_traducidas(locale: str) -> None:
    etiquetas = _etiquetas()
    assert etiquetas, "no extraje ninguna `label: t('…')` — ¿cambió el estilo?"
    cat = json.loads(_leer(_LOCALES / f"{locale}.json"))

    sin_traducir = [
        k for k in sorted(etiquetas)
        if k not in _IGUALES_A_PROPOSITO and cat.get(k) == k
    ]
    assert not sin_traducir, (
        f"{locale}: estas etiquetas de chip siguen idénticas al español: "
        f"{sin_traducir}. Es la pantalla de alergias — un usuario que no lee español "
        f"no puede declarar a qué es alérgico. [{_MARKER}]"
    )


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_toda_etiqueta_existe_en_el_catalogo(locale: str) -> None:
    """Una clave ausente cae al español, que es el estado del que venimos."""
    cat = json.loads(_leer(_LOCALES / f"{locale}.json"))
    faltan = sorted(k for k in _etiquetas() if k not in cat)
    assert not faltan, f"{locale}: etiquetas sin entrada en el catálogo: {faltan} [{_MARKER}]"


def test_los_valores_del_motor_siguen_en_espanol_canonico() -> None:
    """LA MITAD QUE IMPORTA. El riesgo de este P-fix no es olvidar una etiqueta: es que
    alguien arrastre el `val` al traducir la etiqueta de al lado.

    `pantry_names_match`, el guard de coherencia y el backstop de alergias resuelven por
    esta cadena EXACTA. Traducirla rompe las tres, y dos en silencio.
    """
    vals = _valores()
    assert vals, "no extraje ningún `val:` — ¿cambió el estilo?"

    # Un puñado de nombres que tienen que seguir ahí, tal cual, en español.
    #
    # [corregido 2026-08-21] Antes esta lista incluía «Nueces», y «Nueces» NO es un `val`:
    # es la ETIQUETA del chip cuyo valor es «Frutos Secos». O sea, el test que existe para
    # impedir que se confundan valor y etiqueta las estaba confundiendo él mismo. Se toman
    # los valores REALES del formulario.
    canonicos = {"Mariscos", "Hongos", "Pescado", "Frutos Secos", "Huevo", "Soya"}
    perdidos = canonicos - vals
    assert not perdidos, (
        f"Estos `val` ya no están en español canónico: {sorted(perdidos)}. Son los "
        "identificadores con los que el motor resuelve la Nevera, la coherencia "
        f"receta↔lista y el backstop de alergias. [{_MARKER}]"
    )


def test_ninguna_etiqueta_traducida_se_coló_como_valor() -> None:
    """La forma concreta del accidente: que el `val` pase a ser el texto en inglés."""
    cat = json.loads(_leer(_LOCALES / "en-US.json"))
    traducciones_en = {cat.get(k) for k in _etiquetas() if cat.get(k) and cat.get(k) != k}
    colados = sorted(_valores() & traducciones_en)
    assert not colados, (
        f"Estos `val` coinciden con una traducción inglesa: {colados}. El valor tiene "
        f"que ser SIEMPRE el nombre español del catálogo del motor. [{_MARKER}]"
    )
