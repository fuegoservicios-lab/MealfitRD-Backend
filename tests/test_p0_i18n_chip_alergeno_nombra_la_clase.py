"""[P0-I18N-ALERGIA-TEXTO-LIBRE-SOLO-ES · 2026-08-23] El chip de una alergia tiene que
nombrar la CLASE, no uno de sus miembros.

MEDIDO antes de este fix, sobre los cuatro catálogos:

    clave 'Nueces'  ->  en-US "Nuts"              (clase, correcto)
                        it-IT "Frutta a guscio"   (clase, correcto — término del Reglamento)
                        fr-FR "Noix"              *** UN fruto seco: la NUEZ ***
                        pt-BR "Nozes"             *** UN fruto seco: las NUECES ***

Por qué importa, y por qué es parte del P0 y no un gap de traducción: el chip es la vía
SEGURA de declarar una alergia —manda `val` canónico español, así que el motor lo entiende
siempre— y el campo libre es la insegura. Un francés alérgico a las ALMENDRAS que lee
«Noix» no marca ese chip, porque «noix» es la nuez y no la almendra. Se va al campo libre,
escribe «amandes», y cae justo en el hueco que este P-fix acaba de cerrar por el otro lado.
O sea: la etiqueta estrecha EMPUJA al usuario del camino seguro al peligroso.

Es la observación de `P2-ALLERGEN-CHIPS-REACH-ENGINE` —«media seguridad de un chip es que
el usuario no tenga que acordarse»— aplicada a la etiqueta en vez de a la existencia del
chip: de nada sirve tener el chip si no se llama como lo que cubre.

LO QUE NO SE TOCA: el `val`. Sigue siendo `"Frutos Secos"` en español canónico en los cinco
idiomas, porque es lo que resuelve el motor. Aquí sólo se vigila lo que el usuario LEE.

tooltip-anchor: P0-I18N-CHIP-ALERGENO-NOMBRA-LA-CLASE
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_MARKER = "P0-I18N-CHIP-ALERGENO-NOMBRA-LA-CLASE"
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LOCALES = _FRONTEND / "src" / "i18n" / "locales"
_IDIOMAS = ("en-US", "fr-FR", "it-IT", "pt-BR")

# Miembros concretos que NO pueden ser, ellos solos, la etiqueta de la clase entera.
# (Si la etiqueta los CONTIENE junto a algo más —«Castanhas e nozes»— es correcta: nombra
#  la clase y de paso el miembro que la gente busca.)
_MIEMBRO_SUELTO = {
    "fr-FR": {"noix", "amande", "amandes", "noisette", "noisettes", "cajou", "pistache"},
    "pt-BR": {"nozes", "noz", "amendoa", "amendoas", "avela", "avelas", "castanha"},
    "en-US": {"walnut", "walnuts", "almond", "almonds", "hazelnut", "cashew"},
    "it-IT": {"noci", "noce", "mandorla", "mandorle", "nocciola", "anacardi"},
}


def _catalogo(idioma: str) -> dict:
    return json.loads((_LOCALES / f"{idioma}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("idioma", _IDIOMAS)
def test_el_chip_de_frutos_secos_no_se_llama_como_un_solo_fruto(idioma) -> None:
    """La etiqueta de la clase no puede ser el nombre de UNO de sus miembros."""
    valor = _catalogo(idioma).get("Nueces")
    assert valor, f"falta la clave 'Nueces' en {idioma} [{_MARKER}]"
    normalizado = valor.strip().lower()
    assert normalizado not in _MIEMBRO_SUELTO[idioma], (
        f"{idioma}: el chip de frutos secos se llama «{valor}», que es UN fruto seco y no la "
        f"clase. Un alérgico a las almendras no lo marca, se va al campo libre y cae en el "
        f"hueco que este P-fix cierra por el otro lado. [{_MARKER}]"
    )


# NOTA para quien venga detrás: aquí había un segundo test que afirmaba que los `val`
# canónicos («Mariscos», «Pescado»…) NO podían aparecer como CLAVE en los catálogos. Es
# falso, y lo cazó su propia ejecución: el chip es
#
#     { val: "Mariscos", label: t('Mariscos') }
#
# o sea que la MISMA cadena hace de identificador y de clave de traducción, y que sea clave
# es correcto. Lo que importa no es que la cadena no esté en el catálogo, sino que el `val`
# se mande SIN pasar por `t()` — y eso ya lo ancla `test_p1_i18n_chips_traducidos.py`, que
# lee el JSX. Se escribe en vez de borrarse porque la confusión es fácil de repetir.
