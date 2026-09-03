"""[P2-I18N-DISPLAY-PODA-INERTE-EN-SU-VALOR-MINIMO · 2026-08-23] El ajuste MÁS agresivo de la
poda de idiomas no podaba nada.

`_podar_locales` conserva el idioma activo más los `tope - 1` últimos. Con `tope = 1` eso es

    conservar = [k for k in disp_map if k != activo][-(1 - 1):]   ->   lista[-0:]

y `lista[-0:]` en Python **es la lista ENTERA**: `-0 == 0`, así que el slice arranca en el
principio, no en el final. O sea que poner `MEALFIT_PLAN_DISPLAY_I18N_MAX_LOCALES=1` —el
valor que un operador elegiría justo cuando el jsonb se le está yendo de las manos— deja la
poda en NO-OP total y conserva los cinco idiomas.

Es el modo de fallo más caro de un knob: el que sólo aparece en el extremo al que recurres en
una emergencia, y en el que además parece que has hecho algo.

Por qué P2 y no P1: el default es 2, y con 2 la poda funciona. Esto no está roto para nadie
hoy; está roto para quien lo necesite.

⚠️ NO se cambia el DEFAULT. `P1-DISPLAY-PODA-TIRA-TRABAJO-PAGADO` ya dejó escrito que el 2
es un cruce calculado entre re-pagar traducciones y multiplicar el jsonb, y que subirlo
«revierte un cruce que alguien ya calculó». Aquí sólo se arregla la aritmética del extremo.

tooltip-anchor: P2-I18N-DISPLAY-PODA-INERTE-EN-SU-VALOR-MINIMO
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from plan_display_i18n import _podar_locales

_MARKER = "P2-I18N-DISPLAY-PODA-INERTE-EN-SU-VALOR-MINIMO"

_CINCO = {"es-DO": 1, "en-US": 2, "pt-BR": 3, "fr-FR": 4, "it-IT": 5}


def _con_tope(n):
    return patch("plan_display_i18n._max_locales_display", return_value=n)


def test_con_tope_1_solo_queda_el_activo() -> None:
    """El extremo. `lista[-0:]` es la lista entera: el ajuste más agresivo era un no-op."""
    with _con_tope(1):
        fuera = _podar_locales(dict(_CINCO), "fr-FR")
    assert set(fuera) == {"fr-FR"}, (
        f"con tope=1 quedaron {sorted(fuera)} en vez de sólo el activo. `lista[-0:]` devuelve "
        f"la lista ENTERA, así que el valor al que recurre un operador con el jsonb "
        f"desbordado no poda nada. [{_MARKER}]"
    )


def test_con_tope_2_se_conserva_el_activo_y_uno_mas() -> None:
    """El default, que es el que gobierna hoy: no se toca su conducta."""
    with _con_tope(2):
        fuera = _podar_locales(dict(_CINCO), "fr-FR")
    assert "fr-FR" in fuera and len(fuera) == 2, (
        f"con tope=2 salieron {sorted(fuera)}: la conducta por defecto cambió. [{_MARKER}]"
    )


def test_el_activo_nunca_se_poda_aunque_sea_el_mas_viejo() -> None:
    with _con_tope(1):
        fuera = _podar_locales(dict(_CINCO), "es-DO")
    assert set(fuera) == {"es-DO"}, f"se podó el idioma ACTIVO. [{_MARKER}]"


@pytest.mark.parametrize("tope", [1, 2, 3, 4, 5, 9])
def test_nunca_se_devuelven_mas_de_tope_idiomas(tope) -> None:
    """La propiedad, para cualquier valor del knob — que es lo que un test del extremo suelto
    no da: con `tope` idiomas o menos no se toca nada, y con más nunca sobra ninguno."""
    with _con_tope(tope):
        fuera = _podar_locales(dict(_CINCO), "pt-BR")
    assert len(fuera) <= max(tope, 1), (
        f"tope={tope} devolvió {len(fuera)} idiomas. [{_MARKER}]"
    )
    assert "pt-BR" in fuera, f"tope={tope} podó el activo. [{_MARKER}]"


def test_no_toca_nada_si_ya_cabe() -> None:
    dos = {"fr-FR": 1, "es-DO": 2}
    with _con_tope(2):
        assert _podar_locales(dict(dos), "fr-FR") == dos
