"""[P2-I18N-PAYPAL-LOCALE · 2026-08-21] El widget de pago hablaba otro idioma.

`PayPalScriptProvider` sin `locale` deja que PayPal deduzca del navegador o de la IP.
Así que un francés con el móvil en inglés, que había puesto la app en francés, veía
aparecer el pago en inglés: el único punto del producto donde se entrega dinero, en un
idioma que el usuario ya había rechazado explícitamente.

POR QUÉ UN MAPA EXPLÍCITO Y NO `getLocale().replace('-', '_')`:

PayPal usa `xx_XX`, así que la transformación mecánica *parece* funcionar. No lo hace:
su lista de locales soportados NO es la nuestra. `es_DO` no existe en PayPal, y un
locale no soportado no degrada al default — rompe el widget. La conversión automática
es precisamente la forma de meter un valor inválido sin enterarse.

POR QUÉ EL ESPAÑOL SE OMITE EN VEZ DE MAPEARSE A `es_ES`:

Omitir es la conducta de HOY (PayPal deduce), y para un dominicano deducir acierta.
Mandarle `es_ES` sería cambiarla, y a peor. La omisión es una decisión, no un hueco —
por eso este test la ancla en positivo: si alguien "completa el mapa" añadiendo
`'es-DO': 'es_ES'`, el test falla y le cuenta por qué.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_MODAL = _ROOT / "frontend" / "src" / "components" / "dashboard" / "PaymentModal.jsx"

_MARKER = "P2-I18N-PAYPAL-LOCALE"

_ESPERADO = {"en-US": "en_US", "pt-BR": "pt_BR", "fr-FR": "fr_FR", "it-IT": "it_IT"}


def _fuente() -> str:
    if not _MODAL.exists():
        pytest.skip(f"{_MODAL} no existe en este checkout (repos hermanos)")
    return _MODAL.read_text(encoding="utf-8")


def _mapa() -> dict[str, str]:
    s = _fuente()
    m = re.search(r"const\s+_PAYPAL_LOCALE\s*=\s*\{(.*?)\}\s*\[", s, re.S)
    assert m, (
        f"no encontré `const _PAYPAL_LOCALE = {{…}}[locale]` en PaymentModal.jsx. "
        f"Si el mapa se renombró, actualiza este guard. [{_MARKER}]"
    )
    return dict(re.findall(r"['\"]([\w-]+)['\"]\s*:\s*['\"](\w+)['\"]", m.group(1)))


def test_el_mapa_cubre_los_cuatro_idiomas_traducidos() -> None:
    mapa = _mapa()
    faltan = {k: v for k, v in _ESPERADO.items() if mapa.get(k) != v}
    assert not faltan, (
        f"El mapa de locales de PayPal no cubre (o cubre mal): {faltan}. Un usuario "
        f"que eligió ese idioma pagaría en otro. [{_MARKER}]"
    )


def test_el_espanol_se_omite_a_proposito() -> None:
    """La omisión es la decisión, no el hueco.

    `es_DO` no existe en PayPal y `es_ES` cambiaría a peor la conducta actual de un
    dominicano. Si alguien "completa" el mapa, que se entere aquí.
    """
    mapa = _mapa()
    colado = {k: v for k, v in mapa.items() if k.startswith("es")}
    assert not colado, (
        f"El español entró en el mapa de PayPal: {colado}. `es_DO` no existe en su "
        f"lista (un locale no soportado ROMPE el widget, no degrada) y `es_ES` le "
        f"mandaría España a un dominicano, que es peor que dejar deducir. La omisión "
        f"es deliberada. [{_MARKER}]"
    )


def test_no_se_deriva_el_locale_por_transformacion_mecanica() -> None:
    """`replace('-', '_')` sobre nuestro locale es la forma de colar `es_DO`."""
    s = _fuente()
    ventana = s[max(0, s.find("_PAYPAL_LOCALE") - 400): s.find("initialOptions") + 400]
    sospechoso = re.search(r"locale[^;\n]{0,60}\.replace\(\s*['\"]-['\"]", ventana)
    assert not sospechoso, (
        f"Se está derivando el locale de PayPal transformando el nuestro "
        f"(`{sospechoso.group(0) if sospechoso else ''}`). Su lista NO es la nuestra: "
        f"el mapa tiene que ser explícito. [{_MARKER}]"
    )


def test_el_locale_llega_a_initialOptions_y_solo_si_existe() -> None:
    """Un `locale: undefined` en las opciones es distinto de no mandarlo: el spread
    condicional es lo que preserva la conducta de hoy para el español."""
    s = _fuente()
    assert re.search(
        r"\.\.\.\(\s*_PAYPAL_LOCALE\s*\?\s*\{\s*locale:\s*_PAYPAL_LOCALE\s*\}\s*:\s*\{\s*\}\s*\)",
        s,
    ), (
        f"`_PAYPAL_LOCALE` no se está esparciendo condicionalmente dentro de "
        f"`initialOptions`. Sin el condicional, el español manda `locale: undefined` "
        f"en vez de no mandar nada. [{_MARKER}]"
    )


def test_el_componente_esta_suscrito_al_cambio_de_idioma() -> None:
    """Leer `getLocale()` una vez no basta: el modal tiene que re-renderizar cuando el
    usuario cambia de idioma, y eso lo da el hook."""
    s = _fuente()
    assert "useI18n(" in s, (
        f"PaymentModal ya no usa `useI18n()`. Sin la suscripción, el widget se queda "
        f"en el idioma que hubiera al montar. [{_MARKER}]"
    )
