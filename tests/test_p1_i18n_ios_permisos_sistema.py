"""[P1-I18N-IOS-PERMISOS-SISTEMA-SOLO-ES · 2026-08-23] Los tres modales de permiso de iOS
salían en español a los cinco idiomas, y ese texto ya viaja dentro del binario que está en
TestFlight.

`Info.plist` tenía `CFBundleDevelopmentRegion=es`, ningún `CFBundleLocalizations`, sólo
`Base.lproj`, y los tres `NS*UsageDescription` en español fijo. iOS pinta esas cadenas en
un modal del SISTEMA que la capa web no puede traducir: un francés que toca «Escanear
comida» lee, en el momento exacto de decidir si concede la cámara, una frase en español.
Y como el permiso se pide una sola vez y la negativa se recuerda, el precio de un «no» por
no entender la frase se paga para siempre.

El plan v2 lo clasificó como «checklist previo al envío, no gap abierto». Esa clasificación
CADUCÓ: la app está subida a App Store Connect (build #6, 2026-08-22).

CÓMO LO RESUELVE iOS, y por qué cada pieza:
  · `<lang>.lproj/InfoPlist.strings` — iOS busca ahí las claves `NS*UsageDescription` en el
    idioma del dispositivo antes de caer al `Info.plist`. Un fichero por idioma.
  · `CFBundleLocalizations` — declara qué idiomas soporta el bundle; sin él, iOS no mira
    los `.lproj` aunque existan.
  · El `Info.plist` se queda en ESPAÑOL como fallback: es el idioma base del producto y el
    que ve quien tenga el dispositivo en un idioma que no ofrecemos.
  · Los ficheros tienen que estar REGISTRADOS en `project.pbxproj` (un `PBXVariantGroup`
    igual que los storyboards) o Xcode no los empaqueta. Un `.lproj` en disco que no esté
    en el proyecto es un fichero que no viaja.

LO QUE NO SE TRADUCE: el nombre de la app (`CFBundleDisplayName=Bioboros`), que es marca.

tooltip-anchor: P1-I18N-IOS-PERMISOS-SISTEMA-SOLO-ES
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P1-I18N-IOS-PERMISOS-SISTEMA-SOLO-ES"
_IOS = Path(__file__).resolve().parents[2] / "frontend" / "ios" / "App" / "App"
_PLIST = _IOS / "Info.plist"
_PBXPROJ = _IOS.parent / "App.xcodeproj" / "project.pbxproj"

# Los cinco idiomas del producto, en el código que iOS entiende (el primario del tag).
_IDIOMAS_IOS = {"es": "es-DO", "en": "en-US", "pt-BR": "pt-BR", "fr": "fr-FR", "it": "it-IT"}

_PERMISOS = (
    "NSCameraUsageDescription",
    "NSPhotoLibraryUsageDescription",
    "NSPhotoLibraryAddUsageDescription",
)


def _saltar_si_no_hay_ios() -> None:
    if not _PLIST.exists():
        pytest.skip(f"no existe {_PLIST} (¿repo hermano sin clonar?)")


def _plist() -> str:
    _saltar_si_no_hay_ios()
    return _PLIST.read_text(encoding="utf-8")


def _strings(lang: str) -> dict:
    p = _IOS / f"{lang}.lproj" / "InfoPlist.strings"
    assert p.exists(), (
        f"falta {p.relative_to(_IOS.parent.parent)}: el modal de permisos saldrá en español "
        f"para {_IDIOMAS_IOS[lang]}. [{_MARKER}]"
    )
    txt = p.read_text(encoding="utf-8")
    return dict(re.findall(r'^"([A-Za-z]+)"\s*=\s*"((?:[^"\\]|\\.)*)";', txt, re.M))


def test_el_bundle_declara_los_cinco_idiomas() -> None:
    plist = _plist()
    m = re.search(r"<key>CFBundleLocalizations</key>\s*<array>(.*?)</array>", plist, re.S)
    assert m, (
        f"`CFBundleLocalizations` no está en Info.plist: sin él iOS no mira los `.lproj` "
        f"aunque existan. [{_MARKER}]"
    )
    declarados = set(re.findall(r"<string>([^<]+)</string>", m.group(1)))
    faltan = sorted(set(_IDIOMAS_IOS) - declarados)
    assert not faltan, f"idiomas sin declarar en CFBundleLocalizations: {faltan} [{_MARKER}]"


@pytest.mark.parametrize("lang", sorted(k for k in _IDIOMAS_IOS if k != "es"))
def test_cada_idioma_traduce_los_tres_permisos(lang: str) -> None:
    """La CONDUCTA que importa: cada idioma no-español tiene las tres cadenas, y distintas
    del español (una copia del español con otro nombre de fichero es el mismo defecto)."""
    _saltar_si_no_hay_ios()
    plist = _plist()
    es = {
        k: re.search(rf"<key>{k}</key>\s*<string>([^<]*)</string>", plist).group(1)
        for k in _PERMISOS
    }
    traducidas = _strings(lang)
    for k in _PERMISOS:
        assert k in traducidas and traducidas[k].strip(), (
            f"{lang}.lproj/InfoPlist.strings no trae `{k}`. [{_MARKER}]"
        )
        assert traducidas[k].strip() != es[k].strip(), (
            f"{lang}: `{k}` es el español tal cual — eco, no traducción. [{_MARKER}]"
        )
        assert "Bioboros" in traducidas[k], (
            f"{lang}: `{k}` perdió el nombre de la app; Apple exige que la frase diga QUIÉN "
            f"pide el permiso y PARA QUÉ. [{_MARKER}]"
        )


def test_el_espanol_del_plist_sigue_siendo_el_fallback() -> None:
    """El Info.plist NO se vacía: es lo que ve un dispositivo en un idioma que no ofrecemos."""
    plist = _plist()
    for k in _PERMISOS:
        m = re.search(rf"<key>{k}</key>\s*<string>([^<]*)</string>", plist)
        assert m and m.group(1).strip(), f"`{k}` desapareció del Info.plist [{_MARKER}]"
        assert "Bioboros" in m.group(1), f"`{k}` del Info.plist perdió el nombre de la app [{_MARKER}]"


def test_los_lproj_estan_registrados_en_el_proyecto_de_xcode() -> None:
    """Un `.lproj` en disco que no esté en `project.pbxproj` es un fichero que NO viaja en
    el .ipa. Xcode los agrupa en un `PBXVariantGroup` con un hijo por idioma."""
    _saltar_si_no_hay_ios()
    assert _PBXPROJ.exists(), f"no existe {_PBXPROJ} [{_MARKER}]"
    pbx = _PBXPROJ.read_text(encoding="utf-8")
    assert re.search(r"name = InfoPlist\.strings;", pbx), (
        f"`InfoPlist.strings` no tiene `PBXVariantGroup` en project.pbxproj: los `.lproj` "
        f"existen en disco pero Xcode no los empaqueta. [{_MARKER}]"
    )
    for lang in _IDIOMAS_IOS:
        if lang == "es":
            continue
        assert re.search(rf"path = {re.escape(lang)}\.lproj/InfoPlist\.strings;", pbx), (
            f"`{lang}.lproj/InfoPlist.strings` no está referenciado en project.pbxproj. [{_MARKER}]"
        )
    assert "InfoPlist.strings in Resources" in pbx, (
        f"`InfoPlist.strings` no está en la fase de Resources: se registra pero no se copia. [{_MARKER}]"
    )
    regiones = re.search(r"knownRegions = \((.*?)\);", pbx, re.S)
    assert regiones, f"desapareció `knownRegions` [{_MARKER}]"
    faltan = [l for l in _IDIOMAS_IOS if l != "es" and not re.search(rf"\b{re.escape(l)}\b", regiones.group(1))]
    assert not faltan, f"`knownRegions` no incluye {faltan} [{_MARKER}]"
