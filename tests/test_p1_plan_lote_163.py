"""[P1-PLAN-LOTE-163 · 2026-09-22] El APK 103 (y el build de iOS): el gesto «atrás» y el icono de Bioboros.

1. **El gesto «atrás» de Android cerraba la app.** El APK no traía `@capacitor/app` y el `BridgeActivity` de
   Capacitor 8 no maneja el botón: Android mandaba la app al fondo (12+) o la cerraba (11 y anteriores, perdiendo lo
   escrito) desde cualquier pantalla. Los oyentes de `popstate` del formulario y del login solo funcionaban en Chrome.
   Ahora `src/native/botonAtras.js` cierra el diálogo de arriba (por el camino de Escape), minimiza en las pestañas
   raíz y retrocede en el resto. Es código nativo: no viaja por OTA; por eso el APK 103.
2. **El icono y la pantalla de arranque eran el logo de Capacitor**, en iOS desde agosto y en Android desde el lote 152
   (que los derivó del de iOS). Ahora salen de `brand/favicon-source.png`.
3. Google en Android manda el mensaje original del «cancelado»: Credential Manager usa esa excepción también para
   fallos de configuración, y ese texto es lo único que los distingue.

Tooltip-anchor: P1-PLAN-LOTE-163
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _f(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"frontend ausente: {rel}")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_el_gesto_atras_tiene_quien_lo_escuche():
    assert json.loads(_f("package.json"))["dependencies"].get("@capacitor/app")
    assert "include ':capacitor-app'" in _f("android/capacitor.settings.gradle")
    assert "implementation project(':capacitor-app')" in _f("android/app/capacitor.build.gradle")
    atras = _f("src/native/botonAtras.js")
    assert "await App.addListener('backButton'," in atras
    assert "nativePluginAvailable('App')" in atras          # un APK viejo recibe el JS por OTA y no hace nada
    assert "return mod.App" not in atras                    # el Proxy jamás sale de una función async (lote 133)
    assert "import('./native/botonAtras')" in _f("src/main.jsx")


def test_version_del_apk_sin_tocar_el_umbral_de_ota():
    gradle = _f("android/app/build.gradle")
    assert int(re.search(r"versionCode\s+(\d+)", gradle).group(1)) == 103
    assert 'versionName "1.0.103"' in gradle
    assert json.loads(_f("ota.config.json"))["minNativeBuild"] == 13


def test_el_icono_es_el_de_bioboros_y_no_el_de_capacitor():
    PIL = pytest.importorskip("PIL.Image")
    icono = _FRONT / "ios" / "App" / "App" / "Assets.xcassets" / "AppIcon.appiconset" / "AppIcon-512@2x.png"
    if not icono.exists():
        pytest.skip("frontend ausente")
    img = PIL.open(icono)
    assert img.size == (1024, 1024)
    assert img.mode == "RGB", "El App Store rechaza un icono con canal alfa"
    px = img.load()
    # a sangre: las cuatro esquinas son el azul de la marca (el de Capacitor era blanco con rejilla)
    for x, y in ((2, 2), (1021, 2), (2, 1021), (1021, 1021)):
        r, g, b = px[x, y]
        assert (r, g, b) != (255, 255, 255) and b < 60 and r < 20, f"esquina {x},{y} = {(r, g, b)}"
    # el brote (azul claro): el tallo pasa por el centro de la mitad de abajo, y el punto por encima de las hojas
    for x, y in ((512, 700), (512, 310)):
        r, g, b = px[x, y]
        assert b > 200 and r > 80, f"({x},{y}) no es el brote: {(r, g, b)}"
    assert "#01071D" in _f("android/app/src/main/res/values/ic_launcher_background.xml")


def test_google_manda_el_detalle_del_cancelado():
    java = _f("android/app/src/main/java/com/bioboros/app/MfGoogleId.java")
    assert 'datos.put("detalle", String.valueOf(e.getMessage()));' in java


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 163
