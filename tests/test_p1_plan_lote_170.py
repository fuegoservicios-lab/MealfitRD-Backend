# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-170 · 2026-09-23] «En la app Android el usuario no siente cambios en el teclado, le sigue igual
tapando todo al abrirlo; en iPhone se ve bien.» El cambio vive en el binario Android (este test es el contrato).

Lo que dicen los datos: nginx muestra que los dos Android de prueba (Samsung S23 Ultra con Android 16 y Xiaomi con
Android 13, WebView 152) BAJARON el paquete OTA con el trabajo de teclado y reabrieron la app, y que lo que usaban era
el FORMULARIO (telemetría del wizard), no el chat. La red de seguridad web del formulario (lote 166) solo actúa si el
`visualViewport` delata el teclado; si la WebView no cambia de tamaño, no ve nada.

La causa es nativa: SystemBars (Capacitor 8) en modo 'css' pone un escuchador de márgenes en la DecorView que
SUSTITUYE el manejo de Android. Reconstruye los márgenes sin el teclado (el `adjustResize` del manifiesto deja de
actuar), encoge la pantalla con su propio padding solo si `isVisible(ime())`, y deja pasar el margen del teclado a la
WebView, que con `viewport-fit=cover` y WebView ≥ 140 puede encogerse otra vez (Capacitor #8601, #8611). 'disable'
devuelve el `adjustResize` de siempre. Solo es seguro porque la ventana NO es de borde a borde (targetSdk 34): con 35,
Android lo impone y alguien tiene que volver a manejar los márgenes — por eso el test ata las dos cosas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _bloque_plugins(cap: str) -> str:
    return cap[cap.index("plugins: {"):]


def test_systembars_sin_escuchador_de_margenes():
    cap = _front("capacitor.config.ts")
    assert "SystemBars: { insetsHandling: 'disable' }," in _bloque_plugins(cap)
    assert "insetsHandling: 'css'" not in cap


def test_el_adjustresize_de_android_sigue_puesto():
    """'disable' devuelve el manejo de Android: sin `adjustResize` en la actividad, nadie encogería la WebView."""
    manifest = _front("android/app/src/main/AndroidManifest.xml")
    assert 'android:windowSoftInputMode="adjustResize"' in manifest


def test_la_ventana_no_es_de_borde_a_borde():
    """Con targetSdk 35+ Android impone el borde a borde y 'disable' dejaría la app debajo de las barras y del
    teclado. Si subes el target, este test te obliga a volver a pensar el manejo de márgenes."""
    variables = _front("android/variables.gradle")
    m = re.search(r"targetSdkVersion\s*=\s*(\d+)", variables)
    assert m and int(m.group(1)) <= 34
    nativo = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (_FRONT / "android" / "app" / "src" / "main" / "java").rglob("*.java")
    )
    assert "setDecorFitsSystemWindows" not in nativo and "EdgeToEdge" not in nativo


def test_apk_nuevo_sin_tocar_el_umbral_de_ota():
    """Es config del binario (no viaja por OTA): hace falta el APK 104. La web funciona con los dos binarios, así que
    `minNativeBuild` no sube (subirlo dejaría sin OTA a los APK 100-103 y a los iPhone)."""
    import json
    gradle = _front("android/app/build.gradle")
    code = int(re.search(r"versionCode\s+(\d+)", gradle).group(1))
    assert code >= 104
    assert f'versionName "1.0.{code}"' in gradle
    ota = json.loads(_front("ota.config.json"))
    assert ota["minNativeBuild"] == 13
    assert "insetsHandling" in ota["_nota_lote_170"]


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 170 and m.group(2) >= "2026-09-23"   # la serie sigue; el marker nunca baja
    assert "[P1-PLAN-LOTE-170 · 2026-09-23]" in app
