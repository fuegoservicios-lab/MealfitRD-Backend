# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-128 · 2026-09-19] La app nativa quita la barra de accesorios del teclado de iOS (flechas y ✓).

El dueño: «quita la barra de flechas de iOS sobre el teclado». iOS la pone sobre el teclado en cualquier campo de un
WKWebView; en el chat (un solo campo) no sirve de nada y se come ~45 pt de conversación.

  · Es código NATIVO (`frontend/ios/App/App/SceneDelegate.swift`): subclase en tiempo de ejecución de la vista de
    contenido del WKWebView cuyo `inputAccessoryView` devuelve nil. Viaja en el BINARIO: hace falta un build de
    Codemagic; por OTA no llega.
  · Va en `sceneDidBecomeActive` y no en AppDelegate: la app usa ciclo de vida por ESCENAS (UIApplicationSceneManifest),
    y con escenas UIKit no llama a `applicationDidBecomeActive` ni rellena `AppDelegate.window`.
  · NO se usa @capacitor/keyboard, que también la quita: su `load` retira los observadores del teclado del WebView en
    todos sus modos, `visualViewport` dejaría de encoger y toda la geometría del chat quedaría ciega (lote 111).

Sin compilar aquí (no hay Xcode en Windows): lo confirma el build. Contrato fino: `frontend/src/__tests__/lote128.test.js`."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_la_barra_se_quita_donde_corre_el_ciclo_de_vida():
    assert "<key>UIApplicationSceneManifest</key>" in _front("ios/App/App/Info.plist"), "la app es de escenas"
    escena = _front("ios/App/App/SceneDelegate.swift")
    assert "func sceneDidBecomeActive(_ scene: UIScene) {" in escena
    assert "bridgeVC.webView?.ocultarBarraDeAccesorios()" in escena
    assert "@objc var inputAccessoryView: UIView? { return nil }" in escena
    assert "if nombreActual.hasSuffix(sufijo) { return }" in escena, "repetirlo en cada activación no debe apilar subclases"


def test_el_binario_retransmite_el_teclado_sin_retirar_observadores():
    # «¿no puedes hacer más fluido el cerrar y abrir el teclado?»: la web solo ve el teclado cuando la animación ya
    # terminó; UIKit avisa ANTES, con alto y duración. El binario lo retransmite; de momento solo lo apunta la sonda,
    # para medir antes de tocar la geometría del chat.
    escena = _front("ios/App/App/SceneDelegate.swift")
    assert "UIResponder.keyboardWillShowNotification" in escena and "UIResponder.keyboardWillHideNotification" in escena
    assert "mf:teclado-nativo" in escena
    assert "removeObserver" not in escena, "AÑADE observadores; retirar los de WebKit cegaría visualViewport"
    assert "export const EVENTO_TECLADO_NATIVO = 'mf:teclado-nativo';" in _front("src/utils/keyboardProbe.js")


def test_el_mas_del_chat_es_un_glifo_limpio():
    ap = _front("src/pages/AgentPage.jsx")
    css = ap[ap.index(".attachment-btn {"):ap.index(".chat-mic-btn {")]
    assert "background: transparent;" in css and "var(--text-main) 9%" not in css
    assert re.search(r"@media \(hover: hover\) and \(pointer: fine\) \{\n\s+\.attachment-btn:not\(\.disabled\):hover \{", ap), (
        "el hover solo con puntero fino: en táctil se queda pegado tras el toque")


def test_sin_el_plugin_de_teclado():
    pkg = json.loads(_front("package.json"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "@capacitor/keyboard" not in deps, "retira los observadores del teclado: la geometría del chat quedaría ciega"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 128
