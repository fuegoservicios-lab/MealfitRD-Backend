# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-111 · 2026-09-19] El teclado del chat en la app nativa: sube CON el teclado y adjuntar no lo cierra.

El dueño, ya con el build nativo y el OTA funcionando: «tiene delay cuando lo abro y cuando abro lo de subir foto me
cierra el teclado». (1) La apertura se decidía solo por geometría e iOS la entrega al TERMINAR la animación: el teclado
tapaba la caja ~300 ms y el chat saltaba; ahora el foco anticipa el inset recordado, solo en nativo. (2) Adjuntar con
el teclado abierto abre un menú anclado al «+» que no toca el foco (como Gemini); sin teclado, hoja inferior.
Se descartó `@capacitor/keyboard`: su `load` quita los observadores de teclado del WebView, de los que depende todo
el ajuste actual. Contrato fino: `frontend/src/__tests__/lote111.test.jsx`."""
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
    return p.read_text(encoding="utf-8")


def test_la_apertura_se_anticipa_solo_en_nativo_y_con_cerrojo():
    ap = _front("src/pages/AgentPage.jsx")
    i = ap.index("const alGanarElFoco = (e) => {")
    cuerpo = ap[i:i + 1600]
    assert "if (!isNativeApp() ||" in cuerpo
    assert "if (recordado < KB_UMBRAL_PX ||" in cuerpo
    # [P1-PLAN-LOTE-129] colocar el chat pasó a `anticiparApertura`, que comparten el foco y el aviso nativo de UIKit
    assert "anticiparApertura(recordado);" in cuerpo
    j = ap.index("const anticiparApertura = (inset) => {")
    assert "abriendoRef.current = true;" in ap[j:j + 1400]
    assert re.search(r"if \(abriendoRef\.current\) \{\s*if \(!abiertoMedido\) return;", ap)
    assert "document.addEventListener('focusin', alGanarElFoco);" in ap
    assert "document.removeEventListener('focusin', alGanarElFoco);" in ap


def test_adjuntar_no_cierra_el_teclado_en_nativo():
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (abierto && !isNativeApp()) chatInputRef.current?.blur();" in ap
    assert "anchorRect={attachmentAnchorRect}" in ap
    hoja = _front("src/components/agent/AttachmentSourceSheet.jsx")
    assert "if (!esMenu) firstActionRef.current?.focus({ preventScroll: true });" in hoja
    assert "const noRobarFoco = (event) => { if (esMenu) event.preventDefault(); };" in hoja


def test_no_se_instalo_el_plugin_de_teclado():
    """Su `load` hace removeObserver de UIKeyboardWillShow sobre el WebView: visualViewport dejaría de encoger."""
    pkg = json.loads(_front("package.json"))
    assert "@capacitor/keyboard" not in pkg["dependencies"]


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 111
