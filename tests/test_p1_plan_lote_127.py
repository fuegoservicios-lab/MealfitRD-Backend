# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-127 · 2026-09-19] El micrófono no se lleva el teclado, y la caja del chat es simétrica.

El dueño, ya con el build nativo que trae el micrófono (el dictado FUNCIONA en el iPhone): «cuando abro el teclado y
prendo el micrófono se cierra el teclado, corrige eso… veo que no hay simetría en la parte izquierda donde está el +».

  · TECLADO. El toque no le quita el foco a la caja (`preventDefault` en pointerdown). Quien esconde el teclado es iOS al
    arrancar la sesión de audio (o su aviso de permiso, la primera vez), y lo hace sin desenfocar el campo: la misma
    forma que al volver del selector de fotos (`restoreChatKeyboardAfterAttachment`), y el mismo remedio. Si HABÍA
    teclado al tocar el micrófono y, ya escuchando, la geometría dice que se fue, se suelta y se retoma el foco — una
    sola vez; se mira a los 350/700/1200/2000 ms porque iOS no lo esconde en un instante fijo. Sin teclado al empezar no
    se abre ninguno. Visto en el arnés con un visualViewport falso. DEDUCIDO para iOS, no medido: por eso cada paso deja
    marca en la sonda (`/sonda`): `micKB`, `micON`, `kbRepon`.
  · SIMETRÍA. Medido: 28 px del borde de la caja al «+» contra 11 px del borde a ENVIAR (relleno 16/8 y un glifo suelto
    dentro de un botón invisible). Ahora relleno igual y el «+» es un círculo visible del tamaño de ENVIAR: 11 y 11.

Contrato fino: `frontend/src/__tests__/lote127.test.js`. Cero cambios de backend."""
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
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_encender_el_microfono_repone_el_teclado_que_ios_esconde():
    ap = _front("src/pages/AgentPage.jsx")
    assert "reponerTecladoTrasMicRef.current = Boolean(tecladoAbiertoRef.current || medirTecladoDeVentana(window).abierto);" in ap
    assert "const MIC_REPONER_TECLADO_MS = [350, 700, 1200, 2000];" in ap
    assert "onClick={handleMicClick}" in ap
    efecto = ap[ap.index("if (!isListening || !reponerTecladoTrasMicRef.current) return undefined;"):ap.index("// lo dictado crece por abajo")]
    assert "if (!campo || medirTecladoDeVentana(window).abierto) return;" in efecto, "solo si el teclado se FUE"
    assert "relojes.forEach(clearTimeout);" in efecto, "una sola vez"
    assert "campo.focus({ preventScroll: true });" in efecto


def test_la_sonda_acepta_marcas_con_nombre():
    sonda = _front("src/utils/keyboardProbe.js")
    assert "export function marcarSondaTeclado(nombre) {" in sonda
    assert "if (!_pararSonda || typeof document === 'undefined') return;" in sonda, "sin sonda encendida no cuesta nada"


def test_la_caja_del_chat_es_simetrica():
    ap = _front("src/pages/AgentPage.jsx")
    assert "'0.5rem 0.5rem 0.5rem 1rem'" not in ap, "volvió el relleno de 16 px a la izquierda"
    css = ap[ap.index(".attachment-btn {"):ap.index(".chat-mic-btn {")]
    assert "background: color-mix(in srgb, var(--text-main) 9%, transparent);" in css, "el «+» es un círculo visible"
    assert "margin-left: 2px;" in css, "el mismo margen al borde que ENVIAR (marginRight: 2px)"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 127
