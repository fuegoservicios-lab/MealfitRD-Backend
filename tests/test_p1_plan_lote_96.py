# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-96 · 2026-09-17] El lote 95 («Continuar con Google» pregunta siempre qué cuenta usar) no hacía nada en
producción y se revirtió.

Verificado en producción: la URL que devuelve Neon al iniciar el acceso con Google es un salto en SU dominio
(`…/neondb/auth/sign-in/social/init?token=<uuid>`) que redirige a Google desde el servidor; ni añadiéndole
`prompt=select_account` llega a Google (las URLs finales de Google no traen `prompt`). El proveedor de Google de Neon solo
acepta client ID y secret. El envoltorio del 95 solo tocaba URLs de `accounts.google.com`, que nunca pasan por el cliente:
código que no puede actuar y un test que afirmaba lo contrario. Este test impide que vuelva sin la vía para que funcione."""
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


def test_el_envoltorio_del_95_no_vuelve():
    js = _front("src/authClient.js")
    for rastro in ("conSelectorDeCuenta", "_conGoogleQuePregunta", "disableRedirect: true"):
        assert rastro not in js, f"{rastro}: el selector de cuenta no se puede forzar desde el cliente con Neon"


def test_queda_escrito_por_que():
    js = _front("src/authClient.js")
    assert "[P1-PLAN-LOTE-96" in js and "sign-in/social/init?token" in js
    assert not (_FRONT / "src" / "__tests__" / "authClient.google_select_account.test.js").exists()


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 96
    doc = (_BACKEND / "docs" / "sesion_first_party_marcador.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-96" in doc and "sign-in/social/init" in doc
