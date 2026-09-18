# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-95 · 2026-09-17] «Continuar con Google» pregunta SIEMPRE qué cuenta usar (decisión del dueño).

En su iPhone, Google entró sin preguntar con la otra cuenta de Gmail del teléfono y creó una identidad nueva y vacía
(`0f3ca99f…`, 17-sep 20:34 UTC). El endpoint `/sign-in/social` de Better Auth no acepta `prompt` por petición y el adaptador
Supabase no reenvía `queryParams`; el cliente pide la URL con `disableRedirect: true`, le añade `prompt=select_account` y
navega. Ancla cross-repo del contrato."""
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


def test_el_cliente_envuelve_el_oauth_de_google():
    js = _front("src/authClient.js")
    assert "export function conSelectorDeCuenta(url)" in js
    assert "if (u.hostname !== 'accounts.google.com') return url;" in js
    assert "valores.push('select_account');" in js
    assert "ba.signIn.social({ provider, callbackURL, disableRedirect: true })" in js
    assert "window.location.href = conSelectorDeCuenta(url);" in js
    assert "c.auth.signInWithOAuth = _conGoogleQuePregunta(c);" in js


def test_sin_cliente_better_auth_el_acceso_no_depende_del_arreglo():
    js = _front("src/authClient.js")
    i = js.index("function _conGoogleQuePregunta(c)")
    cuerpo = js[i:js.index("\n}\n", i)]
    assert "return original(credentials);" in cuerpo, "si no hay cliente Better Auth, el camino de siempre"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 95
    assert "P1-PLAN-LOTE-95" in (_BACKEND / "docs" / "sesion_first_party_marcador.md").read_text(encoding="utf-8")
