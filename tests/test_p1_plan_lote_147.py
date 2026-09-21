# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-147 · 2026-09-21] «Continuar con Google» NATIVO en iOS — anclas cross-repo.

Los casos funcionales viven en `test_p1_plan_lote_147_google.py`. Aquí: que el binario y la web no se separen."""
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


def test_el_client_id_es_el_mismo_en_los_dos_lados():
    """Si divergen, el canje falla con `invalid_client` y nadie sabría por qué."""
    import google_auth
    assert f"'{google_auth.google_ios_client_id()}'" in _front("src/utils/googleSignInNative.js")


def test_el_redirect_lo_fija_el_servidor_y_coincide_con_el_de_la_web():
    import google_auth
    esperado = google_auth._redirect_uri_esperado()
    assert esperado.startswith("com.googleusercontent.apps.") and esperado.endswith(":/oauth2redirect")
    assert "${GOOGLE_IOS_SCHEME}:/oauth2redirect" in _front("src/utils/googleSignInNative.js")


def test_el_plugin_del_binario_solo_abre_https():
    swift = _front("ios/App/App/SceneDelegate.swift")
    assert 'guard url.scheme?.lowercased() == "https" else {' in swift
    assert "registerPluginInstance(MfWebAuthPlugin())" in swift


def test_ningun_sdk_de_google_entro_al_binario():
    """ASWebAuthenticationSession es del sistema: el SDK habría sido un pod nuevo y un mínimo de build más alto."""
    import json
    ota = json.loads(_front("ota.config.json"))
    assert not any("google" in d.lower() for d in ota["nativeDeps"]), ota["nativeDeps"]


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 147
