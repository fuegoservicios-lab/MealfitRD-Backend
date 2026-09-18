# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-97 · 2026-09-18] El aviso «¿Es la cuenta que querías?» tras un acceso con Google.

Google no pregunta qué cuenta usar y desde aquí no se le puede obligar (lote 96). En el iPhone del dueño eso creó una
cuenta nueva y vacía con su otro correo. El dispositivo recuerda con qué cuentas se entra (correo ENMASCARADO, máx. 5) y,
si un acceso con Google cae en una desconocida mientras conocía otra, pregunta. Ancla cross-repo del contrato."""
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


def test_el_dispositivo_nunca_guarda_el_correo_completo():
    js = _front("src/utils/cuentasDelDispositivo.js")
    assert "export function enmascararCorreo(correo)" in js
    assert "const lista = [{ id: perfil.id, correo, visto: ahora }, ...resto]" in js
    assert "const correo = enmascararCorreo(perfil.email);" in js
    assert "const MAX_CUENTAS = 5;" in js


def test_solo_pregunta_tras_google_y_con_otra_cuenta_conocida():
    js = _front("src/utils/cuentasDelDispositivo.js")
    assert "if (!perfil?.id || !vieneDeGoogle || yaConocida || !otra || !actual) {" in js
    # el marcador de Google se consume SIEMPRE: un aviso por acceso, no uno por recarga
    i = js.index("export function evaluarAccesoGoogle")
    assert js.index("safeLocalStorageRemove(CLAVE_INICIO_GOOGLE);", i) < js.index("if (!perfil?.id", i)


def test_el_login_marca_google_y_app_monta_el_aviso():
    assert "if (provider === 'google') marcarInicioGoogle();" in _front("src/pages/Login.jsx")
    app = re.sub(r"\s+", " ", _front("src/App.jsx"))
    assert "const AvisoCuentaGoogle = lazy(() => import('./components/auth/AvisoCuentaGoogle'));" in app
    assert "{!IS_APEX_HOST && ( <Suspense fallback={null}> <AvisoCuentaGoogle /> </Suspense> )}" in app


def test_la_lista_sobrevive_al_cierre_de_sesion():
    """Es su razón de ser: el aviso compara con la cuenta de ANTES del cierre. Si alguien la mete en la limpieza del
    logout, el aviso deja de poder dispararse sin que ningún otro test lo note."""
    ctx = _front("src/context/AssessmentContext.jsx")
    assert "mf_cuentas_dispositivo" not in ctx


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 97
    assert "P1-PLAN-LOTE-97" in (_BACKEND / "docs" / "sesion_first_party_marcador.md").read_text(encoding="utf-8")
