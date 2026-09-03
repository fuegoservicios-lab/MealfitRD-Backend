"""[P1-IOS-CORS-NATIVE-ORIGIN · 2026-08-22] La app nativa (Capacitor) vive en el
origen `capacitor://localhost` y llama a `https://app.bioboros.com/api/...`:
CROSS-ORIGIN de verdad, con preflight. En la web nunca hubo preflight (nginx
sirve frontend y API en el MISMO origen), así que dos huecos de la lista CORS
eran inertes y en nativo son un muro:

  1. `capacitor://localhost` no estaba en `allow_origins` → toda fetch de la
     app muere en el preflight.
  2. `X-MF-Session` no estaba en `allow_headers` → el ÚNICO mecanismo de sesión
     que funciona en nativo (el token de localStorage en header, porque la
     cookie `__Host-mf_session` es same-site) se rechaza en el preflight.
     El propio comentario del bloque decía «NO custom X-* headers — si se añade
     alguno, extender esta lista»: el header se añadió (P1-FIRST-PARTY-SESSION)
     y la lista no se extendió. Comentario-vence-guard, otra vez.

Sin esto «Continuar con correo» en el iPhone fallaría igual que Google, y con
un error mudo. Funcional (TestClient con preflight real) + parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = _REPO_ROOT / "backend" / "app.py"

NATIVE_ORIGIN = "capacitor://localhost"


def _cors_block() -> str:
    src = _APP.read_text(encoding="utf-8")
    m = re.search(r"app\.add_middleware\(\s*CORSMiddleware\s*,(.*?)\n\)", src, re.DOTALL)
    assert m is not None
    return m.group(1)


def test_el_origen_nativo_esta_en_la_lista():
    block = _cors_block()
    assert f'"{NATIVE_ORIGIN}"' in block, (
        "capacitor://localhost debe estar en allow_origins: es el origen de la app "
        "nativa iOS y sin él toda fetch a /api muere en el preflight."
    )


def test_el_header_de_sesion_esta_permitido():
    block = _cors_block()
    # `[^\]]+` se frenaría en el primer `]` de un marker `[P1-...]` dentro de
    # un comentario de la lista: se lee hasta el cierre `],` de la lista entera.
    m = re.search(r"allow_headers\s*=\s*\[(.*?)\n\s*\],", block, re.DOTALL)
    assert m is not None
    assert '"X-MF-Session"' in m.group(1), (
        "X-MF-Session debe estar en allow_headers: en nativo la cookie __Host- no "
        "viaja y el token va en ese header (P1-FIRST-PARTY-SESSION)."
    )


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import app as app_module

    return TestClient(app_module.app)


def test_preflight_desde_capacitor_con_x_mf_session_pasa(client):
    """Preflight REAL: el middleware responde con el origen y el header ecoados."""
    r = client.options(
        "/api/auth/me",
        headers={
            "Origin": NATIVE_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-mf-session",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("access-control-allow-origin") == NATIVE_ORIGIN
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    assert "x-mf-session" in allowed, allowed
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_preflight_desde_un_origen_ajeno_sigue_rechazado(client):
    """Abrir capacitor:// no abre la puerta a cualquiera (P2-CORS-NARROW)."""
    r = client.options(
        "/api/auth/me",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-mf-session",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
