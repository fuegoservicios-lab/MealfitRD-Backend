"""[P2-CORS-NARROW · 2026-05-12] Anchor + regression guard.

`backend/app.py` CORS middleware NO debe volver a usar `allow_methods=["*"]`
ni `allow_headers=["*"]` con `allow_credentials=True`. El wildcard `*` con
credentials habilitado es defense-in-depth gap: cualquier verbo HTTP (TRACE,
CONNECT) o header custom de un script third-party puede ser exfiltrado.

Defensas que el test enforza:
  1. Anchor `P2-CORS-NARROW` en app.py.
  2. `allow_methods` es lista explícita de verbos REST + OPTIONS, NO `["*"]`.
  3. `allow_headers` es lista explícita, NO `["*"]`.
  4. `allow_credentials=True` preservado (auth.py JWT requiere cookies/auth).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = _REPO_ROOT / "backend" / "app.py"


def _read() -> str:
    return _APP.read_text(encoding="utf-8")


def _extract_cors_block(src: str) -> str:
    """Aísla el bloque app.add_middleware(CORSMiddleware, ...)."""
    m = re.search(
        r"app\.add_middleware\(\s*CORSMiddleware\s*,(.*?)\n\)",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró bloque `app.add_middleware(CORSMiddleware, ...)`."
    return m.group(1)


def test_anchor_present():
    src = _read()
    assert "P2-CORS-NARROW" in src, (
        "Falta anchor `P2-CORS-NARROW` en backend/app.py. "
        "Sin anchor, un futuro reader puede 'simplificar' la config "
        "revirtiendo a `[\"*\"]`."
    )


def test_allow_methods_is_explicit_list():
    block = _extract_cors_block(_read())
    # Match `allow_methods=[ "GET", ... ]`
    m = re.search(r'allow_methods\s*=\s*\[([^\]]+)\]', block)
    assert m is not None, "No se encontró `allow_methods=[...]` en CORS block."
    methods = m.group(1)
    # Debe contener verbos explícitos NO solo "*"
    assert '"*"' not in methods and "'*'" not in methods, (
        f"`allow_methods` contiene wildcard `*`: {methods!r}. "
        "Debe ser lista explícita de verbos REST. P2-CORS-NARROW lo cierra."
    )
    # Verificar que al menos GET, POST, OPTIONS están whitelisteados
    for required in ('"GET"', '"POST"', '"OPTIONS"'):
        assert required in methods, (
            f"`allow_methods` falta verbo crítico {required}: {methods!r}"
        )


def test_allow_headers_is_explicit_list():
    block = _extract_cors_block(_read())
    m = re.search(r'allow_headers\s*=\s*\[([^\]]+)\]', block, re.DOTALL)
    assert m is not None, "No se encontró `allow_headers=[...]` en CORS block."
    headers = m.group(1)
    assert '"*"' not in headers and "'*'" not in headers, (
        f"`allow_headers` contiene wildcard `*`: {headers!r}. "
        "Debe ser lista explícita. P2-CORS-NARROW lo cierra."
    )
    # Authorization es crítico — sin él el JWT auth rompe
    assert '"Authorization"' in headers, (
        "`allow_headers` falta `Authorization`. Sin esto el JWT auth del "
        "frontend recibe CORS rejected en preflight → auth roto en prod."
    )
    assert '"Content-Type"' in headers, (
        "`allow_headers` falta `Content-Type`. Sin esto JSON bodies rompen."
    )


def test_allow_credentials_preserved():
    block = _extract_cors_block(_read())
    assert re.search(r'allow_credentials\s*=\s*True', block), (
        "`allow_credentials=True` debe preservarse. Sin esto el frontend "
        "no puede enviar cookies/JWT en cross-origin requests."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-CORS-NARROW" in src
