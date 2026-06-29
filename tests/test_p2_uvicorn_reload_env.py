"""[P2-UVICORN-RELOAD-ENV · 2026-05-12] Anchor + regression guard.

`backend/app.py:__main__` NO debe hardcodear `reload=True` en `uvicorn.run(...)`.
Pre-fix tenía `reload=True` literal: si alguien arrancase el server vía
`python app.py` en producción (no es el path actual del VPS Oracle
pero un futuro script-change podría reintroducirlo), uvicorn watchea el
filesystem y re-importa módulos en cada cambio. Eso rompe el estado
in-process (cache de knobs, _SCHEDULER_*, connection pools).

Defensas que el test enforza:
  1. Anchor `P2-UVICORN-RELOAD-ENV` en app.py.
  2. `reload=` toma un valor que viene de env var, NO literal True.
  3. La env var por default cae a OFF (no auto-reload).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = _REPO_ROOT / "backend" / "app.py"


def _read() -> str:
    return _APP.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read()
    assert "P2-UVICORN-RELOAD-ENV" in src, (
        "Falta anchor `P2-UVICORN-RELOAD-ENV` en app.py."
    )


def test_uvicorn_run_does_not_hardcode_reload_true():
    src = _read()
    # Buscar `uvicorn.run(...)` call — NO debe contener `reload=True` literal
    pat = re.compile(r"uvicorn\.run\([^)]*reload\s*=\s*True[^)]*\)", re.DOTALL)
    bad = pat.search(src)
    assert bad is None, (
        f"`uvicorn.run(...)` contiene `reload=True` hardcoded: {bad.group(0)[:120]!r}. "
        "Cambiar a env-gated (UVICORN_RELOAD=0 default)."
    )


def test_uvicorn_run_uses_env_gated_reload():
    """Debe haber un `reload=<variable>` donde la variable viene de env var."""
    src = _read()
    # Verifica que se lee `UVICORN_RELOAD` env var en alguna parte cerca de uvicorn.run
    assert "UVICORN_RELOAD" in src, (
        "Falta referencia a env var `UVICORN_RELOAD`. Debe leerse con default '0' "
        "(off) y solo activarse en dev local."
    )
    # Verificar default OFF (la lectura debe contener "0" como fallback)
    pat = re.compile(
        r'os\.environ\.get\(\s*[\"\']UVICORN_RELOAD[\"\']\s*,\s*[\"\']0[\"\']',
    )
    assert pat.search(src), (
        "El default de `UVICORN_RELOAD` debe ser `'0'` (off). Sin esto, "
        "un binary sin la env var configurada arrancaría con reload=True."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-UVICORN-RELOAD-ENV" in src
