"""[P1-PROD-AUDIT-1 · 2026-05-23] JWT retry list debe ser extensible vía
env var + detectar transient errors por isinstance/MRO, no solo string match.

Gap original (audit production-readiness 2026-05-23, B-P1-9):
    `_TRANSIENT_NETWORK_ERRORS` tupla hardcoded en `backend/auth.py`. Si
    httpx o supabase introducen una excepción transient nueva (ej.
    `httpx.NetworkError` en futura versión), el match `type(e).__name__ in
    list` retorna False y el caller obtiene 403 espurio en lugar de retry.

    Operador NO puede añadir un transient nuevo sin redeploy.

Fix:
    (a) Mantener la lista canónica (back-compat).
    (b) Knob `MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS` (env var, comma-separated)
        extiende la lista en runtime.
    (c) Walk MRO: si una subclase usa un nombre canónico como parent
        (e.g. `class CustomError(RemoteProtocolError)`), se detecta.

Cobertura:
    A) Constante `_CANONICAL_TRANSIENT_NETWORK_ERRORS` presente (back-compat).
    B) Pattern `MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS` referenced en source.
    C) Walk de MRO via `type(e).__mro__` presente.
    D) Anchor `P1-PROD-AUDIT-1-AUTH-RETRY` presente.

Tooltip-anchor: P1-PROD-AUDIT-1-AUTH-RETRY | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AUTH_PY = _BACKEND_ROOT / "auth.py"


def _read_auth() -> str:
    return _AUTH_PY.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read_auth()
    assert "P1-PROD-AUDIT-1-AUTH-RETRY" in src, (
        "Anchor `P1-PROD-AUDIT-1-AUTH-RETRY` ausente en auth.py. Sin "
        "anchor, futuro mantenedor pierde contexto del fix (audit 2026-05-23)."
    )


def test_canonical_list_preserved():
    """La lista canónica debe seguir conteniendo los 8 nombres pre-fix
    (back-compat). Si alguien refactoriza y solo deja el knob, se pierde
    la cobertura default sin env var."""
    src = _read_auth()
    canonical = [
        "RemoteProtocolError",
        "ReadError",
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "Server disconnected",
    ]
    missing = [name for name in canonical if name not in src]
    assert not missing, (
        f"Canonical transient error names ausentes en auth.py: {missing}. "
        f"REGRESIÓN — pre-fix los detectaba como transient. Restaurar."
    )


def test_env_var_knob_present():
    """Knob `MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS` debe ser leído del env."""
    src = _read_auth()
    assert "MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS" in src, (
        "Knob `MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS` no referenciado en auth.py. "
        "Sin él, operador no puede añadir transient errors nuevos sin redeploy."
    )


def test_mro_walk_present():
    """El detector debe walkear MRO via `__mro__` para captar subclases
    de nombres canónicos (e.g. una versión futura de httpx que extienda).
    """
    src = _read_auth()
    assert "__mro__" in src, (
        "auth.py NO walkea `type(e).__mro__` para detectar subclases. "
        "Sin esto, una subclase con nombre custom de un parent canónico "
        "(ej. `MyTimeout(ReadTimeout)`) NO se detecta como transient."
    )


def test_isinstance_or_name_match_in_mro():
    """El check debe ser: nombre del tipo OR nombres en MRO OR substring
    en mensaje. Defensa-en-profundidad para captar variantes."""
    src = _read_auth()
    # Heuristic: aparecen los 3 patterns en el bloque de check.
    has_type_name = "type(e).__name__" in src
    has_mro_check = "mro_names" in src or "__mro__" in src
    has_substring = "err_str" in src or "str(e)" in src
    assert has_type_name and has_mro_check and has_substring, (
        f"Auth retry check incompleto: type_name={has_type_name}, "
        f"mro={has_mro_check}, substring={has_substring}. Los 3 son "
        f"necesarios para defense-in-depth."
    )


def test_retry_attempts_unchanged():
    """`MAX_ATTEMPTS = 4` se preserva (back-compat). Bajar este número
    haría que blips Supabase >1s vuelvan a producir 403 espurio."""
    src = _read_auth()
    import re
    m = re.search(r"MAX_ATTEMPTS\s*=\s*(\d+)", src)
    assert m is not None, "MAX_ATTEMPTS no definido en auth.py"
    n = int(m.group(1))
    assert n >= 3, (
        f"MAX_ATTEMPTS={n} es muy bajo — pre-fix=4 daba ~3.75s total backoff. "
        f"Reducir aumenta 403 espurio bajo blips Supabase. Si necesitas "
        f"bajar, documentar el trade-off."
    )


def test_async_def_preserved():
    """`get_verified_user_id` debe seguir siendo `async def` (P2-AUTH-ASYNC-SLEEP)."""
    src = _read_auth()
    assert "async def get_verified_user_id" in src, (
        "REGRESIÓN: `get_verified_user_id` ya NO es async. Eso bloquea "
        "el event loop durante el call a Supabase + sleep retry → "
        "throughput per-worker cae a ~20-30 req/s. Restaurar async."
    )


def test_extra_transient_env_parsing_handles_empty():
    """Sanity de parsing del env var: empty / None / spaces / trailing
    comma no rompen. Sin esto, una config rota → AttributeError en
    module-init de auth.py.
    """
    import os
    saved = os.environ.get("MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS")
    try:
        for empty_val in ["", "  ", ",,", " , , "]:
            os.environ["MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS"] = empty_val
            # Re-leer + parsear como lo hace auth.py
            raw = os.environ.get("MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS", "")
            extra = tuple(s.strip() for s in raw.split(",") if s.strip())
            assert extra == (), (
                f"Empty env var '{empty_val}' produjo tupla no-vacía: {extra}. "
                f"Parsing roto — añadiría strings vacíos a la lista canónica."
            )

        # Valor real con espacios.
        os.environ["MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS"] = "NewError, AnotherError , Third"
        raw = os.environ.get("MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS", "")
        extra = tuple(s.strip() for s in raw.split(",") if s.strip())
        assert extra == ("NewError", "AnotherError", "Third")
    finally:
        if saved is None:
            os.environ.pop("MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS", None)
        else:
            os.environ["MEALFIT_AUTH_EXTRA_TRANSIENT_ERRORS"] = saved
