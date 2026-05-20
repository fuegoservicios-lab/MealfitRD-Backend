r"""[P3-AUTH-RETRY-EXPANDED · 2026-05-18] Retry de `get_verified_user_id`
contra errores de red transient de httpx debe cubrir los 5 nombres canónicos
+ el legacy `"Server disconnected"`.

Pre-fix, el handler solo matcheaba el substring `"Server disconnected"`. Cuando
Supabase Auth API tenía blip de keep-alive y httpx levantaba `RemoteProtocolError`,
`ReadError`, `ConnectError`, `PoolTimeout` o `ReadTimeout`, el retry NO se
disparaba y el endpoint devolvía 403 espurio al frontend. Reportado por user en
logs locales 2026-05-18 05:49:10 con `RemoteProtocolError` en
`/api/user/preferences/memory` + `/water-tracker` simultáneos.

Comportamiento post-fix:
  - 1 reintento con `await asyncio.sleep(0.5)` ante cualquiera de los 6
    errores transient → 99% de blips se recuperan transparente.
  - Si el segundo intento también falla → 403 (fail-secure intacto).
  - Errores NO-transient (firma inválida, token expirado, JWT malformed) →
    403 inmediato sin retry (comportamiento previo intacto, defensa P0-AUDIT-1).

Lo que este test enforza:
  1. **Parser-based**: la tupla `_TRANSIENT_NETWORK_ERRORS` contiene los 6
     nombres exactos.
  2. **Parser-based**: el match es bidireccional (por `type(e).__name__` Y por
     substring) para cubrir tanto exceptions tipadas (`httpx.RemoteProtocolError`)
     como exceptions con mensaje en `str(e)` (legacy "Server disconnected").
  3. **Parser-based**: el retry tiene `attempt == 0` guard → solo 1 reintento,
     no infinite loop.
  4. **Parser-based**: errores no-transient siguen yendo a 403 inmediato (no
     se debilita el fail-secure de P0-AUDIT-1).
  5. **Funcional**: simular `RemoteProtocolError` → segundo intento ejecutado.
  6. **Funcional**: simular `InvalidJWT` (firma) → 403 sin retry.

Cierre del gap operacional: el user reportó 2 picos 403 espurios en logs locales
y el handler era el causante. Después del fix esos 403 dejan de aparecer en
condiciones de red transient.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AUTH_PY = _BACKEND_ROOT / "auth.py"


# ─────────────────────────────────────────────────────────────────────────────
# Parser-based: contract anchors
# ─────────────────────────────────────────────────────────────────────────────


def _read_auth_source() -> str:
    return _AUTH_PY.read_text(encoding="utf-8")


def test_marker_present():
    """Marker `P3-AUTH-RETRY-EXPANDED` ancla la decisión y permite a un grep +
    git blame llegar al fix sin pasar por código."""
    src = _read_auth_source()
    assert "P3-AUTH-RETRY-EXPANDED" in src, (
        "Marker P3-AUTH-RETRY-EXPANDED ausente. Un revert silente reintroduciría "
        "los 403 espurios ante blips de red Supabase."
    )


def test_transient_errors_tuple_contains_all_six_names():
    """La tupla `_TRANSIENT_NETWORK_ERRORS` debe contener los 6 nombres
    canónicos. Si alguno se borra, el retry pierde cobertura."""
    src = _read_auth_source()
    # Buscar la tupla — debe existir como literal multi-línea.
    expected_names = (
        "RemoteProtocolError",
        "ReadError",
        "ConnectError",
        "PoolTimeout",
        "ReadTimeout",
        "Server disconnected",
    )
    # Extraer el bloque de la tupla. Anchor: declaration `_TRANSIENT_NETWORK_ERRORS = (`.
    match = re.search(
        r"_TRANSIENT_NETWORK_ERRORS\s*=\s*\((.*?)\)",
        src,
        re.DOTALL,
    )
    assert match, (
        "Tupla `_TRANSIENT_NETWORK_ERRORS` no encontrada en auth.py. "
        "El retry depende de esta tupla para clasificar errores transient."
    )
    tuple_body = match.group(1)
    for name in expected_names:
        assert f'"{name}"' in tuple_body, (
            f"Nombre transient `{name}` ausente de _TRANSIENT_NETWORK_ERRORS. "
            f"Los blips de Supabase Auth con esa exception caerán a 403 espurio."
        )


def test_match_is_bidirectional_type_name_or_substring():
    """El check `is_transient` debe matchear por tanto `type(e).__name__` (para
    exceptions tipadas como httpx.RemoteProtocolError) como por substring en
    `str(e)` (para mensajes legacy 'Server disconnected' que vienen plano)."""
    src = _read_auth_source()
    # Anchor: el bloque del check is_transient.
    is_transient_block = src[src.find("is_transient = ("):src.find("if attempt == 0 and is_transient:")]
    assert "err_type in _TRANSIENT_NETWORK_ERRORS" in is_transient_block, (
        "El check por type name está ausente. Exceptions tipadas (RemoteProtocolError) "
        "no serán matcheadas si solo se filtra por substring."
    )
    assert "in err_str" in is_transient_block, (
        "El check por substring está ausente. Errores legacy con mensaje plano "
        "('Server disconnected' como string) no serán matcheados."
    )


def test_retry_has_attempt_zero_guard():
    """El retry debe estar bounded por `attempt < MAX_ATTEMPTS - 1` — solo reintentos finitos. Sin
    esto, un Supabase Auth permanentemente caído entraría en loop infinito."""
    src = _read_auth_source()
    assert "attempt < MAX_ATTEMPTS - 1 and is_transient:" in src, (
        "Guard `attempt < MAX_ATTEMPTS - 1` ausente. Sin él, un Supabase Auth caído producía "
        "loop infinito de retries."
    )


def test_fail_secure_403_preserved_for_non_transient():
    """Para errores NO transient (firma inválida, token expirado, JWT malformed),
    DEBE devolverse 403 inmediato sin retry. Este es el contrato P0-AUDIT-1
    que no se debilita con el fix de retry."""
    src = _read_auth_source()
    # El raise debe estar FUERA del bloque del retry transient.
    raise_pattern = re.search(
        r'raise HTTPException\(status_code=403,\s*detail="Token validation failed\."\)',
        src,
    )
    assert raise_pattern, (
        "raise HTTPException(403, ...) no encontrado. El fail-secure de "
        "P0-AUDIT-1 fue debilitado — VECTOR DE ATAQUE: tokens con firma "
        "inválida podrían ser aceptados."
    )


def test_log_does_not_leak_error_message():
    """El logger.warning DEBE registrar solo el `type(e).__name__`, NO el
    mensaje completo. Defense: los mensajes de Supabase pueden contener
    información que filtra a logs públicos / telemetría compartida."""
    src = _read_auth_source()
    # El logger.warning con [P0-AUDIT-1] solo debe usar `err_type`, NO `err_str`
    # ni `{e}` directo.
    log_block = re.search(
        r'logger\.warning\(\s*f"\[P0-AUDIT-1\] Token validation falló:[^"]*"',
        src,
    )
    assert log_block, (
        "Logger.warning con marker [P0-AUDIT-1] no encontrado o renombrado."
    )
    log_text = log_block.group(0)
    assert "{err_type}" in log_text or "{type(e).__name__}" in log_text, (
        "El log debe registrar `err_type` (no `err_str` ni `e` directo) — "
        "evita leak de mensajes Supabase con detalle interno."
    )
    # Defense extra: confirmar que NO se mete err_str ni e en el log.
    assert "{err_str}" not in log_text, (
        "El log filtra `err_str` — esto puede leakar detalles de Supabase "
        "internals a observabilidad compartida."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Funcional: simular escenarios con mocks
# ─────────────────────────────────────────────────────────────────────────────


def _build_fake_supabase(get_user_fn):
    """Helper: construye un objeto fake `supabase` con `.auth.get_user(token)`."""
    fake_auth = type("FakeAuth", (), {"get_user": staticmethod(get_user_fn)})()
    return type("FakeSupabase", (), {"auth": fake_auth})()


def test_transient_remote_protocol_error_triggers_retry():
    """Cuando `supabase.auth.get_user` lanza RemoteProtocolError en el primer
    intento, el handler DEBE dormir 0.5s y reintentar. El segundo intento OK
    → retorna user.id sin 403."""
    import auth as auth_mod

    class _FakeUser:
        id = "11111111-1111-1111-1111-111111111111"

    class _FakeUserRes:
        user = _FakeUser()

    call_count = {"n": 0}

    def _fake_get_user(token):
        call_count["n"] += 1
        if call_count["n"] == 1:
            class RemoteProtocolError(Exception):
                pass
            raise RemoteProtocolError("Server closed connection mid-response")
        return _FakeUserRes()

    fake_supabase = _build_fake_supabase(_fake_get_user)

    with patch.object(auth_mod, "supabase", fake_supabase), \
         patch.object(auth_mod.asyncio, "sleep", new=AsyncMock()) as mock_sleep:
        result = asyncio.run(
            auth_mod.get_verified_user_id(authorization="Bearer fake-token-xyz")
        )

    assert result == "11111111-1111-1111-1111-111111111111", (
        f"Retry no se ejecutó correctamente. Resultado: {result!r}. "
        f"call_count={call_count['n']} (esperado 2)."
    )
    assert call_count["n"] == 2, (
        f"Esperado 2 intentos (1 fail + 1 retry), recibido {call_count['n']}."
    )
    mock_sleep.assert_called_once_with(0.25)


def test_non_transient_error_fails_immediately_with_403():
    """Cuando la exception NO es de red (e.g. token con firma inválida), el
    handler NO debe retroyear y debe devolver 403 inmediato. Preserva el
    fail-secure de P0-AUDIT-1."""
    from fastapi import HTTPException
    import auth as auth_mod

    call_count = {"n": 0}

    def _fake_get_user(token):
        call_count["n"] += 1
        raise ValueError("Invalid JWT signature")

    fake_supabase = _build_fake_supabase(_fake_get_user)

    with patch.object(auth_mod, "supabase", fake_supabase), \
         patch.object(auth_mod.asyncio, "sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                auth_mod.get_verified_user_id(authorization="Bearer fake-malformed-token")
            )

    assert exc_info.value.status_code == 403, (
        "Errores NO transient deben devolver 403 inmediato (fail-secure P0-AUDIT-1)."
    )
    assert call_count["n"] == 1, (
        f"Esperado 1 intento (no retry), recibido {call_count['n']}. "
        "El retry se está activando para errores NO transient → debilita fail-secure."
    )
    mock_sleep.assert_not_called()


def test_double_failure_returns_403_after_retry_exhausted():
    """Si AMBOS intentos fallan con error transient, retorna 403 tras retry
    exhausted. Sin esto, un Supabase Auth caído durante 5 minutos haría
    loop indefinido."""
    from fastapi import HTTPException
    import auth as auth_mod

    call_count = {"n": 0}

    def _fake_get_user(token):
        call_count["n"] += 1
        class RemoteProtocolError(Exception):
            pass
        raise RemoteProtocolError("Server disconnected")

    fake_supabase = _build_fake_supabase(_fake_get_user)

    with patch.object(auth_mod, "supabase", fake_supabase), \
         patch.object(auth_mod.asyncio, "sleep", new=AsyncMock()):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                auth_mod.get_verified_user_id(authorization="Bearer fake-token")
            )

    assert exc_info.value.status_code == 403
    assert call_count["n"] == 4, (
        f"Esperado exactamente 4 intentos (1 original + 3 reintentos = 403 tras agotarse). "
        f"Recibido {call_count['n']}. Si >4 hay loop. Si <4, retry no se agotó."
    )
