"""[P1-TTS-FINALLY-LOG · 2026-05-15] Anchor parser-based: el endpoint
`/api/chat/tts` debe cobrar `log_api_usage` en un bloque `finally` (no solo
en path de éxito) + emitir `pipeline_metrics(node='tts_timeout')` en timeout.

Contexto del bug original:
    Pre-fix, `log_api_usage(verified_user_id, "elevenlabs_tts")` vivía
    DENTRO del `async with httpx.AsyncClient` SOLO en path de éxito. Si
    la request a ElevenLabs lanzaba TimeoutError / HTTPStatusError /
    ConnectError, el accounting NO se ejecutaba. ElevenLabs factura POR
    CARÁCTER SUBMITIDO (no por response devuelto): si la request llegó al
    servidor antes del timeout, ellos cobraron al owner y el usuario quedó
    sin charge en cuota mensual.

Fix:
    - `_tts_billed` + `_tts_request_started` flags para dedupe defensivo
      y para distinguir "request iniciada" vs "request abortada antes de
      enviar bytes" (auth fail, missing api_key).
    - Bloque `finally` cobra `log_api_usage` cuando
      `_tts_request_started AND not _tts_billed AND verified_user_id`.
    - Timeout específico emite `pipeline_metrics(node='tts_timeout', ...)`.

Tooltip-anchor: P1-TTS-FINALLY-LOG-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CHAT_PY = _BACKEND / "routers" / "chat.py"


@pytest.fixture(scope="module")
def chat_src() -> str:
    return _CHAT_PY.read_text(encoding="utf-8")


def _extract_tts_handler_body(src: str) -> str:
    """Devuelve el cuerpo de `api_chat_tts` desde su `async def` hasta el
    siguiente `@router.post` o `async def`/`def` top-level."""
    anchor = re.search(r"async def api_chat_tts\b", src)
    assert anchor is not None, (
        "P1-TTS-FINALLY-LOG regresión: `async def api_chat_tts` ya no aparece "
        "en chat.py. ¿Renombrado? Actualizar test."
    )
    start = anchor.end()
    rest = src[start:]
    next_decl = re.search(r"\n(?:@router\.|async def |def )", rest)
    end = start + (next_decl.start() if next_decl else len(rest))
    return src[start:end]


def test_finally_block_present(chat_src: str):
    body = _extract_tts_handler_body(chat_src)
    assert re.search(r"\n    finally:\s*\n", body), (
        "P1-TTS-FINALLY-LOG: el bloque `finally:` no aparece dentro de "
        "`api_chat_tts`. log_api_usage volvería a vivir solo en path de éxito."
    )


def test_log_api_usage_in_finally(chat_src: str):
    """`log_api_usage(verified_user_id, "elevenlabs_tts")` debe estar dentro
    del bloque finally (post-request).
    """
    body = _extract_tts_handler_body(chat_src)
    finally_match = re.search(r"\n    finally:\s*\n([\s\S]+)$", body)
    assert finally_match is not None, "finally: block not found"
    finally_body = finally_match.group(1)
    assert "log_api_usage" in finally_body, (
        "P1-TTS-FINALLY-LOG: `log_api_usage` no aparece en el bloque finally — "
        "el accounting vuelve a depender del path de éxito."
    )
    assert "elevenlabs_tts" in finally_body, (
        "P1-TTS-FINALLY-LOG: la clave de billing 'elevenlabs_tts' no aparece "
        "en el finally — accounting roto."
    )


def test_request_started_flag_gates_billing(chat_src: str):
    """El finally debe gatear el cobro con `_tts_request_started` para no
    cobrar errores pre-network (auth fail, missing api_key).
    """
    body = _extract_tts_handler_body(chat_src)
    assert "_tts_request_started" in body, (
        "P1-TTS-FINALLY-LOG: flag `_tts_request_started` ausente — el finally "
        "cobraría incluso cuando la request nunca alcanzó ElevenLabs."
    )
    assert "_tts_billed" in body, (
        "P1-TTS-FINALLY-LOG: flag `_tts_billed` ausente — billing podría "
        "duplicarse en reentradas defensivas."
    )


def test_timeout_emits_pipeline_metric(chat_src: str):
    """TimeoutError catch debe emitir `node='tts_timeout'` a pipeline_metrics."""
    body = _extract_tts_handler_body(chat_src)
    assert "tts_timeout" in body, (
        "P1-TTS-FINALLY-LOG: el bucket 'tts_timeout' no aparece — sin signal "
        "estructurada, SRE no puede graficar incidencia de timeouts."
    )
    assert re.search(
        r"except\s*\([^)]*Timeout[^)]*\)|except\s+httpx\.TimeoutException|except\s+asyncio\.TimeoutError",
        body,
    ), (
        "P1-TTS-FINALLY-LOG: el catch específico de timeout (httpx.TimeoutException "
        "o asyncio.TimeoutError) no aparece — timeouts caen al except genérico."
    )


def test_marker_tooltip_present(chat_src: str):
    assert "P1-TTS-FINALLY-LOG" in chat_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
