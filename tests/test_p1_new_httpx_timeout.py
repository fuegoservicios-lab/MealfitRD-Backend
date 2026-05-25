"""[P1-NEW-HTTPX-TIMEOUT · 2026-05-15] Anchor + regression guard.

`backend/routers/billing.py` tiene 4 callsites a `httpx.AsyncClient(...)` que
hablan con `api-m.paypal.com`:

  1. `/api/subscription/verify` → OAuth + GET `/v1/billing/subscriptions/<id>`
     (líneas ~302).
  2. `/api/subscription/verify` (rama upgrade) → POST `/cancel` de la sub vieja
     (líneas ~382).
  3. `/api/subscription/cancel` → OAuth + POST `/cancel` (líneas ~496).
  4. `/api/webhooks/paypal` → OAuth + POST `/verify-webhook-signature`
     (líneas ~674).

Pre-fix los 4 instanciaban `httpx.AsyncClient()` SIN parámetro `timeout=`,
dejando reads colgados indefinidamente bajo tail-latency PayPal. Bajo
incidente PayPal regional el worker FastAPI quedaba bloqueado por request
hasta que el cliente abortaba la conexión, agotando el pool de workers
(503 cascada visible al usuario, alertas Sentry sin diagnóstico claro).

Defensas que el test enforza:
  1. Anchor `P1-NEW-HTTPX-TIMEOUT` presente en `billing.py`.
  2. Knob `MEALFIT_HTTPX_TIMEOUT_S` resuelto via `_env_float` con default 15.0
     y validator `lambda v: 5.0 <= v <= 60.0`.
  3. Cero callsites `httpx.AsyncClient()` sin `timeout=` en `billing.py`.
  4. Los 4 callsites usan `timeout=_HTTPX_TIMEOUT_S` (NO literal numérico).
  5. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).

Test parser-based — no levanta el server, solo escanea source con regex.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BILLING = _REPO_ROOT / "backend" / "routers" / "billing.py"
_CHAT = _REPO_ROOT / "backend" / "routers" / "chat.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_billing():
    src = _read(_BILLING)
    assert "P1-NEW-HTTPX-TIMEOUT" in src, (
        "Falta anchor `P1-NEW-HTTPX-TIMEOUT` en backend/routers/billing.py. "
        "Sin anchor, un futuro reader que vea `timeout=_HTTPX_TIMEOUT_S` no "
        "sabrá el modo de fallo que cierra (workers FastAPI bloqueados bajo "
        "tail-latency PayPal → pool exhausted → 503 cascada)."
    )


def test_knob_registered_with_default_and_validator():
    """`_HTTPX_TIMEOUT_S` debe resolverse via `_env_float("MEALFIT_HTTPX_TIMEOUT_S",
    15.0, validator=lambda v: 5.0 <= v <= 60.0)`. Sin default explícito un
    deploy sin la env var caería al implicit None → TypeError en httpx.
    Sin validator un override accidental a `MEALFIT_HTTPX_TIMEOUT_S=0.001`
    fallaría requests legítimos en latencia normal (5s mínimo) o anularía
    la defensa con valor absurdo (60s máximo).
    """
    src = _read(_BILLING)
    pat = re.compile(
        r"_env_float\(\s*[\"']MEALFIT_HTTPX_TIMEOUT_S[\"']\s*,\s*15\.0\s*,",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Knob `MEALFIT_HTTPX_TIMEOUT_S` debe resolverse via "
        "`_env_float(\"MEALFIT_HTTPX_TIMEOUT_S\", 15.0, ...)`. "
        "Default 15.0s = patrón pre-existente en `chat.py:261`."
    )
    # Validator clamp [5, 60]
    validator_pat = re.compile(
        r"MEALFIT_HTTPX_TIMEOUT_S.*?validator\s*=\s*lambda\s+\w+\s*:\s*5\.0\s*<=\s*\w+\s*<=\s*60\.0",
        re.DOTALL,
    )
    assert validator_pat.search(src), (
        "Knob `MEALFIT_HTTPX_TIMEOUT_S` debe tener "
        "`validator=lambda v: 5.0 <= v <= 60.0`. "
        "Sin clamp un override accidental tira la defensa o rompe latencias normales."
    )


def test_zero_async_client_without_timeout():
    """Cero `httpx.AsyncClient(...)` sin `timeout=` en `billing.py`. Esto es
    el guard principal: cualquier nuevo callsite que olvide el parámetro
    explícito reabre el modo de fallo. Match `httpx.AsyncClient(` seguido de
    cualquier cosa que NO contenga `timeout` antes del primer `)`.
    """
    src = _read(_BILLING)
    # Encuentra todos los `httpx.AsyncClient(...)` y verifica timeout=
    bad_callsites = []
    for m in re.finditer(r"httpx\.AsyncClient\(([^)]*)\)", src):
        args_blob = m.group(1)
        if "timeout" not in args_blob:
            # Reportar línea para diagnóstico
            line_no = src[: m.start()].count("\n") + 1
            bad_callsites.append(f"línea {line_no}: httpx.AsyncClient({args_blob})")
    assert not bad_callsites, (
        "`billing.py` contiene callsites `httpx.AsyncClient(...)` sin parámetro "
        "`timeout=`. Cada uno deja reads colgados indefinidamente bajo "
        "tail-latency PayPal. Callsites:\n  " + "\n  ".join(bad_callsites)
    )


def test_four_callsites_use_knob_variable():
    """Los 4 callsites a PayPal deben usar `timeout=_HTTPX_TIMEOUT_S`, NO
    literal numérico (`timeout=15.0` o similar). Pasar la variable garantiza
    que el knob `MEALFIT_HTTPX_TIMEOUT_S` aplica a TODOS los callsites — un
    literal hardcoded ignora silenciosamente el override SRE.
    """
    src = _read(_BILLING)
    callsites = re.findall(
        r"httpx\.AsyncClient\(\s*timeout\s*=\s*_HTTPX_TIMEOUT_S\s*\)",
        src,
    )
    assert len(callsites) == 4, (
        f"Esperados exactamente 4 callsites con `timeout=_HTTPX_TIMEOUT_S` "
        f"en billing.py (OAuth verify, cancel-old-sub, cancel direct, "
        f"webhook verify). Encontrados: {len(callsites)}. "
        f"Si añadiste un 5° callsite a PayPal, actualiza este conteo y "
        f"asegúrate que también usa el knob."
    )
    # Defensa: cero literales numéricos en `httpx.AsyncClient(timeout=...)`
    bad = re.search(r"httpx\.AsyncClient\(\s*timeout\s*=\s*[0-9]", src)
    assert bad is None, (
        f"`billing.py` contiene callsite `httpx.AsyncClient(timeout=<literal>)`: "
        f"{bad.group(0)!r}. Usar `timeout=_HTTPX_TIMEOUT_S` para que el "
        f"knob `MEALFIT_HTTPX_TIMEOUT_S` aplique uniformemente."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: el slug del marker debe matchear
    al menos un archivo `tests/test_<slug>*.py`. Este test ancla el slug
    `p1_new_httpx_timeout` (su nombre lo provee implícito)."""
    src = _read(Path(__file__))
    assert "P1-NEW-HTTPX-TIMEOUT" in src


# ---------------------------------------------------------------------------
# [P1-CHAT-TTS-TIMEOUT-HARDCODED · 2026-05-24] Extensión del blanket a chat.py.
#
# `routers/chat.py:493` exhibía el mismo patrón prohibido que cerró este test
# para billing.py: `httpx.AsyncClient(timeout=15.0)` literal sin knob.
# Sin rollback sin redeploy si ElevenLabs degrada latencia.
# ---------------------------------------------------------------------------


def test_chat_tts_anchor_present():
    """Anchor `P1-CHAT-TTS-TIMEOUT-HARDCODED` presente en `chat.py`."""
    src = _read(_CHAT)
    assert "P1-CHAT-TTS-TIMEOUT-HARDCODED" in src, (
        "Falta anchor `P1-CHAT-TTS-TIMEOUT-HARDCODED` en backend/routers/chat.py. "
        "Sin anchor un futuro reader que vea `timeout=_TTS_HTTPX_TIMEOUT_S` no "
        "sabrá el modo de fallo que cierra (no rollback sin redeploy si "
        "ElevenLabs degrada latencia)."
    )


def test_chat_tts_knob_registered_with_default_and_validator():
    """`_TTS_HTTPX_TIMEOUT_S` debe resolverse via
    `_env_float("MEALFIT_TTS_HTTPX_TIMEOUT_S", 15.0, validator=lambda v: 1.0 <= v <= 60.0)`.
    Default 15.0 preserva comportamiento previo. Clamp [1.0, 60.0]: piso
    contra `=0.001` accidental, techo contra `=120` que excede SLA total-graph.
    """
    src = _read(_CHAT)
    pat = re.compile(
        r"_env_float\(\s*[\"']MEALFIT_TTS_HTTPX_TIMEOUT_S[\"']\s*,\s*15\.0\s*,",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Knob `MEALFIT_TTS_HTTPX_TIMEOUT_S` debe resolverse via "
        "`_env_float(\"MEALFIT_TTS_HTTPX_TIMEOUT_S\", 15.0, ...)`."
    )
    validator_pat = re.compile(
        r"MEALFIT_TTS_HTTPX_TIMEOUT_S.*?validator\s*=\s*lambda\s+\w+\s*:\s*1\.0\s*<=\s*\w+\s*<=\s*60\.0",
        re.DOTALL,
    )
    assert validator_pat.search(src), (
        "Knob `MEALFIT_TTS_HTTPX_TIMEOUT_S` debe tener "
        "`validator=lambda v: 1.0 <= v <= 60.0`."
    )


def test_chat_tts_zero_async_client_without_timeout():
    """Cero `httpx.AsyncClient(...)` sin `timeout=` en `chat.py`. Guard
    principal: cualquier nuevo callsite que olvide el parámetro reabre el
    modo de fallo.
    """
    src = _read(_CHAT)
    bad_callsites = []
    for m in re.finditer(r"httpx\.AsyncClient\(([^)]*)\)", src):
        args_blob = m.group(1)
        if "timeout" not in args_blob:
            line_no = src[: m.start()].count("\n") + 1
            bad_callsites.append(f"línea {line_no}: httpx.AsyncClient({args_blob})")
    assert not bad_callsites, (
        "`chat.py` contiene callsites `httpx.AsyncClient(...)` sin parámetro "
        "`timeout=`. Cada uno deja reads colgados indefinidamente bajo "
        "tail-latency. Callsites:\n  " + "\n  ".join(bad_callsites)
    )


def test_chat_tts_callsite_uses_knob_variable():
    """El callsite TTS de ElevenLabs debe usar `timeout=_TTS_HTTPX_TIMEOUT_S`,
    NO literal numérico. Pasar la variable garantiza que el knob
    `MEALFIT_TTS_HTTPX_TIMEOUT_S` aplique uniformemente."""
    src = _read(_CHAT)
    pat = re.compile(
        r"httpx\.AsyncClient\(\s*timeout\s*=\s*_TTS_HTTPX_TIMEOUT_S\s*\)",
    )
    assert pat.search(src), (
        "Esperado `httpx.AsyncClient(timeout=_TTS_HTTPX_TIMEOUT_S)` en chat.py. "
        "Si añadiste un literal numérico, el knob se ignora silenciosamente."
    )
    # Defensa adicional: cero literales numéricos en `httpx.AsyncClient(timeout=N)`
    # como CÓDIGO ejecutable (excluye matches dentro de comentarios/docstrings
    # — el comment del knob menciona el patrón pre-fix como referencia narrativa).
    bad_callsites = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comentario inline
        # Heurística: backtick `…` indica markdown en docstring/comment.
        # El callsite real nunca está dentro de backticks.
        m = re.search(r"httpx\.AsyncClient\(\s*timeout\s*=\s*[0-9]", line)
        if m and "`" not in line:
            bad_callsites.append(f"línea {line_no}: {line.strip()[:120]}")
    assert not bad_callsites, (
        "`chat.py` contiene callsite `httpx.AsyncClient(timeout=<literal>)` "
        "fuera de comentarios. Usar `timeout=_TTS_HTTPX_TIMEOUT_S`. "
        "Callsites:\n  " + "\n  ".join(bad_callsites)
    )
