"""[P2-CHAT-STREAM-TIMEOUT-TOOLS · 2026-07-12] El stream sobrevive tools de 2-4 min.

Vivo (owner): "inténtalo de nuevo" → modify_single_meal con el retry de
expansión (P1-CHAT-MODIFY-EXPAND-FALLBACK) = dos generaciones LLM DENTRO de
un solo nodo (~2-4 min sin emitir eventos) → el inactivity timeout del stream
(default 25s, clamp máx 120s) mató el turno con el plato YA persistido: el
chat mostró error aunque el plan sí se actualizó (el refresh-recover recogió
los pedazos, pero la UX fue de error falso).

Fix: clamp del knob 120→360 (el env del VPS sube la ventana a 300s; el
default 25s se conserva para conversación normal) + total timeout ya
clampeaba a 600 (env a 420).
tooltip-anchor: P2-CHAT-STREAM-TIMEOUT-TOOLS
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from agent import _chat_stream_inactivity_timeout_s, _chat_stream_total_timeout_s  # noqa: E402


def test_inactivity_clamp_covers_double_swap(monkeypatch):
    # El env del VPS necesita poder subir a ≥300s (dos generaciones + retries).
    monkeypatch.setenv("MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S", "300")
    assert _chat_stream_inactivity_timeout_s() == 300.0, \
        "con el clamp viejo (≤120) el env caía al default y el stream moría"


def test_default_stays_snappy(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S", raising=False)
    assert _chat_stream_inactivity_timeout_s() == 25.0, \
        "sin env, la conversación normal conserva la ventana corta"


def test_total_timeout_env_reachable(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_STREAM_TOTAL_TIMEOUT_S", "420")
    assert _chat_stream_total_timeout_s() == 420.0
