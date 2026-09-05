"""[P2-VISION-USAGE-TOKENS · 2026-09-04] El primer escaneo con Gemini 3.8 Flash dejó en
`llm_usage_events` una fila (model=gemini-3.8-flash, node=vision_scan) con input/output_tokens NULL:
el cliente de visión es PLANO (sin el mixin de costo) y el router anotaba solo el modelo, así que
`compute_llm_cost_micros` no podía ponerle precio a la foto. Un callback captura el usage del
proveedor en `_invoke_structured_vision` (ContextVar de la tarea) y el router lo anota.

Invariantes: la firma de `_resolve_vision_client` y el shape del resultado NO cambian; un cliente
sin `with_config` (fakes) sigue funcionando con usage=None.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def test_usage_capture_reads_usage_metadata_and_token_usage():
    from vision_agent import _VisionUsageCapture
    cb = _VisionUsageCapture()
    msg = SimpleNamespace(usage_metadata={"input_tokens": 1234, "output_tokens": 88, "total_tokens": 1322})
    cb.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None))
    assert cb.usage == {"input_tokens": 1234, "output_tokens": 88}
    cb2 = _VisionUsageCapture()
    cb2.on_llm_end(SimpleNamespace(generations=[], llm_output={"token_usage": {"prompt_tokens": 500, "completion_tokens": 40}}))
    assert cb2.usage == {"input_tokens": 500, "output_tokens": 40}
    # otros hooks son no-op (el manager de langchain llama a varios)
    cb2.on_llm_start(None, None)
    cb2.on_chat_model_start(None, None)


@pytest.mark.asyncio
async def test_invoke_sets_last_usage_in_task_context(monkeypatch):
    import vision_agent as va

    class _Fake:
        def __init__(self):
            self._cbs = []

        def with_config(self, callbacks=None, **kw):
            self._cbs = list(callbacks or [])
            return self

        async def ainvoke(self, messages):
            msg = SimpleNamespace(usage_metadata={"input_tokens": 1100, "output_tokens": 60})
            for cb in self._cbs:
                cb.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None))
            return {"ok": True}

    monkeypatch.setattr(va, "_resolve_vision_client", lambda schema: _Fake())
    monkeypatch.setattr(va, "_vision_llm_timeout_s", lambda: 5.0, raising=False)
    out = await va._invoke_structured_vision(b"\xff\xd8\xff", "prompt", dict)
    assert out == {"ok": True}
    assert va.get_last_vision_usage() == {"input_tokens": 1100, "output_tokens": 60}

    class _NoConfig:
        async def ainvoke(self, messages):
            return {"ok": 2}

    monkeypatch.setattr(va, "_resolve_vision_client", lambda schema: _NoConfig())
    assert await va._invoke_structured_vision(b"\xff\xd8\xff", "prompt", dict) == {"ok": 2}
    assert va.get_last_vision_usage() is None  # se resetea por llamada; sin callback ⇒ None


def test_router_logs_tokens_into_the_cost_book():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = src.index('node="vision_scan",')
    win = src[i - 1600:i + 300]  # el lookup va ANTES del gate (ventana de test_p1_vision_luna)
    assert "get_last_vision_usage" in win
    assert 'input_tokens=_vision_usage.get("input_tokens")' in win
    assert 'output_tokens=_vision_usage.get("output_tokens")' in win


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P2-VISION-USAGE-TOKENS · 2026-09-04" in app
