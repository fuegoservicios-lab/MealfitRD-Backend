"""[P1-PLAN-LOTE-903 · 2026-09-29] La voz de pago del coach exige sesión.

Tras desplegar el 901, `curl` SIN sesión contra `/api/chat/voz/flujo` y `/api/chat/voz` devolvía 200 con audio: el
`Depends(get_verified_user_id)` da None sin token y nadie lo miraba. El tope de gasto es COMÚN a todos los usuarios, así
que un anónimo podía agotarlo y dejar a todos con la voz del teléfono. Sin sesión: 204 (el cliente ya pone la voz del
teléfono ante cualquier respuesta que no sea audio) y ni una llamada a Google.
"""
from __future__ import annotations

import asyncio

import pytest

import coach_voz


@pytest.fixture
def google_prohibido(monkeypatch):
    monkeypatch.delenv("MEALFIT_COACH_VOZ_NUBE", raising=False)
    monkeypatch.setattr(coach_voz, "hay_presupuesto", lambda: True)
    monkeypatch.setattr(coach_voz, "sintetizar", lambda *a: pytest.fail("sin sesión no se llama a Google"))
    monkeypatch.setattr(coach_voz, "abrir_flujo", lambda *a: pytest.fail("sin sesión no se llama a Google"))


class _Tareas:
    def add_task(self, *a, **k):
        pytest.fail("sin sesión no hay gasto que registrar")


@pytest.mark.parametrize("uid", [None, ""])
def test_sin_sesion_el_wav_es_204(google_prohibido, uid):
    from routers import chat
    r = asyncio.run(chat.api_chat_voz(_Tareas(), {"texto": "Hola"}, uid, None))
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "sin_sesion"


@pytest.mark.parametrize("uid", [None, ""])
def test_sin_sesion_el_streaming_es_204(google_prohibido, uid):
    from routers import chat
    r = asyncio.run(chat.api_chat_voz_flujo({"texto": "Hola"}, uid, None))
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "sin_sesion"


def test_la_guarda_va_antes_de_cualquier_gasto():
    import inspect
    from routers import chat
    for fn in (chat.api_chat_voz, chat.api_chat_voz_flujo):
        src = inspect.getsource(fn)
        i = src.index("if not verified_user_id:")
        assert i < src.index("hay_presupuesto)"), fn.__name__
