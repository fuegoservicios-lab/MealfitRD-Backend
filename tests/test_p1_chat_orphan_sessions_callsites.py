"""[P1-CHAT-ORPHAN-SESSIONS · 2026-09-14] Los dos call sites que aún creaban sesiones sin dueño.

`generation_inputs.build_initial_pipeline_inputs` (cola de generación) y
`services._process_swap_rejection_background` («No me gusta») llamaban a
`get_or_create_session(session_id)` sin usuario — el forense encontró las 175 sesiones de
producción con `user_id` NULL, que sobreviven al borrado de la cuenta. Además, con un
`session_id` ajeno en el body, el primero metía la memoria de chat de OTRO usuario en el
plan y el segundo escribía «Rechacé…» en su conversación.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import services  # noqa: E402


def test_rechazo_crea_la_sesion_con_dueno_y_sin_nudge(monkeypatch):
    llamadas = {}
    monkeypatch.setattr(services, "get_or_create_session",
                        lambda sid, user_id=None: llamadas.setdefault("sess", (sid, user_id)) and {"user_id": user_id})
    monkeypatch.setattr(services, "save_message",
                        lambda *a, **kw: llamadas.setdefault("msg", (a, kw)))
    services._process_swap_rejection_background("s1", "u1", "Mangú", "Desayuno")
    assert llamadas["sess"] == ("s1", "u1")
    assert llamadas["msg"][1] == {"user_id": "u1", "process_nudge": False}


def test_rechazo_no_escribe_en_la_sesion_de_otro(monkeypatch):
    escritos = []
    monkeypatch.setattr(services, "get_or_create_session", lambda sid, user_id=None: {"user_id": "otro"})
    monkeypatch.setattr(services, "save_message", lambda *a, **kw: escritos.append(a))
    services._process_swap_rejection_background("s1", "u1", "Mangú", "Desayuno")
    assert escritos == [], "se escribió en la conversación de otro usuario"


def test_cola_de_generacion_pasa_el_dueno_y_no_lee_memoria_ajena():
    src = (_BACKEND / "generation_inputs.py").read_text(encoding="utf-8")
    i = src.index("def build_initial_pipeline_inputs(")
    body = src[i:i + 2500]
    assert "rp.get_or_create_session(session_id, user_id=actual_user_id or None)" in body
    guard = body.index("if _owner and str(_owner) != str(actual_user_id or \"\"):")
    assert guard < body.index("memory = rp.build_memory_context(session_id)"), (
        "la memoria de la sesión se lee antes de comprobar de quién es"
    )
