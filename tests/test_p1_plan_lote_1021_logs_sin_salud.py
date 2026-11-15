"""[P1-PLAN-LOTE-1021 · A-24] Los logs del coach no guardan alergias, condiciones ni el texto del usuario."""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_B = Path(__file__).resolve().parent.parent


def test_huella_estable_y_sin_contenido():
    from log_sin_datos_de_salud import claves, huella

    h = huella("soy alérgico al maní")
    assert h == huella("soy alérgico al maní") and "maní" not in h and h.startswith("len=20 h=")
    assert huella(None) == huella("")
    assert claves({"new_value": "maní", "field": "allergies"}) == "field,new_value"


def test_la_guarda_g20_no_escribe_valores_ni_texto(caplog):
    from langchain_core.messages import HumanMessage
    import agent

    state = {"turn_plate_photos": [{"x": 1}]}
    msgs = [HumanMessage(content="📷 qué tal este plato de Doña Rosa")]
    with caplog.at_level(logging.WARNING):
        r = agent._guard_source_intent_mutation("update_form_field", {"field": "allergies", "new_value": "maní"},
                                                state, msgs)
    assert r and r.startswith("BLOQUEADO")
    log = caplog.text
    assert "G-20 SOURCE-INTENT-GUARD" in log and "campo=allergies" in log
    assert "maní" not in log and "Doña Rosa" not in log


def test_ninguna_linea_tocada_vuelve_a_interpolar_el_contenido():
    a = (_B / "agent.py").read_text(encoding="utf-8")
    assert "texto={clean_user[:60]!r}" not in a
    assert "Query reescrita: '{prompt[:30]}...' → '{result}'" not in a
    assert "no necesita RAG: '{prompt[:30]}'" not in a
    assert "update_form_field({tool_args!r})" not in a
    t = (_B / "tools.py").read_text(encoding="utf-8")
    assert "({_previos} + {_new_field_value} → {_union})" not in t
    assert "'{user_message[:50]}...'" not in (_B / "sentiment_classifier.py").read_text(encoding="utf-8")
    assert "para: '{text[:50]}...'" not in (_B / "vision_agent.py").read_text(encoding="utf-8")


def test_el_aviso_de_fusion_clinica_solo_cuenta(monkeypatch, caplog):
    src = (_B / "tools.py").read_text(encoding="utf-8")
    i = src.index("[P0-CHAT-ALLERGY-MERGE] {field}: se conservan")
    linea = src[i:i + 400]
    assert re.search(r"len\(_previos or \[\]\)", linea) and "len(_union)" in linea
