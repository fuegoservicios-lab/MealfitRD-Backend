"""[P1-PLAN-LOTE-1026 · A-13] El prompt del turno viaja en la config de la ejecución, no en el checkpoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import coach_checkpoint_ligero as ccl  # noqa: E402

_PROMPT = "MANUAL " * 5000   # ~35 KB, como el real


def _grafo():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        sys_prompt: str
        visto: str

    def nodo(state):
        return {"visto": ccl.prompt_del_turno(state)[:7]}

    g = StateGraph(S)
    g.add_node("n", nodo)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    saver = MemorySaver()
    return g.compile(checkpointer=saver), saver


def test_el_nodo_lo_lee_y_el_checkpoint_no_lo_guarda(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_CHECKPOINT_LIGERO", raising=False)
    app, saver = _grafo()
    config = {"configurable": {"thread_id": "t1"}}
    en_estado = ccl.preparar(config, _PROMPT)
    assert en_estado == ""
    list(app.stream({"sys_prompt": en_estado, "visto": ""}, config=config, stream_mode="updates"))
    t = saver.get_tuple({"configurable": {"thread_id": "t1"}})
    assert t.checkpoint["channel_values"]["visto"] == "MANUAL "
    assert t.checkpoint["channel_values"]["sys_prompt"] == ""
    assert ccl.CLAVE not in t.metadata and "MANUAL" not in repr(t.metadata)


def test_un_checkpoint_viejo_con_61_kb_queda_pisado(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_CHECKPOINT_LIGERO", raising=False)
    app, saver = _grafo()
    viejo = {"configurable": {"thread_id": "t2"}}
    list(app.stream({"sys_prompt": _PROMPT, "visto": ""}, config=viejo, stream_mode="updates"))   # turno de antes
    nuevo = {"configurable": {"thread_id": "t2"}}
    list(app.stream({"sys_prompt": ccl.preparar(nuevo, "OTRO"), "visto": ""}, config=nuevo, stream_mode="updates"))
    vals = saver.get_tuple({"configurable": {"thread_id": "t2"}}).checkpoint["channel_values"]
    assert vals["sys_prompt"] == "" and vals["visto"] == "OTRO"


def test_apagado_o_fuera_del_grafo_se_lee_el_estado(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_CHECKPOINT_LIGERO", "0")
    config = {"configurable": {"thread_id": "t3"}}
    assert ccl.preparar(config, "P") == "P" and ccl.CLAVE not in config["configurable"]
    assert ccl.prompt_del_turno({"sys_prompt": "DEL ESTADO"}) == "DEL ESTADO"   # fuera de una ejecución
    assert ccl.prompt_del_turno({}) == "" and ccl.prompt_del_turno(None) == ""


def test_el_grafo_se_compila_una_vez_y_cada_turno_lleva_su_saver(monkeypatch):
    import agent
    import coach_grafo_compilado as cgc
    from langgraph.checkpoint.memory import MemorySaver
    monkeypatch.delenv("MEALFIT_CHAT_GRAFO_COMPILADO_UNA_VEZ", raising=False)
    monkeypatch.setattr(cgc, "_cache", {"builder": None, "base": None})
    compilaciones = []
    real = agent.chat_builder.compile
    monkeypatch.setattr(agent.chat_builder, "compile", lambda **kw: compilaciones.append(1) or real(**kw))
    s1, s2 = MemorySaver(), MemorySaver()
    g1, g2 = cgc.para_turno(agent.chat_builder, s1), cgc.para_turno(agent.chat_builder, s2)
    assert len(compilaciones) == 1
    assert g1.checkpointer is s1 and g2.checkpointer is s2 and g1 is not g2   # el lock del saver no se comparte
    assert set(g1.nodes) == set(agent.chat_builder.compile(checkpointer=s1).nodes)


def test_builder_falso_o_apagado_compila_como_siempre(monkeypatch):
    import coach_grafo_compilado as cgc
    monkeypatch.setattr(cgc, "_cache", {"builder": None, "base": None})

    class Falso:
        n = 0

        def compile(self, **kw):
            Falso.n += 1
            return object()          # sin copy()

    b = Falso()
    cgc.para_turno(b, "s"), cgc.para_turno(b, "s")
    assert Falso.n == 2 and cgc._cache["base"] is None
    monkeypatch.setenv("MEALFIT_CHAT_GRAFO_COMPILADO_UNA_VEZ", "0")
    cgc.para_turno(b, "s")
    assert Falso.n == 3


def test_cableado():
    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    assert s.count('"sys_prompt": __import__("coach_checkpoint_ligero").preparar(config, system_prompt),') == 1
    assert s.count('__import__("coach_checkpoint_ligero").prompt_del_turno(state)') == 2
    assert 'state.get("sys_prompt"' not in s
    assert s.count("_grafo_para_turno(chat_builder, ") == 2 and "chat_builder.compile(" not in s
