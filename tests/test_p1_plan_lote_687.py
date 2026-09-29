"""[P1-PLAN-LOTE-687 · 2026-09-28] La foto SOLA de un plato claro se registra, y el bloque de la foto va al FINAL.

Caso vivo del dueño (28-sep, 02:07 UTC): foto de su cena sin texto. El escáner acertó («2 huevos fritos acompañados de
tajadas de plátano maduro frito», sin dudas), el bloque llegó al system prompt… en la posición 25.595 de 36.550, y el
coach —que venía de pedirle la tabla de un suplemento— contestó «No me llegó el detalle de esa foto». El dueño: «si la
foto es 100 %, que lo anote directo; si no, que pregunte primero».
"""
from __future__ import annotations

import pytest

import agent
from prompts.chat_agent import build_vision_context
from respuestas_de_la_foto import (MARCADOR_SOLO_FOTO, NOTA_REINTENTO, es_solo_foto, foto_para_anotar,
                                   marcar_rotulo)

_CENA = "2 huevos fritos acompañados de tajadas de plátano maduro frito. (Estimación: Calorías: 390, Proteína: 14g, Carbohidratos: 36g, Grasas Saludables: 22g)"


def _vision(desc=_CENA, kind="plato"):
    return {"kind": "multi", "has_text": False, "items": [{"attachment_id": "a1", "kind": kind, "description": desc}]}


def test_solo_foto_es_el_turno_sin_texto():
    assert MARCADOR_SOLO_FOTO == "\U0001f4f7"
    assert es_solo_foto("\U0001f4f7") and es_solo_foto("") and es_solo_foto(None) and es_solo_foto("  ")
    assert not es_solo_foto("Mi cena") and not es_solo_foto("¿esto engorda?")


def test_la_foto_sola_de_un_plato_claro_es_para_anotar():
    assert foto_para_anotar(_vision(), MARCADOR_SOLO_FOTO) is True


def test_con_dudas_manda_la_regla_de_dudas_no_el_registro_forzado():
    dudosa = _CENA + " DUDAS (pregúntale solo esto): ¿Cuántos huevos eran?"
    assert foto_para_anotar(_vision(dudosa), MARCADOR_SOLO_FOTO) is False


def test_sin_plato_o_con_una_pregunta_no_se_fuerza_nada():
    assert foto_para_anotar(_vision(kind="items"), MARCADOR_SOLO_FOTO) is False
    assert foto_para_anotar(_vision(kind="etiqueta"), MARCADOR_SOLO_FOTO) is False
    assert foto_para_anotar(_vision(), "¿esto me conviene para cenar?") is False


def test_marcar_rotulo_marca_la_foto_sola():
    v = marcar_rotulo(_vision(), MARCADOR_SOLO_FOTO)
    assert v.get("solo_foto") is True and "rotulo" not in v
    assert marcar_rotulo(_vision(), "Mi cena").get("rotulo") == "cena"          # el rótulo sigue igual (694)


def test_el_bloque_pide_registrar_la_foto_sola_y_sigue_siendo_proactivo():
    ctx = build_vision_context(marcar_rotulo(_vision(), MARCADOR_SOLO_FOTO))
    assert "SOLO LA FOTO" in ctx and "log_consumed_meal" in ctx and "ESTA foto es su comida" in ctx
    assert "proactiv" in ctx.lower()   # P1-PHOTO-ONLY-TURN
    sin_marca = build_vision_context(_vision())
    assert "SOLO LA FOTO" not in sin_marca


def test_la_nota_de_reintento_cubre_la_foto_sola():
    assert "te la mandó sola" in NOTA_REINTENTO


class _PromptCapturado(Exception):
    pass


def _entradas(monkeypatch, *, prompt, vision):
    from types import SimpleNamespace
    import db_core
    import db_inventory
    import db_plans
    import nevera_opcional as no
    import shopping_calculator

    capturado = {}

    class _Grafo:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            capturado.update(inputs)
            raise _PromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: [])
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda items, **k: list(items))
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a: {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "classify_sentiment", lambda _p: {})
    monkeypatch.setattr(agent, "rag_query_router", lambda _p: {"skip": True})
    monkeypatch.setattr(agent, "get_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    with pytest.raises(_PromptCapturado):
        list(agent.chat_with_agent_stream(session_id="sesion-687", prompt=prompt, user_id="u-687", form_data={},
                                          vision=vision))
    return capturado


def test_en_el_stream_la_foto_va_al_final_y_el_turno_queda_para_anotar(monkeypatch):
    entradas = _entradas(monkeypatch, prompt=MARCADOR_SOLO_FOTO, vision=_vision())
    sp = entradas["sys_prompt"]
    assert sp.count("CONTEXTO DE VARIAS FOTOS") == 1
    i = sp.index("CONTEXTO DE VARIAS FOTOS")
    assert i > sp.index("DIARIO DE HOY"), "detrás del diario, el plan y los días anteriores"
    assert "SOLO LA FOTO" in sp[i:]
    assert entradas["turn_photo_to_log"] is True, "la red del 694 fuerza el registro si el modelo no lo hace"
