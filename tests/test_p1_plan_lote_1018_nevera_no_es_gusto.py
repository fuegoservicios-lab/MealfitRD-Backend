"""[P1-PLAN-LOTE-1018 · D2] «Se me acabó el pollo» no puede escribir dislikes=pollo (caso real de la batería del 14-nov)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guarda_nevera_no_es_gusto import bloqueo_si_es_la_nevera  # noqa: E402

_DISLIKE = {"field": "dislikes", "new_value": "pollo"}


@pytest.mark.parametrize("texto", [
    "se me acabó el pollo", "Se me terminaron los huevos", "ya no tengo arroz", "no me queda leche",
    "se dañó el aguacate", "I ran out of chicken", "acabou o frango", "je n'ai plus de poulet",
])
def test_quedarse_sin_algo_no_es_un_gusto(texto):
    assert bloqueo_si_es_la_nevera("update_form_field", _DISLIKE, texto)
    assert bloqueo_si_es_la_nevera("update_form_field", {"field": "allergies", "new_value": "pollo"}, texto)


@pytest.mark.parametrize("texto", [
    "no me gusta el pollo", "se me acabó el pollo y la verdad no me gusta, quítalo", "soy alérgico al maní",
    "ya no tengo ganas de comer brócoli, lo detesto", "I hate chicken", "agrega pollo a lo que no me gusta",
    "",
])
def test_un_gusto_dicho_de_verdad_pasa(texto):
    assert bloqueo_si_es_la_nevera("update_form_field", _DISLIKE, texto) is None


def test_otros_campos_y_otras_herramientas_no_se_tocan():
    assert bloqueo_si_es_la_nevera("update_form_field", {"field": "weight", "new_value": "80"}, "se me acabó el pollo") is None
    assert bloqueo_si_es_la_nevera("modify_pantry_inventory", _DISLIKE, "se me acabó el pollo") is None
    assert bloqueo_si_es_la_nevera("update_form_field", "no-dict", "se me acabó el pollo") is None


def test_el_bloqueo_manda_a_la_nevera_sin_tocar_el_perfil():
    m = bloqueo_si_es_la_nevera("update_form_field", _DISLIKE, "se me acabó el pollo")
    assert m.startswith("BLOQUEADO") and "modify_pantry_inventory" in m and "No cambies el perfil" in m


def test_esta_enganchada_en_el_unico_punto_de_ejecucion():
    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = s.index("_src_block = _guard_source_intent_mutation(tool_name, tool_args, state, messages)")
    bloque = s[i:i + 1400]
    assert "bloqueo_si_es_la_nevera(tool_name, tool_args, _extraer_ultimo_texto_usuario(messages))" in bloque
    assert bloque.index("bloqueo_si_es_la_nevera") < bloque.index("elif tool_name == \"update_form_field\"")


def test_lee_el_texto_real_del_ultimo_mensaje_del_usuario():
    from langchain_core.messages import AIMessage, HumanMessage
    import agent

    msgs = [HumanMessage(content="no me gusta el brócoli"), AIMessage(content="ok"),
            HumanMessage(content=[{"type": "text", "text": "se me acabó el pollo"}])]
    assert bloqueo_si_es_la_nevera("update_form_field", _DISLIKE, agent._extraer_ultimo_texto_usuario(msgs))
