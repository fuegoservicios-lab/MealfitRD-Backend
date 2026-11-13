# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-998] Recorte sin pérdida del manual del coach: paridad exacta, repetible y sin gastar IA.

Reproduce (offline) una matriz de turnos contra el prompt real y exige que lo único que desaparezca sean
(1) etiquetas de procedencia y (2) el bullet de herramientas que ese turno no liga.

tooltip-anchor: P1-PLAN-LOTE-998-LEAN
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from langchain_core.messages import HumanMessage

from prompts import coach_lean as cl

_FRASES = [
    "me desayuné 2 huevos revueltos", "cuánto me queda de proteína hoy", "cuánta agua llevo", "soy alérgico al maní",
    "a partir de hoy soy vegano", "compré pollo y arroz", "se me acabó la leche", "qué tengo en la nevera",
    "dame mi lista de compras", "necesito más hierro", "qué dicen mis laboratorios", "activa el modo oscuro",
    "cambia el idioma a inglés", "me tomé la creatina", "qué ceno hoy", "corrige el almuerzo de ayer",
    "hola", "gracias", "mi peso es 82 kg", "quiero ir al supermercado", "cuánto potasio tiene mi plan",
    "bórralo, era un ejemplo", "qué comí el domingo", "recuérdame cómo era la receta de la cena",
]


def _prompt_real(stream: bool = True, **kw) -> str:
    from constants import coach_country_context, culinary_knowledge_base_for_country
    from prompts import chat_agent as ca
    if stream:
        return (ca.CHAT_STREAM_INLINE_PROMPT + coach_country_context("DO") + "\n"
                + culinary_knowledge_base_for_country("DO") + ca.build_tools_instructions_stream("U", **kw))
    return (ca.CHAT_AGENT_INLINE_PROMPT + coach_country_context("DO") + "\n"
            + culinary_knowledge_base_for_country("DO") + ca.build_tools_instructions("U", **kw))


@pytest.fixture(autouse=True)
def _knob(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_PROMPT_LEAN", raising=False)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("kw", [{}, {"plan_en_pausa": True}, {"contador_sin_plan": True}])
def test_las_etiquetas_quitadas_son_solo_procedencia(stream, kw):
    orig = _prompt_real(stream, **kw)
    quitadas = [m.group(0) for m in cl._ETIQUETA.finditer(orig)]
    assert len(quitadas) >= 30, "se esperaban decenas de etiquetas en el manual real"
    for q in quitadas:
        # una etiqueta es MAYÚSCULAS, dígitos y separadores: jamás una palabra de una regla
        assert not re.search(r"[a-záéíóúñ]{2}", q), f"parece texto de regla, no una etiqueta: {q!r}"
    out = cl.quitar_etiquetas(orig)
    assert len(out) == len(orig) - sum(len(q) for q in quitadas)
    # nada de lo que el modelo ejecuta o emite se toca
    for tok in re.findall(r"`[^`\n]+`", orig) + re.findall(r"\[UI_ACTION:[^\]]*\]", orig):
        assert out.count(tok) == orig.count(tok), tok
    assert not re.search(r"\[(?:P\d|BB)-[A-Z0-9\-]+", out)


def test_cada_bullet_podable_es_de_una_sola_linea_y_existe():
    orig = _prompt_real(True)
    for nombre in cl.HERRAMIENTAS_PODABLES:
        lineas = [l for l in orig.splitlines() if l.startswith(f"- Usa `{nombre}`")]
        assert len(lineas) == 1, f"{nombre}: esperaba 1 bullet, hay {len(lineas)}"
        assert cl._linea_de_bullet(nombre).search(orig)


def test_poda_exacta_solo_de_las_no_ligadas():
    orig = _prompt_real(True)
    assert cl.podar_herramientas_no_ligadas(orig, None) == orig                           # None ⇒ nada
    assert cl.podar_herramientas_no_ligadas(orig, cl.HERRAMIENTAS_PODABLES) == orig       # todas ligadas ⇒ nada
    solo_nevera = cl.podar_herramientas_no_ligadas(orig, {"modify_pantry_inventory"})
    esperado = "\n".join(l for l in orig.split("\n") if not any(
        l.startswith(f"- Usa `{n}`") for n in cl.HERRAMIENTAS_PODABLES if n != "modify_pantry_inventory"))
    assert solo_nevera == esperado
    assert "- Usa `modify_pantry_inventory`" in solo_nevera


@pytest.mark.parametrize("frase", _FRASES)
def test_replay_el_bullet_sigue_a_la_herramienta_ligada(frase):
    """Misma función que decide el esquema decide el bullet: nunca una herramienta ligada sin su instrucción, ni una
    instrucción de una herramienta que no existe en el turno."""
    import agent
    ligadas = {t.name for t in agent._select_tools_for_turn(agent.agent_tools, {"messages": [HumanMessage(content=frase)]})}
    orig = _prompt_real(True)
    out = cl.adelgazar(orig, ligadas)
    for nombre in cl.HERRAMIENTAS_PODABLES:
        assert (f"- Usa `{nombre}`" in out) == (nombre in ligadas), (frase, nombre)
    # todo lo demás (las reglas de registro/corrección, los núcleos, la interfaz) intacto
    for nombre in ("log_consumed_meal", "correct_consumed_meal", "update_form_field", "check_hydration_today",
                   "proponer_comida", "consultar_dia_del_plan", "search_deep_memory"):
        assert f"`{nombre}`" in out
    assert "UNA LLAMADA POR COMIDA" in out and "[UI_ACTION: REFRESH_INVENTORY]" in out


def test_con_foto_modo_voz_o_gating_apagado_no_se_poda_nada(monkeypatch):
    import agent
    base = _prompt_real(True)
    quitar = cl.quitar_etiquetas(base)
    assert agent._adelgazar_prompt_coach(base, "hola", con_foto=True) == quitar
    assert agent._adelgazar_prompt_coach(base, "hola", modo_voz=True) == quitar
    assert agent._adelgazar_prompt_coach(base, "   ") == quitar
    monkeypatch.setenv("MEALFIT_CHAT_TOOL_GATING_ENABLED", "0")
    assert agent._adelgazar_prompt_coach(base, "hola") == quitar


def test_el_knob_lo_apaga_y_el_ahorro_es_el_medido(monkeypatch):
    import agent
    base = _prompt_real(True)
    con = agent._adelgazar_prompt_coach(base, "me desayuné 2 huevos revueltos")
    assert len(base) - len(con) >= 2500, len(base) - len(con)      # ~1,9 k de etiquetas + ~1,2 k de bullets
    assert "- Usa `check_shopping_list`" not in con
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_LEAN", "0")
    assert agent._adelgazar_prompt_coach(base, "me desayuné 2 huevos revueltos") == base


def test_el_cableado_esta_en_los_dos_caminos():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert src.count("_adelgazar_prompt_coach(system_prompt, prompt, user_id=user_id") == 2
    # va ANTES de los bloques volátiles: solo toca el manual estático
    for m in re.finditer(r"system_prompt = _adelgazar_prompt_coach\(system_prompt", src):
        assert "bloques dinámicos (volátiles)" in src[m.start():m.start() + 600]
