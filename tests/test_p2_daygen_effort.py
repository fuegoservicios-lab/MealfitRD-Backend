"""[P2-DAYGEN-EFFORT · 2026-07-31] Nivel de razonamiento del day-gen, en el
vocabulario de CADA proveedor (el A/B luna-medium/xhigh vs flash-high/max).

La primera versión sólo cubría OpenAI y con el vocabulario clásico
(minimal..high): "luna al máximo" era imposible de pedir — el owner preguntó
"¿no pudiste probar luna max?" y la respuesta era que el knob lo impedía. La
doc oficial de luna (aportada por el owner) da: none/low/medium/high/XHIGH.
GLM usa low/medium/high/MAX vía `extra_body.thinking` (contrato ya usado
por reviewer y fact-checker).

CONTRATO:
  · Nace vacío ⇒ default del proveedor ⇒ encenderlo es explícito.
  · Alias cruzados: `max`→`xhigh` en OpenAI, `xhigh`→`max` en GLM — "el
    tope" se pide igual sin memorizar qué palabra usa cada API.
  · "none": en OpenAI se pasa explícito (apaga el razonamiento de luna); en
    GLM no se inyecta nada (el wrapper ya lo desactiva por default).
  · Con effort pedido, el timeout por llamada sale de
    MEALFIT_DAYGEN_EFFORT_TIMEOUT_S: el day-gen GLM con thinking superó
    los 170 s medidos (2026-06-13) y el tope base es 90 s — sin subirlo a la
    par, el A/B mediría la red de fallback, no el modelo.
  · Valor inválido ⇒ vacío (fail-safe al default, nunca a un esfuerzo alto:
    el razonamiento se factura como OUTPUT).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_knob_nace_apagado_y_acepta_ambos_vocabularios():
    m = re.search(r'DAYGEN_EFFORT = \(_env_str\("MEALFIT_DAYGEN_EFFORT", "(.*?)"\)', _GO)
    assert m, "falta el knob MEALFIT_DAYGEN_EFFORT"
    assert m.group(1) == "", "debe nacer VACÍO (default del proveedor)"
    i = _GO.index('DAYGEN_EFFORT = (_env_str("MEALFIT_DAYGEN_EFFORT"')
    win = _GO[i:i + 400]
    for nivel in ("none", "low", "medium", "high", "xhigh", "max"):
        assert f'"{nivel}"' in win, (
            f"el nivel {nivel} debe aceptarse — sin 'xhigh' luna no puede ir al "
            f"tope y sin 'max' GLM tampoco (fue el gap de la v1)"
        )
    assert '"minimal"' not in win, (
        "'minimal' no existe en la doc de luna ni en el contrato GLM — "
        "era el vocabulario clásico que dejó el knob corto"
    )
    assert 'DAYGEN_EFFORT = ""' in win, "un valor raro debe caer al default"


# [P1-DAYGEN-TIER-MODEL · 2026-07-31] El effort efectivo ahora se resuelve en
# la variable `_eff` (global de experimentos GANA; si no, el del tier gateado
# al modelo primario) — los asserts siguen a la variable resuelta, no al knob
# crudo, y la ventana creció con el bloque de resolución (900 → 1400).
def test_alias_cruzados():
    i = _GO.index("_es_openai = is_openai_model(_model)")
    win = _GO[i:i + 1400]
    assert '"xhigh" if _eff == "max" else _eff' in win, (
        "OpenAI no conoce 'max': debe traducirse a 'xhigh'"
    )
    # [P0-GLM-MIGRATION · 2026-09-02] La traducción al vocabulario de Z.ai vive en el
    # wrapper (`_glm_reasoning_effort`), no en el fork: xhigh→max, medium→high, none→low.
    from llm_provider import _glm_reasoning_effort
    assert _glm_reasoning_effort("xhigh") == "max"
    assert _glm_reasoning_effort("medium") == "high"
    assert _glm_reasoning_effort("none") == "low"
    assert _glm_reasoning_effort("max") == "max"


def test_glm_recibe_reasoning_effort():
    i = _GO.index("_es_openai = is_openai_model(_model)")
    win = _GO[i:i + 1400]
    assert '_kw["reasoning_effort"] = _eff' in win, (
        "GLM razona SIEMPRE: el effort del tier viaja como reasoning_effort y el "
        "wrapper lo traduce; sin él, day-gen correría con el default low del wrapper"
    )
    assert 'elif _eff != "none"' not in win, (
        "'none' ya no es 'no inyectar': GLM no puede apagar el razonamiento"
    )


def test_timeout_acompana_al_effort():
    assert '_env_int("MEALFIT_DAYGEN_EFFORT_TIMEOUT_S", 90' in _GO, (
        "falta el knob del timeout: el day-gen GLM con thinking midió "
        ">170 s y el tope base es 90 s — sin poder subirlo, cada llamada del "
        "A/B muere en timeout y se mide la red, no el modelo"
    )
    i = _GO.index("_es_openai = is_openai_model(_model)")
    win = _GO[i:i + 1400]
    assert '_kw["timeout"] = DAYGEN_EFFORT_TIMEOUT_S' in win


def test_reasoning_effort_es_parametro_real_del_cliente():
    """Verificado contra el paquete instalado, no asumido: si un upgrade de
    langchain-openai lo renombra, el knob quedaría inerte en silencio."""
    from langchain_openai import ChatOpenAI

    assert "reasoning_effort" in ChatOpenAI.model_fields, (
        "ChatOpenAI ya no acepta `reasoning_effort` — el knob quedaría inerte"
    )


def test_registrado_en_knobs():
    """Sin registro no aparece en /health/version y nadie sabe qué corre."""
    assert '_env_str("MEALFIT_DAYGEN_EFFORT"' in _GO
    assert '_env_int("MEALFIT_DAYGEN_EFFORT_TIMEOUT_S"' in _GO
