"""[P2-DAYGEN-EFFORT · 2026-07-31] Nivel de razonamiento del day-gen cuando
corre en un modelo OpenAI (el A/B "luna con esfuerzo medium" que pidió el owner).

El canario de modelo (`MEALFIT_DAYGEN_CANARY_MODEL`) ya existía, pero sólo
elegía el MODELO: no había forma de pedir un nivel de razonamiento, así que
"luna con medium" no se podía ni probar.

CONTRATO:
  · Nace vacío ⇒ default del proveedor ⇒ encenderlo es explícito.
  · Sólo se aplica a modelos OpenAI: DeepSeek gobierna el razonamiento con
    `extra_body.thinking`, que es otro contrato. Mezclarlos haría que el A/B
    comparase dos configuraciones a la vez en vez de dos modelos.
  · Valor inválido ⇒ vacío (fail-safe al default, nunca a un esfuerzo alto:
    el razonamiento se factura como OUTPUT).

⚠️ CONTEXTO MEDIDO EN ESTE REPO: el corrector quirúrgico con razonamiento pasó
de 17 s a TIMEOUT de 120 s. El day-gen escribe un día entero con tope de 90 s —
la misma clase de superficie. Por eso el default es "no razonar de más".
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_knob_nace_apagado_y_valida():
    m = re.search(r'DAYGEN_EFFORT = \(_env_str\("MEALFIT_DAYGEN_EFFORT", "(.*?)"\)', _GO)
    assert m, "falta el knob MEALFIT_DAYGEN_EFFORT"
    assert m.group(1) == "", "debe nacer VACÍO (default del proveedor)"
    i = _GO.index('DAYGEN_EFFORT = (_env_str("MEALFIT_DAYGEN_EFFORT"')
    win = _GO[i:i + 500]
    for nivel in ("minimal", "low", "medium", "high"):
        assert f'"{nivel}"' in win, f"el nivel {nivel} debe ser aceptado"
    assert 'DAYGEN_EFFORT = ""' in win, "un valor raro debe caer al default, no a un esfuerzo alto"


def test_solo_se_aplica_a_openai():
    i = _GO.index("_es_openai = is_openai_model(_model)")
    win = _GO[i:i + 400]
    assert 'if _es_openai and DAYGEN_EFFORT:' in win, (
        "el esfuerzo NO puede aplicarse a DeepSeek: allí el razonamiento va por "
        "extra_body.thinking (otro contrato)"
    )
    assert '_kw["reasoning_effort"] = DAYGEN_EFFORT' in win


def test_reasoning_effort_es_parametro_real_del_cliente():
    """Verificado contra el paquete instalado, no asumido: si un upgrade de
    langchain-openai lo renombra, el knob quedaría inerte en silencio."""
    from langchain_openai import ChatOpenAI

    assert "reasoning_effort" in ChatOpenAI.model_fields, (
        "ChatOpenAI ya no acepta `reasoning_effort` — el knob quedaría inerte"
    )


def test_registrado_en_knobs():
    """Sin registro no aparece en /health/version y nadie sabe qué corre."""
    assert '_env_str("MEALFIT_DAYGEN_EFFORT"' in _GO, (
        "debe leerse con _env_str para auto-registrarse en _KNOBS_REGISTRY"
    )
