"""[P1-SWAP-LUNA · 2026-08-05] El swap sube a gpt-5.6-luna, con el effort por superficie.

POR QUÉ. El plan NACE con luna (`day_generator`) y cada plato que el usuario sustituía
después lo escribía `deepseek-v4-flash`: cada actualización cambiaba un plato del modelo
bueno por uno del barato, dentro de un día ya cuadrado.

MEDIDO contra la API real desde el VPS (mismo prompt de swap, 3 corridas):

    flash  temperature=0.3        8,1 s   → "Salmón glaseado…" LAS TRES VECES
    luna   reasoning_effort=low   8,2 s   → 3 platos dominicanos distintos
    luna   reasoning_effort=med  16,5 s   → 3 platos distintos

Luna en `low` cuesta lo MISMO en espera que flash. `medium` la dobla — y el día completo
es un bucle EN SERIE de 4-5 swaps, así que ahí `medium` lo llevaría de ~35 s a 66-83 s.
De ahí el effort por superficie.

⚠️ EL BUG QUE ESTO CIERRA NO ERA EL MODELO, ERA EL CLIENTE. `MEALFIT_CHAT_AGENT_SWAP_MODEL`
ya existía y parecía bastar, pero el callsite construía `ChatDeepSeek` FIJO: poner el knob
a un ID de OpenAI habría mandado cada swap al base_url de DeepSeek con la key equivocada.
Por eso el test que importa es `test_construye_el_cliente_del_proveedor_correcto`.

tooltip-anchor: P1-SWAP-LUNA
"""
import os
import re
from pathlib import Path
from unittest import mock

import pytest

import agent
from llm_provider import GPT56_LUNA

_AGENT_SRC = Path(agent.__file__).read_text(encoding="utf-8")


def _cuerpo_de_swap_meal() -> str:
    """Cuerpo de `swap_meal`, acotado por la SIGUIENTE definición top-level.

    Deliberadamente NO es una ventana de N caracteres: `swap_meal` pasa de 30k y la
    primera versión de este test falló por eso. Una ventana fija caduca sola en cuanto
    la función crece — ya pasó tres veces en este repo con otros guards.
    """
    i = _AGENT_SRC.index("def swap_meal(form_data")
    j = _AGENT_SRC.find("\ndef ", i + 1)
    assert j > i, "no se encontró el final de swap_meal"
    return _AGENT_SRC[i:j]


# ---------------------------------------------------------------- modelo

def test_el_default_del_swap_es_luna():
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        assert agent._chat_agent_swap_model_name("u1") == GPT56_LUNA


def test_sin_openai_key_degrada_en_vez_de_reventar():
    """Fail-safe: un swap es user-facing.

    `build_chat_llm` LEVANTA si le piden un modelo OpenAI sin `OPENAI_API_KEY`. Sin este
    degradado, una env var ausente convertiría cada swap en un 500 — el usuario no podría
    cambiar ni un plato. Mismo criterio que la red post-fallo P1-NET-LUNA.
    """
    entorno = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with mock.patch.dict(os.environ, entorno, clear=True):
        elegido = agent._chat_agent_swap_model_name("u1")
    assert elegido != GPT56_LUNA, "sin key NO puede seguir pidiendo el modelo de OpenAI"
    assert elegido, "y tampoco puede devolver vacío: el swap tiene que seguir funcionando"


def test_el_knob_de_modelo_sigue_ganando():
    """Rollback sin redeploy (convención P3-PREVIEW-MODEL-KNOB)."""
    with mock.patch.dict(os.environ, {"MEALFIT_CHAT_AGENT_SWAP_MODEL": "deepseek-v4-flash"}):
        assert agent._chat_agent_swap_model_name("u1") == "deepseek-v4-flash"


# ---------------------------------------------------------------- effort

def test_effort_por_superficie():
    """El día NO puede heredar el effort del plato individual: son 4-5 swaps en serie."""
    entorno = {k: v for k, v in os.environ.items()
               if k not in ("MEALFIT_SWAP_EFFORT_INDIVIDUAL", "MEALFIT_SWAP_EFFORT_DAY")}
    with mock.patch.dict(os.environ, entorno, clear=True):
        assert agent._swap_reasoning_effort("individual") == "medium"
        assert agent._swap_reasoning_effort("day") == "low"


def test_cada_superficie_tiene_su_knob():
    with mock.patch.dict(os.environ, {"MEALFIT_SWAP_EFFORT_DAY": "medium"}):
        assert agent._swap_reasoning_effort("day") == "medium"
        assert agent._swap_reasoning_effort("individual") == "medium"
    with mock.patch.dict(os.environ, {"MEALFIT_SWAP_EFFORT_INDIVIDUAL": "high"}):
        assert agent._swap_reasoning_effort("individual") == "high"


def test_un_knob_invalido_degrada_al_default_y_no_a_un_400():
    """⚠️ Verificado CONTRA LA API: `minimal` NO existe en gpt-5.6-luna (400: los válidos
    son none/low/medium/high/xhigh). Un valor inválido aquí no degradaría la calidad —
    rompería el swap entero."""
    for basura in ("minimal", "", "  ", "MEDIO", "42"):
        with mock.patch.dict(os.environ, {"MEALFIT_SWAP_EFFORT_DAY": basura}):
            assert agent._swap_reasoning_effort("day") == "low", basura
    assert "minimal" not in agent._SWAP_EFFORT_VALID
    assert set(agent._SWAP_EFFORT_VALID) == {"none", "low", "medium", "high", "xhigh"}


def test_una_superficie_desconocida_no_revienta():
    assert agent._swap_reasoning_effort("inventada") == "medium"
    assert agent._swap_reasoning_effort("") == "medium"


# ------------------------------------------------- el bug real: el cliente

def test_construye_el_cliente_del_proveedor_correcto():
    """EL test de este P-fix.

    El defecto no era el nombre del modelo: era que el callsite instanciaba `ChatDeepSeek`
    fijo. Se verifica que `swap_meal` pide el cliente a la fábrica por proveedor y que le
    pasa el effort — no que el código esté escrito de cierta forma.
    """
    cuerpo = _cuerpo_de_swap_meal()
    j = cuerpo.index("_swap_base_llm =")
    constructor = cuerpo[j:j + 200]
    assert "build_chat_llm(" in constructor, (
        "el swap volvió a instanciar un cliente concreto; con un modelo OpenAI eso lo manda "
        "al base_url de DeepSeek con la key equivocada"
    )
    assert "ChatDeepSeek(" not in constructor


def test_no_le_pasa_temperatura_a_los_modelos_openai():
    """LangChain DESCARTA la temperatura EN SILENCIO en estos modelos.

    Pasarla igualmente dejaría en el código una garantía que el runtime no cumple.
    """
    cuerpo = _cuerpo_de_swap_meal()
    i = cuerpo.index("_swap_effort_kwargs = (")
    bloque = cuerpo[i:i + 400]
    m = re.search(r"\((.*?)if is_openai_model\([^)]*\)\s*else(.*?)\)", bloque, re.S)
    assert m, "se perdió la bifurcación por proveedor en los kwargs del swap"
    rama_openai, rama_deepseek = m.group(1), m.group(2)
    assert "reasoning_effort" in rama_openai
    assert "temperature" not in rama_openai, (
        "a un modelo OpenAI no se le pasa temperature: LangChain la descarta en silencio y "
        "el código quedaría afirmando una garantía que el runtime no cumple"
    )
    assert "temperature" in rama_deepseek
    assert "reasoning_effort" not in rama_deepseek

    # Y el timeout sigue en el callsite, LITERAL: el tripwire P0-CHAT-LLM-TIMEOUT lo busca
    # ahí. Esconderlo dentro de un dict lo dejaba fuera de su vigilancia (pasó en la primera
    # versión de este fix).
    j = cuerpo.index("build_chat_llm(")
    assert "timeout=" in cuerpo[j:j + 260]


# ------------------------------------------------- el día pide su effort

def test_regenerate_day_pide_el_effort_del_dia():
    """Sin esto el bucle del día heredaría `medium` y pasaría de ~35 s a 66-83 s."""
    plans = Path(agent.__file__).parent / "routers" / "plans.py"
    src = plans.read_text(encoding="utf-8")
    llamadas = re.findall(r"=\s*swap_meal\((.*?)\)\n", src)
    assert llamadas, "no se encontró ninguna llamada a swap_meal en el router"
    for args in llamadas:
        assert 'surface="day"' in args, (
            "una llamada de regenerate-day dejó de pedir el effort del día: %r" % args
        )


def test_swap_meal_acepta_la_superficie_sin_romper_a_los_callers():
    import inspect
    sig = inspect.signature(agent.swap_meal)
    assert "surface" in sig.parameters
    assert sig.parameters["surface"].default == "individual", (
        "el default tiene que dejar intactos a los callers individuales"
    )


@pytest.mark.parametrize("superficie,esperado", [("individual", "medium"), ("day", "low")])
def test_tabla_resumen(superficie, esperado):
    entorno = {k: v for k, v in os.environ.items() if not k.startswith("MEALFIT_SWAP_EFFORT_")}
    with mock.patch.dict(os.environ, entorno, clear=True):
        assert agent._swap_reasoning_effort(superficie) == esperado
