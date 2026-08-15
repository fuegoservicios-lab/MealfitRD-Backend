"""[P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] Los bloques PRESCRIPTIVOS del system
prompt no reciben un plan que el usuario pausó.

EL AGUJERO QUE CIERRA. El arreglo de esta misma mañana (P1-AGENT-WELCOME-TRACKING)
enseñó la pausa a UN bloque —`_plan_context_for_chat`— y la auditoría encontró que
otros cuatro del MISMO prompt la contradecían unas líneas más abajo. `grep
plan_mode agent.py` daba 6 aciertos, todos dentro de ese único helper. Con el plan
pausado, el prompt seguía diciendo:

  - «HOY es el día N del menú; asume HOY → day_number=N y NO le preguntes»
    (`_build_plan_today_context`), más el contador de ciclo que al agotarse
    empuja a RENOVAR — que es exactamente reencender el gasto que el usuario apagó.
  - «DÍAS QUE FALTAN POR GENERARSE… se generan por etapas» y días «ATRASADOS»
    (`_build_pending_days_lines_block`) sobre una cola que la pausa CANCELÓ.
  - «Hoy te quedan N comida(s) del plan» (`_build_today_remaining_context` parte b).
  - Y el presupuesto de kcal cayendo a `current_plan['calories']` del plan
    congelado, en vez de las metas que el propio dashboard del contador pinta.

LA FORMA DEL ARREGLO, y por qué NO es otro gate por call site. Gatear call sites es
lo que produjo este bug: se arregla el que se ve y quedan los demás. Aquí el modo
se resuelve UNA vez por turno y se deriva un dato — `plan_vigente`, que es `None`
en pausa — que reciben las secciones prescriptivas. Los builders no aprenden nada
de modos: simplemente no tienen plan del que hablar, y sus guardas de shape (que ya
existían, `isinstance(current_plan, dict)`) los apagan solas.

  `current_plan`  → el plan de verdad. Lo sigue recibiendo `_plan_context_for_chat`,
                    que en pausa lo entrega con el encuadre correcto (PAUSADO ≠
                    AMPUTADO: si el usuario pregunta por su plan, hay que poder
                    responderle y ofrecerle reanudar).
  `plan_vigente`  → el plan que GOBIERNA el día. `None` mientras esté en pausa.

Un bloque prescriptivo futuro que reciba `plan_vigente` queda gateado sin que nadie
se acuerde de gatearlo.

Tooltip-anchor: P1-CHAT-PAUSED-PROMPT-BLOCKS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_AGENT = Path(__file__).resolve().parent.parent / "agent.py"

PLAN = {
    "calories": 2100,
    "generation_status": "paused_by_user",
    "days": [{"day": 1, "date": "2026-08-14", "meals": [
        {"meal": "Desayuno", "name": "Mangú con Huevo", "calories": 400},
        {"meal": "Cena", "name": "Pasta Integral Salteada", "calories": 700},
    ]}],
}


@pytest.fixture()
def agente():
    import agent
    return agent


def _fuente() -> str:
    return _AGENT.read_text(encoding="utf-8")


def _sin_comentarios(t: str) -> str:
    return re.sub(r"^\s*#.*$", "", t, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# 1. El derivador del dato
# ---------------------------------------------------------------------------

def test_en_pausa_no_hay_plan_vigente(agente, monkeypatch):
    monkeypatch.setattr(agente, "_plan_mode_for_chat", lambda uid: "tracking", raising=True)
    assert agente._plan_vigente_para_prompt("u1", PLAN) is None, (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] En modo contador `plan_vigente` debe ser "
        "None: es el dato que apaga TODAS las secciones prescriptivas a la vez, sin "
        "que cada una tenga que acordarse del modo."
    )


def test_con_el_plan_activo_el_plan_vigente_es_el_plan(agente, monkeypatch):
    monkeypatch.setattr(agente, "_plan_mode_for_chat", lambda uid: "plan", raising=True)
    assert agente._plan_vigente_para_prompt("u1", PLAN) is PLAN, (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] Con el plan vigente no cambia nada: el "
        "contrato de siempre no se toca."
    )


def test_si_el_modo_no_se_puede_leer_se_asume_activo(agente, monkeypatch):
    """Fail-open: un fallo de DB no puede dejar mudo el chat de TODOS."""
    def revienta(uid):
        raise RuntimeError("db caída")
    monkeypatch.setattr(agente, "_plan_mode_for_chat", revienta, raising=True)
    assert agente._plan_vigente_para_prompt("u1", PLAN) is PLAN


# ---------------------------------------------------------------------------
# 2. Comportamiento de los bloques cuando no hay plan vigente
# ---------------------------------------------------------------------------

def test_sin_plan_vigente_no_se_mapea_hoy_al_dia_del_menu(agente):
    assert agente._build_plan_today_context(None) == "", (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] `_build_plan_today_context` debe salir "
        "vacío sin plan vigente. Es el bloque que dice «HOY es el día N del menú, "
        "asume day_number=N y NO le preguntes» y el que empuja a RENOVAR."
    )


def test_sin_plan_vigente_desaparecen_las_comidas_pero_NO_el_presupuesto(agente):
    """La parte (a) —kcal restantes— SÍ sirve en el contador; la (b) no."""
    consumido = [{"meal_type": "Desayuno", "calories": 400}]
    salida = agente._build_today_remaining_context(None, consumido, 2100, 400.0)
    assert "Pasta Integral" not in salida and "Mangú" not in salida, (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] Sin plan vigente siguen apareciendo las "
        "comidas del plan («hoy te quedan N comida(s) del plan»)."
    )
    assert salida.strip(), (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] El bloque quedó COMPLETAMENTE mudo. El "
        "presupuesto de calorías restantes es justo lo que un contador de macros "
        "necesita — apagarlo entero es pasarse de frenada y empeorar el modo que "
        "estamos arreglando."
    )


def test_un_plan_pausado_no_anuncia_dias_por_generarse(agente):
    """«DÍAS QUE FALTAN POR GENERARSE… se generan por etapas» / «ATRASADO».

    Este bloque se gatea por el DATO del propio plan (`generation_status`) y no
    por el modo de sesión: llega anidado dentro de `_build_past_days_context`, y
    hacerle llegar el modo exigiría enhebrar un parámetro por dos funciones y dos
    paths — justo el tipo de hilo que se rompe en el siguiente refactor. El plan
    ya sabe que está pausado; se le pregunta a él. Además cuesta cero (sin
    roundtrip a DB) y sigue siendo correcto aunque el bloque se invoque desde un
    camino nuevo que nadie acordó gatear.
    """
    from datetime import date
    import chat_history_context as chc

    # Sin esto el bloque sale por sus guardas baratas (el knob apagado y
    # `plan_cycle_pending_days`=0) y el test pasaría SIN arreglo: un veredicto
    # que no puede fallar no informa. Se le abren las puertas para que el único
    # motivo de silencio posible sea la pausa.
    _knob = chc.upcoming_days_signal_enabled
    _pend = chc.plan_cycle_pending_days
    chc.upcoming_days_signal_enabled = lambda: True
    chc.plan_cycle_pending_days = lambda *a, **k: 3
    try:
        salida = agente._build_pending_days_lines_block("u1", PLAN, date(2026, 8, 14))
    finally:
        chc.upcoming_days_signal_enabled = _knob
        chc.plan_cycle_pending_days = _pend

    assert salida == "", (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] Con `generation_status='paused_by_user'` el "
        "bloque sigue anunciando días pendientes. La pausa CANCELA la cola: "
        "prometer días «que se generan por etapas», o declararlos «ATRASADOS», es "
        "hablar de un trabajo que nadie va a hacer."
    )


# ---------------------------------------------------------------------------
# 3. Los dos paths consumen el dato derivado (no `current_plan`)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [
    "_build_plan_today_context",
    "_build_today_remaining_context",
])
def test_los_builders_prescriptivos_reciben_plan_vigente(builder):
    codigo = _sin_comentarios(_fuente())
    llamadas = re.findall(rf"{builder}\(\s*([A-Za-z_][\w]*)", codigo)
    # La primera aparición es la definición (`def ...(current_plan, ...)`).
    invocaciones = [a for a in llamadas if a != "current_plan"] + \
                   [a for a in llamadas if a == "current_plan"]
    reales = llamadas[1:]  # saltamos la firma
    assert reales, f"No se encontraron invocaciones de {builder}"
    malas = [a for a in reales if a != "plan_vigente"]
    assert not malas, (
        f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] `{builder}` se invoca con {malas} en vez de "
        "`plan_vigente`. Ese argumento es el gate: con `current_plan` el bloque "
        "vuelve a recitar el plan pausado. Hay DOS paths (stream y no-stream) y los "
        "dos tienen que pasar el dato derivado — la divergencia entre ambos ya "
        "costó bugs antes (P1-CHAT-PAST-DAYS)."
    )


def test_el_presupuesto_no_cae_al_plan_pausado():
    """`target_calories` no puede degradar a las kcal de un plan en pausa."""
    codigo = _sin_comentarios(_fuente())
    malas = re.findall(r"target_calories\s*=\s*current_plan\.get\(\s*[\"']calories", codigo)
    assert not malas, (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] El presupuesto de kcal sigue cayendo a "
        "`current_plan['calories']`. En modo contador eso son las calorías del plan "
        "CONGELADO presentadas como la meta de hoy, mientras el dashboard pinta "
        "otras: el coach y su propia pantalla dirían cifras distintas. Usa "
        "`plan_vigente` y, sin él, `get_nutrition_targets(form_data)` — la MISMA "
        "función pura que sirve /api/nutrition/targets."
    )


@pytest.mark.parametrize("funcion", ["chat_with_agent", "chat_with_agent_stream"])
def test_plan_vigente_se_asigna_antes_de_usarse(funcion):
    """El dato derivado tiene que nacer ANTES de su primer consumidor.

    Casi se despliega al revés: la asignación quedó 41 líneas por DEBAJO de la
    primera lectura (`build_inventory_context`), que es un `NameError` en caliente
    — y no en un rincón, sino en el ensamblado del prompt de los DOS paths del
    chat. Ningún test de comportamiento lo habría cazado, porque todos ejercitan
    los helpers por separado y jamás la función de 600 líneas que los cose.
    """
    import ast
    arbol = ast.parse(_fuente())
    nodo = next((n for n in ast.walk(arbol)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == funcion), None)
    assert nodo is not None, f"No existe {funcion}"

    asignaciones, lecturas = [], []
    for n in ast.walk(nodo):
        if isinstance(n, ast.Name) and n.id == "plan_vigente":
            (asignaciones if isinstance(n.ctx, ast.Store) else lecturas).append(n.lineno)
    if not lecturas:
        pytest.skip(f"{funcion} no consume plan_vigente")
    assert asignaciones, (
        f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] `{funcion}` lee `plan_vigente` "
        f"(línea {min(lecturas)}) sin asignarlo nunca: NameError en caliente."
    )
    assert min(asignaciones) < min(lecturas), (
        f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] En `{funcion}` `plan_vigente` se asigna en "
        f"la línea {min(asignaciones)} pero ya se lee en la {min(lecturas)}. Eso es "
        "un NameError que tumba el ensamblado del system prompt entero."
    )


def test_el_contexto_honesto_sigue_recibiendo_el_plan_de_verdad():
    """PAUSADO != AMPUTADO: el bloque que encuadra la pausa conserva el plan."""
    codigo = _sin_comentarios(_fuente())
    assert re.search(r"_plan_context_for_chat\(\s*user_id\s*,\s*current_plan", codigo), (
        "[P1-CHAT-PAUSED-PROMPT-BLOCKS] `_plan_context_for_chat` dejó de recibir "
        "`current_plan`. Ese es el ÚNICO bloque que debe seguir viendo el plan real: "
        "si el usuario pregunta «¿qué tenía mi plan?» hay que poder responderle y "
        "ofrecerle reanudar. Apagarlo también sería arreglar el bug rompiendo la "
        "función."
    )
