"""[P1-CHAT-TODAY-CONTEXT · 2026-07-12] "actualiza el desayuno" = HOY, sin preguntar.

Vivo (owner, domingo): "actualiza el desayuno" → el agente preguntó "¿Opción A
(Domingo) u Opción B (Lunes)?". Tenía la HORA (build_temporal_context) pero no
el mapeo hoy→día-del-menú ni la posición del ciclo (día k de 7/15/30 — al
agotarse, el usuario debe RENOVAR).

`_build_plan_today_context`: (1) HOY es <weekday> → día N del menú (match por
day_name, soporta planes shifteados) + orden de asumir HOY sin preguntar y de
NO usar el legacy 'Opción A/B/C'; (2) ciclo día k de {7,15,30} con días
restantes, recordatorio suave a ≤3 días y aviso de ciclo VENCIDO.
tooltip-anchor: P1-CHAT-TODAY-CONTEXT
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from agent import _build_plan_today_context  # noqa: E402

_PLAN = {
    "days": [
        {"day": 1, "day_name": "Domingo", "meals": []},
        {"day": 2, "day_name": "Lunes", "meals": []},
    ],
    "cycle_start_date": "2026-07-11T11:56:04.347837+00:00",
    "calc_grocery_duration": "monthly",
}


def test_today_maps_to_menu_day_and_forbids_asking():
    # 2026-07-12 fue domingo (el caso vivo del owner).
    out = _build_plan_today_context(_PLAN, local_date_str="2026-07-12")
    assert "HOY es domingo 2026-07-12" in out
    assert "day_number=1" in out and "'Domingo'" in out
    assert "NO le preguntes" in out
    assert "Opción A/B/C" in out, "prohibición explícita del vocabulario legacy"


def test_cycle_position_matches_dashboard_chip():
    # cycle_start 2026-07-11 (11:56 UTC → día local 11-jul) + monthly=30 →
    # el 12-jul es día 2 de 30 y quedan 29 INCLUYENDO hoy — exactamente el
    # "30d mensual · 29d" del chip del Dashboard (mismo cálculo, cero drift).
    out = _build_plan_today_context(_PLAN, local_date_str="2026-07-12")
    assert "día 2 de 30" in out
    assert "quedan 29 día(s) incluyendo hoy" in out


def test_cycle_ending_and_expired():
    ending = _build_plan_today_context(_PLAN, local_date_str="2026-08-08")  # día 29/30
    assert "RENOVAR" in ending
    expired = _build_plan_today_context(_PLAN, local_date_str="2026-08-20")  # >30 días
    assert "YA TERMINÓ" in expired


def test_weekday_without_menu_day_still_gives_cycle():
    plan = dict(_PLAN, days=[{"day": 1, "day_name": "Martes", "meals": []}])
    out = _build_plan_today_context(plan, local_date_str="2026-07-12")  # domingo
    assert "day_number=" not in out, "sin match de día no se inventa mapeo"
    assert "día 2 de 30" in out, "el ciclo no depende del match del día"


def test_fail_open_empty():
    assert _build_plan_today_context(None) == ""
    assert _build_plan_today_context({"days": []}) == ""


def test_injected_in_both_paths():
    """El contexto de HOY se inyecta en los DOS paths del chat.

    [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] El argumento pasó de `current_plan`
    a `plan_vigente`, y el cambio es deliberado: `plan_vigente` es `None` mientras
    el usuario tenga el plan en pausa, así que las secciones PRESCRIPTIVAS del
    prompt («HOY es el día N del menú, no le preguntes») se apagan solas por sus
    guardas de shape. `current_plan` sigue existiendo para el bloque que SÍ debe
    hablar del plan pausado (PAUSADO ≠ AMPUTADO).

    Por eso este test ancla la INVARIANTE —que ambos paths inyecten el bloque— y no
    el nombre de la variable. Un test pegado al nombre convierte un rediseño
    correcto en un rojo, y la respuesta natural a un rojo injusto es relajar el
    test, no mejorarlo.
    """
    with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
        src = f.read()
    inyecciones = len(re.findall(
        r"system_prompt \+= _build_plan_today_context\(\s*(?:current_plan|plan_vigente)\b",
        src,
    ))
    assert inyecciones >= 2, (
        f"Sólo {inyecciones} path(s) del chat inyectan `_build_plan_today_context`. "
        "Los dos (welcome y turno normal) tienen que hacerlo: la divergencia entre "
        "ambos ya ha causado bugs antes (P1-CHAT-PAST-DAYS)."
    )
