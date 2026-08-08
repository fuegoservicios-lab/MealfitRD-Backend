"""[P1-PANTRY-EXIST-WAIVER · 2026-08-08] La TERCERA guarda de nevera también
consulta la SSOT — o el modo flexible castiga el permiso que él mismo dio.

## El lazo, medido en producción (plan f380821a, semana 2)

    07:39:48  recovery TTL resucita w2 en flexible_mode (execute_after=NOW)
    07:39-08:00  pipeline completo: day_generator (Luna) ×N, self_critique,
                 culinary_judge, surgical_marker, planner  ← ~21 min de LLM
    08:00:49  re-pausa `pending_user_action:pantry_violation_after_retries`
              con `_pantry_flexible_mode=true` EN el snapshot y en form_data

P1-PANTRY-GATE-SSOT (2026-07-26) unificó la guarda PRE-pipeline y el gate de
reservas POST-merge en `_pantry_gate_waiver_reason`. Pero la validación de
existencia post-generación ([P0-3], la que produce
`pantry_violation_after_retries`) quedó fuera: decidía sola con
`if _pantry_snapshot:` — no leía NINGÚN flag, y por eso el blanket del SSOT
(que prohíbe lecturas sueltas de `_pantry_flexible_mode`) jamás la vio. Una
guarda puede desincronizarse por leer el flag por su cuenta O por no leerlo
en absoluto.

## Qué ancla este archivo

1. El call site existe y usa los MISMOS argumentos que `_res_waiver`.
2. El gate no vuelve a decidir solo: entre el snapshot y el `if` hay waiver.
3. La combinación del incidente (initial_plan + flexible) queda exenta; la
   estricta (rolling_refill sin flags) queda ÍNTEGRA.

tooltip-anchor: P1-PANTRY-EXIST-WAIVER
"""
from __future__ import annotations

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _assigns_de(nombre: str) -> list[ast.Assign]:
    tree = ast.parse(_CRON)
    worker = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_chunk_worker"
    )
    return [
        n for n in ast.walk(worker)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == nombre for t in n.targets)
    ]


def test_el_gate_de_existencia_consulta_la_ssot():
    asignaciones = _assigns_de("_exist_waiver")
    assert len(asignaciones) == 1, (
        f"esperaba exactamente 1 asignación de _exist_waiver en _chunk_worker, "
        f"hay {len(asignaciones)}. Sin ella, la validación de existencia vuelve "
        "a decidir sola y el modo flexible vuelve a castigar su propio permiso."
    )
    call = asignaciones[0].value
    assert isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_pantry_gate_waiver_reason"


def test_mismos_argumentos_que_el_gate_de_reservas():
    """Si un futuro arg nuevo entra a `_res_waiver` y no aquí (o viceversa),
    las guardas vuelven a discrepar — la clase exacta de P1-PANTRY-GATE-SSOT."""
    kw_exist = {k.arg for k in _assigns_de("_exist_waiver")[0].value.keywords}
    kw_res = {k.arg for k in _assigns_de("_res_waiver")[0].value.keywords}
    assert kw_exist == kw_res, (
        f"los dos call sites del waiver divergen: existencia={sorted(kw_exist)} "
        f"vs reservas={sorted(kw_res)}"
    )


def test_el_waiver_va_entre_el_snapshot_y_el_if():
    """El orden ES el fix: snapshot → waiver → salida temprana → if estricto.
    Si alguien reordena y el `if _pantry_snapshot:` vuelve a quedar pegado al
    snapshot, el gate decide sin waiver aunque la asignación siga existiendo."""
    ini = _CRON.index('_pantry_snapshot = form_data.get("current_pantry_ingredients", [])')
    fin = _CRON.index("if _pantry_snapshot:", ini)
    tramo = _CRON[ini:fin]
    assert "_exist_waiver" in tramo and "_pantry_gate_waiver_reason(" in tramo, (
        "entre la captura del snapshot y el gate estricto no se consulta el "
        "waiver: la tercera guarda volvió a quedar ciega"
    )
    assert "_pantry_ok = True" in tramo and "break" in tramo, (
        "el chunk waived debe salir por el camino del path advisory "
        "(_pantry_ok=True + break) — si cae al else, el drift check "
        "`_finalize_live_pantry_validation` lo pausa por otra puerta"
    )


def test_la_combinacion_del_incidente_queda_exenta_y_la_estricta_integra():
    from cron_tasks import _pantry_gate_waiver_reason

    # w2 de f380821a: initial_plan resucitado en flexible — DEBE quedar exento
    # (dos motivos independientes; cualquiera basta).
    assert _pantry_gate_waiver_reason(
        chunk_kind="initial_plan",
        snapshot={"_pantry_flexible_mode": True},
        form_data={"_pantry_flexible_mode": True},
    ) is not None

    # Refill a mitad de plan sin degradación: la promesa "cocinar con lo que
    # hay" sigue enforzada — el fix NO desactiva la validación estricta.
    assert _pantry_gate_waiver_reason(
        chunk_kind="rolling_refill", snapshot={}, form_data={},
    ) is None


def test_marker_anclado_en_el_fuente():
    assert _CRON.count("P1-PANTRY-EXIST-WAIVER") >= 2, (
        "el anchor del fix desapareció de cron_tasks.py"
    )
