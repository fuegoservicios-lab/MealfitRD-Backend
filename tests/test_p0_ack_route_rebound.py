"""[P0-ACK-ROUTE-REBOUND · 2026-08-10] Un decorador decora la SIGUIENTE función.

EL DEFECTO. `@router.post("/pending-status/ack")` estaba pegado a
`_emit_change_outcome_metric`, un helper de telemetría que alguien insertó ENTRE el
decorador y su handler (P1-CHANGE-OUTCOME-TELEMETRY, 2026-08-05). El decorador no
busca «la función que le corresponde»: toma la de abajo. Resultado:
  · la ruta quedó apuntando al helper, que exige `kind` y `outcome` ⇒ **422 en el
    100% de las llamadas** (verificado en los logs de producción),
  · `api_pending_pipeline_ack` se quedó SIN decorador: código muerto,
  · y el helper de telemetría quedó expuesto como endpoint HTTP.

CONSECUENCIA PARA EL USUARIO: el acuse nunca limpiaba el KV, así que cada carga de
página releía `status='complete'` y relanzaba el aviso «Tu plan está listo» de un plan
entregado hacía días.

POR QUÉ NADIE LO VIO: el frontend hacía `catch { /* best-effort */ }` sin mirar el
status. *Un fallo que se traga su propio error es indistinguible del éxito* — y este
llevaba desde el 5 de agosto.

Este test es estructural a propósito: comprueba que CADA decorador de ruta de este
router queda pegado a un handler de verdad, no solo el que se rompió. La clase entera
de fallo desaparece, en vez de quedar arreglada una vez.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _rutas_y_su_funcion():
    """(línea, ruta, nombre de la función que realmente decora)."""
    lineas = _SRC.split("\n")
    out = []
    for i, ln in enumerate(lineas):
        m = re.match(r'\s*@router\.(get|post|patch|put|delete)\("([^"]+)"', ln)
        if not m:
            continue
        # La primera línea que declara función tras el decorador (saltando otros
        # decoradores apilados y líneas en blanco).
        for j in range(i + 1, min(i + 12, len(lineas))):
            s = lineas[j].strip()
            if not s or s.startswith("@") or s.startswith("#"):
                continue
            f = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", s)
            out.append((i + 1, m.group(2), f.group(1) if f else None))
            break
    return out


def test_el_acuse_apunta_a_su_handler():
    rutas = [r for r in _rutas_y_su_funcion() if r[1] == "/pending-status/ack"]
    assert len(rutas) == 1, f"se esperaba una sola ruta de acuse; hay {len(rutas)}"
    _linea, _ruta, fn = rutas[0]
    assert fn == "api_pending_pipeline_ack", (
        f"la ruta del acuse decora `{fn}` en vez del handler. Si es un helper, el "
        "endpoint devolverá 422 y el aviso «Tu plan está listo» volverá en cada carga."
    )


def test_ninguna_ruta_decora_un_helper_privado():
    """La clase entera, no solo el caso que se rompió: un handler HTTP nunca empieza
    por `_`. Si un decorador acaba pegado a un helper privado, es que alguien insertó
    código entre el decorador y su función."""
    intrusos = [(ln, ruta, fn) for ln, ruta, fn in _rutas_y_su_funcion()
                if fn and fn.startswith("_")]
    assert not intrusos, (
        "hay decoradores de ruta pegados a helpers privados — el decorador toma la "
        f"función SIGUIENTE, no la que se pretendía: {intrusos}"
    )


def test_toda_ruta_decora_una_funcion():
    huerfanos = [(ln, ruta) for ln, ruta, fn in _rutas_y_su_funcion() if fn is None]
    assert not huerfanos, f"decoradores sin función debajo: {huerfanos}"


def test_el_helper_de_telemetria_ya_no_es_un_endpoint():
    i = _SRC.find("def _emit_change_outcome_metric")
    assert i > 0, "desapareció el helper de telemetría"
    antes = _SRC[max(0, i - 400):i]
    assert "@router." not in antes, (
        "el helper de telemetría volvió a quedar expuesto como endpoint HTTP: acepta "
        "`kind` y `outcome` sin auth y escribe en pipeline_metrics"
    )
