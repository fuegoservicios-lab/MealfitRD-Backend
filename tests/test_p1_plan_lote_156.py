"""[P1-PLAN-LOTE-156 · 2026-09-22] Que el proveedor de IA esté caído deje de ser algo que se
descubre por los usuarios.

EL HUECO. El `LLMCircuitBreaker` abre tras 3 fallos seguidos y persiste `is_open=true` en
`app_kv_store`. A partir de ese momento el coach contesta «temporalmente saturado» a TODO el
mundo sin llegar a llamar al proveedor — y eso es todo lo que pasa: ni alert, ni correo, ni
una fila que alguien mire. El 16-sep Z.ai se quedó sin saldo a las 21:30 y toda la IA cayó; el
dueño se enteró porque estaba usándola. Con 5 testers y UN solo proveedor vivo, la misma caída
de madrugada son horas de coach muerto en silencio.

Y el único rastro se borraba solo: `_sweep_stale_llm_circuit_breakers` (P2-NEW-D) resetea la
fila a las 2 h, por una razón buena (un `is_open=true` viejo confunde a quien lee la tabla).
Sumadas, las dos cosas daban una avería invisible y sin huella.

  *Una condición que solo se ve mientras dura, y cuyo rastro se limpia solo, es una condición
  que nadie va a ver nunca.*

Estos tests EJECUTAN el cron con la base fingida: lo que se afirma es lo que hace, no lo que
dice su docstring.

Tooltip-anchor: P1-PLAN-LOTE-156
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import cron_tasks

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DOC = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"

_AHORA = 1_790_000_000.0  # epoch fijo: los tests no dependen del reloj de quien los corre


def _correr(filas):
    """Corre el cron con `execute_sql_query` devolviendo `filas`. Devuelve las escrituras."""
    escrituras = []

    def _write(sql, params=()):
        escrituras.append((" ".join(str(sql).split()), params))
        return None

    class _RelojFijo:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone
            return datetime.fromtimestamp(_AHORA, tz or timezone.utc)

    with patch("cron_tasks.execute_sql_query", return_value=filas), \
         patch("cron_tasks.execute_sql_write", side_effect=_write), \
         patch("cron_tasks.datetime", _RelojFijo):
        cron_tasks._llm_breaker_open_alert_job()
    return escrituras


def _alertas(escrituras):
    return [(sql, p) for sql, p in escrituras if "INSERT INTO system_alerts" in sql]


def _resoluciones(escrituras):
    return [(sql, p) for sql, p in escrituras if "UPDATE system_alerts" in sql]


def test_un_breaker_abierto_emite_alerta_con_su_modelo():
    """El caso del 16-sep: el modelo del chat deja de responder y hay que enterarse."""
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 3, "last_failure": _AHORA - 300},
    ])
    alertas = _alertas(escrituras)
    assert len(alertas) == 1, "Un breaker abierto tiene que emitir exactamente un alert."
    _sql, params = alertas[0]
    assert params[0] == "llm_circuit_breaker_open:deepseek-flash", (
        "La key lleva el MODELO: colapsarlos escondería que el chat está caído mientras otra "
        "superficie respira."
    )
    assert "'critical'" in _sql, "El coach caído para todo el mundo no es un warning."
    metadata = json.loads(params[3])
    assert metadata["modelo"] == "deepseek-flash"
    assert metadata["failures"] == 3
    assert metadata["minutos_desde_el_ultimo_fallo"] == 5


def test_el_mensaje_dice_que_mirar():
    """Un alert que no dice qué hacer se archiva sin leer."""
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 4, "last_failure": _AHORA - 60},
    ])
    mensaje = _alertas(escrituras)[0][1][2]
    assert "saldo" in mensaje.lower(), "La causa número uno (saldo agotado) tiene que estar."
    assert "MEALFIT_LLM_PROVIDER" in mensaje, "Y el knob con el que se cambia de proveedor."


def test_la_clave_legacy_sin_sufijo_se_reporta_como_default():
    """`llm_circuit_breaker` a secas (global, pre-P1-Q3) sigue existiendo en producción."""
    escrituras = _correr([
        {"key": "llm_circuit_breaker", "failures": 3, "last_failure": _AHORA - 120},
    ])
    assert _alertas(escrituras)[0][1][0] == "llm_circuit_breaker_open:default"


def test_varios_modelos_abiertos_dan_varias_alertas():
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 3, "last_failure": _AHORA - 60},
        {"key": "llm_circuit_breaker:gpt-5.6-luna", "failures": 5, "last_failure": _AHORA - 90},
    ])
    claves = {p[0] for _s, p in _alertas(escrituras)}
    assert claves == {
        "llm_circuit_breaker_open:deepseek-flash",
        "llm_circuit_breaker_open:gpt-5.6-luna",
    }


def test_sin_breakers_abiertos_no_alerta_y_cierra_lo_que_hubiera():
    """El caso normal, que es el 99,9 % de los ticks: nada que decir y el pasado se cierra."""
    escrituras = _correr([])
    assert _alertas(escrituras) == [], "Sin avería no se emite nada."
    resoluciones = _resoluciones(escrituras)
    assert len(resoluciones) == 1
    sql, _params = resoluciones[0]
    assert "resolved_at = NOW()" in sql and "IS NULL" in sql, (
        "Una alerta que no se cierra sola envenena el panel: la siguiente de verdad se ignora."
    )
    # `NOT IN ()` es SQL inválido; la rama de «ninguno abierto» tiene que ser la suya.
    assert "ALL(" not in sql


def test_con_algunos_abiertos_solo_se_cierran_los_demas():
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 3, "last_failure": _AHORA - 60},
    ])
    sql, params = _resoluciones(escrituras)[0]
    assert "<> ALL(" in sql, "Cerrar TODO en el mismo tick borraría la alerta recién emitida."
    assert params[0] == ["llm_circuit_breaker_open:deepseek-flash"]


def test_la_consulta_pide_las_filas_explicitamente():
    """`execute_sql_query` SIN `fetch_all` devuelve `[]` aunque la consulta traiga filas.

    No es una preferencia de estilo: el helper lo dice en su propia docstring y ya enmascaró 8
    callsites en producción. Escribí este cron sin el flag y ninguno de los tests de arriba lo
    vio, porque todos mockean el helper: habría corrido cada 10 min informando de que todo
    está bien, para siempre.

      *Un mock del transporte no prueba el transporte.*
    """
    with patch("cron_tasks.execute_sql_query", return_value=[]) as q, \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._llm_breaker_open_alert_job()
    assert q.call_args.kwargs.get("fetch_all") is True, (
        "Sin `fetch_all=True` el vigilante lee una lista vacía siempre: nunca alertaría."
    )


def test_la_ventana_acota_la_consulta_y_es_configurable():
    """Sin ventana, una fila de hace días alertaría como si fuese de ahora."""
    with patch("cron_tasks.execute_sql_query", return_value=[]) as q, \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._llm_breaker_open_alert_job()
    sql, params = q.call_args[0][0], q.call_args[0][1]
    assert "is_open' = 'true'" in sql
    assert "last_failure" in sql and "EXTRACT(EPOCH FROM NOW())" in sql
    assert params == (90,), "Ventana por defecto de 90 min."

    with patch.dict("os.environ", {"MEALFIT_LLM_BREAKER_ALERT_WINDOW_MIN": "30"}), \
         patch("cron_tasks.execute_sql_query", return_value=[]) as q2, \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._llm_breaker_open_alert_job()
    assert q2.call_args[0][1] == (30,)


def test_el_cron_no_toca_el_kv_del_breaker():
    """Leer es leer. El reseteo es del sweep de P2-NEW-D, que es su dueño: dos escritores
    sobre la misma fila con reglas distintas es como nacen los estados imposibles."""
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 3, "last_failure": _AHORA - 60},
    ])
    for sql, _p in escrituras:
        assert "app_kv_store" not in sql, f"El vigilante escribió en el KV del breaker: {sql[:120]}"


def test_un_fallo_de_base_no_tumba_el_cron():
    """Un vigilante que revienta deja de vigilar justo cuando más falta hace."""
    with patch("cron_tasks.execute_sql_query", side_effect=RuntimeError("base caída")), \
         patch("cron_tasks.execute_sql_write") as w:
        cron_tasks._llm_breaker_open_alert_job()   # no levanta
    tick = [c for c in w.call_args_list if "pipeline_metrics" in str(c[0][0])]
    assert tick, "Aun fallando tiene que dejar su rastro en pipeline_metrics."
    assert json.loads(tick[0][0][1][1])["job_failed"] is True


def test_esta_registrado_en_el_scheduler_y_mas_a_menudo_que_el_barrendero():
    """El intervalo no es una preferencia: el sweep borra la fila a las 2 h."""
    fuente = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
    assert 'scheduler.get_job("llm_breaker_open_alert_job")' in fuente
    assert "_llm_breaker_open_alert_job," in fuente
    assert 'MEALFIT_LLM_BREAKER_ALERT_INTERVAL_MIN", 10' in fuente, (
        "Default 10 min. Con uno horario el cron podría no ver nunca la avería que vigila, "
        "porque `_sweep_stale_llm_circuit_breakers` resetea la fila a las 2 h."
    )


def test_la_alerta_esta_documentada_en_la_tabla_canonica():
    """`test_p2_audit_4_alert_keys_documented.py` exige paridad; esto ancla el CONTENIDO."""
    doc = _DOC.read_text(encoding="utf-8")
    fila = [l for l in doc.splitlines() if "`llm_circuit_breaker_open:<modelo>`" in l]
    assert fila, "La alerta nueva no está en la tabla canónica."
    assert "Auto (explicit)" in fila[0], "Su modelo de resolución es el cron cerrándola."


@pytest.mark.parametrize("campo", ["modelo", "failures", "ventana_min"])
def test_la_metadata_lleva_lo_necesario_para_diagnosticar(campo):
    escrituras = _correr([
        {"key": "llm_circuit_breaker:deepseek-flash", "failures": 3, "last_failure": _AHORA - 60},
    ])
    assert campo in json.loads(_alertas(escrituras)[0][1][3])
