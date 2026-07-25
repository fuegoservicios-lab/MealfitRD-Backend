"""[P1-TEST-ALERT-POLLUTION · 2026-07-25] La suite escribía incidentes FALSOS en el tablero.

La suite corre contra la MISMA base Neon que producción (no hay staging) y ~9 módulos escriben
`system_alerts` con SQL directo. Una noche de corridas dejó 3 alertas falsas ABIERTAS:

    plan_persist_failed:user-abc        meta: 'invalid input syntax for type uuid: "user-abc"'
    plan_data_corrupted:plan-corrupt:_last_chunk_learning
    plan_data_corrupted:plan-corrupt:_recent_chunk_lessons

`user-abc` y `plan-corrupt` son fixtures literales (`test_chunk_recovery_7day_plan.py`,
`test_chunked_learning_propagation.py`). Así es como se acaba ignorando el canal de alertas: se
llena de ruido y el operador deja de mirarlo. Encontradas revisando logs — la metadata de la
propia alerta delataba al fixture.

La guarda vive en el único cuello de botella (`execute_sql_write`) en vez de en los 9 call sites.
"""
import os

import pytest

import db_core


ALERT_INSERT = "INSERT INTO system_alerts (alert_key, message) VALUES (%s, %s)"


def test_bajo_pytest_se_descarta_la_escritura():
    # Este test corre BAJO pytest, así que PYTEST_CURRENT_TEST está puesta por definición.
    assert os.environ.get("PYTEST_CURRENT_TEST"), "pytest siempre exporta esta variable"
    assert db_core._suppress_test_alert_write(ALERT_INSERT) is True


def test_cubre_update_no_solo_insert():
    """Un test que 'resuelve' alertas podría cerrar una REAL — peor que crear una falsa."""
    assert db_core._suppress_test_alert_write("UPDATE system_alerts SET resolved_at = now()") is True
    assert db_core._suppress_test_alert_write("DELETE FROM system_alerts WHERE id = %s") is True


def test_no_toca_otras_tablas():
    """La guarda es quirúrgica: el resto de la suite escribe en la base con normalidad."""
    for q in ("INSERT INTO meal_plans (id) VALUES (%s)",
              "UPDATE user_inventory SET quantity = 0",
              "INSERT INTO pipeline_metrics (node) VALUES (%s)"):
        assert db_core._suppress_test_alert_write(q) is False, q


def test_fuera_de_pytest_no_suprime_nada(monkeypatch):
    """El señal es 'esto es un test', NO el entorno: un dev corriendo la app a mano apunta a la
    misma base y SÍ quiere sus alertas reales. Por eso no se usa ENVIRONMENT."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert db_core._suppress_test_alert_write(ALERT_INSERT) is False


def test_opt_in_explicito_para_quien_quiera_probar_el_camino_real(monkeypatch):
    monkeypatch.setenv("MEALFIT_ALLOW_TEST_SYSTEM_ALERTS", "true")
    assert db_core._suppress_test_alert_write(ALERT_INSERT) is False


def test_query_vacia_o_none_no_revienta():
    assert db_core._suppress_test_alert_write(None) is False
    assert db_core._suppress_test_alert_write("") is False


def test_el_writer_respeta_el_contrato_de_retorno(monkeypatch):
    """Descartar no puede parecer un fallo: `returning=False` → True (éxito),
    `returning=True` → lista vacía. Si devolviera None, los callers que hacen
    `if not execute_sql_write(...)` interpretarían un error inexistente."""
    monkeypatch.setattr(db_core, "connection_pool", None, raising=False)
    # Con pool=None, cualquier otra query levantaría RuntimeError: llegar a un retorno
    # limpio demuestra que la guarda corta ANTES de tocar el pool.
    assert db_core.execute_sql_write(ALERT_INSERT, ("k", "m")) is True
    assert db_core.execute_sql_write(ALERT_INSERT, ("k", "m"), returning=True) == []
    with pytest.raises(RuntimeError):
        db_core.execute_sql_write("INSERT INTO meal_plans (id) VALUES (%s)", ("x",))


def test_los_tests_que_afirman_sobre_la_insercion_no_se_rompen():
    """Parchean `execute_sql_write` a nivel de módulo, así que reemplazan la función entera y
    nunca ven la guarda. Se ancla el patrón para que nadie 'arregle' la guarda moviéndola a los
    call sites y rompa 60 archivos de tests."""
    from pathlib import Path
    ct = (Path(db_core.__file__).resolve().parent / "cron_tasks.py").read_text(encoding="utf-8")
    assert "from db_core import" in ct or "from db import" in ct, (
        "los módulos importan el writer por nombre: parchearlo por módulo sigue funcionando"
    )
