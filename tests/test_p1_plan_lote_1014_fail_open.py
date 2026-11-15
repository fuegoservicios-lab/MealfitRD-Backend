"""[P1-PLAN-LOTE-1014 · A-04] Un fail-open nunca es silencioso."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _limpio():
    import fail_open
    fail_open.reset_para_tests()
    yield
    fail_open.reset_para_tests()


def test_cuenta_y_avisa_una_vez_por_hora_y_sitio(caplog, monkeypatch):
    import fail_open as fo

    ahora = {"t": 1000.0}
    monkeypatch.setattr(fo, "_ahora", lambda: ahora["t"])
    caplog.set_level(logging.WARNING, logger="fail_open")
    for _ in range(5):
        fo.fail_open("sitio_a", RuntimeError("boom"))
    assert fo.snapshot()["sitio_a"]["n"] == 5
    assert sum("FAIL-OPEN" in r.message for r in caplog.records) == 1     # no inunda el log
    ahora["t"] += 3601
    fo.fail_open("sitio_a", RuntimeError("otra vez"))
    assert sum("FAIL-OPEN" in r.message for r in caplog.records) == 2     # pasada la hora, vuelve a avisar
    assert fo.total() == 6 and fo.total(solo_criticos=True) == 0


def test_un_sitio_critico_abre_alerta_como_mucho_una_vez_por_hora(monkeypatch):
    import fail_open as fo

    escrituras = []
    monkeypatch.setattr(fo, "_abrir_alerta", lambda sitio, n, error: escrituras.append((sitio, n)))
    for _ in range(4):
        fo.fail_open("tope", ValueError("columna inexistente"), critico=True)
    assert escrituras == [("tope", 1)]
    assert fo.total(solo_criticos=True) == 4


def test_nunca_lanza_aunque_la_alerta_o_el_texto_fallen(monkeypatch):
    import fail_open as fo

    class Raro:
        def __str__(self):
            raise RuntimeError("no se puede imprimir")

    fo.fail_open("x", Raro())             # exc no es BaseException: se formatea con str(); aun así no debe lanzar
    monkeypatch.setattr(fo, "_abrir_alerta", lambda *a: (_ for _ in ()).throw(RuntimeError("db caída")))
    fo.fail_open("y", RuntimeError("e"), critico=True)


def test_la_alerta_usa_columnas_reales_de_system_alerts():
    """`system_alerts`: id, alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at."""
    src = (_BACKEND / "fail_open.py").read_text(encoding="utf-8")
    i = src.index("INSERT INTO system_alerts")
    sql = src[i:src.index('""",', i)]
    for col in ("alert_key", "alert_type", "severity", "title", "message", "metadata", "triggered_at", "resolved_at"):
        assert col in sql, col
    assert "created_at" not in sql           # esa columna NO existe (error real del 14-nov al consultar la tabla)


def test_tope_de_sitios():
    import fail_open as fo

    for i in range(200):
        fo.fail_open(f"s{i}", "x")
    assert len(fo.snapshot()) == 64


# ── los sitios del chat ya no se tragan el error ─────────────────────────────────────────────────────────────────
def test_el_tope_de_gasto_cuenta_el_fallo_en_vez_de_callar():
    import chat_hard_turn
    import fail_open as fo

    os.environ["MEALFIT_CHAT_USER_DAILY_BUDGET_USD"] = "0.25"
    try:
        def q(*a, **k):
            raise RuntimeError('column "estimated_cost_micros" does not exist')

        r = chat_hard_turn.check_user_daily_chat_budget("u-1", execute_sql_query=q)
    finally:
        os.environ.pop("MEALFIT_CHAT_USER_DAILY_BUDGET_USD", None)
    assert r["status"] == "ok" and r["block_turn"] is False          # sigue siendo fail-open
    s = fo.snapshot()["chat_budget_check"]
    assert s["n"] == 1 and s["critico"] and "estimated_cost_micros" in s["ultimo_error"]


def test_las_alertas_de_salud_cuentan_cada_comprobacion_rota():
    import chat_observability_alerts as coa
    import fail_open as fo

    def q(*a, **k):
        raise RuntimeError("columna fantasma")

    coa.evaluar_salud_coach_y_alertar(execute_sql_query=q, execute_sql_write=lambda *a, **k: None, include_hourly_and_cost=True)
    sitios = set(fo.snapshot())
    assert {"chat_health_reliability", "chat_health_feedback", "chat_health_burst", "chat_health_cost_spike"} <= sitios


def test_ningun_except_del_chat_queda_en_debug_silencioso():
    """Los archivos del lote 1011/1010 no pueden volver a tragarse un error con `logger.debug(...)` a secas."""
    for nombre in ("chat_hard_turn.py", "chat_observability_alerts.py"):
        src = (_BACKEND / nombre).read_text(encoding="utf-8")
        assert "omitido (best-effort)" not in src and "fail-open: %r" not in src, nombre
    src = (_BACKEND / "chat_phase_metrics.py").read_text(encoding="utf-8")
    assert "best-effort): %r" not in src.split("def query_chat_phase_percentiles")[0] or "fail_open" in src


def test_health_version_expone_solo_cifras():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert '"fail_open_total": fail_open_total' in src and '"fail_open_critical": fail_open_critical' in src
    assert "fail_open.snapshot" not in src          # el detalle por sitio NO va en el endpoint público
    s2 = (_BACKEND / "routers" / "system.py").read_text(encoding="utf-8")
    assert '"fail_open": _fo_snapshot()' in s2


# ── el guion de comprobación del deploy ──────────────────────────────────────────────────────────────────────────
def test_el_guion_detecta_una_consulta_rota():
    import verificar_chat_contra_base as v

    def q_roto(sql, params=(), **k):
        raise RuntimeError('column "endpoint" does not exist')

    problemas = v.verificar(q_roto, "u-1")
    assert problemas and any("endpoint" in p for p in problemas)
    assert any("fail_open[chat_budget_check]" in p for p in problemas)


def test_el_guion_da_ok_con_una_base_que_responde_bien(monkeypatch):
    import db_core
    import verificar_chat_contra_base as v

    def q_ok(sql, params=(), fetch_one=False, fetch_all=False, **k):
        return {} if fetch_one else []

    monkeypatch.setattr(db_core, "execute_sql_query", q_ok)   # las query_chat_* lo importan de db_core al llamarse
    problemas = v.verificar(q_ok, "u-1")
    assert problemas == [], problemas
