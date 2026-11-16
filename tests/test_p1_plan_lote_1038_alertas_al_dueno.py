"""[P1-PLAN-LOTE-1038 · A-25 · 2026-11-16] Las alertas que importan llegan al dueño por push, sin abrir el panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import alertas_al_dueno as aad  # noqa: E402

ADMIN = "61a13831-0000-4000-8000-000000000001"
T0 = 1_800_000_000.0


class Base:
    """`system_alerts` y `app_kv_store` de mentira: devuelve las alertas dadas y recuerda la clave de avisadas."""

    def __init__(self, alertas, avisadas=None):
        self.alertas = alertas
        self.kv = json.dumps(avisadas) if avisadas is not None else None
        self.consultas = []

    def query(self, sql, params=None, **k):
        self.consultas.append((" ".join(sql.split()), params))
        if "FROM app_kv_store" in sql:
            return {"value": json.loads(self.kv)} if self.kv is not None else None
        return list(self.alertas)

    def write(self, sql, params=None, **k):
        assert "INSERT INTO app_kv_store" in sql and params[0] == aad.CLAVE_KV
        self.kv = params[1]
        return True


def _alerta(clave, tipo="fail_open", severidad="warning", titulo="Protección del coach fallando en silencio: chat_budget_check"):
    return {"alert_key": clave, "alert_type": tipo, "severity": severidad, "title": titulo}


def _correr(base, ahora=T0, admins=(ADMIN,), acepta=True):
    enviados = []

    def enviar(uid, titulo, cuerpo, url="/dashboard", tag=None, **k):
        enviados.append({"uid": uid, "titulo": titulo, "cuerpo": cuerpo, "url": url, "tag": tag})
        return acepta
    r = aad.avisar_pendientes(execute_sql_query=base.query, execute_sql_write=base.write, enviar=enviar,
                              admins=admins, ahora=ahora)
    return r, enviados


def test_la_consulta_elige_lo_critico_y_lo_del_coach_de_las_ultimas_24_h():
    sql = " ".join(aad._SQL_PENDIENTES.split())
    assert "resolved_at IS NULL" in sql and "triggered_at >= NOW() - INTERVAL '24 hours'" in sql
    assert "severity = 'critical' OR alert_type = ANY(%s) OR alert_key LIKE 'chat%%'" in sql
    assert aad.TIPOS_DEL_COACH == ("fail_open", "cost_guardrail", "openai_ratelimit")


def test_una_alerta_nueva_sale_por_push_una_sola_vez_en_24_h():
    base = Base([_alerta("fail_open:chat_budget_check")])
    r, enviados = _correr(base)
    assert r == {"enviadas": 1, "nuevas": 1, "destinatarios": 1}
    assert enviados == [{"uid": ADMIN, "titulo": aad.TITULO_PUSH, "url": "/admin", "tag": "alerta:fail_open:chat_budget_check",
                         "cuerpo": "Protección del coach fallando en silencio: chat_budget_check"}]
    assert base.consultas[0][1] == (["fail_open", "cost_guardrail", "openai_ratelimit"],)

    r, enviados = _correr(base, ahora=T0 + 3600)               # una hora después: sigue abierta, no se repite
    assert r["nuevas"] == 0 and enviados == []
    r, enviados = _correr(base, ahora=T0 + 24 * 3600 + 1)      # al día siguiente sigue abierta: se recuerda
    assert r["nuevas"] == 1 and len(enviados) == 1


def test_como_mucho_tres_push_y_el_resto_se_resume_sin_goteo():
    base = Base([_alerta(f"fail_open:sitio_{i}", titulo=f"Alerta {i}") for i in range(7)])
    r, enviados = _correr(base)
    assert r["nuevas"] == 7 and len(enviados) == 3
    assert [e["cuerpo"] for e in enviados] == ["Alerta 0", "Alerta 1", "Alerta 2 (y 4 más en el panel)"]
    r, enviados = _correr(base, ahora=T0 + 3600)
    assert enviados == [], "las resumidas ya quedaron avisadas: no gotean 3 por hora"


def test_cada_administrador_recibe_el_suyo_y_un_fallo_no_corta_el_resto():
    base = Base([_alerta("chat_reliability_degraded", tipo="reliability_degradation", titulo="El coach está usando el respaldo")])
    llamados = []

    def enviar(uid, titulo, cuerpo, **k):
        llamados.append(uid)
        if uid == "a-roto":
            raise RuntimeError("apns")
        return True
    r = aad.avisar_pendientes(execute_sql_query=base.query, execute_sql_write=base.write, enviar=enviar,
                              admins=("b-bien", "a-roto", " ", "b-bien"), ahora=T0)
    assert llamados == ["a-roto", "b-bien"] and r == {"enviadas": 1, "nuevas": 1, "destinatarios": 2}
    assert json.loads(base.kv) == {"chat_reliability_degraded": T0}, "queda avisada aunque un dispositivo falle"


def test_el_texto_no_lleva_identificadores_y_es_una_linea():
    t = aad.texto_del_push("Plan 808c7e26-3b55-4a52-83b7-378ccd91b7fa\n  de la cuenta 865e95c9-dbb1-4d72-89e4-252fdeea4d14 atascado")
    assert t == "Plan … de la cuenta … atascado"
    assert len(aad.texto_del_push("x" * 400)) == 140
    assert aad.texto_del_push(None) == "Alerta sin título"


def test_sin_alertas_sin_administradores_o_apagado_no_hace_nada(monkeypatch):
    r, enviados = _correr(Base([]))
    assert r["enviadas"] == 0 and enviados == []
    r, enviados = _correr(Base([_alerta("fail_open:x")]), admins=())
    assert r == {"enviadas": 0, "nuevas": 0, "destinatarios": 0} and enviados == []
    monkeypatch.setenv("MEALFIT_ALERTAS_PUSH_AL_DUENO", "0")
    base = Base([_alerta("fail_open:x")])
    r, enviados = _correr(base)
    assert r["enviadas"] == 0 and enviados == [] and base.consultas == []


def test_el_estado_se_poda_a_siete_dias_y_tolera_basura():
    base = Base([_alerta("fail_open:nueva")], avisadas={"vieja": T0 - 8 * 24 * 3600, "reciente": T0 - 3600, "rara": "x"})
    _correr(base)
    assert json.loads(base.kv) == {"reciente": T0 - 3600, "fail_open:nueva": T0}


def test_si_la_base_falla_no_rompe_el_cron_y_queda_contado(monkeypatch):
    import fail_open as fo
    contados = []
    monkeypatch.setattr(fo, "fail_open", lambda sitio, exc, **k: contados.append(sitio))

    def rota(*a, **k):
        raise RuntimeError("db")
    r = aad.avisar_pendientes(execute_sql_query=rota, execute_sql_write=rota, enviar=lambda *a, **k: True,
                              admins=(ADMIN,), ahora=T0)
    assert r["enviadas"] == 0 and contados == ["alertas_al_dueno"]


def test_corre_dentro_del_vigilante_de_gasto_sin_job_nuevo():
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = cron.index("def _llm_daily_spend_alert_job(")
    cuerpo = cron[i:cron.index("\ndef ", i + 10)]
    assert '__import__("alertas_al_dueno").avisar_pendientes(' in cuerpo
    assert cuerpo.index("evaluar_salud_coach_y_alertar(") < cuerpo.index("avisar_pendientes("), \
        "primero se abren las alertas del coach y después se avisa"
    assert "alertas_al_dueno" not in cron.replace(cuerpo, ""), "no registra un job propio"
