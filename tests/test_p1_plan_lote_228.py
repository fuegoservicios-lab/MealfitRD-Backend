# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-228 · 2026-09-25] «Tu plan está listo» con la app cerrada.

El dueño: «cuando se esté generando un plan en el móvil, quiero que se pueda salir de la app y que cuando termine
llegue una notificación, como la de hidratación o la de comer».

Salir ya se podía: la generación corre en el servidor y el plan se guarda igual (cola o tarea del SSE que no se cancela
al desconectarse). Lo que faltaba era ENTERARSE. Web/PWA: una Web Push al pasar `pending_pipeline` a `complete`
(o `failed`) — `aviso_plan_listo.py`, disparada desde `db_plans.upsert_pending_pipeline`, el cuello de botella de las
dos vías, y solo en la transición. App nativa: notificación local (`frontend/src/__tests__/lote228.test.js`).

Tooltip-anchor: P1-PLAN-LOTE-228
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_UID = "61a13831-0000-4000-8000-000000000001"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_aviso_solo_en_la_transicion():
    import aviso_plan_listo as a
    assert a.aviso_para("complete", "generating")[0] == "Tu plan está listo 🎉"
    assert a.aviso_para("complete", None)[0] == "Tu plan está listo 🎉"      # KV borrado: el plan terminó igual
    assert a.aviso_para("complete", "complete") is None                     # segundo marcado del mismo run
    assert a.aviso_para("failed", "generating")[0] == "No pudimos terminar tu plan"
    assert a.aviso_para("failed", None) is None                             # sin generación en curso: no es noticia
    assert a.aviso_para("generating", None) is None


@pytest.fixture
def envios(monkeypatch):
    import bg_executor
    import utils_push
    enviados = []
    monkeypatch.setattr(bg_executor, "submit_bg_task", lambda fn, *a, task_name, **k: fn())
    monkeypatch.setattr(utils_push, "send_push_notification",
                        lambda uid, title, body, url="/dashboard", tag=None, solo_si_no_mira=False:
                        enviados.append((uid, title, body, url, tag, solo_si_no_mira)) or True)
    return enviados


def test_la_push_sale_con_etiqueta_y_solo_si_no_mira(envios):
    import aviso_plan_listo as a
    assert a.avisar_fin_de_generacion(_UID, "complete", "generating") is True
    assert envios == [(_UID, "Tu plan está listo 🎉", "Toca para verlo.", "/dashboard", "plan-listo", True)]


def test_invitados_y_knob_apagado_no_envian(envios, monkeypatch):
    import aviso_plan_listo as a
    assert a.avisar_fin_de_generacion("guest-session-123", "complete", "generating") is False
    monkeypatch.setenv("MEALFIT_PLAN_READY_PUSH", "false")
    assert a.avisar_fin_de_generacion(_UID, "complete", "generating") is False
    assert envios == []


def test_upsert_pending_pipeline_avisa_una_vez(envios, monkeypatch):
    import db_plans
    kv = {}
    monkeypatch.setattr(db_plans, "get_pending_pipeline", lambda uid: kv.get(uid))

    def _write(sql, params):
        import json
        kv[_UID] = json.loads(params[1])
    monkeypatch.setattr(db_plans, "execute_sql_write", _write)
    assert db_plans.upsert_pending_pipeline(_UID, status="generating")
    assert envios == []
    assert db_plans.upsert_pending_pipeline(_UID, status="complete", plan_id_final="p1")
    assert db_plans.upsert_pending_pipeline(_UID, status="complete", plan_id_final="p1")   # el fallback repite
    assert len(envios) == 1 and envios[0][1] == "Tu plan está listo 🎉"


def test_el_fallo_de_la_push_no_rompe_el_upsert(monkeypatch):
    import db_plans
    import aviso_plan_listo
    monkeypatch.setattr(db_plans, "get_pending_pipeline", lambda uid: {"status": "generating"})
    monkeypatch.setattr(db_plans, "execute_sql_write", lambda *a, **k: None)

    def revienta(*a, **k):
        raise RuntimeError("pool lleno")
    import bg_executor
    monkeypatch.setattr(bg_executor, "submit_bg_task", revienta)
    assert db_plans.upsert_pending_pipeline(_UID, status="complete", plan_id_final="p1") is True
    assert aviso_plan_listo.avisar_fin_de_generacion(_UID, "complete", "generating") is False


def test_el_payload_lleva_la_marca_para_el_service_worker():
    src = _src("utils_push.py")
    assert "solo_si_no_mira: bool = False" in src and '_payload["solo_si_no_mira"] = True' in src


def test_textos_traducidos_en_el_catalogo_de_push():
    import push_i18n
    import aviso_plan_listo as a
    for texto in (a.TITULO_LISTO, a.CUERPO_LISTO, a.TITULO_FALLO, a.CUERPO_FALLO):
        assert texto in push_i18n.push_catalog_keys(), texto
        assert push_i18n.translate_push_text(texto, "en-US") != texto


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 228 and m.group(2) >= "2026-09-25"


def test_frontend_vigia_nativo():
    front = _BACKEND.parent / "frontend" / "src"
    if not (front / "__tests__" / "lote228.test.js").exists():
        pytest.skip("el frontend de este checkout no trae el lote 228")
    assert "iniciarVigiaPlanListo" in (front / "main.jsx").read_text(encoding="utf-8")
    assert "if (data.solo_si_no_mira) {" in (front / "custom-sw.js").read_text(encoding="utf-8")
