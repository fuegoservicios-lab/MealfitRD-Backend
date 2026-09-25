# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-291 · 2026-09-25] El coach guarda suplementos en la Alacena y registra scoops con la etiqueta del pote.
Spec: docs/superpowers/specs/2026-09-25-suplementos-alacena-design.md §2. Tooltip-anchor: P1-PLAN-LOTE-291"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
WHEY = {"serving_g": 31, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5}


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ── Task 7: guardar_suplemento ──────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db(monkeypatch):
    import nevera_opcional
    import suplementos
    filas, escr = {}, []
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda uid, forzar=False: "activa")
    monkeypatch.setattr(suplementos, "buscar", lambda uid, nombre: filas.get(nombre.lower()))

    def _upsert(uid, nombre, marca, porciones, unidad, etiqueta, fuente):
        filas[nombre.lower()] = {"id": 1, "ingredient_name": nombre, "quantity": porciones or 0,
                                 "serving_unit": unidad, "serving_label": etiqueta, "label_source": fuente}
        escr.append((nombre, porciones, unidad, etiqueta, fuente))
    monkeypatch.setattr(suplementos, "_upsert", _upsert)
    return suplementos, filas, escr


def test_guardar_con_foto(db):
    s, filas, escr = db
    r = s.guardar("u", "Proteína Whey", "Optimum", 30, "scoop", WHEY, "foto")
    assert r["ok"] and r["fuente"] == "foto" and escr[-1][3] == WHEY


def test_etiqueta_absurda_cae_al_estimado_o_a_nada(db):
    s, filas, escr = db
    r = s.guardar("u", "Proteína Rara", None, 10, "scoop", {**WHEY, "kcal": 1200}, "foto")
    assert r["etiqueta"] is None and r["fuente"] == "estimado"      # sin clave: sin etiqueta
    r = s.guardar("u", "Whey genérica", None, 10, "scoop", {**WHEY, "kcal": 1200}, "foto", clave="whey_protein")
    assert r["etiqueta"]["protein_g"] == 24 and r["fuente"] == "estimado"


def test_sin_etiqueta_usa_el_estimado_de_la_clave(db):
    s, filas, escr = db
    r = s.guardar("u", "Creatina", None, 60, "g", None, "estimado", clave="creatine")
    assert r["etiqueta"]["kcal"] == 0 and r["fuente"] == "estimado"


def test_sin_estimado_si_no_se_quiere(db):
    s, filas, escr = db
    r = s.guardar("u", "Creatina", None, None, "g", None, "estimado", clave="creatine", usar_estimado=False)
    assert r["etiqueta"] is None and escr[-1][3] is None


def test_nevera_apagada_a_mano_pregunta(db, monkeypatch):
    s, filas, escr = db
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda uid, forzar=False: "preguntar")
    assert s.guardar("u", "Proteína Whey", None, 30, "scoop", WHEY, "foto")["ok"] is False and escr == []


def test_la_tool_esta_registrada_y_documentada():
    import tools
    assert "guardar_suplemento" in [t.name for t in tools.agent_tools]
    assert "guardar_suplemento" in _src("docs/agent_tools_user_id_table.md")


def test_la_tool_responde_y_avisa_del_estimado(db):
    import tools
    out = tools.guardar_suplemento.func("u", "Creatina", porciones=60, unidad="g", clave="creatine")
    assert "Guardado en su Alacena: Creatina" in out and "ESTIMADO" in out and "[UI_ACTION: REFRESH_INVENTORY]" in out
