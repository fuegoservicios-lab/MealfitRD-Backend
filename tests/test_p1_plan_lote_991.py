# backend/tests/test_p1_plan_lote_991.py
# [P1-PLAN-LOTE-991 · 2026-11-12] «Me tomé la creatina» → SU toma (6 cápsulas), no «1 cápsula» (caso vivo del dueño).
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import suplementos as sp

_POTE = {"id": 1, "ingredient_name": "Creatine 5000 Creatine Monohydrate Capsules", "brand": "Natural Sport",
         "quantity": 0, "serving_unit": "capsula", "label_source": "estimado",
         "serving_label": {"kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0, "gramos_porcion": 0.83,
                           "unidades_por_toma": 6}}
_ARGS = {"meal_name": "Creatine 5000 Creatine Monohydrate Capsules (Natural Sport), 1 cápsula", "calories": 0,
         "protein": 0, "suplemento": "Creatine 5000", "porciones": 1, "meal_type": "suplemento",
         "ingredients": ["Creatina monohidratada, 1 cápsula"]}


def test_la_etiqueta_conserva_las_unidades_por_toma():
    assert sp.etiqueta_valida(_POTE["serving_label"])["unidades_por_toma"] == 6


def test_sin_cantidad_registra_su_toma(monkeypatch):
    monkeypatch.setattr(sp, "buscar", lambda uid, nombre: _POTE)
    out = sp.ajustar_toma("u", dict(_ARGS), "Me tome la creatina")
    assert out["porciones"] == 6
    assert "6 capsulas" in out["meal_name"] and "1 cápsula" not in out["meal_name"]
    assert out["ingredients"] == ["Creatina monohidratada, 6 capsulas"]


def test_con_cantidad_manda_el_usuario(monkeypatch):
    monkeypatch.setattr(sp, "buscar", lambda uid, nombre: _POTE)
    assert sp.ajustar_toma("u", dict(_ARGS), "me tomé 3 cápsulas de creatina") is None
    assert sp.ajustar_toma("u", dict(_ARGS), "me tomé una cápsula nada más") is None


def test_ya_era_su_toma_no_toca(monkeypatch):
    monkeypatch.setattr(sp, "buscar", lambda uid, nombre: _POTE)
    assert sp.ajustar_toma("u", dict(_ARGS, porciones=6), "me tomé la creatina") is None


def test_sin_etiqueta_usa_lo_que_registro_antes(monkeypatch):
    pote = dict(_POTE, serving_label={"kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0, "gramos_porcion": 5})
    import db_core
    filas = [{"meal_name": "Creatine 5000 Creatine Monohydrate Capsules (Natural Sport), 1 cápsula"},
             {"meal_name": "5 cápsulas de Creatine 5000 Creatine Monohydrate Capsules (Natural Sport)"},
             {"meal_name": "Creatine 5000 Creatine Monohydrate Capsules (Natural Sport) — 5 cápsulas"}]
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: filas)
    assert sp.dosis_habitual("u", pote) == {"n": 5, "fuente": "lo que registró antes"}


def test_el_bloque_del_chat_dice_su_toma(monkeypatch):
    monkeypatch.setattr(sp, "_potes", lambda uid: [_POTE])
    b = sp.bloque_para_chat("u")
    assert "SU TOMA: 6 capsula" in b
    assert "NUNCA 1 por defecto" in sp.BLOQUE_CONOCIMIENTO


def test_enganchado_en_execute_tools():
    src = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    assert "from suplementos import ajustar_toma as _ajustar_toma" in src
