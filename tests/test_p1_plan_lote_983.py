"""[P1-PLAN-LOTE-983 · 2026-11-11] Cantidades en los registros desde foto + aviso de exceso de kcal.

Caso vivo del dueño: la cena del mangú quedó «salami frito (incluyendo el salami consumido)» sin cantidad, y con
+865 kcal sobre la meta el coach no dijo nada (con +225 sí lo dijo).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ingredientes_con_cantidad as icc  # noqa: E402


def setup_function(_):
    icc._YA_AVISADO.clear()


def test_detecta_las_lineas_del_caso_del_dueno():
    faltan = icc.sin_cantidad([
        "mangú de auyama o batata (porción estimada)", "1 huevo frito",
        "salami frito (incluyendo el salami consumido)", "1 lasca de queso frito",
    ])
    assert faltan == ["mangú de auyama o batata (porción estimada)", "salami frito (incluyendo el salami consumido)"]


def test_cantidades_validas_y_condimentos_pasan():
    assert icc.sin_cantidad([
        "pan de agua (3 unidades, ≈180 g)", "una taza de arroz", "media taza de habichuelas", "3 lascas de salami",
        "sal al gusto", "aceite de oliva", "pizca de orégano", {"name": "pollo (150 g)"},
    ]) == []


def test_solo_en_foto_y_una_vez_por_usuario():
    ings = ["salami frito"]
    assert icc.rechazo("u", "Cena", ings, es_foto=False) is None
    txt = icc.rechazo("u", "Cena", ings, es_foto=True, ahora=1000.0)
    assert txt and "salami frito" in txt and not txt.startswith("⚠️ NO REGISTRADO:")
    assert icc.rechazo("u", "Otro nombre", ings, es_foto=True, ahora=1010.0) is None   # el reintento pasa
    assert icc.rechazo("u", "Cena", ings, es_foto=True, ahora=1200.0) is not None      # ventana vencida
    assert icc.rechazo("v", "Cena", ings, es_foto=True, ahora=1010.0) is not None      # otro usuario


def test_knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_PHOTO_INGREDIENT_QTY_GUARD", "0")
    assert icc.rechazo("u", "Cena", ["salami frito"], es_foto=True) is None


def test_no_se_confunde_con_aviso_de_duplicado():
    from agent import _AVISOS_DE_DUPLICADO
    txt = icc.rechazo("w", "Cena", ["salami frito"], es_foto=True, ahora=5.0)
    assert not any(a in txt for a in _AVISOS_DE_DUPLICADO)


def test_cableado_en_log_consumed_meal_y_aviso_de_exceso():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i_guard = src.index("_icc.rechazo(user_id, meal_name, ingredients, es_foto=_es_foto)")
    i_insert = src.index("# [P1-CONSUMED-BACKDATE · 2026-07-12] Fecha real de consumo + guard dup.")
    assert i_guard < i_insert
    assert 'if "kcal: YA SE PASÓ" in _m:' in src and "díselo SIEMPRE en una frase con la cifra" in src


def test_el_texto_de_exceso_casa_con_margen_del_dia():
    import coach_day_context as cdc
    m = cdc.margen_del_dia({"kcal": 2050, "protein_g": 120, "carbs_g": 250, "fats_g": 70},
                           {"kcal": 2915, "protein_g": 170, "carbs_g": 300, "fats_g": 120, "registros": 4})
    assert "kcal: YA SE PASÓ" in m


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 983
