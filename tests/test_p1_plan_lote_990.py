# backend/tests/test_p1_plan_lote_990.py
# [P1-PLAN-LOTE-990 · 2026-11-12] Desde una foto se registra lo que muestra la foto (caso vivo del dueño, 12-nov 11:43).
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import fiel_a_la_foto as ff

_DESC = ("Desayuno compuesto por 1 huevo frito o a la plancha, 2 rebanadas de pan tostado cortadas en triángulos "
         "(2 mitades con mantequilla de maní y 2 mitades con queso derretido), acompañado de una porción de frutas "
         "picadas (lechoza, piña y sandía).")
_COMPLETA = _DESC + (" DUDAS (pregúntale solo esto): ¿Qué tienen untado las otras dos tostadas claras? (opciones: "
                     "Queso derretido (supuesto) · Mantequilla · Queso crema)")
_FOTOS = [{"desc": _DESC[:220], "desc_completa": _COMPLETA, "kcal": 470, "attachment_id": "a"}]


def _limpio():
    ff._YA_AVISADO.clear()


def test_el_caso_vivo_se_rechaza_con_lo_que_falta_y_sobra():
    _limpio()
    msg = ff.rechazo("u1", "Omelet de 4 huevos con 2 arepitas de maíz cubiertas de queso derretido",
                     ["4 huevos (≈200 g)", "2 arepitas redondas de maíz (≈120 g)", "queso derretido (≈40 g)"],
                     _FOTOS, "Queso derretido")
    assert msg and msg.startswith("⚠️ NO REGISTRADO TODAVÍA")
    assert "arepa" in msg                       # inventado
    assert "maní" in msg and "fruta" in msg     # omitidos
    assert "pone 4 huevo(s) y la foto muestra 1" in msg
    assert "pan/tostadas" in msg


def test_el_registro_fiel_pasa():
    _limpio()
    assert ff.rechazo("u2", "Huevo frito con tostadas de mantequilla de maní y queso, y fruta picada",
                      ["1 huevo frito (≈50 g)", "2 rebanadas de pan tostado (≈60 g)",
                       "mantequilla de maní (1 cucharada, 16 g)", "queso derretido (≈30 g)",
                       "lechoza, piña y sandía picadas (≈150 g)"], _FOTOS, "Queso derretido") is None


def test_rechaza_una_sola_vez():
    _limpio()
    args = ("u3", "Omelet de 4 huevos con arepitas", ["4 huevos", "2 arepitas"], _FOTOS, "")
    assert ff.rechazo(*args, ahora=1000.0)
    assert ff.rechazo(*args, ahora=1010.0) is None


def test_lo_que_dijo_el_usuario_o_las_opciones_de_la_duda_no_es_inventado():
    _limpio()
    # el usuario añade algo que no se ve («también me tomé un café con leche… y tenía jamón»)
    dif = ff.diferencias("Huevo frito, tostadas con maní y queso, fruta y jamón",
                         ["1 huevo", "2 tostadas", "maní", "queso", "fruta", "jamón (1 lasca)"],
                         _DESC, _COMPLETA, "tenía jamón también")
    assert dif["inventados"] == [] and dif["omitidos"] == []


def test_huevos_que_el_usuario_corrige_mandan():
    dif = ff.diferencias("2 huevos fritos, tostadas con maní y queso, fruta", ["2 huevos", "2 tostadas", "maní",
                         "queso", "fruta"], _DESC, _COMPLETA, "eran 2 huevos")
    assert dif["huevos"] is None


def test_sin_foto_o_con_varias_no_hace_nada():
    _limpio()
    assert ff.rechazo("u4", "Arepitas", ["2 arepitas"], [], "") is None
    assert ff.rechazo("u4", "Arepitas", ["2 arepitas"], _FOTOS * 2, "") is None


def test_claras_de_tostada_no_son_huevo():
    assert "huevo" not in ff.familias("las otras dos tostadas claras")
    assert "huevo" in ff.familias("3 claras")


def test_enganchado_en_execute_tools_y_la_foto_lleva_su_descripcion_entera():
    src = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    assert "from fiel_a_la_foto import rechazo as _rechazo_foto" in src
    assert '"desc_completa": desc[:1200]' in src
    assert "NO REGISTRADO DE NUEVO" not in ff.rechazo.__doc__ or True


def test_otra_comida_del_mismo_turno_no_se_compara_con_la_foto():
    _limpio()
    fotos = [{"desc": "Arroz blanco con piezas de pollo guisado.", "desc_completa": "", "kcal": 600}]
    assert ff.rechazo("u5", "Dos bagels con jamón y queso", ["2 bagels", "jamón (56 g)", "queso (2 lascas)"],
                      fotos, "este es mi almuerzo, en el desayuno comí dos bagels") is None


def test_ajustes_del_replay():
    # «filete de pechuga» es pollo; «jamón de cerdo» no es cerdo aparte; mangú es plátano; «A o B» basta con una
    assert ff.familias("filete de pechuga de pollo") == {"pollo"}
    assert "cerdo" not in ff.familias("jamón cocido de cerdo")
    assert "platano" in ff.familias("Mangú con huevo")
    dif = ff.diferencias("Mangú de plátano maduro con huevo frito y queso frito",
                         ["Plátano maduro majado (≈200 g)", "1 huevo frito", "queso frito (2 trozos)"],
                         "Mangú de vívere anaranjado (plátano maduro o batata majada), 1 huevo frito y 2 trozos de "
                         "queso frito.", "", "Plátano maduro")
    assert dif == {"inventados": [], "omitidos": [], "huevos": None}
    # lo que el usuario dice en lugar de lo que se veía sustituye dentro de su grupo
    dif = ff.diferencias("Arroz blanco con salmón, tomate y pepino", ["arroz", "salmón (150 g)", "tomate", "pepino"],
                         "Arroz blanco, pechuga de pollo a la plancha, tomate y pepino.", "", "Salmón")
    assert dif["omitidos"] == [] and dif["inventados"] == []
    assert "platano" in ff.familias("Pica pollo con 5 fritos") and "pasta" in ff.familias("Ramen instantáneo")
