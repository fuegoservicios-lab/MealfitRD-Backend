# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] Conversación del dueño del 21-nov (00:40-00:44), cuatro roturas:

1. Las preguntas del escáner sin «¿» se leían como UNA: el atún se reemplazó por «manteca de cerdo».
2. La creatina y la multivitamina se fusionaron en la cena (franja «cena», lote 982); la multivitamina entró.
3. Lo anotado de madrugada quedaba en el día nuevo (el 20 sin cena).
4. La Nevera no leía «(≈200 g)»: todo caía en la porción típica (Lay's de 20 g → 150 g de papa).
"""
from datetime import datetime, timezone
from pathlib import Path

import dia_biologico as dbio
import gramos_del_renglon as gr
import respuestas_de_la_foto as rf
import toma_de_suplemento as tds

_BACKEND = Path(__file__).resolve().parent.parent

_ALMUERZO = (
    "Arroz blanco cocido; atún guisado o salteado con cebolla roja; plátano maduro frito (maduros al caldero). "
    "DUDAS (pregúntale solo esto): El atún es en agua o en aceite? (opciones: En agua / escurrido · En aceite "
    "(supuesto) · Manteca de cerdo · Aceite de oliva · Sin grasa / a la plancha) Cómo se frieron los maduros? "
    "(opciones: Aceite vegetal (supuesto) · Manteca de cerdo · Aceite de oliva · Mantequilla · Air fryer / sin grasa) "
    "(Estimación: Calorías: 675, Proteína: 34g, Carbohidratos: 88g, Grasas Saludables: 19g)"
)


# ── 1. Preguntas sin «¿» ────────────────────────────────────────────────────────────────────────────────────────────
def test_dos_preguntas_sin_signo_de_apertura():
    bloque = _ALMUERZO.split("esto): ")[1].split(" (Estimación")[0]
    preguntas = rf._parse_dudas_y_opciones(bloque)
    assert [p["supuesto"] for p in preguntas] == ["En aceite", "Aceite vegetal"]
    assert preguntas[0]["opciones"][0] == "En agua / escurrido"
    assert "Cómo se frieron" not in " ".join(preguntas[0]["opciones"])


def test_el_atun_sigue_y_cada_respuesta_dice_a_que_responde():
    desc, _, parejas = rf.aplicar_respuestas_multiples(_ALMUERZO, "En agua / escurrido · Manteca de cerdo")
    pre = desc.split("DUDAS")[0]
    assert "atún guisado" in pre
    assert "manteca de cerdo guisado" not in pre
    assert "El atún es en agua o en aceite: en agua / escurrido" in pre
    assert "Cómo se frieron los maduros: manteca de cerdo" in pre
    assert len(parejas) == 2


def test_una_grasa_no_reemplaza_la_proteina():
    assert rf._reemplazar_proteina_por_sujeto("atún guisado", "¿Con qué aceite?", "Aceite vegetal",
                                              "Manteca de cerdo") is None


# ── 2. Toma de suplemento ────────────────────────────────────────────────────────────────────────────────────────────
_POTES = ["DESIGNS FOR SPORT MULTI + PHYTO", "Creatine 5000 Creatine Monohydrate Capsules"]


def test_es_toma_por_su_alacena_y_por_palabras():
    multi = {"meal_name": "DESIGNS FOR SPORT MULTI + PHYTO (Designs for Sport)", "calories": 0, "meal_type": "cena"}
    assert tds.es_toma("u", multi, potes=_POTES)
    assert tds.es_toma("u", {"meal_name": "Creatina 5 g", "calories": 0, "meal_type": "cena"}, potes=[])
    assert tds.es_toma("u", {"meal_name": "x", "suplemento": "whey", "calories": 120}, potes=[])
    assert tds.es_toma("u", {"meal_name": "x", "meal_type": "suplemento", "calories": 0}, potes=[])


def test_comida_con_palabra_de_suplemento_no_es_toma():
    plato = {"meal_name": "Bowl alto en proteína con pollo", "calories": 520, "meal_type": "almuerzo"}
    assert not tds.es_toma("u", plato, potes=_POTES)


def test_nombra_pote_nuevo():
    orig = "Mangú con huevo y salami"
    assert tds.nombra_pote_nuevo("u", orig + " y DESIGNS FOR SPORT MULTI + PHYTO", orig, potes=_POTES)
    assert not tds.nombra_pote_nuevo("u", orig + " y aguacate", orig, potes=_POTES)


def test_la_toma_no_pasa_por_la_fusion():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "_es_toma_supl = _tds.es_toma(_trusted_uid, tool_args)" in src
    assert 'if tool_name == "log_consumed_meal" and not _es_toma_supl:\n                    _fused_part_args' in src
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "_tds.nombra_pote_nuevo(user_id, meal_name, _orig_name)" in tools
    assert 'if suplemento:   # [P1-PLAN-LOTE-1141]' in tools


# ── 3. Madrugada = ayer ──────────────────────────────────────────────────────────────────────────────────────────────
_RD = 240
_0042_RD = datetime(2026, 11, 21, 4, 42, tzinfo=timezone.utc)


def test_cena_de_madrugada_va_a_ayer_a_las_23(monkeypatch):
    monkeypatch.setenv("MEALFIT_MADRUGADA_ES_AYER", "1")
    assert dbio.hora_en_el_dia_de_ayer("cena", _0042_RD, _RD) == "2026-11-21T03:00:00+00:00"   # 20-nov 23:00 RD
    assert dbio.hora_en_el_dia_de_ayer("suplemento", _0042_RD, _RD) == "2026-11-21T03:45:00+00:00"
    assert dbio.hora_en_el_dia_de_ayer("almuerzo", _0042_RD, _RD) == "2026-11-20T17:00:00+00:00"


def test_desayuno_y_horas_normales_no_se_mueven(monkeypatch):
    monkeypatch.setenv("MEALFIT_MADRUGADA_ES_AYER", "1")
    assert dbio.hora_en_el_dia_de_ayer("desayuno", _0042_RD, _RD) is None
    assert dbio.hora_en_el_dia_de_ayer("cena", datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc), _RD) is None
    monkeypatch.setenv("MEALFIT_MADRUGADA_ES_AYER", "0")
    assert dbio.hora_en_el_dia_de_ayer("cena", _0042_RD, _RD) is None


def test_log_consumed_meal_usa_la_madrugada():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "_dbio.hora_en_el_dia_de_ayer(_meal_type" in tools
    assert "_hora_guardada = _hora_madrugada or (_consumed_at if _days_ago > 0 else None)" in tools


# ── 4. La Nevera lee la cantidad del renglón ────────────────────────────────────────────────────────────────────────
def test_cantidad_al_frente():
    assert gr.al_frente("papas fritas Lay's Clásicas (1 paquete, 20 g)") == "20 g papas fritas Lay's Clásicas"
    assert gr.al_frente("huevo frito en aceite de oliva (1 unidad, ≈50 g)") == "1 huevo frito en aceite de oliva"
    assert gr.al_frente("arroz blanco cocido (≈200 g)") == "200 g arroz blanco cocido"
    assert gr.al_frente("8 rodajas de plátano maduro (≈180 g)") == "180 g plátano maduro"
    assert gr.al_frente("galletas (1 paquete, tamaño no especificado)") == "galletas (1 paquete, tamaño no especificado)"
    assert gr.al_frente("Avena") == "Avena"


def test_el_descuento_usa_al_frente():
    src = (_BACKEND / "db_inventory.py").read_text(encoding="utf-8")
    assert "_item_parse = _al_frente(item)" in src
    assert "_parse_quantity(_item_parse)" in src


# ── 5. El escáner sabe qué fotos mandó antes («de las que te envié») ────────────────────────────────────────────────
def test_fotos_recientes_resuelve_la_referencia():
    import fotos_recientes as fr
    fr._memoria.clear()
    fr.anotar("u-fr", "1 paquete de papas fritas Lay's Clásicas")
    txt = "Me voy a comer 3 papitas de las que te envié con 3 rodajas de queso"
    assert "Lay's" in fr.contexto("u-fr", txt)
    assert "Lay's" in fr.contexto("u-fr", "Tú sabes, la que recién envié")
    assert fr.contexto("u-fr", "Me comí 3 papitas con queso") == ""      # sin referencia, sin contexto
    assert fr.contexto("otro", txt) == ""


def test_el_prompt_del_escaner_lleva_las_fotos_previas():
    import fotos_recientes as fr
    import vision_agent as va
    with fr.scope("«1 paquete de papas fritas Lay's Clásicas»"):
        p = va._prompt_de_escaneo("3 papitas de las que te envie")
    assert "FOTOS QUE EL USUARIO MANDO ANTES" in p and "Lay's" in p
    assert "FOTOS QUE EL USUARIO MANDO ANTES" not in va._prompt_de_escaneo("algo")
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "_fr.scope(_fotos_previas)" in src and "_fr_anotar.anotar(actual_user_id" in src
