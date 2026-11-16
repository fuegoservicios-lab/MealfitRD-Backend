# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1047] Conversación real del 16-nov: «No» guardó «Yogur» en los gustos, «el último yogur» agotó
las rondas día por día, y el atún no se pudo sumar a un yogur «Protein» tomado por suplemento."""
from datetime import date
from pathlib import Path

import diario_buscar as db_
import gustos_con_evidencia as g

_BACKEND = Path(__file__).resolve().parent.parent


# ── gustos ──────────────────────────────────────────────────────────────────────────────────────────────
def test_un_no_a_otra_pregunta_no_cambia_los_gustos():
    ok, msg = g.revisar("dislikes", "Yogur", "No", ["Cilantro"])
    assert not ok and "NO CAMBIÉ" in msg


def test_nombrar_el_alimento_no_es_rechazarlo():
    assert not g.revisar("dislikes", "Yogur", "Quería que me recuerdes el último yogurt que registré", [])[0]


def test_rechazo_explicito_suma_sin_perder_los_previos():
    ok, val = g.revisar("dislikes", "Yogur", "no me gusta el yogur", ["Cilantro", "Hígado"])
    assert ok and val == "Yogur, Cilantro, Hígado"
    assert g.revisar("dislikes", "Brócoli", "tampoco el brócoli", ["Yogur"]) == (True, "Brócoli, Yogur")


def test_quitar_reemplaza():
    assert g.revisar("dislikes", "Cilantro", "ya me gusta el hígado, quítalo", ["Cilantro", "Hígado"]) == (True, "Cilantro")


def test_otros_campos_pasan_intactos():
    assert g.revisar("weight", "70", "No", None) == (True, "70")


def test_agent_frena_antes_de_ejecutar():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "from gustos_con_evidencia import revisar as _revisar_gustos" in src
    assert '[LOTE-1047 GUSTOS] update_form_field(dislikes) sin evidencia' in src


# ── búsqueda en el diario ───────────────────────────────────────────────────────────────────────────────
_FILAS = [
    {"meal_name": "Yogur Ratio Protein sabor strawberry cheesecake", "meal_type": "merienda", "calories": 170,
     "protein": 25, "carbs": 7, "healthy_fats": 3, "consumed_at": "2026-11-10T21:00:00+00:00"},
    {"meal_name": "Desayuno", "ingredients": ["avena", "yogurt griego"], "meal_type": "desayuno", "calories": 400,
     "protein": 20, "carbs": 50, "healthy_fats": 10, "consumed_at": "2026-11-03T12:00:00+00:00"},
    {"meal_name": "Mass Gainer", "meal_type": "suplemento", "calories": 600, "protein": 50,
     "consumed_at": "2026-11-15T15:00:00+00:00"},
]


def test_busca_por_nombre_e_ingredientes_mas_reciente_primero():
    out = db_.buscar("u", "yogurt", 240, hoy=date(2026, 11, 16), filas=_FILAS)
    assert out.index("2026-11-10") < out.index("2026-11-03")
    assert "Ratio Protein" in out and "Mass Gainer" not in out


def test_sin_coincidencias_dice_la_verdad_y_no_recorre_dias():
    out = db_.buscar("u", "atún", 240, hoy=date(2026, 11, 16), filas=_FILAS)
    assert "ningún registro" in out and "día por día" in out


def test_la_tool_y_el_prompt_lo_ofrecen():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert 'def consultar_dia_del_plan(user_id: str, fecha: str = "", fuente: str = "plan", buscar: str = "")' in tools
    assert "from diario_buscar import buscar as _buscar_diario" in tools
    prompt = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert "buscar='yogur'" in prompt


# ── suplemento por palabra clave ────────────────────────────────────────────────────────────────────────
def test_fila_en_franja_de_comida_no_es_suplemento_por_la_palabra_protein():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "_orig_type not in _FRANJAS_DE_COMIDA" in tools
    import tools as t
    assert {"merienda", "desayuno", "almuerzo", "cena", "snack"} <= t._FRANJAS_DE_COMIDA
