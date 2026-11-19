"""[P1-PLAN-LOTE-1109 · 1110 · 2026-11-19] El turno de la cena del dueño (19-nov 00:33): la batida se sumó DOS veces
(1200 kcal / 101 g en vez de ~960) y el multivitamínico quedó con una tabla de foto ilegible que bloqueaba la de
internet. 1110: lo que ya está en la fila tocada en este turno no se suma otra vez. 1109: las vitaminas se anotan en
el diario (0 kcal) y cada toma cuenta sus porciones reales."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import aviso_suplemento as s  # noqa: E402
import fusion_sin_duplicar as f  # noqa: E402

FILA = {
    "id": "fila-cena", "meal_type": "cena", "calories": 960, "protein": 81, "carbs": 90, "healthy_fats": 30,
    "meal_name": "Plátano maduro con chuleta, huevo y queso + Batida de guineo con leche evaporada",
    "ingredients": ["Plátano maduro frito (1 mediano)", "Chuleta de cerdo (1 pieza)", "Huevo frito (1 unidad)",
                    "Queso frito (2 lascas)", "Guineo maduro (1 mediano)", "Leche evaporada (1/2 taza)"],
}
BATIDA = {"meal_name": "Batida de guineo con leche evaporada", "calories": 240, "protein": 20,
          "ingredients": ["Guineo (1 mediano)", "Leche evaporada (½ taza)"]}


def _turno(*despues, antes=()):
    return [*antes, HumanMessage(content="esta es mi cena y la batida"), *despues]


def _tool(fid):
    return ToolMessage(content=f"Registrado. [ID_REGISTRO_DIARIO: {fid} — uso interno tuyo]", tool_call_id="t1")


def test_la_batida_que_ya_entro_en_la_cena_no_se_suma_otra_vez():
    msgs = _turno(AIMessage(content=""), _tool("fila-cena"), AIMessage(content=""), _tool("fila-cena"))
    assert f.ids_tocados_en_el_turno(msgs) == {"fila-cena"}
    assert f.ya_esta_en_la_fila(FILA, BATIDA, msgs) is True


def test_fuera_del_turno_o_con_algo_nuevo_se_suma_como_siempre():
    # la fila se tocó en un turno ANTERIOR: «y también una batida» horas después sigue sumando
    viejo = _turno(antes=(HumanMessage(content="mi cena"), _tool("fila-cena")))
    assert f.ya_esta_en_la_fila(FILA, BATIDA, viejo) is False
    msgs = _turno(_tool("fila-cena"))
    assert f.ya_esta_en_la_fila(FILA, {"meal_name": "Aguacate", "ingredients": ["Aguacate (1/4)"]}, msgs) is False
    assert f.ya_esta_en_la_fila(FILA, {"meal_name": "Batida de guineo con avena"}, msgs) is False, "avena es nueva"
    assert f.ya_esta_en_la_fila(FILA, {"meal_name": ""}, msgs) is False
    assert f.ya_esta_en_la_fila(None, BATIDA, msgs) is False


def test_sin_cambios_es_un_no_op_de_correct_consumed_meal():
    a = f.sin_cambios("u1", FILA, "cena")
    assert a["meal_id"] == "fila-cena" and a["calories"] == 960 and a["protein"] == 81
    assert a["meal_name"] == FILA["meal_name"] and a["meal_type"] == "cena"
    assert "ingredients" not in a and "nutrition" not in a, "con ellos BB-044 reescribe la fila"


def test_cableado_de_la_fusion_y_del_prompt():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("_fsd.ya_esta_en_la_fila(orig, tool_args, messages)")
    assert i < src.index('    orig_id = str(orig.get("id"))\n    orig_name = str(orig.get("meal_name") or "").strip()')
    p = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert "[P1-PLAN-LOTE-1109]" in p and "meal_type='suplemento'" in p
    assert "sin crear una comida de 0 kcal" not in p


AHORA = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)


def test_cada_toma_cuenta_sus_porciones_reales():
    reg = [[(AHORA - timedelta(days=d)).isoformat(), 0.5] for d in range(14)]       # 2 de 4 cápsulas al día
    reg.append([(AHORA - timedelta(days=20)).isoformat(), 1.0])                       # fuera de la ventana
    n, gastado = s.tomas_en_ventana(reg, AHORA)
    assert (n, gastado) == (14, 7.0)
    assert s.dias_restantes(5, n, gastado) == 10.0, "a media porción al día, 5 porciones son 10 días"
    assert s.dias_restantes(5, n) == 5.0, "sin registro, 1 por toma (como antes)"
    assert s.tomas_en_ventana([["basura"], None, ["2026-11-18T00:00:00", "x"]], AHORA) == (0, 0.0)


def test_la_toma_se_anota_al_descontar_del_pote():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i = src.index('_sup.descontar(user_id, _pote.get("id"), porciones or 1)')
    assert '__import__("aviso_suplemento").anotar_toma(_pote.get("id"), porciones or 1)' in src[i:i + 300]


def test_anotar_toma_guarda_las_ultimas_60(monkeypatch):
    import db_core
    guardado = {}
    previo = [["2026-11-01T00:00:00+00:00", 1.0]] * 60
    monkeypatch.setattr(db_core, "execute_sql_query", lambda q, p, **k: {"value": previo})
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p: guardado.update(k=p[0], v=p[1]))
    s.anotar_toma("pote-1", 0.5, AHORA)
    import json
    v = json.loads(guardado["v"])
    assert guardado["k"] == "suplemento_tomas:pote-1" and len(v) == 60 and v[-1] == [AHORA.isoformat(), 0.5]
