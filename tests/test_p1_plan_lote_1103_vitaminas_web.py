"""[P1-PLAN-LOTE-1103 · 2026-11-18] La tabla de vitaminas de un suplemento, buscada en internet, y el pote guardado.

«Es mío, me metí dos cápsulas» con la foto del Multi+ Phyto: el coach pidió una foto de la tabla y no lo guardó.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import etiqueta_web_vitaminas as v  # noqa: E402

FILA_A = {"name": "Vitamin A", "amount": 1500, "unit": "mcg RAE", "form": "Palmitate",
          "label_text": "Vitamin A (as Mixed Carotenoids and Palmitate) 1500 mcg RAE"}
FILA_C = {"name": "Vitamin C", "amount": 400, "unit": "mg", "form": "Ascorbic Acid",
          "label_text": "Vitamin C (as Ascorbic Acid and Acerola) 400 mg"}
RESPUESTA = ('```json\n{"encontrado": true, "serving_text": "4 Capsules", "units_per_serving": 4, '
             '"serving_unit": "capsule", "porciones_por_envase": 30, "nutrients": [%s, %s, '
             '{"name": "Zinc", "amount": 15, "unit": "mg", "label_text": ""}], "fuente": "https://x.com/f.pdf"}\n```')


def _respuesta():
    import json
    return RESPUESTA % (json.dumps(FILA_A), json.dumps(FILA_C))


def test_lee_la_tabla_y_descarta_cifras_sin_su_fila():
    h = v.leer_tabla(_respuesta())
    f = h["supplement_facts"]
    assert (f["serving_text"], f["units_per_serving"], f["serving_unit"]) == ("4 Capsules", 4.0, "capsule")
    assert f["table_readable"] is True and f["amount_basis"] == "per_serving"
    zinc = [n for n in f["nutrients"] if n["name"] == "Zinc"][0]
    assert zinc["amount"] is None, "sin la fila que la respalda, la cantidad no vale (igual que en una foto)"
    assert h["porciones"] == 30
    assert v.leer_tabla('{"encontrado": false}') is None
    assert v.leer_tabla('{"encontrado": true, "nutrients": []}') is None


def test_dos_capsulas_de_una_porcion_de_cuatro_es_media_porcion():
    from supplement_labels import analyze
    a = analyze(v.leer_tabla(_respuesta())["supplement_facts"], 2, None, "cápsulas")
    assert a["servings_taken"] == 0.5
    assert [d["amount"] for d in a["dose"][:2]] == [750, 200]


def test_la_etiqueta_sin_calorias_la_acepta_la_alacena():
    import suplementos
    e = suplementos.etiqueta_valida(v.etiqueta_de(v.leer_tabla(_respuesta())))
    assert e["kcal"] == 0 and e["unidades_por_toma"] == 4 and e["supplement_facts"]["nutrients"]


def test_urls_resueltas_sin_google_y_la_ficha_en_pdf_primero():
    texto = ("https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA\n"
             "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB\n"
             "https://www.google.com/search?q=x\nhttps://tienda.com/p/multi\nhttps://tienda.com/p/multi")
    destino = {"AAA": "https://marca.com/producto", "BBB": "https://cdn.marca.com/ficha.pdf"}

    def resolver(u):
        return destino.get(u.rsplit("/", 1)[-1], u) if "vertexaisearch" in u else u

    assert v.urls_de_la_respuesta(texto, resolver) == [
        "https://cdn.marca.com/ficha.pdf", "https://marca.com/producto", "https://tienda.com/p/multi"]


def test_lanzar_usa_la_cache_y_si_no_la_busca_en_segundo_plano(monkeypatch):
    import bg_executor
    lanzadas = []
    monkeypatch.setattr(v._ew, "activa", lambda: True)
    monkeypatch.setattr(v, "en_cache", lambda m, p: None)
    monkeypatch.setattr(v, "hay_cupo", lambda u: True)
    monkeypatch.setattr(bg_executor, "submit_bg_task", lambda fn, *a, **k: lanzadas.append((fn, a)))
    assert v.lanzar("u1", "Multi+ Phyto", "Designs for Sport", "Multi+ Phyto") == {"estado": "buscando"}
    assert lanzadas and lanzadas[0][1] == ("u1", "Multi+ Phyto", "Designs for Sport", "Multi+ Phyto")
    monkeypatch.setattr(v, "hay_cupo", lambda u: False)
    assert v.lanzar("u1", "x", "m", "p") == {"estado": "sin_cupo"}
    monkeypatch.setattr(v, "en_cache", lambda m, p: {"estado": "encontrada", "supplement_facts": {}})
    assert v.lanzar("u1", "x", "m", "p")["estado"] == "encontrada"


def test_mientras_busca_el_coach_guarda_el_pote_sin_pedir_foto():
    t = v.texto_para_el_coach({"estado": "buscando"}, "Multi+ Phyto")
    assert "«Multi+ Phyto»" in t and "SIN etiqueta" in t and "sin pedirle foto" in t
    assert "foto" in v.texto_para_el_coach({"estado": "no_encontrada"}, "x")


def test_completa_el_pote_solo_si_no_tiene_tabla(monkeypatch):
    import suplementos
    import db_core
    escrituras = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p: escrituras.append(p) or True)
    h = v.leer_tabla(_respuesta())
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: {"id": "p1", "serving_label": None})
    assert v.completar_pote("u1", "Multi+ Phyto", h) is True
    assert escrituras and escrituras[0][-2:] == ("p1", "u1")
    legible = {"table_readable": True, "nutrients": [{"name": "Vitamina C", "amount": 90}]}
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: {"id": "p1", "quantity": 30, "serving_label": {"supplement_facts": legible}})
    assert v.completar_pote("u1", "Multi+ Phyto", h) is False, "una tabla de foto LEGIBLE nunca se pisa"
    # [P1-PLAN-LOTE-1109] la del Multi+ Phyto del dueño (19-nov): foto ilegible -> la de internet sí la completa
    ilegible = {"table_readable": False, "nutrients": []}
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: {"id": "p1", "serving_label": {"supplement_facts": ilegible}})
    assert v.completar_pote("u1", "Multi+ Phyto", h) is True
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: None)
    assert v.completar_pote("u1", "Multi+ Phyto", h) is False


def test_cableado_herramienta_recorte_y_prompt():
    tools_src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "vitaminas: bool = False) -> str:" in tools_src
    assert "_ewv.lanzar(user_id, _nombre, marca, producto)" in tools_src
    prompt = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert "la búsqueda de proteína no aporta estos datos" not in prompt
    assert "con vitaminas=true" in prompt and "[P1-PLAN-LOTE-1103]" in prompt

    from unittest.mock import MagicMock
    from agent import _select_tools_for_turn

    class T:
        def __init__(self, n):
            self.name = n

    todas = [T(n) for n in ("log_consumed_meal", "correct_consumed_meal", "check_hydration_today", "log_water_glass",
                            "proponer_comida", "consultar_dia_del_plan", "search_deep_memory", "update_form_field",
                            "analizar_suplemento", "guardar_suplemento", "buscar_etiqueta_en_internet",
                            "check_shopping_list")]
    estado = {"messages": [MagicMock(type="user", content="Es mío, me metí dos cápsulas")],
              "turn_plate_photos": [], "turn_photo_to_log": None}
    nombres = {t.name for t in _select_tools_for_turn(todas, estado)}
    assert {"analizar_suplemento", "guardar_suplemento", "buscar_etiqueta_en_internet"} <= nombres
