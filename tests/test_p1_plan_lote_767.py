"""[P1-PLAN-LOTE-767 · 2026-09-29] La etiqueta de un suplemento sin su tabla: el FRENTE del envase y, si no, internet.

El dueño: «¿el agente no puede investigar la tabla nutricional de esa proteína? En la imagen se ve toda la info para
buscar en internet». Probado contra la API real (VPS, 29-sep) antes de escribirlo: Gemini 3.8 Flash busca en Google por
`/v1beta/interactions` (por `generateContent` contestaba sin buscar); sin tope hizo 43 búsquedas, con «COMO MÁXIMO 3»
hace 3; el Atlas Gainer de Patriot Nutrition no está publicado y el Gold Standard de Optimum sí. Orden del coach: la
tabla de la foto → las cifras del frente → internet → la foto de la tabla.
"""
import json
from pathlib import Path

import pytest

import etiqueta_web as ew
import suplementos
from prompts import chat_agent as ca

_BACKEND = Path(__file__).resolve().parents[1]
_RESPUESTA_REAL = {   # la forma que devolvió la API el 29-sep (pasos de búsqueda + el texto final)
    "id": "v1_x", "status": "completed",
    "usage": {"total_input_tokens": 1342, "total_output_tokens": 92, "total_cached_tokens": 1059,
              "grounding_tool_count": [{"type": "google_search", "count": 3, "search_query_count": 3}]},
    "steps": [
        {"id": "call_1", "signature": "x"}, {"call_id": "call_1", "signature": "y"},
        {"content": [{"type": "text", "text": "```json\n{\n  \"encontrado\": true, \"porcion_texto\": \"1 scoop (31 g)\", "
                                              "\"gramos_porcion\": 31, \"kcal\": 120, \"protein_g\": 24, \"carbs_g\": 3, "
                                              "\"fats_g\": 1.5, \"porciones_por_envase\": 74, "
                                              "\"fuente\": \"optimumnutrition.com\"\n}\n```"}],
         "type": "model_output"},
    ],
}


# ---------- lo que contesta Gemini ----------

def test_lee_el_texto_final_de_la_interaccion_y_su_etiqueta():
    texto = ew.texto_de_la_respuesta(_RESPUESTA_REAL)
    h = ew.leer_etiqueta(texto)
    assert h["etiqueta"] == {"gramos_porcion": 31.0, "kcal": 120.0, "protein_g": 24.0, "carbs_g": 3.0, "fats_g": 1.5}
    assert h["porciones"] == 74 and h["porcion_texto"] == "1 scoop (31 g)" and h["fuente"] == "optimumnutrition.com"


def test_no_encontrado_o_imposible_no_es_una_etiqueta():
    assert ew.leer_etiqueta('{"encontrado": false, "kcal": null}') is None
    assert ew.leer_etiqueta("no hay json aquí") is None
    # 1250 kcal en 31 g no cabe (9 kcal/g): la misma validación que una etiqueta de foto
    assert ew.leer_etiqueta('{"encontrado": true, "gramos_porcion": 31, "kcal": 1250, "protein_g": 50, '
                            '"carbs_g": 250, "fats_g": 4}') is None


def test_la_clave_de_la_cache_es_del_producto_y_estable():
    a = ew.clave_de_producto("Patriot Nutrition", "Atlas Gainer Advanced Mass Vanilla")
    b = ew.clave_de_producto("PATRIOT  nutrition", "Atlas Gainer Advanced Mass Vanilla", sabor="vanilla")
    assert a == b == "etiqueta_web:patriot nutrition|atlas gainer advanced mass vanilla"
    assert ew.clave_de_producto("ON", "Gold Standard", sabor="Chocolate").endswith("|chocolate")


# ---------- la búsqueda, sin red ----------

class _Resp:
    def __init__(self, status, datos):
        self.status_code, self._datos = status, datos

    def json(self):
        return self._datos


@pytest.fixture
def _sin_db(monkeypatch):
    estado = {"cache": {}, "usos": [], "peticiones": [], "hoy": 0}
    monkeypatch.setattr(ew, "_leer_cache", lambda clave: estado["cache"].get(clave))
    monkeypatch.setattr(ew, "_escribir_cache", lambda clave, valor: estado["cache"].__setitem__(clave, valor))
    monkeypatch.setattr(ew, "_busquedas_de_hoy", lambda uid: estado["hoy"])
    monkeypatch.setattr(ew, "_clave", lambda: "k")
    import db
    monkeypatch.setattr(db, "log_llm_usage_event", lambda **kw: estado["usos"].append(kw), raising=False)
    import httpx

    def _post(url, **kw):
        estado["peticiones"].append((url, kw))
        return _Resp(200, _RESPUESTA_REAL)
    monkeypatch.setattr(httpx, "post", _post)
    return estado


def test_busca_una_vez_con_tope_de_busquedas_y_la_segunda_sale_de_la_cache(_sin_db):
    r = ew.buscar("u1", "Optimum Nutrition", "Gold Standard 100% Whey", "Double Rich Chocolate")
    assert r["estado"] == "encontrada" and r["cache"] is False and r["etiqueta"]["kcal"] == 120
    url, kw = _sin_db["peticiones"][0]
    assert url.endswith("/v1beta/interactions")
    assert kw["json"]["tools"] == [{"type": "google_search"}]
    assert "COMO MÁXIMO 3 búsquedas" in kw["json"]["input"]
    assert _sin_db["usos"][0]["node"] == "etiqueta_web" and _sin_db["usos"][0]["metadata"]["busquedas"] == 3
    r2 = ew.buscar("u2", "Optimum Nutrition", "Gold Standard 100% Whey", "Double Rich Chocolate")
    assert r2["estado"] == "encontrada" and r2["cache"] is True
    assert len(_sin_db["peticiones"]) == 1   # el segundo usuario no paga otra búsqueda


def test_sin_cupo_no_busca_y_apagada_tampoco(_sin_db, monkeypatch):
    _sin_db["hoy"] = 99
    assert ew.buscar("u1", "X", "Y")["estado"] == "sin_cupo"
    monkeypatch.setenv("MEALFIT_ETIQUETA_WEB", "false")
    assert ew.buscar("u1", "X", "Y")["estado"] == "apagada"
    assert _sin_db["peticiones"] == []


def test_un_no_encontrado_tambien_se_guarda(_sin_db, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp(200, {"usage": {}, "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": '{"encontrado": false}'}]}]}))
    assert ew.buscar("u1", "Patriot Nutrition", "Atlas Gainer")["estado"] == "no_encontrada"
    assert ew.buscar("u1", "Patriot Nutrition", "Atlas Gainer") == {"estado": "no_encontrada", "cache": True}


# ---------- el frente del envase ----------

def test_el_frente_completa_carbohidratos_grasa_y_gramos():
    e = suplementos.completar_desde_el_frente({"kcal": 625, "protein_g": 50})
    assert e["fats_g"] == pytest.approx(4.2) and e["carbs_g"] == pytest.approx(96.8)
    assert e["gramos_porcion"] == 159   # lo que pesan sus macros + 5 %
    assert suplementos.etiqueta_valida(e) is not None   # sin gramos, >600 kcal se rechazaba
    # lo que ya viene completo y cuadra, tal cual; lo imposible sigue siendo imposible
    completa = {"gramos_porcion": 31, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5}
    assert suplementos.completar_desde_el_frente(completa) == completa
    assert suplementos.etiqueta_valida(suplementos.completar_desde_el_frente(
        {"gramos_porcion": 125, "kcal": 1250, "protein_g": 60})) is None


def test_guardar_con_fuente_frente_completa_antes_de_validar(monkeypatch):
    escrito = {}
    monkeypatch.setattr(suplementos, "_upsert", lambda *a: escrito.setdefault("fila", a))
    monkeypatch.setattr(suplementos, "buscar", lambda *a, **k: None)
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda *a, **k: "activa")
    r = suplementos.guardar("u1", "Atlas Gainer", "Patriot Nutrition", 56, "porcion",
                            {"kcal": 625, "protein_g": 50}, "frente")
    assert r["ok"] and r["fuente"] == "frente" and r["etiqueta"]["kcal"] == 625
    assert escrito["fila"][-1] == "frente"


# ---------- la tool, el prompt y el escáner ----------

def test_la_tool_esta_en_el_chat_y_documentada():
    import tools
    assert "buscar_etiqueta_en_internet" in [t.name for t in tools.agent_tools]
    doc = (_BACKEND / "docs" / "agent_tools_user_id_table.md").read_text(encoding="utf-8")
    assert "| 16 | `buscar_etiqueta_en_internet` |" in doc


def test_la_tool_le_dice_al_coach_que_guardar_y_de_donde_salio(monkeypatch):
    import tools
    fn = tools.buscar_etiqueta_en_internet.func
    monkeypatch.setattr(ew, "buscar", lambda *a, **k: {"estado": "encontrada", "etiqueta": {
        "gramos_porcion": 31.0, "kcal": 120.0, "protein_g": 24.0, "carbs_g": 3.0, "fats_g": 1.5},
        "porciones": 74.0, "porcion_texto": "1 scoop (31 g)", "fuente": "optimumnutrition.com"})
    txt = fn(user_id="u1", marca="Optimum Nutrition", producto="Gold Standard 100% Whey")
    assert "120 kcal" in txt and "74 porciones" in txt and "fuente='web'" in txt and "salieron de internet" in txt
    monkeypatch.setattr(ew, "buscar", lambda *a, **k: {"estado": "no_encontrada"})
    assert "guárdalo SIN etiqueta" in fn(user_id="u1", marca="Patriot Nutrition", producto="Atlas Gainer")
    assert "no digas que buscaste" in fn(user_id="guest-1", marca="X", producto="Y")


def test_el_coach_sigue_el_orden_tabla_frente_internet_foto():
    ctx = ca.build_vision_context({"kind": "multi", "has_text": False, "items": [
        {"kind": "etiqueta", "description": "Patriot Nutrition; Atlas Gainer; no se lee la tabla nutricional."}]})
    i_frente = ctx.index("DEL FRENTE del envase")
    i_web = ctx.index("`buscar_etiqueta_en_internet`")
    i_sin = ctx.index("guárdalo SIN etiqueta")
    assert i_frente < i_web < i_sin
    assert "divide SOLO la cifra que el frente da para varias porciones" in ctx
    assert "búscala antes con buscar_etiqueta_en_internet" in ca._CHAT_RESOLVE_RULES
    assert "buscar_etiqueta_en_internet" in suplementos.BLOQUE_CONOCIMIENTO


def test_el_escaner_copia_las_cifras_del_frente_sin_tomarlas_por_la_tabla():
    src = (_BACKEND / "vision_agent.py").read_text(encoding="utf-8")
    assert "si el FRENTE del envase imprime cifras" in src and "precedidas de 'del frente:'" in src
    assert "los numeros siguen en 0" in src   # el registro del escáner no toma un frente por una tabla


def test_la_costura_de_gemini_esta_marcada():
    src = (_BACKEND / "etiqueta_web.py").read_text(encoding="utf-8")
    linea = [l for l in src.splitlines() if l.startswith("_MODELO_DEFAULT")][0]
    assert "[P1-PLAN-LOTE-767-ETIQUETA-WEB]" in linea
    assert "google.genai" not in src and "langchain_google_genai" not in src   # REST, sin SDK


def test_la_bateria_cubre_los_tres_caminos():
    casos = {c["id"]: c for c in json.loads((_BACKEND / "scripts" / "coach_battery" / "battery.json")
                                            .read_text(encoding="utf-8"))["casos"]}
    assert casos["P6"]["expect"]["tools"] == ["buscar_etiqueta_en_internet", "guardar_suplemento"]
    assert "buscar_etiqueta_en_internet" in casos["P9"]["expect"]["no_tools"]
    assert casos["P10"]["expect"]["tools"] == ["buscar_etiqueta_en_internet", "guardar_suplemento"]
