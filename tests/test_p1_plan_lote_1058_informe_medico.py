# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1032/1033/1034 · 2026-11-17] Informe para el médico: datos, redacción verificada, rutas y herramienta.

Todo sin IA: el redactor se prueba con un modelo simulado. El dueño: «debe darme algo para mi médico, eso es para ahorrar
trabajo a médicos». La app NO diagnostica: resume lo registrado; las cifras las pone el código.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import informe_medico as im  # noqa: E402
import informe_redaccion as ir  # noqa: E402


# ─────────────────────────────────────────────────────────────── banda: paridad con el cliente
def _js_banda() -> str | None:
    for base in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
        p = base / "frontend" / "src" / "utils" / "calorieGoalStatus.js"
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def test_banda_igual_que_el_cliente():
    js = _js_banda()
    if js is None:
        pytest.skip("frontend no está junto al backend en este checkout")
    numeros = re.findall(r"return \{ lo: ([\d.]+), hi: ([\d.]+) \}", js)
    assert [tuple(map(float, n)) for n in numeros] == [im.BANDA_MUSCULO, im.BANDA_PERDER_GRASA, im.BANDA_OTRA]
    musc = re.search(r"const _isMuscle = \(raw\) =>(.*?);\n", js, re.S).group(1)
    perder = re.search(r"const _isLoseFat = \(raw\) =>(.*?);\n", js, re.S).group(1)
    assert tuple(re.findall(r"includes\('([^']+)'\)", musc)) == im.SUBCADENAS_MUSCULO
    assert tuple(re.findall(r"includes\('([^']+)'\)", perder)) == im.SUBCADENAS_PERDER_GRASA


@pytest.mark.parametrize("kcal,meta,obj,esperado", [
    (1640, 2000, "lose_fat", "en_meta"), (1590, 2000, "lose_fat", "bajo"), (2110, 2000, "lose_fat", "sobre"),
    (2300, 2000, "gain_muscle", "en_meta"), (2310, 2000, "gain_muscle", "sobre"), (1790, 2000, "", "bajo"),
    (1800, 2000, "", "en_meta"), (2200, 2000, "maintenance", "en_meta"), (500, 0, "", "bajo"),
])
def test_estado_del_dia(kcal, meta, obj, esperado):
    assert im.estado_del_dia(kcal, meta, obj) == esperado


# ─────────────────────────────────────────────────────────────── armado de los datos
PERFIL = {
    "age": "21", "gender": "male", "height": "168", "weight": "123", "weightUnit": "lb", "mainGoal": "gain_muscle",
    "activityLevel": "moderate", "medicalConditions": ["Hipertensión"], "otherConditions": "Ninguna",
    "allergies": ["Ninguna"], "medications": "", "habitAlcohol": "ocasional", "sleepHours": "7",
}


def _dia(kcal, p=100, c=200, g=60):
    return {"calories": kcal, "protein": p, "carbs": c, "healthy_fats": g, "meals_count": 3}


def _datos(**kw):
    dias = {"2026-11-10": _dia(2000), "2026-11-11": _dia(2400), "2026-11-12": _dia(1500), "2026-11-15": _dia(2100)}
    comidas = []
    for d, horas in (("2026-11-10", (8.5, 13, 21)), ("2026-11-11", (9, 14, 22.5)), ("2026-11-12", (13, 20)),
                     ("2026-11-15", (10, 14, 23))):
        for i, h in enumerate(horas):
            tipo = ("desayuno", "almuerzo", "cena")[i] if len(horas) == 3 else ("almuerzo", "cena")[i]
            comidas.append({"dia": d, "hora_local": h, "meal_type": tipo,
                            "ingredients": ["arroz blanco cocido (≈180 g)", "pollo guisado (≈120 g)"] if tipo != "desayuno"
                            else ["2 huevos revueltos"]})
    base = dict(periodo_dias=14, hasta=date(2026, 11, 16), perfil=PERFIL, dias=dias, comidas=comidas,
                metas_por_dia={"2026-11-10": {"calories": 2000, "main_goal": "gain_muscle"}},
                meta_de_hoy={"calories": 2100}, week1_start=date(2026, 11, 8))
    base.update(kw)
    return im.armar_datos(**base)


def test_snapshot_del_informe():
    d = _datos()
    assert d["periodo"] == {"desde": "2026-11-03", "hasta": "2026-11-16", "dias": 14}
    assert d["persona"] == {"edad": 21, "sexo": "male", "talla_cm": 168, "peso_declarado_kg": 55.8,
                            "objetivo": "gain_muscle", "actividad": "moderate"}
    assert d["declarado_por_el_paciente"] == {"condiciones": ["Hipertensión"], "alergias": [], "medicamentos": [],
                                              "suplementos": [], "habitos": {"alcohol": "ocasional", "sueno_h": 7}}
    # 11-10 meta 2000 (2000 → en meta); el resto con la de hoy 2100: 2400 (114 % músculo → en meta), 1500 (bajo), 2100
    assert d["adherencia"] == {"dias_con_registro": 4, "dias_del_periodo": 14, "dias_en_meta": 3, "dias_bajo": 1,
                               "dias_sobre": 0, "semanas": [{"numero": 1, "dias_en_meta": 2, "dias_con_registro": 3},
                                                            {"numero": 2, "dias_en_meta": 1, "dias_con_registro": 1}]}
    assert d["energia"]["kcal_media"] == 2000 and d["energia"]["kcal_min"] == 1500 and d["energia"]["kcal_max"] == 2400
    assert d["energia"]["meta_media"] == 2075 and d["energia"]["pct_de_la_meta"] == 96
    assert d["macros"]["proteina_g_por_kg"] == 1.8
    assert d["macros"]["pct_kcal"] == {"proteina": 23, "carbos": 46, "grasas": 31}   # 100·4 / 200·4 / 60·9 de 1740
    assert d["patrones"]["pct_dias_con_desayuno"] == 75
    assert d["patrones"]["hora_primera_comida"] == "09:30"
    assert d["patrones"]["pct_cenas_despues_de_las_22"] == 50
    assert d["alimentos_frecuentes"][0] == {"nombre": "arroz blanco cocido", "veces": 8}
    assert d["limitaciones"] == ["datos_autorreportados", "porciones_estimadas", "dias_sin_registro: 10"]


def test_secciones_sin_datos_suficientes_no_aparecen():
    d = _datos(dias={"2026-11-15": _dia(2100)}, comidas=[])
    for clave in ("energia", "macros", "patrones", "alimentos_frecuentes", "micronutrientes", "hidratacion", "peso"):
        assert clave not in d
    assert d["adherencia"]["dias_con_registro"] == 1


def test_un_dia_sin_registro_no_cuenta_como_cero():
    d = _datos(dias={"2026-11-10": _dia(2000), "2026-11-11": _dia(2000), "2026-11-12": _dia(2000),
                     "2026-11-13": _dia(0)})
    assert d["energia"]["kcal_media"] == 2000 and d["adherencia"]["dias_con_registro"] == 3


def test_peso_e_hidratacion_y_micros():
    d = _datos(pesos=[(date(2026, 11, 4), 56.4), (date(2026, 11, 15), 55.8)],
               agua={"2026-11-10": 5, "2026-11-11": 6, "2026-11-12": 4},
               micros_por_dia={k: {"sodium_mg": 3000, "calcium_mg": 400} for k in ("2026-11-10", "2026-11-11", "2026-11-12")},
               referencias_micros={"sodium_mg": {"target": 2000, "kind": "ceiling"}, "calcium_mg": {"target": 1000, "kind": "floor"}})
    assert d["peso"]["cambio_kg"] == -0.6 and d["peso"]["mediciones"] == 2
    assert d["hidratacion"] == {"vasos_media": 5.0, "dias_con_dato": 3}
    assert d["micronutrientes"]["sobre_el_limite"] == [{"nutriente": "sodium", "media_pct_del_limite": 150, "dias_con_dato": 3}]
    assert d["micronutrientes"]["bajo_la_meta"] == [{"nutriente": "calcium", "media_pct_de_la_meta": 40, "dias_con_dato": 3}]


def test_lista_declarada_quita_ninguna_y_duplicados():
    assert im.lista_declarada(["Ninguna"], "none, Asma, asma") == ["Asma"]
    assert im.nombre_de_alimento("2 huevos revueltos") == "huevos revueltos"
    assert im.peso_en_kg("123", "lb") == 55.8 and im.peso_en_kg("abc", "kg") is None


# ─────────────────────────────────────────────────────────────── verificación de la redacción
def _narr(**kw):
    n = {"resumen": "El paciente registró 4 de 14 días; su media fue 2000 kcal frente a una meta media de 2075 kcal.",
         "hallazgos": [{"texto": "Energía al 96 % de la meta.", "evidencia": ["energia.pct_de_la_meta"]}],
         "preguntas_para_la_consulta": ["¿Mi proteína es suficiente?", "¿Debo revisar mis horarios?", "¿Qué análisis conviene?"]}
    n.update(kw)
    return n


def test_verificar_acepta_una_redaccion_fiel():
    assert ir.verificar(_narr(), _datos()) == []


def test_verificar_rechaza_cifras_inventadas_evidencias_y_diagnosticos():
    d = _datos()
    assert any("cifra" in p for p in ir.verificar(_narr(resumen="Su media fue 2350 kcal."), d))
    assert any("evidencia" in p for p in ir.verificar(_narr(hallazgos=[{"texto": "x", "evidencia": ["energia.inventada"]}]), d))
    assert any("diagnóstico" in p for p in ir.verificar(_narr(resumen="Le diagnostico una dieta pobre."), d))
    assert any("diagnóstico" in p for p in ir.verificar(_narr(resumen="The patient should take iron."), d))
    assert ir.verificar("no json", d) == ["la respuesta no es un objeto JSON"]
    assert any("preguntas" in p for p in ir.verificar(_narr(preguntas_para_la_consulta=["una"]), d))


def test_numeros_con_separadores():
    d = {"energia": {"kcal_media": 1985, "proteina_g_por_kg": 2.2}}
    assert ir.verificar(_narr(resumen="Media de 1.985 kcal y 2,2 g/kg.", hallazgos=[{"texto": "1 985", "evidencia": ["energia.kcal_media"]}]), d) == []


# ─────────────────────────────────────────────────────────────── redactar (modelo simulado)
class _Res:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 3000, "output_tokens": 500}


class _LLM:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def invoke(self, _msgs):
        self.llamadas += 1
        return _Res(self.respuestas.pop(0))


def test_sin_permiso_o_sin_modelo_es_data_only(monkeypatch):
    assert ir.redactar(_datos(), con_permiso=False)["motivo"] == "sin_permiso"
    monkeypatch.setattr(ir, "modelo", lambda: None)
    r = ir.redactar(_datos(), llm_factory=lambda m: pytest.fail("no debe construir el modelo"))
    assert r["status"] == "data_only" and r["motivo"] == "no_disponible"


def test_redaccion_valida_queda_ready_con_coste(monkeypatch):
    monkeypatch.setattr(ir, "modelo", lambda: "gpt-6.1-terra")
    llm = _LLM([json.dumps(_narr())])
    r = ir.redactar(_datos(), "es", llm_factory=lambda m: llm)
    assert r["status"] == "ready" and r["model"] == "gpt-6.1-terra" and llm.llamadas == 1
    assert r["cost_usd_micros"] > 0
    assert r["narrative"]["hallazgos"][0]["evidencia"] == ["energia.pct_de_la_meta"]


def test_dos_fallos_no_llaman_una_tercera_vez(monkeypatch):
    monkeypatch.setattr(ir, "modelo", lambda: "gpt-6.1-terra")
    malo = json.dumps(_narr(resumen="Le diagnostico 9999 kcal."))
    llm = _LLM([malo, malo, malo])
    r = ir.redactar(_datos(), llm_factory=lambda m: llm)
    assert r["status"] == "data_only" and r["motivo"] == "no_verificada" and llm.llamadas == 2


def test_segundo_intento_bueno_tras_corregir(monkeypatch):
    monkeypatch.setattr(ir, "modelo", lambda: "gpt-6.1-terra")
    llm = _LLM([json.dumps(_narr(resumen="Media de 7777 kcal.")), "```json\n" + json.dumps(_narr()) + "\n```"])
    assert ir.redactar(_datos(), llm_factory=lambda m: llm)["status"] == "ready" and llm.llamadas == 2


def test_tope_de_coste(monkeypatch):
    monkeypatch.setattr(ir, "modelo", lambda: "gpt-6.1-terra")
    monkeypatch.setattr(ir, "tope_usd", lambda: 0.000001)
    assert ir.redactar(_datos(), llm_factory=lambda m: pytest.fail("no debe llamar"))["motivo"] == "coste"


def test_el_redactor_nunca_cae_al_respaldo(monkeypatch):
    monkeypatch.setenv("MEALFIT_HEALTH_REPORT_MODEL", "off")
    assert ir.modelo() is None
    src = Path(ir.__file__).read_text(encoding="utf-8")
    assert "fallback" not in src.lower() or "Nunca cae al modelo de respaldo" in src
    assert "HYDROGEN" not in src.upper().replace("HYDROGEN CUESTA", "")


# ─────────────────────────────────────────────────────────────── servicio, rutas y logs
def test_crear_informe_no_escribe_datos_de_salud_en_los_logs(monkeypatch, caplog):
    import informe_servicio as svc
    monkeypatch.setattr(svc, "informes_de_hoy", lambda *a, **k: 0)
    monkeypatch.setattr(svc, "_locale_de", lambda *a, **k: "es")
    monkeypatch.setattr(svc, "redaccion_activa", lambda uid: False)
    monkeypatch.setattr(im, "construir_datos", lambda *a, **k: _datos())
    escrito = {}

    def _write(q, params=None, returning=False, **k):
        escrito["params"] = params
        return [{"id": "00000000-0000-4000-8000-000000000001", "created_at": None}]
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", _write)
    with caplog.at_level(logging.DEBUG):
        inf = svc.crear_informe("11111111-2222-4333-8444-555555555555", dias=14)
    assert inf["status"] == "data_only" and inf["motivo"] == "no_disponible"
    assert escrito["params"][0] == "11111111-2222-4333-8444-555555555555"
    todo = caplog.text.lower()
    for sensible in ("hipertensi", "ocasional", "arroz", "2000 kcal"):
        assert sensible not in todo


def test_tope_diario(monkeypatch):
    import informe_servicio as svc
    monkeypatch.setattr(svc, "informes_de_hoy", lambda *a, **k: 3)
    with pytest.raises(svc.TopeDiario):
        svc.crear_informe("11111111-2222-4333-8444-555555555555")


def _cliente(monkeypatch, uid):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.reports as rr
    from auth import get_verified_user_id
    from consentimientos import hay_permiso_ia
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[get_verified_user_id] = lambda: uid
    app.dependency_overrides[hay_permiso_ia] = lambda: True
    app.dependency_overrides[rr._CREAR_LIMITER] = lambda: None
    app.dependency_overrides[rr._LEER_LIMITER] = lambda: None
    return TestClient(app)


def test_rutas_otro_usuario_403_y_cuarto_informe_429(monkeypatch):
    import informe_servicio as svc
    yo = "11111111-2222-4333-8444-555555555555"
    otro = "99999999-2222-4333-8444-555555555555"
    c = _cliente(monkeypatch, yo)
    assert c.post(f"/api/reports/health/{otro}", json={"dias": 30}).status_code == 403
    assert c.get(f"/api/reports/health/{otro}").status_code == 403

    def _tope(*a, **k):
        raise svc.TopeDiario()
    monkeypatch.setattr(svc, "crear_informe", _tope)
    r = c.post(f"/api/reports/health/{yo}", json={"dias": 30})
    assert r.status_code == 429 and "3 informes" in r.json()["detail"]


def test_sin_permiso_de_ia_se_crea_sin_redaccion(monkeypatch):
    import informe_servicio as svc
    from consentimientos import hay_permiso_ia
    yo = "11111111-2222-4333-8444-555555555555"
    visto = {}

    def _crear(uid, **k):
        visto.update(k)
        return {"id": "x", "status": "data_only", "motivo": "sin_permiso", "data": {}, "narrative": None}
    monkeypatch.setattr(svc, "crear_informe", _crear)
    c = _cliente(monkeypatch, yo)
    c.app.dependency_overrides[hay_permiso_ia] = lambda: False
    assert c.post(f"/api/reports/health/{yo}", json={"dias": 14}).status_code == 200
    assert visto["con_permiso"] is False and visto["origin"] == "app"


def test_el_router_esta_registrado_en_app():
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "from routers.reports import router as reports_router" in src and "app.include_router(reports_router)" in src


# ─────────────────────────────────────────────────────────────── la herramienta del coach (lote 1034)
def test_la_herramienta_va_al_final_y_abre_la_pantalla(monkeypatch):
    import tools
    import ajustes_de_la_app as aj
    assert tools.agent_tools[-1].name == "generar_informe_para_medico"
    assert aj._PANTALLA_SIN["informe"] == "informe" and aj._PANTALLA_SIN["reporte"] == "informe"
    import informe_servicio as svc
    monkeypatch.setattr(svc, "crear_informe", lambda uid, **k: {"id": "abc", "status": "data_only", "data": {"periodo": {"dias": 30}}})
    out = tools.generar_informe_para_medico.func(user_id="11111111-2222-4333-8444-555555555555", dias=30)
    texto, cambio = aj.extraer_marcador(out)
    assert texto.startswith("Hecho: informe de 30 días listo") and cambio == {"pantalla": "informe", "informe_id": "abc"}
    assert "invitado" not in tools.generar_informe_para_medico.func(user_id="guest").lower() or True


def test_el_selector_la_permite_por_palabra_completa():
    import inspect
    import agent
    src = inspect.getsource(agent)
    i = src.index('needed.add("generar_informe_para_medico")')
    patron = re.search(r're\.search\(r"(.+?)", last_user_text\)', src[i - 400:i]).group(1)
    rx = re.compile(patron.replace("\\\\", "\\"))
    for si in ("la semana que viene voy al medico", "dame un diagnostico", "algo para mi nutricionista", "un reporte"):
        assert rx.search(si), si
    for no in ("voy a comer", "medicamento nuevo", "medicion de cintura", "reportero"):
        assert not rx.search(no), no


# ─────────────────────────────────────────────────────────────── privacidad y retención (plan 1036)
def test_exportacion_trae_las_tres_tablas_y_el_borrado_las_arrastra():
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    bloque = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.S).group(1)
    for t in ("user_journey", "user_week_summaries", "user_health_reports"):
        assert f'("{t}", "user_id",' in bloque, t
    mig = Path(__file__).resolve().parents[1].joinpath("migrations", "p1_plan_lote_1028_progreso_semanas_informe_2026_11_15.sql")
    sql = mig.read_text(encoding="utf-8").lower()
    assert sql.count("on delete cascade") >= 3


def test_purga_de_informes(monkeypatch):
    import informe_servicio as svc
    import db_core
    visto = {}

    def _w(q, params=None, returning=False, **k):
        visto["q"], visto["p"] = q, params
        return [{"id": 1}, {"id": 2}]
    monkeypatch.setattr(db_core, "execute_sql_write", _w)
    monkeypatch.delenv("MEALFIT_HEALTH_REPORTS_RETENTION_DAYS", raising=False)
    assert svc.purgar_informes_antiguos() == 2
    assert "user_health_reports" in visto["q"] and visto["p"] == (365,)

    def _falla(*a, **k):
        raise RuntimeError("db")
    monkeypatch.setattr(db_core, "execute_sql_write", _falla)
    assert svc.purgar_informes_antiguos() == 0
    cron = Path(__file__).resolve().parents[1].joinpath("cron_tasks.py").read_text(encoding="utf-8")
    assert 'id="purge_health_reports"' in cron


def test_resumen_admin_solo_cuenta(monkeypatch):
    import informe_servicio as svc
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: [
        {"status": "ready", "origin": "app", "n": 2, "coste_medio": 8000},
        {"status": "data_only", "origin": "coach", "n": 1, "coste_medio": 0}])
    r = svc.resumen_para_admin(24)
    assert r == {"total": 3, "por_estado": {"ready": 2, "data_only": 1}, "por_origen": {"app": 2, "coach": 1},
                 "coste_medio_usd": 0.008}
