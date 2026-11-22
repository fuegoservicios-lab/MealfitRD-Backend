# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1059 · 2026-11-17] El juego de las semanas en el servidor (plan 1029 + eventos 1035).

La banda de «día en meta» ya tiene su paridad con `calorieGoalStatus.js` en el test del lote 1058 (la usan los dos
módulos). Aquí: la tabla de verdad del cliente (`semanaEnProgreso.test.jsx`), el resumen, el día 1 que solo va hacia
atrás, el enganche que nunca tumba un registro, las rutas y los eventos que salen una sola vez.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import juego_puntos as jp
import progreso_semanas as ps

D = date.fromisoformat
YO = "11111111-2222-4333-8444-555555555555"
OTRO = "99999999-2222-4333-8444-555555555555"


# ═════════════════════════════════ tabla de verdad (copiada de frontend/src/__tests__/semanaEnProgreso.test.jsx)
def test_semanas_como_el_cliente():
    assert ps.numero_de_semana(D("2026-10-28"), D("2026-10-28")) == 1
    assert ps.numero_de_semana(D("2026-10-28"), D("2026-11-03")) == 1
    assert ps.numero_de_semana(D("2026-10-28"), D("2026-11-04")) == 2
    assert ps.numero_de_semana(D("2026-10-28"), D("2026-11-15")) == 3
    assert ps.numero_de_semana(D("2020-01-01"), D("2026-11-15")) > 350
    assert ps.domingo_de(D("2026-11-15")) == D("2026-11-15")
    assert ps.domingo_de(D("2026-11-18")) == D("2026-11-15")
    assert ps.domingo_de(D("2026-11-01")) == D("2026-11-01")
    assert ps.domingo_de(D("2026-10-31")) == D("2026-10-25")
    assert ps.numero_de_semana(ps.domingo_de(D("2026-10-28")), D("2026-11-15")) == 4
    assert [ps.estrellas(n) for n in (0, 2, 3, 4, 5, 6, 7)] == [0, 0, 1, 1, 2, 2, 3]
    assert [ps.faltan_para_otra_estrella(n) for n in (0, 2, 3, 5, 6, 7)] == [3, 1, 2, 2, 1, 0]
    assert ps.fechas_de_la_semana(D("2026-11-15"), 2)[0] == D("2026-11-22")


def test_racha_hoy_en_curso_no_rompe_un_fallado_si():
    assert ps.racha(["meta", "meta", "meta", "bajo"]) == 3        # hoy (600 kcal) aún en curso
    assert ps.racha(["meta", "meta", "meta", "meta"]) == 4
    assert ps.racha(["meta", "bajo", "meta", "vacio"]) == 1
    assert ps.mejor_racha(["meta", "meta", "bajo", "meta"]) == 2


def test_estado_vacio_y_nombres_del_cliente():
    assert ps.estado(0, 0, 2000, "") == "vacio"
    assert ps.estado(2000, 3, 2000, "") == "meta"
    assert ps.estado(900, 1, 2000, "") == "bajo"
    assert ps.estado(2300, 3, 2000, "") == "pasado"
    assert ps.estado(0, 1, 2000, "") == "bajo"                   # registró algo sin kcal: no es vacío
    assert ps.estado(1650, 2, 2000, "lose_fat") == "meta"        # 82 %: en meta perdiendo grasa


def test_rangos():
    assert ps.rango(0) == {"clave": "bronce", "siguiente": "plata", "estrellas_para_el_siguiente": 10}
    assert ps.rango(10)["clave"] == "plata"
    assert ps.rango(49) == {"clave": "oro", "siguiente": "platino", "estrellas_para_el_siguiente": 1}
    assert ps.rango(100) == {"clave": "diamante", "siguiente": None, "estrellas_para_el_siguiente": None}


# ═════════════════════════════════ el resumen
def _dias(desde: date, kcal: list[int]) -> dict:
    return {(desde + timedelta(days=i)).isoformat(): {"calories": k, "protein": 100, "meals_count": 3 if k else 0}
            for i, k in enumerate(kcal) if k}


def test_resumen_dos_semanas():
    # día 1 = martes 3-nov ⇒ Semana 1 empieza el domingo 1-nov. Semana 1: 5 en meta ⇒ ★★. Hoy = lunes 9-nov (S2, día 2).
    dias = _dias(D("2026-11-03"), [2000, 2000, 2000, 2000, 2000]) | _dias(D("2026-11-08"), [2000, 600])
    r = ps.armar_resumen(hoy=D("2026-11-09"), dia_1=D("2026-11-03"), dias=dias, metas_por_dia={},
                         meta_de_hoy={"calories": 2000}, objetivo_del_perfil="maintain")
    assert r["semana_1_empieza"] == "2026-11-01"
    # [P1-PLAN-LOTE-1161] estrellas por PUNTOS y racha de días REGISTRADOS (7: del 3 al 9, hoy incluido).
    sem = {k: v for k, v in r["semana"].items() if k not in ("puntos", "faltan_puntos_para_otra_estrella", "dias", "estrellas", "bono")}
    assert sem == {"numero": 2, "empieza": "2026-11-08", "termina": "2026-11-14", "dia": 2, "dias_en_meta": 1,
                   "dias_registrados": 2, "faltan_para_otra_estrella": 2}
    assert r["racha"] == 7 and r["mejor_racha"] == 7
    assert r["semana_pasada"]["numero"] == 1 and r["semana_pasada"]["dias_en_meta"] == 5
    assert r["semana_pasada"]["estrellas"] == jp.estrellas_por_puntos(r["semana_pasada"]["puntos"])
    assert r["total"]["semanas"] == 2 and r["total"]["dias_en_meta"] == 6
    assert r["semanas"][0]["cerrada"] and not r["semanas"][1]["cerrada"]
    assert r["semanas"][0]["dias_bajo"] == 0 and r["semanas"][0]["dias_registrados"] == 5


def test_meta_de_cada_dia_y_no_la_de_hoy():
    # El 3-nov la meta era 1500 (2000 kcal = pasado); hoy es 2000.
    dias = _dias(D("2026-11-03"), [2000])
    r = ps.armar_resumen(hoy=D("2026-11-04"), dia_1=D("2026-11-03"), dias=dias,
                         metas_por_dia={"2026-11-03": {"calories": 1500}}, meta_de_hoy={"calories": 2000})
    assert r["semanas"][0]["dias_sobre"] == 1 and r["semanas"][0]["dias_en_meta"] == 0


def test_sin_dia_1_vacio():
    assert ps.armar_resumen(hoy=D("2026-11-04"), dia_1=None, dias={}, metas_por_dia={}, meta_de_hoy=None) == {}


# ═════════════════════════════════ día 1 y enganche
class _Graba:
    def __init__(self):
        self.sql = []

    def __call__(self, q, params=None, **k):
        self.sql.append((q, params))
        return []


def test_asegurar_dia_1_solo_hacia_atras(monkeypatch):
    import db_core
    g = _Graba()
    monkeypatch.setattr(db_core, "execute_sql_write", g)
    ps.asegurar_dia_1(YO, "2026-11-03", 240)
    q, params = g.sql[0]
    assert "LEAST(user_journey.first_log_date, EXCLUDED.first_log_date)" in q
    assert "WHERE EXCLUDED.first_log_date < user_journey.first_log_date" in q
    assert "week1_start" not in q                                  # columna generada: nunca se escribe
    assert params == (YO, D("2026-11-03"), 240)


def test_borrar_no_toca_el_dia_1(monkeypatch):
    import db_core
    g = _Graba()
    monkeypatch.setattr(db_core, "execute_sql_write", g)
    monkeypatch.setattr("diary_targets.resolve_user_effective_off_min", lambda uid, tz=None: 480)
    ps.tras_escribir_el_diario(YO, "2026-11-10T15:00:00+00:00")   # borrado/corrección: es_registro=False
    assert not any("user_journey" in q for q, _ in g.sql)
    q, params = g.sql[-1]
    assert "UPDATE user_week_summaries SET closed = false" in q and "user_id = %s" in q
    assert params == (YO, D("2026-11-08"))                         # 15:00 UTC − 8 h = 10-nov ⇒ domingo 8-nov


def test_el_enganche_nunca_lanza(monkeypatch):
    import db_core

    def _rompe(*a, **k):
        raise RuntimeError("neon caído")
    monkeypatch.setattr(db_core, "execute_sql_write", _rompe)
    monkeypatch.setattr("diary_targets.resolve_user_effective_off_min", lambda uid, tz=None: 480)
    llamados = []
    monkeypatch.setattr("fail_open.fail_open", lambda sitio, exc, **k: llamados.append(sitio))
    ps.tras_escribir_el_diario(YO, None, es_registro=True)        # no lanza
    assert llamados == ["progreso_dia_1"]


def test_db_facts_engancha_los_tres_caminos():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "db_facts.py").read_text(encoding="utf-8")
    assert src.count('__import__("progreso_semanas").tras_escribir_el_diario(') == 3
    assert 'tras_escribir_el_diario(user_id, now, es_registro=True)' in src
    assert "RETURNING id, consumed_at" in src


def test_sincronizar_solo_escribe_lo_que_cambio(monkeypatch):
    import db_core
    g = _Graba()
    monkeypatch.setattr(db_core, "execute_sql_write", g)
    monkeypatch.setattr(db_core, "execute_sql_query", lambda q, p=None, **k: [
        {"week_number": 1, "days_logged": 5, "days_on_goal": 5, "days_under": 0, "days_over": 0, "stars": 2,
         "closed": True}])
    sem = [{"numero": 1, "empieza": "2026-11-01", "dias_registrados": 5, "dias_en_meta": 5, "dias_bajo": 0,
            "dias_sobre": 0, "estrellas": 2, "kcal_media": 2000, "proteina_media": 100, "cerrada": True},
           {"numero": 2, "empieza": "2026-11-08", "dias_registrados": 6, "dias_en_meta": 1, "dias_bajo": 5,
            "dias_sobre": 0, "estrellas": 0, "kcal_media": 1500, "proteina_media": 90, "cerrada": True},
           {"numero": 3, "empieza": "2026-11-15", "dias_registrados": 1, "dias_en_meta": 0, "dias_bajo": 1,
            "dias_sobre": 0, "estrellas": 0, "kcal_media": 900, "proteina_media": 50, "cerrada": False}]
    assert ps._sincronizar_semanas(YO, sem) == 1
    assert g.sql[0][1][:2] == (YO, 2)                              # solo la 2 (la 1 ya estaba igual; la 3 en curso)


# ═════════════════════════════════ eventos (lote 1035)
def _res(semanas: list[tuple[int, int]], racha: int = 0) -> dict:
    """semanas: [(numero, dias_en_meta)], la última en curso."""
    ss = [{"numero": n, "dias_en_meta": m, "estrellas": ps.estrellas(m), "cerrada": i < len(semanas) - 1}
          for i, (n, m) in enumerate(semanas)]
    tot = sum(s["estrellas"] for s in ss)
    return {"semanas": ss, "racha": racha, "total": {"estrellas": tot}, "rango": ps.rango(tot),
            "semana": {"numero": semanas[-1][0]}}


def test_primera_vez_se_siembra_y_no_inunda():
    r = _res([(1, 5), (2, 7), (3, 0)], racha=7)
    pend, kv = ps.eventos_nuevos(ps.hitos_cumplidos(r), {}, "2026-11-17")
    assert pend == []                                              # cuenta vieja: nada de golpe
    assert "semana_cerrada:2" in kv["vistos"]


def test_un_evento_sale_una_sola_vez_y_marcar_visto():
    kv0 = {"emitidos": {}, "vistos": [], "version": ps.EVENTOS_VERSION}
    r = _res([(1, 5), (2, 0)])
    pend, kv = ps.eventos_nuevos(ps.hitos_cumplidos(r), kv0, "2026-11-15")
    assert {e["clave"] for e in pend} == {"semana_cerrada:1", "primera_estrella"}
    assert pend[0]["datos"] == {"numero": 1, "estrellas": 2, "dias_en_meta": 5, "siguiente": 2, "puntos": None}   # [1161] + puntos
    pend2, kv2 = ps.eventos_nuevos(ps.hitos_cumplidos(r), kv, "2026-11-16")
    assert {e["clave"] for e in pend2} == {"semana_cerrada:1", "primera_estrella"}   # pendientes hasta verlos
    assert kv2["emitidos"]["semana_cerrada:1"] == "2026-11-15"                        # la fecha no cambia
    kv2["vistos"] = ["semana_cerrada:1", "primera_estrella"]
    pend3, _ = ps.eventos_nuevos(ps.hitos_cumplidos(r), kv2, "2026-11-16")
    assert pend3 == []
    assert ps.eventos_de_hoy(kv2, "2026-11-15") == ["primera_estrella", "semana_cerrada:1"]


def test_dos_semanas_cerradas_de_golpe_solo_la_ultima():
    kv0 = {"emitidos": {}, "vistos": [], "version": ps.EVENTOS_VERSION}
    pend, _ = ps.eventos_nuevos(ps.hitos_cumplidos(_res([(1, 0), (2, 1), (3, 2), (4, 0)])), kv0, "2026-11-29")
    assert [e["clave"] for e in pend if e["codigo"] == "semana_cerrada"] == ["semana_cerrada:3"]
    assert any(e["clave"] == "semana_numero:4" for e in pend)


def test_hitos_racha_perfecta_seguidas_y_rango():
    r = _res([(1, 3), (2, 3), (3, 3), (4, 7), (5, 0)], racha=14)
    claves = {e["clave"] for e in ps.hitos_cumplidos(r)}
    assert {"racha:3", "racha:7", "racha:14", "primera_semana_perfecta", "semanas_seguidas_con_estrella:4",
            "semana_numero:4"} <= claves
    assert "racha:30" not in claves
    r2 = _res([(n, 7) for n in range(1, 5)] + [(5, 0)])
    assert "rango:plata" in {e["clave"] for e in ps.hitos_cumplidos(r2)}   # 12 ★


# ═════════════════════════════════ rutas
def _cliente(uid):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.progress as rp
    from auth import get_verified_user_id
    app = FastAPI()
    app.include_router(rp.router)
    app.dependency_overrides[get_verified_user_id] = lambda: uid
    app.dependency_overrides[rp._LEER_LIMITER] = lambda: None
    app.dependency_overrides[rp._VISTOS_LIMITER] = lambda: None
    return TestClient(app)


def test_rutas_otro_usuario_403_invitado_vacio(monkeypatch):
    c = _cliente(YO)
    assert c.get(f"/api/progress/summary/{OTRO}").status_code == 403
    assert c.get(f"/api/progress/weeks/{OTRO}").status_code == 403
    assert c.post(f"/api/progress/events/seen/{OTRO}", json={"claves": ["x"]}).status_code == 403
    assert c.get("/api/progress/summary/guest").json() == {}
    assert c.get("/api/progress/weeks/guest").json() == {"semanas": []}


def test_ruta_resumen_y_vistos(monkeypatch):
    monkeypatch.setattr(ps, "resumen", lambda uid, tz=None, **k: {"semana": {"numero": 3}, "eventos": []})
    vistos = []
    monkeypatch.setattr(ps, "marcar_vistos", lambda uid, claves: vistos.append(list(claves)) or 1)
    c = _cliente(YO)
    assert c.get(f"/api/progress/summary/{YO}?tzOffset=240").json()["semana"]["numero"] == 3
    assert c.post(f"/api/progress/events/seen/{YO}", json={"claves": ["semana_cerrada:2"]}).json() == {"ok": True, "marcados": 1}
    assert vistos == [["semana_cerrada:2"]]


def test_router_registrado_en_la_app():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "from routers.progress import router as progress_router" in src
    assert "app.include_router(progress_router)" in src


# ═════════════════════════════════ caché y logs
def test_resumen_invitado_y_cache_invalidable(monkeypatch):
    assert ps.resumen("guest") == {}
    llamadas = []

    def _leer(uid, tz):
        llamadas.append(uid)
        return ps.armar_resumen(hoy=D("2026-11-09"), dia_1=D("2026-11-03"), dias=_dias(D("2026-11-03"), [2000]),
                                metas_por_dia={}, meta_de_hoy={"calories": 2000})
    monkeypatch.setattr(ps, "_leer", _leer)
    monkeypatch.setattr(ps, "_sincronizar_semanas", lambda uid, s: 0)
    monkeypatch.setattr(ps, "_estado_kv", lambda uid: {"emitidos": {}, "vistos": []})
    monkeypatch.setattr(ps, "_guardar_kv", lambda uid, v: None)
    ps.invalidar(YO)
    a = ps.resumen(YO, 240)
    b = ps.resumen(YO, 240)
    assert len(llamadas) == 1 and a == b and "semanas" not in a and "eventos" in a
    ps.invalidar(YO)
    ps.resumen(YO, 240)
    assert len(llamadas) == 2
    ps.invalidar(YO)


def test_logs_sin_cifras_de_salud():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "progreso_semanas.py").read_text(encoding="utf-8")
    for linea in src.splitlines():
        if "logger." in linea:
            assert "kcal" not in linea and "calories" not in linea and "meal" not in linea, linea


# ═════════════════════════════════ el coach (plan 1031)
def _res_coach():
    return ps.armar_resumen(hoy=D("2026-11-16"), dia_1=D("2026-09-15"),
                            dias=_dias(D("2026-11-08"), [2000, 2300, 2300, 2300, 2300, 2300, 2300, 2000]),
                            metas_por_dia={}, meta_de_hoy={"calories": 2000})


def test_bloque_cifras_del_resumen_y_corto():
    import coach_progreso as cp
    r = _res_coach()
    t = cp.texto_del_bloque(r)
    # [P1-PLAN-LOTE-1161] el bloque habla de puntos, racha registrando con multiplicador, escudos y división.
    assert t.startswith("\n\n🏆 PROGRESO (cifras del servidor, juego por puntos): Semana 10 (15 nov–21 nov), día 2 de 7.")
    assert "Esta semana: 39 pts, 0★; faltan 46 pts para la 1.ª estrella." in t
    assert "Racha: 8 días registrando (×1,3; mejor: 8); escudos 1/2." in t
    assert "Semana pasada (S9): 133 pts, 1★." in t
    assert "rango Bronce III (faltan 98 pts para Bronce II)" in t      # [1163] 4 divisiones, IV→I
    assert len(t.split()) <= 75                                    # ~70 tokens
    assert cp.texto_del_bloque({}) == ""


def test_bloque_hoy_celebra_una_vez():
    import coach_progreso as cp
    r = dict(_res_coach(), eventos_de_hoy=["semana_cerrada:9", "racha:3"])
    t = cp.texto_del_bloque(r)
    assert "HOY: cerró la S9 con 1★; racha de 3 días registrando (multiplicador ×1,15) (celébralo una vez)." in t


def test_bloque_invitado_canario_y_fallo(monkeypatch):
    import coach_progreso as cp
    assert cp.bloque_para_chat("guest") == ""
    assert cp.bloque_para_chat(None) == ""
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_PROGRESO_COACH", "canary")
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_USER_IDS", YO)
    assert cp.bloque_para_chat(OTRO) == ""                         # fuera del canario
    monkeypatch.setattr(ps, "resumen", lambda *a, **k: _res_coach())
    assert cp.bloque_para_chat(YO, 240).startswith("\n\n🏆 PROGRESO")

    def _rompe(*a, **k):
        raise RuntimeError("neon")
    monkeypatch.setattr(ps, "resumen", _rompe)
    assert cp.bloque_para_chat(YO, 241) == ""                      # fail_open


def test_nota_tras_registrar_solo_si_sube_una_estrella(monkeypatch):
    import coach_progreso as cp
    antes = {"semana": {"numero": 10, "estrellas": 0, "dias_en_meta": 2, "puntos": 70}}
    despues = {"semana": {"numero": 10, "estrellas": 1, "dias_en_meta": 3, "puntos": 95}}
    assert "ganó la 1.ª estrella de la Semana 10 (95 pts)" in cp.nota_de_estrella(antes, despues)
    # [P1-PLAN-LOTE-1161] subir de división también se avisa
    a2 = dict(antes, rango={"clave": "bronce", "division": 3}); d2 = dict(antes, rango={"clave": "plata", "division": 1})
    assert "subió a Plata I" in cp.nota_de_estrella(a2, d2)
    assert cp.nota_de_estrella(antes, antes) == ""
    assert cp.nota_de_estrella(antes, {"semana": {"numero": 11, "estrellas": 1}}) == ""
    assert "perfecta" in cp.nota_de_estrella({"semana": {"numero": 3, "estrellas": 2}},
                                             {"semana": {"numero": 3, "estrellas": 3, "dias_en_meta": 7}})
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_PROGRESO_COACH", "on")
    cp._ANTES.clear()
    assert cp.nota_tras_registrar(YO) == ""                        # sin «antes» no se sabe si es nueva
    cp._guardar_antes(YO, antes)
    monkeypatch.setattr(ps, "resumen", lambda *a, **k: despues)
    assert "1.ª estrella" in cp.nota_tras_registrar(YO)
    assert cp.nota_tras_registrar(YO) == ""                        # la misma estrella no se anuncia dos veces
    cp._ANTES.clear()


def test_enganches_en_agent_y_tools():
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    ag = (base / "agent.py").read_text(encoding="utf-8")
    i_bloque = ag.index('__import__("coach_progreso").bloque_para_chat(user_id, tz_offset)')
    i_split = ag.index("system_prompt += _VOLATILE_SYS_SPLIT_MARKER")
    assert i_split < i_bloque                                      # va en la zona volátil (no rompe el prefijo)
    assert '__import__("coach_progreso").precargar(uid, self.tz_offset)' in ag
    tl = (base / "tools.py").read_text(encoding="utf-8")
    assert 'msg += __import__("coach_progreso").nota_tras_registrar(user_id)' in tl


def test_herramienta_al_final_y_selector():
    import tools
    from agent import _select_tools_for_turn
    assert tools.agent_tools[-1].name == "consultar_progreso"
    from langchain_core.messages import HumanMessage

    def _sel(txt):
        out = _select_tools_for_turn(tools.agent_tools, {"messages": [HumanMessage(content=txt)]})
        return {getattr(t, "name", t) for t in out}
    assert "consultar_progreso" in _sel("¿cómo voy esta semana?")
    assert "consultar_progreso" in _sel("cuál fue mi mejor racha")
    assert "consultar_progreso" not in _sel("voy a comer arroz")


def test_herramienta_lee_y_corta(monkeypatch):
    import tools
    r = dict(_res_coach())
    monkeypatch.setattr(ps, "resumen", lambda *a, **k: r)
    out = tools.consultar_progreso.func(user_id=YO, semanas=2)
    assert out.splitlines()[0].startswith("S9 (2026-11-08): 133 pts, 1★, 7 con registro, 1/7 en meta")
    assert "(en curso)" in out.splitlines()[1]
    assert tools.consultar_progreso.func(user_id="guest").startswith("El progreso por semanas necesita")


def test_regla_r5_en_el_manual():
    from prompts import chat_agent
    src = __import__("inspect").getsource(chat_agent)
    assert "R5. PROGRESO [P1-PLAN-LOTE-1059]" in src
    assert "nunca los calcules ni estimes" in src


# ═════════════════════════════════ aviso del domingo (plan 1035.4)
def test_aviso_texto_y_hora():
    from datetime import datetime, timezone
    import aviso_semanal as av
    assert av.texto_del_aviso("es-DO", 10, {"numero": 9, "estrellas": 2, "dias_en_meta": 5}) == (
        "Tu semana en Bioboros", "Empieza tu Semana 10. La 9 cerró con ★★.")
    assert av.texto_del_aviso("en-US", 10, {"numero": 9, "estrellas": 0, "dias_en_meta": 1})[1] == (
        "Week 10 starts today. Week 9 closed with 1 of 7 days on goal.")
    assert av.texto_del_aviso("xx", 10, {"numero": 9, "estrellas": 1})[0] == "Tu semana en Bioboros"
    assert av.texto_del_aviso("es-DO", 1, None) is None
    domingo_9_rd = datetime(2026, 11, 22, 13, 30, tzinfo=timezone.utc)      # 09:30 en RD (UTC-4)
    assert av.toca_ahora(domingo_9_rd, 240, 9)
    assert not av.toca_ahora(domingo_9_rd, 0, 9)                            # en UTC son las 13
    assert not av.toca_ahora(datetime(2026, 11, 21, 13, 30, tzinfo=timezone.utc), 240, 9)   # sábado


def test_aviso_solo_a_quien_lo_encendio_y_una_vez(monkeypatch):
    from datetime import datetime, timezone
    import aviso_semanal as av
    import db_core
    consultas, escritos, enviados = [], [], []
    kv = {}

    def _q(sql, params=None, **k):
        consultas.append(sql)
        if "FROM user_profiles" in sql:
            return [{"id": YO, "locale": "es-DO"}]
        if "FROM app_kv_store" in sql:
            return {"value": kv.get(params[0])} if params[0] in kv else None
        return None

    def _w(sql, params=None, **k):
        escritos.append(params)
        kv[params[0]] = params[1]
    monkeypatch.setattr(db_core, "execute_sql_query", _q)
    monkeypatch.setattr(db_core, "execute_sql_write", _w)
    monkeypatch.setattr("diary_targets.resolve_user_effective_off_min", lambda uid, tz=None: 480)
    monkeypatch.setattr(ps, "resumen", lambda *a, **k: {"semana": {"numero": 10},
                                                       "semana_pasada": {"numero": 9, "estrellas": 1, "dias_en_meta": 3}})
    monkeypatch.setattr("utils_push.send_push_notification", lambda uid, t, b, **k: enviados.append((t, b)) or True)
    ahora = datetime(2026, 11, 22, 13, 5, tzinfo=timezone.utc)
    assert av.enviar_avisos_del_domingo(ahora) == 1
    assert "health_profile->>'aviso_semanal' = 'true'" in consultas[0]          # solo quien lo encendió
    assert enviados == [("Tu semana en Bioboros", "Empieza tu Semana 10. La 9 cerró con ★.")]
    assert av.enviar_avisos_del_domingo(ahora) == 0                               # misma semana: no repite
    monkeypatch.setenv("MEALFIT_PROGRESO_AVISO_DOMINGO", "false")
    kv.clear()
    assert av.enviar_avisos_del_domingo(ahora) == 0


def test_aviso_registrado_en_cron_y_en_ajustes():
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    assert '__import__("crons_informe_y_progreso").registrar(' in (base / "cron_tasks.py").read_text(encoding="utf-8")
    import crons_informe_y_progreso as cr
    assert ("progreso_aviso_domingo", 1, "aviso_semanal", "enviar_avisos_del_domingo") in cr.JOBS

    class _Sched:
        def __init__(self): self.ids = {"purge_health_reports"}
        def get_job(self, i): return i in self.ids
    altas = []
    # [P1-PLAN-LOTE-1105] + el aviso del peso, en el mismo registro
    assert cr.registrar(_Sched(), lambda s, fn, tipo, **kw: altas.append(kw["id"]), 30) == ["progreso_aviso_domingo", "aviso_peso", "estancamiento_peso", "aviso_suplemento"]
    assert altas == ["progreso_aviso_domingo", "aviso_peso", "estancamiento_peso", "aviso_suplemento"] and callable(__import__("aviso_semanal").enviar_avisos_del_domingo)


# ═════════════════════════════════ medición (plan 1036.5)
def test_medicion_solo_recuentos(monkeypatch):
    import coach_progreso as cp
    import db_core
    vistos = []

    def _q(sql, params=None, **k):
        vistos.append((sql, params))
        return {"total": 40, "con_progreso": 3}
    monkeypatch.setattr(db_core, "execute_sql_query", _q)
    assert cp.medicion_para_admin(168) == {"respuestas": 40, "con_progreso": 3, "pct": 7.5}
    sql, params = vistos[0]
    assert "count(*)" in sql and "content ~*" in sql and "SELECT content" not in sql
    assert params[1] == 168


def test_compartido_solo_si_es_suyo(monkeypatch):
    import informe_servicio as svc
    import db_core
    escritos = []
    monkeypatch.setattr(db_core, "execute_sql_query", lambda q, p=None, **k: None)
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: escritos.append(p))
    assert svc.anotar_compartido(YO, "aaaaaaaa-2222-4333-8444-555555555555") is False and escritos == []
    monkeypatch.setattr(db_core, "execute_sql_query", lambda q, p=None, **k: {"ok": 1})
    assert svc.anotar_compartido(YO, "aaaaaaaa-2222-4333-8444-555555555555") is True
    assert escritos[0][0] == "informes_compartidos"


def test_ruta_compartido_otro_usuario_403():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.reports as rr
    from auth import get_verified_user_id
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[get_verified_user_id] = lambda: YO
    app.dependency_overrides[rr._LEER_LIMITER] = lambda: None
    c = TestClient(app)
    assert c.post(f"/api/reports/health/{OTRO}/aaaaaaaa-2222-4333-8444-555555555555/shared").status_code == 403
