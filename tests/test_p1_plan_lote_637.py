# backend/tests/test_p1_plan_lote_637.py
"""[P1-PLAN-LOTE-637 · 2026-09-28] El panel se entiende: el dueño, ante /admin, «se ve feo y poco entendible».

Medido en producción antes del cambio: «3 activas» incluía al dueño probando (sin él, 2); «11 alertas abiertas» en
grande eran 6 notas informativas y 5 avisos, ninguna crítica; «Bloques en cola 11» eran los CREADOS en el periodo
(la cola real: 8, todos programados para el futuro); el Escáner llenaba media pantalla con porcentajes de 2 fotos; y
ninguna cifra decía contra qué compararse.
"""
import datetime as dt
import json

import pytest

import admin_metricas as am
from tests.test_p1_plan_lote_576_admin_metricas import UUID, _bloque, _fake

ADMIN_ENV = "61a13831-2a70-4437-a084-0d3e09b653e4"
ADMIN_TIER = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _admins(monkeypatch):
    monkeypatch.setattr(am, "admin_ids", lambda: frozenset({ADMIN_ENV}))


def _con(**respuestas):
    """El fake del 576 con algunas consultas reemplazadas: clave = trozo del SQL, valor = respuesta."""
    capturadas = []

    def f(query, params=None, fetch_one=False, fetch_all=False):
        q = " ".join(query.split())
        capturadas.append((q, params))
        for trozo, valor in respuestas.items():
            if trozo in q:
                return valor(q, params) if callable(valor) else valor
        return _fake(query, params, fetch_one=fetch_one, fetch_all=fetch_all)
    return f, capturadas


_ADMIN_TIER = "plan_tier = 'admin'"
_ALERTAS = "FROM public.system_alerts"
_COLA = "FROM public.plan_chunk_queue"


def test_las_cuentas_admin_no_cuentan_como_usuarios(monkeypatch):
    f, capturadas = _con(**{_ADMIN_TIER: [{"id": ADMIN_TIER}]})
    monkeypatch.setattr(am, "execute_sql_query", f)
    am.metricas(7)
    de_personas = [(q, p) for q, p in capturadas
                   if any(t in q for t in ("public.consumed_meals", "public.agent_messages", "public.meal_plans",
                                           "scan_outcome", "vision_scan_resultado' AND metadata"))
                   or ("public.user_profiles" in q and "plan_tier = 'admin'" not in q)]
    assert de_personas
    for q, p in de_personas:
        assert "<> ALL(%s::text[])" in q, q[:90]
        listas = [x for x in (p or ()) if isinstance(x, list)]
        assert listas and all(sorted(x) == sorted([ADMIN_ENV, ADMIN_TIER]) for x in listas), q[:90]


def test_el_gasto_si_cuenta_al_admin(monkeypatch):
    f, capturadas = _con()
    monkeypatch.setattr(am, "execute_sql_query", f)
    am.metricas(7)
    gasto = [q for q, _ in capturadas if "llm_usage_events" in q]
    assert gasto and not any("<> ALL" in q for q in gasto)       # el dinero sale igual, sea quien sea


def test_las_alertas_por_tipo_en_llano_y_sin_ids(monkeypatch):
    ahora = dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc)
    alertas = [
        {"alert_key": f"plan_quality_degraded:{ADMIN_ENV}:x", "severity": "warning", "triggered_at": ahora},
        {"alert_key": f"plan_quality_degraded:{ADMIN_TIER}:y", "severity": "warning", "triggered_at": ahora},
        {"alert_key": "temporal_gate_proactive:juan@correo.com:3", "severity": "info", "triggered_at": ahora},
        {"alert_key": "scheduler_error_nightly_refresh", "severity": "critical", "triggered_at": ahora},
        {"alert_key": "scheduler_error_otra_tarea", "severity": "high", "triggered_at": ahora},
        {"alert_key": "tipo_nuevo_sin_traducir", "severity": "low", "triggered_at": ahora},
        {"alert_key": "Algo Raro@x:1", "severity": "warning", "triggered_at": ahora},
    ]
    f, _ = _con(**{_ALERTAS: alertas})
    monkeypatch.setattr(am, "execute_sql_query", f)
    r = am.metricas(7)
    items = _bloque(r, "atencion")["items"]
    por_titulo = {i["titulo"]: i for i in items}
    assert por_titulo["Una tarea programada falló"]["valor"] == "2"              # la familia es UNA fila
    assert por_titulo["Una tarea programada falló"]["nivel"] == "critico"         # manda la más grave
    assert por_titulo["Plan entregado sin aprobar la revisión"]["valor"] == "2"
    assert por_titulo["Bloque de plan aplazado"]["nivel"] == "info"
    assert "Aviso técnico: tipo nuevo sin traducir" in por_titulo
    assert por_titulo["Otro aviso"]["nivel"] == "aviso"
    assert [i["nivel"] for i in items] == sorted((i["nivel"] for i in items), key=am._ORDEN_NIVEL.get)
    assert items[0]["nivel"] == "critico"
    texto = json.dumps(r, ensure_ascii=False)
    assert not UUID.search(texto) and "@" not in texto


@pytest.mark.parametrize("alertas,atrasados,esperado,tono", [
    ([], 0, "Todo bien", "bueno"),
    ([{"alert_key": "temporal_gate_proactive:a", "severity": "info"}] * 6, 0, "Todo bien", "bueno"),
    ([{"alert_key": "dream_contradiction:a", "severity": "warning"}], 0, "1 aviso", "aviso"),
    ([], 3, "3 avisos", "aviso"),
    ([{"alert_key": "plan_persist_failed:a", "severity": "critical"},
      {"alert_key": "dream_contradiction:a", "severity": "warning"}], 0, "1 crítico", "malo"),
])
def test_el_estado_del_sistema_no_alarma_por_notas(monkeypatch, alertas, atrasados, esperado, tono):
    cola = {"programados": 8, "listos": 0, "en_curso": 0, "atrasados": atrasados, "esperan_usuario": 0}
    f, _ = _con(**{_ALERTAS: alertas, _COLA: cola})
    monkeypatch.setattr(am, "execute_sql_query", f)
    estado = _bloque(am.metricas(7), "resumen")["tarjetas"][3]
    assert (estado["valor"], estado["tono"]) == (esperado, tono)
    if len(alertas) == 6:
        assert "6 notas informativas" in estado["ayuda"]


def test_la_cola_atrasada_sale_en_atencion_y_la_programada_no(monkeypatch):
    f, _ = _con(**{_COLA: {"programados": 8, "listos": 0, "en_curso": 0, "atrasados": 0, "esperan_usuario": 0}})
    monkeypatch.setattr(am, "execute_sql_query", f)
    at = _bloque(am.metricas(7), "atencion")
    assert at["items"] == [] and at["vacio"] == "Nada pendiente."
    f, _ = _con(**{_COLA: {"programados": 8, "listos": 0, "en_curso": 0, "atrasados": 2, "esperan_usuario": 0}})
    monkeypatch.setattr(am, "execute_sql_query", f)
    assert _bloque(am.metricas(7), "atencion")["items"][0]["titulo"] == "Bloques de plan atrasados"


def test_cada_cifra_se_compara_con_el_periodo_anterior(monkeypatch):
    def activos(q, p):
        return {"n": 2 if "AND c.consumed_at <" in q else 5}           # el periodo anterior lleva la cota superior
    f, _ = _con(**{"COUNT(DISTINCT u)": activos})
    monkeypatch.setattr(am, "execute_sql_query", f)
    t = _bloque(am.metricas(7), "resumen")["tarjetas"]
    assert t[0]["valor"] == "5" and t[0]["cambio"] == "↑ 3 frente a los 7 días anteriores (2)"
    assert t[0]["tono"] == "bueno"
    assert t[2]["tono"] == "neutro"                               # el gasto se dice, no se juzga
    assert am._cambio(1, 4, 30) == ("↓ 3 frente a los 30 días anteriores (4)", "malo")
    assert am._cambio(3, 3, 7) == ("Igual que los 7 días anteriores", "neutro")


def test_las_series_rellenan_los_dias_sin_datos(monkeypatch):
    hoy = dt.datetime.now(am.ZoneInfo(am._ZONA)).date()
    f, capturadas = _con(**{"date_trunc": lambda q, p: [{"d": hoy, "n": 3}]})
    monkeypatch.setattr(am, "execute_sql_query", f)
    r = am.metricas(7)
    puntos = _bloque(r, "activos_dia")["puntos"]
    assert len(puntos) == 7 and puntos[-1]["valor"] == 3 and all(p["valor"] == 0 for p in puntos[:-1])
    assert puntos[-1]["etiqueta"] == f"{hoy.day} {am._MES[hoy.month]}"
    assert _bloque(r, "gasto_dia")["puntos"][-1]["texto"] == "US$0.00"            # 3 micros
    assert all(p[0] == "day" for q, p in capturadas if "date_trunc" in q)
    r90 = am.metricas(90)
    semanas = _bloque(r90, "activos_dia")["puntos"]
    assert 13 <= len(semanas) <= 14 and all(p["etiqueta"].startswith("sem. ") for p in semanas)
    assert _bloque(r90, "activos_dia")["titulo"] == "Usuarios activos por semana"


def test_el_embudo_de_las_cuentas_nuevas(monkeypatch):
    f, _ = _con(con_plan={"cuentas": 5, "con_plan": 0, "con_comida": 2, "con_coach": 1, "volvieron": 2})
    monkeypatch.setattr(am, "execute_sql_query", f)
    e = _bloque(am.metricas(7), "embudo")
    assert [(p["etiqueta"], p["valor"], p["texto"]) for p in e["pasos"]] == [
        ("Se registraron", "5", "100 %"), ("Generaron un plan", "0", "0 %"), ("Registraron una comida", "2", "40 %"),
        ("Escribieron al coach", "1", "20 %"), ("Volvieron otro día", "2", "40 %")]
    f, _ = _con(con_plan={"cuentas": 0})
    monkeypatch.setattr(am, "execute_sql_query", f)
    e = _bloque(am.metricas(7), "embudo")
    assert e["nota"] == "Nadie se registró en estos 7 días." and e["pasos"][0]["texto"] == "—"


def test_el_escaner_con_poca_muestra_no_pinta_porcentajes(monkeypatch):
    f, _ = _con(**{"vision_scan_resultado' AND metadata": {"n": 2, "fallidos": 1, "p50": 7900.0, "p90": 7900.0}})
    monkeypatch.setattr(am, "execute_sql_query", f)
    esc = _bloque(am.metricas(7), "escaner")
    assert [x["etiqueta"] for x in esc["filas"]] == ["Fotos analizadas", "Platos registrados con el escáner",
                                                     "Tiempo de análisis (mediana)"]
    assert "Muestra pequeña" in esc["nota"] and not esc.get("ancho")
    assert not any("%" in str(x["valor"]) for x in esc["filas"])


def test_el_coach_sin_valoraciones_no_pinta_un_guion(monkeypatch):
    f, _ = _con(**{"FROM public.agent_messages m": {"preguntas": 2, "respuestas": 12, "personas": 1, "up": 0,
                                                   "down": 0}})
    monkeypatch.setattr(am, "execute_sql_query", f)
    coach = {x["etiqueta"]: x for x in _bloque(am.metricas(7), "coach")["filas"]}
    assert coach["Valoraciones"]["valor"] == "Ninguna todavía" and "👎" not in coach


def test_cada_bloque_dice_su_seccion_en_orden(monkeypatch):
    f, _ = _con()
    monkeypatch.setattr(am, "execute_sql_query", f)
    secciones = [b["seccion"] for b in am.metricas(7)["bloques"]]
    vistas = list(dict.fromkeys(secciones))
    assert vistas == ["Resumen", "Requiere atención", "Usuarios", "Producto", "Costes", "Calidad del escáner"]
    assert secciones == sorted(secciones, key=vistas.index)       # contiguas: el pintor abre una sección por cambio


def test_un_bloque_nuevo_roto_no_tumba_nada(monkeypatch):
    f, _ = _con(con_plan=lambda q, p: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(am, "execute_sql_query", f)
    r = am.metricas(7)
    assert _bloque(r, "embudo")["tipo"] == "error" and _bloque(r, "embudo")["seccion"] == "Usuarios"
    assert _bloque(r, "resumen")["tipo"] == "resumen"


def test_sin_poder_leer_los_admin_de_la_base_quedan_los_del_env(monkeypatch):
    f, capturadas = _con(**{_ADMIN_TIER: lambda q, p: (_ for _ in ()).throw(RuntimeError("x"))})
    monkeypatch.setattr(am, "execute_sql_query", f)
    am.metricas(7)
    listas = [x for _, p in capturadas for x in (p or ()) if isinstance(x, list)]
    assert listas and all(x == [ADMIN_ENV] for x in listas)


def test_el_tipo_de_alerta_solo_deja_pasar_codigo():
    assert am._tipo_alerta(f"plan_quality_degraded:{ADMIN_ENV}:x") == "plan_quality_degraded"
    assert am._tipo_alerta("scheduler_missed_job_x") == "scheduler_missed"
    assert am._tipo_alerta("DROP TABLE; --") == "otro"
    assert am._tipo_alerta("x" * 200) == "otro"                   # más de 64: no es un código nuestro
