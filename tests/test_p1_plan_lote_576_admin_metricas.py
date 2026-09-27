# backend/tests/test_p1_plan_lote_576_admin_metricas.py
"""[P1-PLAN-LOTE-576 · 2026-09-27] Métricas del panel: solo agregados, ya redactados, y un bloque roto no tumba nada."""
import datetime as dt
import json
import re

import admin_metricas as am

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _fake(query, params=None, fetch_one=False, fetch_all=False):
    q = " ".join(query.split())
    if "FROM public.user_profiles" in q:
        return {"n": 12}
    if "COUNT(DISTINCT u)" in q:
        return {"n": 5}
    if "FROM public.consumed_meals" in q:
        return {"n": 40}
    if "vision_scan_resultado" in q:
        return {"n": 20, "fallidos": 2, "p50": 4200.0, "p90": 9100.0}
    if "scan_outcome" in q:
        return {"n": 10, "corregidos": 4, "cambiar": 2, "describelo": 1, "cantidades": 3, "dudas": 1, "macros": 0,
                "desvio": 0.18}
    if "FROM public.agent_messages" in q:
        return {"preguntas": 30, "respuestas": 31, "up": 6, "down": 2}
    if "FROM public.meal_plans" in q:
        return [{"estado": "complete", "n": 3}, {"estado": "partial", "n": 1}]
    if "FROM public.plan_chunk_queue" in q:
        return [{"status": "completed", "n": 9}]
    if "FROM public.system_alerts" in q:
        return {"n": 1}
    if "SUM(cost_usd_micros)" in q and "GROUP BY" in q:
        return [{"funcion": "vision_scan", "llamadas": 14, "micros": 30000}]
    if "SUM(cost_usd_micros)" in q:
        return {"micros": 281234, "n": 90}
    if "FROM public.analyzer_benchmark_runs" in q:
        return [{"ran_at": dt.datetime(2026, 9, 27, 18, 5), "model": "gemini-3.8-flash", "n": 150, "ok": 147,
                 "metrics": {"kcal": {"mediana": 0.21}, "proteina_g": {"mediana": 0.26},
                             "recall_componentes": 0.8, "valida": True}}]
    raise AssertionError(f"consulta inesperada: {q[:80]}")


def _bloque(r, bid):
    return next(b for b in r["bloques"] if b["id"] == bid)


def test_los_seis_bloques_redactados(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake)
    r = am.metricas(7)
    assert [b["id"] for b in r["bloques"]] == ["uso", "escaner", "coach", "planes", "gasto", "analizador"]
    uso = {f["etiqueta"]: f["valor"] for f in _bloque(r, "uso")["filas"]}
    assert uso == {"Cuentas": "12", "Activas en 7 días": "5", "Comidas registradas": "40"}
    esc = {f["etiqueta"]: f["valor"] for f in _bloque(r, "escaner")["filas"]}
    assert esc["Análisis fallidos"] == "10 %" and esc["Corregidos por el usuario"] == "40 %"
    assert esc["Tiempo de análisis (mediana / p90)"] == "4.2 s / 9.1 s"
    coach = {f["etiqueta"]: f["valor"] for f in _bloque(r, "coach")["filas"]}
    assert coach["Tasa de 👎"] == "25 %"
    gasto = _bloque(r, "gasto")
    assert gasto["tipo"] == "tabla" and gasto["filas"] == [["Escáner de fotos", "14", "US$0.03"]]  # lote 620
    assert "US$0.28" in gasto["titulo"]
    assert _bloque(r, "analizador")["filas"][0][2:] == ["21 %", "26 %", "80 %"]  # lote 620: sin «Platos»/«Válida»


def test_ningun_bloque_filtra_identidad(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake)
    texto = json.dumps(am.metricas(30), ensure_ascii=False)
    assert not UUID.search(texto) and "@" not in texto


def test_un_bloque_roto_no_tumba_los_demas(monkeypatch):
    def _roto(query, params=None, **kw):
        if "analyzer_benchmark_runs" in query:
            raise RuntimeError("relation does not exist")
        return _fake(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _roto)
    r = am.metricas(7)
    assert _bloque(r, "analizador") == {"id": "analizador", "titulo": "Banco del analizador", "tipo": "error",
                                        "error": "No disponible"}
    assert _bloque(r, "uso")["tipo"] == "kpis"


def test_los_dias_se_acotan(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake)
    assert am.metricas(0)["dias"] == 1 and am.metricas(999)["dias"] == 90


def test_planes_y_gasto_no_pintan_texto_libre(monkeypatch):
    # Revisión final: /restore-local deja al cliente escribir plan_data (y con él generation_status): un correo o un
    # mensaje acabaría como fila del panel. Lista blanca de estados; etiquetas de código solo [a-z0-9_].
    def _fake2(query, params=None, **kw):
        if "FROM public.meal_plans" in query:
            return [{"estado": "juan@correo.com mi mensaje", "n": 1}, {"estado": "complete", "n": 2},
                    {"estado": "otra cosa", "n": 4}]
        if "FROM public.plan_chunk_queue" in query:
            return [{"status": "raro con espacios", "n": 1}]
        if "SUM(cost_usd_micros)" in query and "GROUP BY" in query:
            return [{"funcion": "Nodo Raro@x", "llamadas": 1, "micros": 10000}]
        return _fake(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _fake2)
    r = am.metricas(7)
    planes = {f["etiqueta"]: f["valor"] for f in _bloque(r, "planes")["filas"]}
    # lote 620: en español y sin «· » (la subfila va en `nivel`); lo inseguro sigue siendo «otro»
    assert planes["Otro estado"] == "5" and planes["Completos"] == "2" and planes["Otro"] == "1"
    assert _bloque(r, "gasto")["filas"] == [["Otro", "1", "US$0.01"]]
    assert "@" not in json.dumps(r, ensure_ascii=False)


def test_escaner_cuenta_solo_fallos_reales_y_solo_del_escaner(monkeypatch):
    capturadas = []

    def _fake3(query, params=None, **kw):
        capturadas.append(" ".join(query.split()))
        if "vision_scan_resultado" in query:
            return {"n": 20, "fallidos": 2, "no_comida": 3, "sin_totales": 1, "p50": 4200.0, "p90": 9100.0}
        return _fake(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _fake3)
    esc = {f["etiqueta"]: f["valor"] for f in _bloque(am.metricas(7), "escaner")["filas"]}
    assert esc["Análisis fallidos"] == "10 %" and esc["No era comida"] == "15 %"
    assert esc["Sin totales (compra o etiqueta)"] == "5 %"
    q = next(c for c in capturadas if "vision_scan_resultado" in c)
    assert "metadata->>'resultado' = 'error'" in q and "metadata->>'purpose' = 'diary'" in q
