# backend/tests/test_p1_plan_lote_620.py
"""[P1-PLAN-LOTE-620 · 2026-09-27] El panel se lee de un vistazo: cifras principales, subfilas, etiquetas en español.

El dueño, al abrir /admin: «mejora cómo se ve». Lo que pintaba el servidor: `partial`, `pending` y `day_generator` en
crudo, subfilas marcadas con «· » dentro de la etiqueta, todas las filas con el mismo peso y el banco sin decir qué
corrida era cada fila. Ahora cada fila puede llevar `destacado` (cifra grande) y `nivel` (subfila), los códigos salen
en español, el gasto trae la proporción de cada función para pintar barras y el banco dice qué se midió.
"""
import datetime as dt
import json

import admin_metricas as am
from tests.test_p1_plan_lote_576_admin_metricas import UUID, _bloque, _fake


def _fake620(query, params=None, **kw):
    q = " ".join(query.split())
    if "MIN(created_at)" in q:
        # relativo al reloj real: una fecha fija caducaría dentro de 30 días (la bomba del lote135)
        return {"desde": dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)}
    if "FROM public.meal_plans" in q:
        return [{"estado": "complete", "n": 3}, {"estado": "partial", "n": 1}]
    if "FROM public.plan_chunk_queue" in q:
        return {"programados": 6, "listos": 1, "en_curso": 0, "atrasados": 0, "esperan_usuario": 0}
    if "SUM(cost_usd_micros)" in q and "GROUP BY" in q:
        nodos = ["day_generator", "vision_scan", "planner", "reviewer", "chat_call_model", "culinary_judge",
                 "self_critique", "compressor", "nodo_nuevo_x", "fact_extractor_extract_facts"]
        return [{"funcion": n, "llamadas": 10 + i, "micros": 100000 - i * 9000} for i, n in enumerate(nodos)]
    if "SUM(cost_usd_micros)" in q:
        return {"micros": 1000000, "n": 200}
    if "FROM public.analyzer_benchmark_runs" in q:
        return [{"ran_at": dt.datetime(2026, 9, 27, 20, 58), "model": "gemini-3.8-flash", "n": 150, "ok": 150,
                 "notes": "linea base re-corrida (ruido del banco)", "metrics": {
                     "kcal": {"mediana": 0.288}, "proteina_g": {"mediana": 0.308}, "recall_componentes": 0.839,
                     "valida": True}},
                {"ran_at": dt.datetime(2026, 9, 27, 19, 6), "model": "gemini-3.8-flash", "n": 150, "ok": 120,
                 "notes": "", "metrics": {"kcal": {"mediana": 0.262}, "proteina_g": {"mediana": 0.285},
                                          "recall_componentes": 0.835, "valida": False}}]
    return _fake(query, params, **kw)


def _filas(bloque):
    return {f["etiqueta"]: f for f in bloque["filas"]}


def test_cifras_principales_y_subfilas(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    r = am.metricas(7)
    destacados = {b["id"]: [f["etiqueta"] for f in b["filas"] if f.get("destacado")] for b in r["bloques"]
                  if b["tipo"] == "kpis"}
    # lote 637: las alertas salen de Planes (van a «Requiere atención») y el coach destaca personas, no la tasa
    assert destacados == {"cuentas": [],
                          "escaner": ["Fotos analizadas", "Platos registrados con el escáner"],
                          "coach": ["Mensajes de usuarios", "Personas que lo usaron"],
                          "planes": ["Planes creados"]}
    esc = _filas(_bloque(r, "escaner"))
    assert esc["Cambió un ingrediente"]["nivel"] == 1 and esc["Tecleó las macros"]["nivel"] == 1
    # una fila destacada se pinta aparte, arriba: sus subfilas quedarían colgando de la fila de encima. Por eso las
    # subfilas siguen SIEMPRE a una fila normal, su total
    for b in r["bloques"]:
        if b["tipo"] != "kpis":
            continue
        previa = None
        for f in b["filas"]:
            if f.get("nivel") == 1:
                assert previa is not None and not previa.get("destacado"), (b["id"], f["etiqueta"])
            else:
                previa = f
    etiquetas = [f["etiqueta"] for f in _bloque(r, "escaner")["filas"]]
    i = etiquetas.index("Corregidos por el usuario")
    assert etiquetas[i + 1:i + 6] == ["Cambió un ingrediente", "«Descríbelo»", "Editó cantidades",
                                      "Cambió la respuesta a una duda", "Tecleó las macros"]
    # ninguna etiqueta arrastra ya la marca de subfila dentro del texto
    assert not any(f["etiqueta"].startswith("·") for b in r["bloques"] if b["tipo"] == "kpis" for f in b["filas"])


def test_el_escaner_pide_su_propia_fila(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    r = am.metricas(7)
    assert [b["id"] for b in r["bloques"] if b.get("ancho")] == ["escaner"]


def test_planes_y_cola_en_espanol(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    planes = _filas(_bloque(am.metricas(7), "planes"))
    assert planes["Estado de esos planes"]["valor"] == ""
    assert planes["Completos"]["valor"] == "3" and planes["Completos"]["nivel"] == 1
    assert planes["Parciales"]["valor"] == "1"
    # lote 637: la cola es la de AHORA, no los bloques creados en el periodo
    assert planes["Bloques en la cola ahora"]["valor"] == "7"
    assert planes["Programados para más adelante"]["valor"] == "6" and planes["Programados para más adelante"]["nivel"] == 1
    assert planes["Listos para generarse"]["valor"] == "1"


def test_gasto_con_nombres_legibles_barras_y_resto_agrupado(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    g = _bloque(am.metricas(7), "gasto")
    nombres = [f[0] for f in g["filas"]]
    assert nombres[:4] == ["Generación de días", "Escáner de fotos", "Planificador", "Revisor clínico"]
    assert len(g["filas"]) == 9 and nombres[-1] == "Otras (2 funciones)"
    assert "nodo_nuevo_x" not in nombres                      # entra en «Otras» por coste; si no, saldría su código
    assert len(g["barras"]) == len(g["filas"]) and g["barras"][0] == 0.1 and all(0 <= b <= 1 for b in g["barras"])


def test_codigo_desconocido_sale_tal_cual_y_el_inseguro_como_otro(monkeypatch):
    def _f(query, params=None, **kw):
        if "SUM(cost_usd_micros)" in query and "GROUP BY" in query:
            return [{"funcion": "nodo_nuevo_x", "llamadas": 3, "micros": 50000},
                    {"funcion": "Nodo Raro@x", "llamadas": 1, "micros": 10000}]
        return _fake620(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _f)
    assert [f[0] for f in _bloque(am.metricas(7), "gasto")["filas"]] == ["nodo_nuevo_x", "Otro"]


def test_el_banco_dice_que_se_midio(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    b = _bloque(am.metricas(7), "analizador")
    assert b["columnas"] == ["Fecha (UTC)", "Corrida", "kcal", "Proteína", "Componentes"]
    assert b["filas"][0] == ["2026-09-27 20:58", "linea base re-corrida (ruido del banco)", "29 %", "31 %", "84 %"]
    assert b["filas"][1][1] == "gemini-3.8-flash (no válida)"      # sin notas: el modelo; y avisa si no vale
    assert "±2-4 puntos" in b["nota"]


def test_la_nota_larga_se_corta_en_una_palabra(monkeypatch):
    def _f(query, params=None, **kw):
        if "FROM public.analyzer_benchmark_runs" in query:
            return [{"ran_at": dt.datetime(2026, 9, 27), "model": "m", "n": 1, "ok": 1,
                     "notes": "P1-PLAN-LOTE-601 v2: lo denso se cuenta (frutos secos, tocineta, aceite visible)",
                     "metrics": {"valida": True}}]
        return _fake620(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _f)
    corrida = _bloque(am.metricas(7), "analizador")["filas"][0][1]
    assert corrida == "P1-PLAN-LOTE-601 v2: lo denso se cuenta (frutos secos…"
    assert len(corrida) <= 61


def test_el_escaner_avisa_desde_cuando_registra(monkeypatch):
    monkeypatch.setattr(am, "execute_sql_query", _fake620)
    desde = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)
    assert f"Se registra desde el {desde:%d-%m-%Y}." in _bloque(am.metricas(30), "escaner")["nota"]
    # si ya registraba antes del periodo, no hay nada que avisar (lote 637: la nota sigue diciendo «sin tus cuentas»)
    assert "Se registra" not in _bloque(am.metricas(1), "escaner")["nota"]


def test_sigue_sin_filtrar_identidad(monkeypatch):
    def _f(query, params=None, **kw):
        if "FROM public.analyzer_benchmark_runs" in query:
            return [{"ran_at": dt.datetime(2026, 9, 27), "model": "m", "n": 1, "ok": 1,
                     "notes": "prueba de juan@correo.com 61a13831-2a70-4437-a084-0d3e09b653e4",
                     "metrics": {"valida": True}}]
        return _fake620(query, params, **kw)
    monkeypatch.setattr(am, "execute_sql_query", _f)
    texto = json.dumps(am.metricas(7), ensure_ascii=False)
    assert not UUID.search(texto) and "@" not in texto
