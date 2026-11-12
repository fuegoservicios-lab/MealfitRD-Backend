# backend/tests/test_p1_plan_lote_992.py
"""[P1-PLAN-LOTE-992 · 2026-11-12] El Resumen del panel, «lo más útil posible y sin errores».

Medido en producción el 12-nov antes del cambio: con 2 críticos, «Estado del sistema» escondía 13 avisos y 4 bloques
atrasados; no veía los «Planes fallidos» que sí listaba «Requiere atención»; el rojo del cambio pintaba la cifra
(«Cuentas nuevas: 1» como una alarma); «US$0.16 por usuario activo» dividía el gasto CON las pruebas del dueño entre
los usuarios SIN él (el real: US$0.11); y en 90 días el periodo anterior caía antes del registro del gasto (18-ago) y
salía «↑ frente a US$0.00».
"""
import datetime as dt

import pytest

import admin_metricas as am
from tests.test_p1_plan_lote_576_admin_metricas import _bloque
from tests.test_p1_plan_lote_637 import ADMIN_ENV, _con

_COLA = "FROM public.plan_chunk_queue"
_ALERTAS = "FROM public.system_alerts"
_FALLIDOS = "'generation_status' = 'failed'"
_PARTES = "AS tuyas"
_RETENCION = "AS siguen"
_MIN_GASTO = "MIN(created_at) AS desde FROM public.llm_usage_events"
_MIN_PERSONAS = "MIN(created_at) AS desde FROM public.user_profiles"


@pytest.fixture(autouse=True)
def _admins(monkeypatch):
    monkeypatch.setattr(am, "admin_ids", lambda: frozenset({ADMIN_ENV}))


def _cola(atrasados=0):
    return {"programados": 0, "listos": 0, "en_curso": 0, "atrasados": atrasados, "esperan_usuario": 0}


def _resumen(monkeypatch, dias=7, **respuestas):
    f, capturadas = _con(**respuestas)
    monkeypatch.setattr(am, "execute_sql_query", f)
    r = am.metricas(dias)
    return {t["etiqueta"]: t for t in _bloque(r, "resumen")["tarjetas"]}, r, capturadas


def test_las_seis_tarjetas_en_su_orden(monkeypatch):
    t, _, _ = _resumen(monkeypatch)
    assert list(t) == ["Usuarios activos", "Siguen activos", "Cuentas nuevas", "Comidas registradas", "Gasto de IA",
                       "Estado del sistema"]


def test_con_criticos_el_estado_no_esconde_los_avisos(monkeypatch):
    alertas = [{"alert_key": "dead_lettered_chunk:a", "severity": "critical"},
               {"alert_key": "dream_contradiction:a", "severity": "warning"},
               {"alert_key": "dream_contradiction:b", "severity": "warning"}]
    t, _, _ = _resumen(monkeypatch, **{_ALERTAS: alertas, _COLA: _cola(atrasados=3)})
    e = t["Estado del sistema"]
    assert e["valor"] == "1 crítico" and e["extra"] == "y 5 avisos por revisar"
    assert e["destino"] == "Requiere atención"


def test_el_estado_y_requiere_atencion_cuentan_lo_mismo(monkeypatch):
    # Antes el Resumen no veía los planes fallidos: decía «Todo bien» con un aviso debajo.
    t, r, _ = _resumen(monkeypatch, **{_FALLIDOS: {"n": 2}, _COLA: _cola()})
    assert t["Estado del sistema"]["valor"] == "2 avisos"
    assert [i["titulo"] for i in _bloque(r, "atencion")["items"]] == ["Planes fallidos"]
    assert _bloque(r, "atencion")["items"][0]["valor"] == "2" and "n" not in _bloque(r, "atencion")["items"][0]


def test_los_avisos_se_leen_una_vez_por_carga(monkeypatch):
    _, _, capturadas = _resumen(monkeypatch)
    # lote 993: una lectura de las abiertas y otra de las resueltas recientes, una vez cada una
    assert sum(_ALERTAS in q and "resolved_at IS NULL" in q for q, _ in capturadas) == 1
    assert sum(_ALERTAS in q and "resolved_at >=" in q for q, _ in capturadas) == 1
    assert sum(_FALLIDOS in q for q, _ in capturadas) == 1


def test_solo_el_estado_colorea_su_cifra(monkeypatch):
    t, _, _ = _resumen(monkeypatch, **{_ALERTAS: [{"alert_key": "plan_persist_failed:a", "severity": "critical"}]})
    con_color = [k for k, v in t.items() if "tono_valor" in v]
    assert con_color == ["Estado del sistema"] and t["Estado del sistema"]["tono_valor"] == "malo"


def test_todo_bien_no_lleva_a_ninguna_parte(monkeypatch):
    t, _, _ = _resumen(monkeypatch)
    e = t["Estado del sistema"]
    assert e["valor"] == "Todo bien" and "destino" not in e and "extra" not in e


def test_el_gasto_por_usuario_no_cuenta_tus_pruebas(monkeypatch):
    partes = {"micros": 900_000, "usuarios": 500_000, "tuyas": 300_000}
    t, _, capturadas = _resumen(monkeypatch, **{_PARTES: partes})
    g = t["Gasto de IA"]
    assert g["valor"] == "US$0.90"
    assert "Usuarios: US$0.50 (US$0.10 por activo)" in g["ayuda"]      # 5 activos en el fake
    assert "tus pruebas: US$0.30" in g["ayuda"] and "procesos automáticos: US$0.10" in g["ayuda"]
    q, p = next((q, p) for q, p in capturadas if _PARTES in q)
    assert "<> ALL" not in q and p[0] == p[1] and ADMIN_ENV in p[0]     # el total sigue contando a todos


def test_el_ruido_de_redondeo_no_es_una_parte(monkeypatch):
    t, _, _ = _resumen(monkeypatch, **{_PARTES: {"micros": 802_000, "usuarios": 500_000, "tuyas": 300_000}})
    assert "procesos automáticos" not in t["Gasto de IA"]["ayuda"]


def test_la_proyeccion_mensual(monkeypatch):
    t, _, _ = _resumen(monkeypatch, **{_PARTES: {"micros": 700_000, "usuarios": 700_000, "tuyas": 0}})
    assert t["Gasto de IA"]["extra"] == "≈ US$3.00 al mes al ritmo de estos 7 días"


def test_sin_base_completa_no_se_compara(monkeypatch):
    hace_10 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
    t, _, _ = _resumen(monkeypatch, **{_MIN_GASTO: {"desde": hace_10}, _MIN_PERSONAS: {"desde": hace_10},
                                       _PARTES: {"micros": 700_000, "usuarios": 700_000, "tuyas": 0}})
    for etiqueta in ("Usuarios activos", "Cuentas nuevas", "Comidas registradas", "Gasto de IA"):
        assert t[etiqueta]["cambio"].startswith("Sin comparación: hay datos desde el "), etiqueta
        assert t[etiqueta]["tono"] == "neutro", etiqueta
    assert t["Siguen activos"]["valor"] == "—"
    # el ritmo se mide sobre los 7 días del periodo (todos registrados), no sobre 0
    assert t["Gasto de IA"]["extra"] == "≈ US$3.00 al mes al ritmo de estos 7 días"


def test_el_ritmo_no_divide_por_dias_sin_registro(monkeypatch):
    hace_10 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
    t, _, _ = _resumen(monkeypatch, dias=30, **{_MIN_GASTO: {"desde": hace_10},
                                               _PARTES: {"micros": 1_000_000, "usuarios": 1_000_000, "tuyas": 0}})
    assert t["Gasto de IA"]["extra"] == "≈ US$3.00 al mes al ritmo de estos 30 días"    # 1 US$ en 10 días, no en 30


def test_la_base_completa_si_compara(monkeypatch):
    hace_un_ano = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365)
    t, _, _ = _resumen(monkeypatch, **{_MIN_GASTO: {"desde": hace_un_ano}, _MIN_PERSONAS: {"desde": hace_un_ano}})
    assert not t["Usuarios activos"]["cambio"].startswith("Sin comparación")


def test_siguen_activos(monkeypatch):
    t, _, capturadas = _resumen(monkeypatch, **{_RETENCION: {"antes": 4, "siguen": 3}})
    s = t["Siguen activos"]
    assert s["valor"] == "75 %" and s["extra"] == "3 de 4 personas activas los 7 días anteriores"
    q, p = next((q, p) for q, p in capturadas if _RETENCION in q)
    # el subconjunto de ESTE periodo va primero en el texto, así que sus parámetros también
    assert p[0] == 7 and p[4] == 14 and "<> ALL(%s::text[])" in q


def test_siguen_activos_sin_nadie_antes(monkeypatch):
    t, _, _ = _resumen(monkeypatch, **{_RETENCION: {"antes": 0, "siguen": 0}})
    assert t["Siguen activos"]["valor"] == "—"
    assert t["Siguen activos"]["extra"] == "Nadie estuvo activo los 7 días anteriores"


def test_cada_cifra_lleva_su_minigrafica(monkeypatch):
    hoy = dt.datetime.now(am.ZoneInfo(am._ZONA)).date()
    t, _, _ = _resumen(monkeypatch, **{"date_trunc": lambda q, p: [{"d": hoy, "n": 3}]})
    for etiqueta in ("Usuarios activos", "Cuentas nuevas", "Comidas registradas", "Gasto de IA"):
        serie = t[etiqueta]["serie"]
        assert len(serie) == 7 and serie[-1]["valor"] == 3.0, etiqueta
        assert t[etiqueta]["serie_titulo"] == "Por día"
    assert t["Usuarios activos"]["extra"] == "Hoy: 3"


def test_en_90_dias_la_minigrafica_es_semanal(monkeypatch):
    t, _, _ = _resumen(monkeypatch, dias=90)
    assert t["Usuarios activos"]["serie_titulo"] == "Por semana"
    assert t["Usuarios activos"]["extra"].startswith("Esta semana: ")


def test_comidas_por_usuario_activo(monkeypatch):
    t, _, _ = _resumen(monkeypatch)                                  # fake: 40 comidas, 5 activos
    assert t["Comidas registradas"]["valor"] == "40" and t["Comidas registradas"]["extra"] == "8 por usuario activo"
