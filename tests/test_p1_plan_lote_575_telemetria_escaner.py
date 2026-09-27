# backend/tests/test_p1_plan_lote_575_telemetria_escaner.py
"""[P1-PLAN-LOTE-575 · 2026-09-27] Señales del escáner para el panel: sin texto libre y jamás rompen nada."""
import json

import telemetria_escaner as te
from routers import diary


def test_resultado_del_analisis():
    assert te.resultado_del_analisis({"analysis_failed": True}) == "error"
    assert te.resultado_del_analisis({"is_food": False}) == "no_comida"
    assert te.resultado_del_analisis({"is_food": True, "calories": 0}) == "sin_totales"
    assert te.resultado_del_analisis({"is_food": True, "calories": 420}) == "ok"
    assert te.resultado_del_analisis(None) == "error"


def test_resumen_acota_y_nunca_lanza():
    r = te.resumen_de_correcciones({"cambiados": -3, "cantidades_editadas": 10**9, "porcion": float("nan"),
                                    "kcal_ia": "600", "kcal_final": float("inf"), "redescrito": "sí",
                                    "texto_libre": "mi receta secreta"})
    assert r["cambiados"] == 0 and r["cantidades_editadas"] == 40 and r["porcion"] == 1.0
    assert r["kcal_ia"] == 600 and r["kcal_final"] == 0 and r["redescrito"] is False
    assert "texto_libre" not in r
    assert te.resumen_de_correcciones("basura") is None and te.resumen_de_correcciones(None) is None


def test_resumen_deriva_corregido_y_desvio():
    limpio = te.resumen_de_correcciones({"componentes": 3, "porcion": 1, "kcal_ia": 600, "kcal_final": 600})
    assert limpio["corregido"] is False and limpio["desvio_kcal"] == 0.0
    tocado = te.resumen_de_correcciones({"componentes": 3, "cambiados": 1, "kcal_ia": 600, "kcal_final": 450})
    assert tocado["corregido"] is True and tocado["desvio_kcal"] == 0.25
    assert te.resumen_de_correcciones({"porcion": 1.5})["corregido"] is True


def test_registrar_scan_outcome_solo_conteos_y_traga_errores(monkeypatch):
    escritas = []
    monkeypatch.setattr(te, "execute_sql_write", lambda q, p=None, **kw: escritas.append((q, p)))
    te.registrar_scan_outcome("u1", te.resumen_de_correcciones({"cambiados": 2, "kcal_ia": 500, "kcal_final": 400}))
    q, p = escritas[0]
    assert "INSERT INTO pipeline_metrics" in q and p[2] == "scan_outcome"
    meta = json.loads(p[4])
    assert all(isinstance(v, (int, float, bool)) or v is None for v in meta.values())

    def _rota(*a, **k):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(te, "execute_sql_write", _rota)
    te.registrar_scan_outcome("u1", te.resumen_de_correcciones({}))          # no lanza


def test_registrar_vision_scan(monkeypatch):
    escritas = []
    monkeypatch.setattr(te, "execute_sql_write", lambda q, p=None, **kw: escritas.append((q, p)))
    te.registrar_vision_scan(None, duracion_ms=4321.9, resultado={"is_food": True, "calories": 300,
                                                                  "photo_kind": "plato"}, purpose="diary")
    q, p = escritas[0]
    assert p[0] is None and p[2] == "vision_scan_resultado" and p[3] == 4321
    assert json.loads(p[4]) == {"resultado": "ok", "photo_kind": "plato", "purpose": "diary"}


def _payload(**extra):
    return diary.ConsumedMealRequest(user_id="u1", meal_name="Arroz con pollo", calories=600, **extra)


def test_consumed_encola_la_senal_solo_si_se_guardo(monkeypatch):
    monkeypatch.setattr(diary, "_persist_consumed_meal", lambda **kw: {"success": True, "already_logged": False,
                                                                       "meal_id": "m1"})
    tareas = diary.BackgroundTasks()
    diary.api_log_consumed_meal(_payload(scan_meta={"cambiados": 1, "kcal_ia": 600, "kcal_final": 600}),
                                background_tasks=tareas, verified_user_id="u1")
    assert [t.func for t in tareas.tasks] == [te.registrar_scan_outcome]


def test_consumed_no_cuenta_un_registro_repetido(monkeypatch):
    monkeypatch.setattr(diary, "_persist_consumed_meal", lambda **kw: {"success": True, "already_logged": True,
                                                                       "meal_id": "m1"})
    tareas = diary.BackgroundTasks()
    diary.api_log_consumed_meal(_payload(scan_meta={"cambiados": 1}), background_tasks=tareas, verified_user_id="u1")
    assert tareas.tasks == []


def test_scan_meta_basura_no_rompe_el_registro(monkeypatch):
    monkeypatch.setattr(diary, "_persist_consumed_meal", lambda **kw: {"success": True, "already_logged": False,
                                                                       "meal_id": "m1"})
    r = diary.api_log_consumed_meal(_payload(scan_meta="no-es-un-dict"), background_tasks=diary.BackgroundTasks(),
                                    verified_user_id="u1")
    assert r["success"] is True
