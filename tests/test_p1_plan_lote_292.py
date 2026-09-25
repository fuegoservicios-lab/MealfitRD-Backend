# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-292 · 2026-09-25] El formulario separa «¿qué tomas?» de «¿te recomendamos?», lo que toma va a la
Alacena, y la tarjeta del dashboard desaparece. Spec §4. Tooltip-anchor: P1-PLAN-LOTE-292"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ── Task 11: normalizar y el generador ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fd,esperado", [
    ({"currentSupplements": ["creatine"], "recommendSupplements": True}, {"toma": ["creatine"], "recomendar": True}),
    ({"includeSupplements": True, "selectedSupplements": ["whey_protein"]}, {"toma": ["whey_protein"], "recomendar": False}),
    ({"includeSupplements": True, "selectedSupplements": []}, {"toma": [], "recomendar": True}),
    ({"includeSupplements": False}, {"toma": [], "recomendar": False}),
    ({}, {"toma": [], "recomendar": False}),
])
def test_normalizar(fd, esperado):
    import suplementos
    assert suplementos.normalizar_suplementos(fd) == esperado


def test_el_prompt_incluye_lo_que_toma_y_nunca_recomienda_quemadores():
    from prompts.plan_generator import build_supplements_context
    txt = build_supplements_context({"currentSupplements": ["creatine"], "recommendSupplements": True})
    assert "Creatina" in txt and "NUNCA recomiendes" in txt
    for prohibido in ("Quemador", "Pre-Entreno", "BCAA"):
        assert prohibido in txt.split("NUNCA recomiendes", 1)[1]
    solo = build_supplements_context({"currentSupplements": ["fat_burner"], "recommendSupplements": False})
    assert "Quemador" in solo and "lo toma el usuario" in solo.lower()


def test_nada_que_tomar_ni_recomendar_prohibe_suplementos():
    from prompts.plan_generator import build_supplements_context
    viejo = build_supplements_context({"includeSupplements": False})
    nuevo = build_supplements_context({"currentSupplements": [], "recommendSupplements": False})
    assert nuevo == viejo and viejo


def test_suplementos_dia_con_el_formulario_nuevo():
    import suplementos_dia as sd
    fd = {"currentSupplements": ["creatine"], "recommendSupplements": True}
    assert sd.elegidos(fd) == ["creatine"]
    plan = {"days": [{"supplements": [{"name": "Omega-3 (Aceite de Pescado)", "dose": "1 g", "timing": "Almuerzo", "reason": "x"},
                                      {"name": "Quemador de Grasa Termogénico", "dose": "1", "timing": "x", "reason": "x"}]}]}
    sd.completar(plan, fd)
    nombres = [s["name"] for s in plan["days"][0]["supplements"]]
    assert any("Creatina" in n for n in nombres)          # lo suyo se asegura
    assert any("Omega" in n for n in nombres)             # una recomendación con respaldo se queda
    assert not any("Quemador" in n for n in nombres)      # nunca un quemador recomendado


def test_suplementos_dia_toma_sin_recomendar_es_ni_mas_ni_menos():
    import suplementos_dia as sd
    plan = {"days": [{"supplements": [{"name": "Omega-3 (Aceite de Pescado)", "dose": "1 g", "timing": "x", "reason": "x"}]}]}
    sd.completar(plan, {"currentSupplements": ["creatine"], "recommendSupplements": False})
    assert [s["name"] for s in plan["days"][0]["supplements"]] == ["Creatina Monohidrato"]


def test_el_orquestador_y_el_coach_leen_el_formulario_nuevo():
    go = _src("graph_orchestrator.py")
    assert 'if not form_data.get("includeSupplements"):' not in go
    assert "suplementos_activos(form_data)" in go
    ag = _src("agent.py")
    assert 'form_data.get("includeSupplements")' not in ag and ag.count("normalizar_suplementos(form_data)") == 2


def test_el_router_valida_lo_que_toma():
    from routers import plans
    src = _src("routers/plans.py")
    assert '"currentSupplements"' in src


# ── Task 12: lo que toma, a la Alacena ──────────────────────────────────────────────────────────────────────────────

def test_endpoint_guarda_los_que_toma_y_enciende_la_nevera(monkeypatch):
    import suplementos
    from routers import user_data
    guardados = []
    monkeypatch.setattr(suplementos, "buscar", lambda uid, n: {"id": 1} if n == "Creatina Monohidrato" else None)
    monkeypatch.setattr(suplementos, "guardar", lambda *a, **k: guardados.append((a, k)) or {"ok": True})
    r = user_data.guardar_suplementos_del_formulario(
        user_data.SuplementosFormulario(claves=["creatine", "omega3", "no_existe", "omega3"]), user_id="u")
    assert r == {"guardados": 1}                      # creatina ya estaba; «no_existe» fuera; omega3 una vez
    a, k = guardados[0]
    assert a[1] == "Omega-3 (Aceite de Pescado)" and k["forzar_nevera"] is True and k["usar_estimado"] is False


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 292 and m.group(2) >= "2026-09-25"


# ── Revisión final (fix pass) ───────────────────────────────────────────────────────────────────────────────────────

def test_C2_C3_los_potes_viven_en_su_propio_espacio_de_unidades(monkeypatch):
    """Un suplemento y un alimento con el mismo nombre y unidad no pueden chocar en ON CONFLICT (user_id,
    ingredient_name, unit): la unidad del pote va con prefijo `sup_` (la real queda en serving_unit)."""
    import suplementos
    import db_core
    import nevera_opcional
    escritos = []
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda uid, forzar=False: "activa")
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: escritos.append((sql, params)))
    suplementos.guardar("u", "Creatina Monohidrato", None, 60, "g", None, "estimado", "creatine")
    sql, params = escritos[0]
    assert params[3] == "sup_g" and params[5] == "g"
    assert "kind = 'supplement'" not in sql.split("DO UPDATE", 1)[1], "el upsert no debe convertir filas de alimento"


def test_C1_la_pantalla_lista_los_potes_aunque_no_se_sepan_las_porciones():
    src = _src("routers/user_data.py")
    assert "OR ui.kind = 'supplement'" in src


def test_C1_I1_tener_potes_cuenta_como_usar_la_nevera():
    import nevera_opcional
    assert "i.kind = 'supplement'" in nevera_opcional._SQL_APAGAR
    assert "kind = 'supplement'" in _src("cron_tasks.py").split("def _plan_freeze_sweep", 1)[1][:6000]


def test_C1_sin_porciones_conocidas_no_avisa_de_pocas(monkeypatch):
    import tools
    import suplementos
    import db
    import db_inventory
    fila = {"id": 7, "ingredient_name": "Creatina", "quantity": 0, "serving_unit": "g",
            "serving_label": {"gramos_porcion": 5, "kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}}
    monkeypatch.setattr(suplementos, "buscar", lambda uid, n: fila)
    monkeypatch.setattr(suplementos, "descontar", lambda uid, fid, n: 0.0)
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: "meal-1")
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_nota_comidas_sin_registrar", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, cal, d: mt)
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory", lambda *a, **k: None)
    out = tools.log_consumed_meal.func("u", "Creatina", calories=0, protein=0, suplemento="Creatina", porciones=1,
                                       meal_type="merienda")
    assert "te quedan" not in out


def test_I2_solo_recomendar_nunca_deja_un_bcaa():
    import suplementos_dia as sd
    plan = {"days": [{"supplements": [{"name": "Aminoácidos BCAA / EAA", "dose": "x", "timing": "x", "reason": "x"},
                                      {"name": "Omega-3 (Aceite de Pescado)", "dose": "x", "timing": "x", "reason": "x"}]},
                     {"supplements": []}]}
    sd.completar(plan, {"currentSupplements": [], "recommendSupplements": True})
    for d in plan["days"]:
        assert [s["name"] for s in d["supplements"]] == ["Omega-3 (Aceite de Pescado)"]


def test_I3_etiqueta_sin_gramos_o_con_calorias_imposibles():
    import suplementos
    assert suplementos.etiqueta_valida({"kcal": 1200, "protein_g": 24, "carbs_g": 3, "fats_g": 1}) is None
    assert suplementos.etiqueta_valida({"gramos_porcion": 200, "kcal": 900, "protein_g": 24, "carbs_g": 3, "fats_g": 1}) is None
    assert suplementos.etiqueta_valida({"kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5}) is not None


def test_I5_porciones_negativas_no_rellenan_el_pote(monkeypatch):
    import suplementos
    import db_core
    llamadas = []
    monkeypatch.setattr(db_core, "execute_sql_query", lambda sql, p, **k: llamadas.append(p) or {"quantity": 3})
    suplementos.descontar("u", 7, -2)
    assert llamadas[0][0] == 0.0


def test_I4_toda_escritura_por_nombre_filtra_los_suplementos():
    """El guard de lecturas no veía UPDATE: `consume_inventory_items_completely` ponía a 0 un pote por nombre."""
    import ast as _ast
    faltan = []
    for py in _BACKEND.rglob("*.py"):
        if {"tests", "scripts", "migrations", "__pycache__", "docs"} & set(py.relative_to(_BACKEND).parts):
            continue
        texto = py.read_text(encoding="utf-8")
        if "user_inventory" not in texto:
            continue
        lineas = texto.splitlines()
        for nodo in _ast.walk(_ast.parse(texto)):
            if isinstance(nodo, _ast.Constant) and isinstance(nodo.value, str) \
                    and re.search(r"\bUPDATE\s+(public\.)?user_inventory\b", nodo.value, re.I):
                prev = "\n".join(lineas[max(0, nodo.lineno - 7):nodo.lineno])
                if not re.search(r"kind\s*=\s*'(food|supplement)'", nodo.value) and "[SUPLEMENTOS-OK:" not in prev:
                    faltan.append(f"{py.relative_to(_BACKEND).as_posix()}:{nodo.lineno}")
    assert not faltan, "UPDATE de user_inventory sin kind ni marcador: " + ", ".join(faltan)


def test_I4_el_inventario_solo_trae_potes_a_quien_los_pide(monkeypatch):
    from routers import user_data
    import db
    sqls = []
    monkeypatch.setattr(db, "execute_sql_query", lambda sql, p, **k: sqls.append(sql) or [])
    user_data._fetch_inventory("u", True)
    user_data._fetch_inventory("u", True, incluir_suplementos=True)
    assert "ui.kind = 'food'" in sqls[0] and "OR ui.kind = 'supplement'" not in sqls[0]
    assert "OR ui.kind = 'supplement'" in sqls[1]
    front = _BACKEND.parent / "frontend" / "src" / "pages" / "Pantry.jsx"
    if front.exists():
        assert "/api/inventory?incluir_suplementos=1" in front.read_text(encoding="utf-8")
