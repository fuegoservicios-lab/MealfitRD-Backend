# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-36 · 2026-09-13] B6 · E5 · D7: tres mediciones que dependen de tráfico real, re-ejecutadas y anotadas.

B6 tenía la pregunta escrita («¿bajan las 9 ventanas rotas con la memoria entre días?») y no el instrumento:
`scripts/measure_variety_windows.py` cuenta, sobre los planes persistidos (solo lectura), las ventanas deslizantes de 7
días que sirven una misma comida por encima del tope de su política, con el conteo de `deterministic_day._conteo_ventana`.
E5 (sombra canónica) y D7 (volumen de `magnitude_undersupply`) ya tenían script: se re-ejecutan y se anota la cifra y la
fecha. «Sin muestra» es un resultado, con la fecha de vuelta; nada se «arregla» aquí.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vw():
    spec = importlib.util.spec_from_file_location("measure_variety_windows", _BACKEND / "scripts" / "measure_variety_windows.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dia(*claves):
    return {"meals": [({"_template_id": c} if c.startswith("t") else {"name": c}) for c in claves]}


def test_la_clave_es_la_plantilla_y_si_no_el_nombre_normalizado(vw):
    assert vw.clave({"_template_id": "do.x", "name": "Otro"}) == "plantilla:do.x"
    assert vw.clave({"_recipe_template_id": "do.y"}) == "plantilla:do.y"
    assert vw.clave({"name": "  Mangú   con HUEVO "}) == vw.clave({"name": "mangu con huevo"})
    assert vw.clave({}) is None


def test_una_plantilla_tres_veces_en_siete_dias_rompe_el_tope_de_dos(vw):
    dias = [_dia("t1", "a"), _dia("t1", "b"), _dia("t1", "c"), _dia("d"), _dia("e"), _dia("f"), _dia("g"), _dia("h")]
    r = vw.ventanas(dias, 2)
    assert r["ventanas"] == 2 and r["rotas"] == 1 and r["sobre_tope"] == ["plantilla:t1"]
    assert vw.ventanas(dias, 3)["rotas"] == 0


def test_un_plan_corto_no_tiene_ventana_y_no_cuenta_como_sin_roturas(vw):
    r = vw.ventanas([_dia("t1")] * 6, 2)
    assert r == {"dias": 6, "ventanas": 0, "rotas": 0, "sobre_tope": []}


def test_los_dias_entregados_incluyen_los_archivados_en_orden(vw):
    pd = {"_archived_days": [{"date": "2026-09-02", "meals": []}, {"date": "2026-09-01", "meals": []}],
          "days": [{"date": "2026-09-03", "meals": []}]}
    assert [d["date"] for d in vw.dias_entregados(pd)] == ["2026-09-01", "2026-09-02", "2026-09-03"]


def test_las_cohortes_separan_antes_y_despues_de_la_memoria_entre_dias(vw):
    filas = [("a" * 36, "2026-09-05", {"days": [_dia("t1")] * 8}), ("b" * 36, "2026-09-12", {"days": [_dia("x")] * 8})]
    res = vw.medir(filas)
    assert set(res["cohortes"]) == {"antes_determinista", "despues_modelo"}
    assert res["cohortes"]["antes_determinista"]["rotas"] == 2 and res["cohortes"]["despues_modelo"]["rotas"] == 2


def test_solo_lectura(vw):
    src = inspect.getsource(vw)
    assert "conn.read_only = True" in src
    assert not re.search(r"\b(INSERT INTO|UPDATE meal_plans|DELETE FROM)\b", src)


def test_el_artefacto_declara_corte_y_linea_base():
    a = json.loads((_BACKEND / "scripts" / "data" / "variety_windows_2026_09_13.json").read_text(encoding="utf-8"))
    assert a["corte"] == "2026-09-11" and "9 ventanas" in a["linea_base"] and "cohortes" in a


def test_docs_plan_marker():
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert plan.count("P1-PLAN-LOTE-36") >= 3, "las filas B6, E5 y D7 anotan la re-medición"
    assert "P1-PLAN-LOTE-36" in (_BACKEND / "docs" / "deterministic_day.md").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 36
