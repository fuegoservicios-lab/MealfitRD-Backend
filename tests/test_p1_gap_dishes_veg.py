# -*- coding: utf-8 -*-
"""[P1-GAP-DISHES-VEG · 2026-09-06] Los platos que el embudo de cobertura demostró que faltaban, y el
etiquetado de alérgenos que se corrigió al medirlos.

El 5 de septiembre el embudo dio dos ceros: un vegano en Puerto Rico no tenía NI UNA cena, y un vegetariano sin
lácteos ni huevo no tenía desayuno en cuatro de las seis bibliotecas. Este test ancla que esos ceros no vuelven,
y que la corrección de alérgenos —«Leche de coco» y «Mantequilla de maní» se declaraban LÁCTEOS— sigue puesta.

Lo que NO ancla: cifras concretas de cobertura. Suben y bajan con el catálogo. Lo que se fija es el suelo: que
ninguna combinación de dieta y alergia deje una franja sin ningún plato."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as dr  # noqa: E402

_SNAPSHOTS = (_BACKEND / "data" / "registry" / "dish_registry_pr_v1.json").exists()
pytestmark = pytest.mark.skipif(not _SNAPSHOTS, reason="snapshots no compilados")


def _funnel():
    spec = importlib.util.spec_from_file_location("cf", _BACKEND / "scripts" / "coverage_funnel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cobertura():
    return _funnel().correr()


def test_ninguna_franja_se_queda_sin_platos(cobertura):
    """El suelo del producto: si una combinación deja una franja en cero, el plan no se puede armar."""
    ceros = []
    for pid, p in cobertura["perfiles"].items():
        for etiqueta, por_slot in p["escenarios"].items():
            for slot, d in por_slot.items():
                if d["elegibles"] == 0:
                    ceros.append(f"{pid}/{etiqueta}/{slot}")
    assert not ceros, f"combinaciones sin ningún plato: {ceros}"


def test_el_vegano_puertorriqueno_tiene_cena(cobertura):
    """El cero que abrió este trabajo."""
    n = cobertura["perfiles"]["puertorico_criolla"]["escenarios"]["vegano"]["cena"]["elegibles"]
    assert n >= 3, f"cenas veganas en PR: {n}"


@pytest.mark.parametrize("pid", ["dominican_criolla", "puertorico_criolla", "mexico_casera", "colombia_casera"])
def test_el_vegetariano_sin_lacteos_ni_huevo_desayuna(cobertura, pid):
    """Las cuatro cocinas criollas cuyo desayuno se sostenía entero sobre huevo y queso."""
    n = cobertura["perfiles"][pid]["escenarios"]["veg + sin lácteos ni huevo"]["desayuno"]["elegibles"]
    assert n >= 3, f"{pid}: {n} desayunos"


@pytest.mark.parametrize("nombre, no_debe", [
    ("Leche de coco", "lacteos"),
    ("Leche de almendras", "lacteos"),
    ("Mantequilla de maní", "lacteos"),
    ("Crema de coco", "lacteos"),
])
def test_la_bebida_vegetal_no_es_un_lacteo(nombre, no_debe):
    assert no_debe not in dr.allergen_classes_for([nombre]), nombre


@pytest.mark.parametrize("nombre", ["Leche", "Leche evaporada", "Mantequilla", "Queso crema", "Yogurt"])
def test_el_lacteo_de_verdad_sigue_marcado(nombre):
    """La exención es quirúrgica: quitar de más aquí sirve un alérgeno a quien lo declaró."""
    assert "lacteos" in dr.allergen_classes_for([nombre]), nombre


def test_la_bebida_vegetal_conserva_sus_otros_alergenos():
    """Una bebida de almendras deja de ser láctea pero NO deja de ser fruto seco."""
    assert "frutos secos" in dr.allergen_classes_for(["Leche de almendras"])
    assert "mani" in dr.allergen_classes_for(["Mantequilla de maní"])


def test_un_plato_con_lacteo_y_bebida_vegetal_sigue_siendo_lacteo():
    assert "lacteos" in dr.allergen_classes_for(["Leche de coco", "Queso blanco"])


def test_los_platos_nuevos_resuelven_y_no_traen_animal():
    """Un plato cuyo constituyente no resuelve nace inservible: el compilador no lo deja en `ok`."""
    snap = dr.load_registry("PR") or {}
    nuevas = [t for t in (snap.get("templates") or [])
              if t["name"] in ("Mofongo de plátano verde con garbanzos guisados",
                               "Sopón de garbanzos con calabaza y papa",
                               "Guiso de lentejas con yautía y ñame")]
    assert len(nuevas) == 3, [t["name"] for t in (snap.get("templates") or [])][:5]
    for t in nuevas:
        assert t["status"] == "ok", t["name"]
        al = (t.get("intrinsic_risk_attributes") or {}).get("allergens") or []
        assert "lacteos" not in al and "huevo" not in al, (t["name"], al)
