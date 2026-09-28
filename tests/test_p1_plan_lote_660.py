# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-660 · 2026-09-28] Con la Nevera exigida, una o dos compras pequeñas no tumban el bloque.

Decisión del dueño (28-sep). Producción 14-27 sep: tras el lote 199, los «ERRORES DE DESPENSA» que quedaban eran «15 g
de pepino», «½ plátano maduro», «1 rebanada de pan de trigo (30 g), ¼ pechuga de pollo (≈44 g)» — y cada uno costaba
reintentos del pipeline y, en el worker, la pausa del bloque.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compras_pequenas as cp  # noqa: E402

_MSG = ("ERRORES DE DESPENSA HALLADOS OBLIGANDO A CORREGIR:\n- Ingredientes COMPLETAMENTE INEXISTENTES en inventario: "
        "{}.\nCorrige tu respuesta bajando las porciones estrictamente numéricas al límite exacto, O "
        "eliminando/sustituyendo ingredientes.")


@pytest.mark.parametrize("lineas", [
    ["15 g de pepino"],
    ["½ plátano maduro"],
    ["1 rebanada de pan de trigo (30 g)", "15 g de pepino"],   # dos alimentos
    ["15 g de pepino", "30 g de pepino"],                       # un solo alimento: se suma
    ["1 rebanada de pan integral", "1 rebanada de pan integral"],
    ["1 pepino"],                                                # una pieza: la compra suelta del colmado
])
def test_compras_pequenas_de_produccion(lineas):
    assert cp.pequenas(lineas) == lineas
    assert cp.tolerar(_MSG.format(", ".join(lineas))) is True


@pytest.mark.parametrize("lineas", [
    ["250 g de edamame cocido"],                                 # no es pequeña
    ["1 rebanada de pan de trigo (30 g)", "¼ pechuga de pollo (≈44 g)"],   # la proteína nunca es compra menor
    ["100g camarones"],
    ["¾ pechuga de pollo (≈150 g)"],
    ["3 zanahorias medianas"],                                   # más de una pieza
    ["1 taza de fresas"],                                        # volumen sin gramos: no se sabe
    ["½ guineo maduro (74 g)", "15 g de arroz blanco crudo", "15 g de berenjena"],   # tres alimentos
])
def test_lo_que_no_es_pequeno_sigue_fallando(lineas):
    assert cp.pequenas(lineas) is None
    msg = _MSG.format(", ".join(lineas))
    assert cp.tolerar(msg) == msg


def test_las_cantidades_de_la_nevera_no_se_relajan():
    msg = _MSG.format("15 g de pepino").replace(
        "Corrige", "- Excediste tus CANTIDADES (Tu inventario restringe esto matemáticamente): [300 g de pollo].\nCorrige")
    assert cp.tolerar(msg) == msg


def test_knob_a_cero_es_la_conducta_previa(monkeypatch):
    monkeypatch.setenv("MEALFIT_PANTRY_SMALL_PURCHASES_MAX", "0")
    msg = _MSG.format("15 g de pepino")
    assert cp.tolerar(msg) == msg


_NEVERA = ["1000 g de pechuga de pollo", "1000 g de arroz blanco", "500 g de tomate"]


def _dia(extra):
    return {"day": 8, "meals": [{"meal": "Almuerzo", "name": "Pollo con arroz y ensalada",
                                 "ingredients": ["150 g de pechuga de pollo", "80 g de arroz blanco", "50 g de tomate"] + extra,
                                 "recipe": ["Mise en place: corta.", "Montaje: sirve."]}]}


def test_la_guarda_post_merge_del_worker_acepta_la_compra_pequena():
    import cron_tasks as ct
    ok, _ = ct._validate_merged_days_against_pantry([_dia(["15 g de pepino"])], _NEVERA)
    assert ok is True
    ok, viol = ct._validate_merged_days_against_pantry([_dia(["250 g de edamame cocido"])], _NEVERA)
    assert ok is False and "edamame" in viol[0]["error"]


def test_el_plato_lleva_su_nota_una_vez():
    d = _dia(["15 g de pepino"])
    assert cp.marcar([d], _NEVERA) == 1
    m = d["meals"][0]
    assert m["_compra_pequena"] == ["15 g de pepino"]
    assert m["recipe"][-1] == "🛒 Compra pequeña (no está en tu Nevera; va en tu lista de compras): 15 g de pepino."
    cp.marcar([d], _NEVERA)
    assert sum(1 for p in m["recipe"] if p.startswith("🛒")) == 1


def test_todas_las_guardas_del_bloque_usan_la_misma_regla():
    ct = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    for var in ("result", "existence", "qty_check", "_live_check", "_val_result", "_qty_result", "_safe"):
        assert re.search(rf'\b{var} = __import__\("compras_pequenas"\)\.tolerar\({var}\)', ct), var
    assert 'val_result = __import__("compras_pequenas").tolerar(val_result)' in go
    assert '__import__("compras_pequenas").marcar(' in go and '__import__("compras_pequenas").marcar(' in ct
