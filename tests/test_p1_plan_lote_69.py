# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-69 · 2026-09-16] El suelo cocinable, también como ÚLTIMA palabra.

Salió de generar un plan REAL para comprobar el lote 68: dos líneas seguían por debajo del piso
(`10 g de Maíz dulce en granos`, `10 g de lechosa`) en una comida que YA llevaba `_portion_floor_adjusted`. No era
el arreglo del 68: es un error de ORDEN. El último suelo del motor está anidado en `if _rq_fixed:` (sólo corre si el
recheck post-quantize rebalanceó algo) y después siguen actuando el micro-closer, el recorte de carbos, el de
grasas, el autofix de sodio y el refill de gain-muscle. Medido ejecutando el recorte de carbos sobre ese día: deja
`5 g de Maíz dulce` y `10 g de ñame`.

Es el MISMO error que el repo ya cerró dos veces —`P1-CAPS-LAST-WORD` para los techos y `P1-RECONCILE-LAST-WORD`
para el display↔raw— por una tercera puerta, así que se cierra igual: re-ejecutar el suelo al final del finalize.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_GO = _BACKEND / "graph_orchestrator.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class _DbFalso:
    """Doble con kcal por gramo: suficiente para que el suelo decida headroom sin base."""

    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return None
        g = float(m.group(1).replace(",", "."))
        return {"kcal": g * 1.2, "protein": 0.0, "carbs": g * 0.2, "fats": 0.0}


def _dia(gramos):
    return {"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Arepitas de Maíz con Lechosa", "cals": 400,
        "protein": 20, "carbs": 40, "fats": 10,
        "ingredients": [f"{gramos} g de lechosa", "70 g de habas cocidas", "1 cdta de aceite vegetal"],
        "ingredients_raw": [f"{gramos} g de lechosa", "70 g de habas cocidas", "1 cdta de aceite vegetal"],
        "recipe": ["Mise en place: pica la lechosa.", "El Toque de Fuego: cocina las arepitas 8 minutos.",
                   "Montaje: sirve."]}]}


def _gramos(dia, aguja="lechosa"):
    linea = [str(x) for x in dia["meals"][0]["ingredients"] if aguja in str(x)]
    assert linea, f"la línea de {aguja} desapareció"
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", linea[0]).group(1).replace(",", "."))


# ─────────────────────────────── el orden: caps → suelo → reconciliador

def test_el_suelo_corre_despues_de_los_techos_y_antes_del_reconciliador():
    """Los techos sólo BAJAN, así que el suelo va detrás (si no, subiría algo que el cap va a recortar); y va
    ANTES del reconciliador display↔raw, para que la lista compre la cantidad ya corregida."""
    src = _src(_GO)
    i_caps = src.index("_clw = _cap_unrealistic_portions")
    i_suelo = src.index("_flw = _floor_subservible_portions")
    # el literal `P1-RECONCILE-LAST-WORD` aparece antes en un comentario de otra sección: se ancla en la LLAMADA
    i_rec = src.index("_rlw = ")
    assert i_caps < i_suelo < i_rec, (i_caps, i_suelo, i_rec)


def test_el_ancla_y_el_knob_estan_en_el_fuente():
    src = _src(_GO)
    assert "P1-PLAN-LOTE-69-SUELO-ULTIMA-PALABRA" in src
    assert 'FLOOR_LAST_WORD = _env_bool("MEALFIT_FLOOR_LAST_WORD", True)' in src


def test_el_knob_apaga_la_ultima_palabra_sin_redeploy():
    import graph_orchestrator as go
    assert go.FLOOR_LAST_WORD is True
    src = _src(_GO)
    i = src.index("_flw = _floor_subservible_portions")
    assert "if FLOOR_LAST_WORD and PORTION_SHRINK_FLOOR_ENABLED:" in src[i - 400:i]


# ─────────────────────────────── la invariante que el arreglo apoya

def test_el_suelo_repara_lo_que_un_pase_posterior_dejo_sub_servible():
    from graph_orchestrator import _floor_subservible_portions, PORTION_SHRINK_FLOOR_G
    d = _dia(10)                     # lo que deja el recorte de carbos, medido en el plan real
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    assert _gramos(d) >= float(PORTION_SHRINK_FLOOR_G)
    assert len(d["meals"][0]["ingredients"]) == 3, "reparar no puede costar una línea"


def test_re_ejecutarlo_es_no_op_sobre_un_plato_ya_servible():
    from graph_orchestrator import _floor_subservible_portions
    d = _dia(120)
    n1 = _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    n2 = _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    assert (n1, n2) == (0, 0) and _gramos(d) == 120.0


def test_el_recorte_de_carbos_y_despues_el_suelo(tmp_path):
    """La secuencia REAL del defecto, con catálogo: recorte que deja la línea corta → el suelo la repara."""
    from shopping_calculator import get_master_ingredients
    if not (get_master_ingredients() or []):
        pytest.skip("sin catálogo: el recorte de carbos no puede medir macros (se mide en la pata con base)")
    from nutrition_db import IngredientNutritionDB
    from graph_orchestrator import (_trim_day_carbs_to_target, _floor_subservible_portions,
                                    PORTION_SHRINK_FLOOR_G)
    db = IngredientNutritionDB()
    d = _dia(40)
    carbs = sum(float(m.get("carbs") or 0) for m in d["meals"])
    _trim_day_carbs_to_target(d["meals"], carbs * 0.5, db)
    _floor_subservible_portions([d], day_kcal_target=2000, db=db)
    assert _gramos(d) >= float(PORTION_SHRINK_FLOOR_G)


# ─────────────────────────────── docs y marker

def test_docs_y_marker():
    doc = _src(_BACKEND / "docs" / "culinary_coherence.md")
    assert "P1-PLAN-LOTE-69" in doc and "última palabra" in doc
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src(_BACKEND / "app.py"), re.M)
    assert m and int(m.group(1)) >= 69 and m.group(2) >= "2026-09-16"
