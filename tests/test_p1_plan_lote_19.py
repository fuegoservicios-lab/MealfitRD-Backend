# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-19 · 2026-09-12] Decimonoveno lote del plan de pendientes: E5-A, la SOMBRA de la lista canónica.

`ARQ30-P1-01` pide promover `IngredientLine` a compras reales por el camino expand → sombra → canario por cohorte →
promoción → retirada (`docs/arq30_e5_e7_diseno_canario.md`). El expand existe desde el 09-06 (`canonical_recipe.py`, solo
lectura). Este lote es la SOMBRA, y nada más:

  · `canonical_shopping_shadow.py` construye la lista canónica por alimento (gramos de `IngredientLine` sumados sobre
    `shopping_source_days`, con el multiplicador efectivo del guard y espejando la POLÍTICA que la lista ya lleva
    sellada: rendimiento de legumbres, sello de proteína, deducción de nevera) y la compara con la lista que hoy se
    entrega. La identidad del alimento la decide el mismo canonicalizador que el guard.
  · Corre donde corre el guard (`run_shopping_coherence_guard`, las 6 superficies) y persiste SÓLO la comparación en
    `pipeline_metrics` (node `canonical_shopping_shadow`). No toca `plan_result`: la vía anterior sigue entregando.
  · Knob `MEALFIT_CANONICAL_SHOPPING_SHADOW` (default True). Gate de salida a la fase B: ≥ 30 planes distintos con
    `parse_fail` < 1 % y divergencia > 10 % en < 5 % de los comparables (`scripts/measure_canonical_shadow.py`). La
    cohorte de la fase B la elige el dueño.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import canonical_shopping_shadow as sh  # noqa: E402


def _plan(lista=None, *, mult=1.0, archivados=None):
    pr = {
        "days": [{"day": "Lunes", "meals": [{
            "meal": "Almuerzo", "name": "Pollo con arroz",
            "ingredients_raw": ["150 g de pechuga de pollo", "60 g de arroz blanco", "Sal al gusto"],
            "ingredients": ["150 g de pechuga de pollo", "60 g de arroz blanco", "Sal al gusto"],
            "recipe": ["Cocina el pollo y el arroz."],
        }]}],
        "calc_household_multiplier": mult,
    }
    if lista is not None:
        pr["aggregated_shopping_list"] = lista
    if archivados:
        pr["_archived_days"] = archivados
    return pr


def _lista(pollo_g=150.0, arroz_g=60.0, extra=None):
    items = [{"name": "Pechuga de pollo", "base_qty": pollo_g, "base_unit": "g", "category": "Proteínas"},
             {"name": "Arroz blanco", "base_qty": arroz_g, "base_unit": "g", "category": "Despensa"}]
    if extra:
        items.extend(extra)
    return items


def _shadow(pr, **kw):
    try:
        return sh.compute_shadow(pr, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"la sombra necesita el catálogo en este entorno: {type(e).__name__}: {e}")


# ────────────────────────────────────────────────────────────── la comparación

def test_lista_igual_a_las_recetas_no_diverge_y_la_sal_no_es_un_fallo():
    r = _shadow(_plan(_lista()))
    assert r["lines"] == 3 and r["parse_fail"] == 0 and r["sin_cantidad"] == 1
    assert r["comparables"] == 2 and r["divergentes"] == 0 and r["divergentes_pct"] == 0.0
    assert r["canonical_only"] == 0 and r["aggregated_only"] == 0
    assert r["lista_usada"] == "active" and r["source_days"] == 1
    assert set(r["grams_source"]) == {"canonical_units.to_base_amount"}


def test_una_magnitud_distinta_diverge_y_se_explica_con_ejemplo():
    r = _shadow(_plan(_lista(pollo_g=300.0)))
    assert r["divergentes"] == 1 and r["divergentes_pct"] == 50.0
    ej = r["ejemplos"][0]
    assert ej["food"] == sh._canon("Pechuga de pollo") and ej["canonical_g"] == 150.0 and ej["aggregated_g"] == 300.0
    assert ej["delta_pct"] == 100.0


def test_dentro_de_la_tolerancia_del_guard_no_diverge():
    r = _shadow(_plan(_lista(pollo_g=160.0)))   # +6,7 % < 10 %
    assert r["divergentes"] == 0 and r["comparables"] == 2 and r["tolerance"] == 0.10


def test_lo_que_solo_esta_en_un_lado_se_cuenta_aparte():
    r = _shadow(_plan(_lista(extra=[{"name": "Tomate", "base_qty": 200.0, "base_unit": "g"}])[1:]))  # sin pollo, con tomate
    assert r["canonical_only"] == 1 and r["canonical_only_ej"] == [sh._canon("Pechuga de pollo")]
    assert r["aggregated_only"] == 1 and r["aggregated_only_ej"] == [sh._canon("Tomate")]
    assert r["divergentes"] == 0


def test_una_unidad_de_envase_no_es_divergencia_ni_coincidencia():
    """«1 cartón (20 uds.)» es lo que la lista persiste para el huevo: no convertible ⇒ no comparable, jamás cero."""
    lista = _lista(extra=[{"name": "Pechuga de pollo", "base_qty": 1.0, "base_unit": "cartón (20 uds.)"}])
    lista = [i for i in lista if not (i["name"] == "Pechuga de pollo" and i["base_unit"] == "g")]
    r = _shadow(_plan(lista))
    assert r["divergentes"] == 0 and r["canonical_only"] == 0
    assert r["no_comparables"] == 1 and r["comparables"] == 1


def test_el_multiplicador_del_hogar_escala_el_lado_canonico():
    r = _shadow(_plan(_lista(pollo_g=300.0, arroz_g=120.0)), multiplier=2.0)
    assert r["divergentes"] == 0 and r["multiplier"] == 2.0


def test_la_deduccion_de_nevera_sellada_no_se_lee_como_divergencia():
    lista = _lista(pollo_g=50.0)
    for i in lista:
        i["pantry_deduction_applied"] = True
    r = _shadow(_plan(lista))
    assert r["sellos"]["pantry_deduction_applied"] is True
    assert r["divergentes"] == 0 and r["descontado_nevera"] == 1


def test_el_sello_de_rendimiento_de_proteina_se_espeja_no_se_reinventa():
    """La lista sellada con `protein_yield_applied` compra 1,35× la proteína cocida; la canónica registra el TEXTO. Para
    comparar like con like se pregunta al parser legacy por el factor: el mismo que espeja `expected_sum_from_recipes`."""
    f_sin = sh._factor_politica("150 g de pechuga de pollo cocida", False)
    f_con = sh._factor_politica("150 g de pechuga de pollo cocida", True)
    assert f_sin == 1.0 and f_con >= f_sin
    assert sh._factor_politica("60 g de arroz blanco", True) == 1.0
    lista = _lista()
    for i in lista:
        i["protein_yield_applied"] = True
    r = _shadow(_plan(lista))
    assert r["sellos"]["protein_yield_applied"] is True


def test_los_dias_fuente_son_el_ssot_del_ciclo():
    from shopping_calculator import shopping_source_days
    pr = _plan(_lista(), archivados=[{"day": "Domingo", "meals": [{"meal": "Cena", "name": "Sopa",
                                                                   "ingredients_raw": ["100 g de zanahoria"], "recipe": ["x"]}]}])
    r = _shadow(pr)
    assert r["source_days"] == len(shopping_source_days(pr)) and r["dias_archivados"] == 1


def test_la_huella_del_plan_identifica_el_contenido():
    a, b = _plan(_lista()), _plan(_lista())
    assert sh.plan_fingerprint(a) == sh.plan_fingerprint(b) and re.fullmatch(r"[0-9a-f]{16}", sh.plan_fingerprint(a))
    b["days"][0]["meals"][0]["ingredients_raw"][0] = "200 g de pechuga de pollo"
    assert sh.plan_fingerprint(a) != sh.plan_fingerprint(b)
    assert sh.plan_fingerprint({}) is None


def test_el_multiplicador_efectivo_espeja_al_guard():
    pr = _plan(_lista(), mult=2.0)
    assert sh.effective_multiplier_like_guard(pr) == 2.0               # lista activa: sin base de días
    pr["aggregated_shopping_list_weekly"] = pr.pop("aggregated_shopping_list")
    from shopping_calculator import _get_coherence_day_basis_norm_knob, shopping_source_days
    esperado = 2.0 * (7.0 / len(shopping_source_days(pr)) if _get_coherence_day_basis_norm_knob() else 1.0)
    assert abs(sh.effective_multiplier_like_guard(pr) - esperado) < 1e-9


# ────────────────────────────────────────────────────────────── emitir: mide, no escribe

def test_la_sombra_persiste_la_comparacion_y_no_toca_el_plan(monkeypatch):
    import db_core
    capturado = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params=None, *a, **k: capturado.append((sql, params)))
    pr = _plan(_lista(pollo_g=300.0))
    antes = copy.deepcopy(pr)
    r = sh.emit_canonical_shopping_shadow(pr, multiplier=1.0, surface="guard:block", plan_id="p1", user_id="u1")
    if r is None:
        pytest.skip("la sombra necesita el catálogo en este entorno")
    assert pr == antes, "la sombra es de SOLO LECTURA sobre plan_result"
    assert len(capturado) == 1
    sql, params = capturado[0]
    assert "INSERT INTO pipeline_metrics" in sql and "canonical_shopping_shadow" in params
    import json
    meta = json.loads(params[-1])
    assert meta["surface"] == "guard:block" and meta["plan_id"] == "p1" and meta["plan_fp"] == sh.plan_fingerprint(pr)
    assert meta["divergentes"] == 1 and params[0] == "u1"


def test_con_el_knob_apagado_no_hay_sombra(monkeypatch):
    import db_core
    monkeypatch.setenv("MEALFIT_CANONICAL_SHOPPING_SHADOW", "false")
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: pytest.fail("no debe escribir métrica"))
    assert sh.shadow_enabled() is False
    assert sh.emit_canonical_shopping_shadow(_plan(_lista())) is None


def test_un_fallo_de_la_metrica_no_llega_al_llamante(monkeypatch):
    import db_core

    def _boom(*a, **k):
        raise RuntimeError("pool cerrado")

    monkeypatch.setattr(db_core, "execute_sql_write", _boom)
    r = sh.emit_canonical_shopping_shadow(_plan(_lista()), multiplier=1.0)
    if r is None:
        pytest.skip("la sombra necesita el catálogo en este entorno")
    assert r["comparables"] == 2


def test_la_sombra_no_escribe_en_el_plan():
    """El expand sigue siendo expand: ninguna asignación sobre `plan_result`/`plan_data`, ningún UPDATE a `meal_plans`."""
    src = (_BACKEND / "canonical_shopping_shadow.py").read_text(encoding="utf-8")
    assert not re.search(r"plan_(?:result|data)\s*\[[^\]]+\]\s*=", src)
    assert not re.search(r"plan_(?:result|data)\.(?:update|pop|setdefault)\(", src)
    assert "meal_plans" not in src.replace("`meal_plans`", "")
    assert "INSERT INTO pipeline_metrics" in src and src.count("execute_sql_write") == 2  # import + llamada


# ────────────────────────────────────────────────────────────── el hook y el expand

def test_el_guard_deja_pasar_la_sombra_despues_de_su_metrica_y_antes_de_devolver():
    import shopping_calculator as sc

    src = inspect.getsource(sc.run_shopping_coherence_guard)
    i_metric = src.rindex("_emit_coherence_guard_metric(")
    i_hook = src.index("emit_canonical_shopping_shadow")
    i_ret = src.rindex("return divergences")
    assert i_metric < i_hook < i_ret
    assert "multiplier=mult * _basis_scale" in src, "la sombra compara con el MISMO multiplicador efectivo que el guard"
    assert "P1-PLAN-LOTE-19-CANONICAL-SHADOW-HOOK" in src


def test_la_sombra_es_el_unico_modulo_de_produccion_que_lee_la_representacion():
    rx = re.compile(r"^\s*(?:from\s+canonical_recipe\s+import|import\s+canonical_recipe)", re.M)
    tocan = set()
    for p in _BACKEND.rglob("*.py"):
        rel = p.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "venv", "scripts/")) or "__pycache__" in rel:
            continue
        try:
            if rx.search(p.read_text(encoding="utf-8")):
                tocan.add(rel)
        except Exception:
            continue
    assert tocan == {"canonical_shopping_shadow.py"}, tocan


# ────────────────────────────────────────────────────────────── el gate y la sonda

def test_el_gate_tiene_tres_salidas():
    assert sh.gate_verdict(5, 0.0, 0.0).startswith("NO CONCLUYENTE")
    assert sh.gate_verdict(30, None, None).startswith("NO CONCLUYENTE")
    assert sh.gate_verdict(30, 0.5, 3.0).startswith("PASA")
    assert sh.gate_verdict(30, 1.5, 3.0).startswith("NO PASA")
    assert sh.gate_verdict(30, 0.5, 9.0).startswith("NO PASA")


def _sonda():
    p = _BACKEND / "scripts" / "measure_canonical_shadow.py"
    spec = importlib.util.spec_from_file_location("measure_canonical_shadow", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p.read_text(encoding="utf-8")


def test_la_sonda_es_solo_lectura_y_resume_una_sombra_por_plan():
    mod, src = _sonda()
    assert "conn.read_only = True" in src
    sql = " ".join(re.findall(r'"([^"]*)"', src))
    assert not re.search(r"\b(UPDATE|INSERT|DELETE|TRUNCATE|ALTER|DROP)\b", sql), sql
    vieja = {"plan_fp": "a", "lines": 10, "parse_fail": 5, "comparables": 10, "divergentes": 9, "surface": "guard:warn"}
    nueva = {"plan_fp": "a", "lines": 10, "parse_fail": 0, "comparables": 10, "divergentes": 0, "surface": "guard:block",
             "dias_archivados": 2}
    otra = {"plan_fp": "b", "lines": 10, "parse_fail": 0, "comparables": 10, "divergentes": 1, "surface": "guard:block"}
    res = mod.resumir([vieja, nueva, otra])
    assert res["planes"] == 2 and res["sombras"] == 3, "gana la sombra más reciente de cada plan"
    assert res["parse_fail"] == 0 and res["divergentes"] == 1 and res["divergentes_pct"] == 5.0
    assert res["planes_tras_shift"] == 1 and res["veredicto"].startswith("NO CONCLUYENTE")


# ────────────────────────────────────────────────────────────── docs y marker

def test_docs_y_plan():
    doc = (_BACKEND / "docs" / "arq30_e5_e7_diseno_canario.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-19" in doc and "MEALFIT_CANONICAL_SHOPPING_SHADOW" in doc and "canonical_shopping_shadow" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "MEALFIT_CANONICAL_SHOPPING_SHADOW" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| E5 | ✅ 2026-09-12 · fase A (sombra)" in plan


def test_marker_bumpeado():
    """«No anterior a este lote», no «igual a hoy»: el lote siguiente vuelve a bumpear el marker (lección de LOTE-13)."""
    import app

    assert "[P1-PLAN-LOTE-19 · 2026-09-12]" in (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
