# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-813 · 2026-09-29] El escudo pre-INSERT bajaba la banda porque recibía un estado que ya no era la
salida de la cadena de calidad.

La cadena de la cola de assemble (`apply_plan_quality_finalize_chain`, surface `assemble-tail`) se declaraba «tras la
ÚLTIMA mutación de assemble», pero detrás seguían cinco mutadores de contenido (ingrediente fantasma de los pasos,
«queso» genérico con nombre, lácteo que promete el nombre, cocido→seco y fusión de duplicados). Sus líneas no pasaban
por los topes, el cerrador ni el contrato hasta el pre-INSERT, que recortaba y sacaba días de banda.

  1. Con `MEALFIT_ASSEMBLE_MUTATORS_BEFORE_CHAIN` (default True) los cinco corren ANTES de la cadena; apagado, en su
     sitio viejo. Los re-autofix tardíos, el reconciliador display↔raw y el AUTO-PATCH se quedan detrás.
  2. La fila `clinical_band_final` lleva `final_score` como double (el float4 de `confidence` fabricaba empates que
     contaban como bajadas), `plan_id`/`run_id` y `input_equals_chain_out`.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]


def _mdc():
    import mutadores_de_contenido as m
    return m


# ─────────────── 1. el orden en la cola de assemble ───────────────
def test_los_mutadores_van_antes_de_la_cadena_de_assemble_tail():
    """tooltip-anchor: P1-PLAN-LOTE-813-MUTADORES-ANTES"""
    asm = inspect.getsource(go.assemble_plan_node)
    i_antes = asm.index('_mdc.en_posicion(result, "antes"')
    i_chain = asm.index('_apqfc, result, surface="assemble-tail"')
    i_despues = asm.index('_mdc.en_posicion(result, "despues"')
    i_list = asm.index("# Calcular shopping lists")
    assert i_antes < i_chain < i_despues < i_list


def test_ya_no_se_llaman_sueltos_en_assemble():
    asm = inspect.getsource(go.assemble_plan_node)
    for fn in ("_repair_declared_but_unlisted_ingredients(", "_repair_name_phantom_dairy(",
               "_normalize_cooked_grain_lines(", "_merge_duplicate_food_lines(",
               '_ccr.nombrar_quesos_genericos(result.get("days")'):
        assert fn not in asm, f"{fn} sigue suelto en assemble: correría dos veces o fuera del knob"


def test_knob_apagado_es_el_sitio_viejo():
    """Apagado: detrás de los re-autofix tardíos y antes del reconciliador display↔raw, como antes."""
    asm = inspect.getsource(go.assemble_plan_node)
    i_chain = asm.index('_apqfc, result, surface="assemble-tail"')
    i_fs = asm.index('_fruit_savory_autofix(result.get("days")', i_chain)
    i_trace = asm.index('_trace_misalign(result.get("days"), "pre_shopping_passes")')
    i_despues = asm.index('_mdc.en_posicion(result, "despues"')
    i_rc = asm.index('_reconcile_display_raw_lines(result.get("days")')
    assert i_chain < i_fs < i_trace < i_despues < i_rc


def test_lo_que_se_queda_detras_de_la_cadena():
    """Los re-autofix tardíos detectan lo que la cadena reintroduce y el reconciliador ya tiene gemelo en la cadena."""
    asm = inspect.getsource(go.assemble_plan_node)
    i_chain = asm.index('_apqfc, result, surface="assemble-tail"')
    assert asm.index("_protein_repeat_autofix(result.get", i_chain) > i_chain
    assert asm.index('_reconcile_display_raw_lines(result.get("days")') > i_chain


def test_el_orden_interno_de_los_cinco_no_cambia():
    src = inspect.getsource(_mdc().aplicar)
    orden = [src.index(s) for s in ("_repair_declared_but_unlisted_ingredients(", "nombrar_quesos_genericos(",
                                    "_repair_name_phantom_dairy(", "_normalize_cooked_grain_lines(",
                                    "_merge_duplicate_food_lines(")]
    assert orden == sorted(orden)


def test_el_comentario_de_la_cadena_ya_no_dice_ultima_mutacion():
    asm = inspect.getsource(go.assemble_plan_node)
    i_chain = asm.index('_apqfc, result, surface="assemble-tail"')
    bloque = asm[max(0, i_chain - 2500):i_chain]
    assert "tras la ÚLTIMA mutación de" not in bloque
    assert "P1-PLAN-LOTE-813" in bloque


def test_knob_registrado_default_true():
    m = _mdc()
    assert m.ASSEMBLE_MUTATORS_BEFORE_CHAIN is True
    assert 'ASSEMBLE_MUTATORS_BEFORE_CHAIN = _env_bool("MEALFIT_ASSEMBLE_MUTATORS_BEFORE_CHAIN", True)' in \
        inspect.getsource(m)
    assert "MEALFIT_ASSEMBLE_MUTATORS_BEFORE_CHAIN" in go.get_knobs_registry_snapshot()


# ─────────────── 2. funcional: la línea del mutador pasa por los topes ───────────────
class _DB:
    def macros_from_ingredient_string(self, _s):
        return {}


@pytest.fixture
def _catalogo(monkeypatch):
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE",
                        {"queso blanco": "Queso blanco", "queso": "Queso blanco", "huevo": "Huevo"}, raising=False)
    yield
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", None, raising=False)


def _plan():
    meal = {"meal": "Almuerzo", "name": "Revoltillo criollo", "cals": 500, "protein": 30, "carbs": 20, "fats": 30,
            "ingredients": ["2 huevos", "1 tomate"], "ingredients_raw": ["2 huevos", "1 tomate"],
            "recipe": ["Mise en place: bate los huevos.", "Agrega 250 g de queso blanco rallado y revuelve."]}
    return {"days": [{"day": 1, "meals": [meal]}]}


def _cadena(result):
    """El tope de queso que la cadena corre en su bucle de topes (`_cap_cheese_dumps_final`)."""
    go._cap_cheese_dumps_final(result["days"], db=_DB())


def _cola(result):
    """La cola de assemble tal como la ordena el knob: antes → cadena → después."""
    m = _mdc()
    m.en_posicion(result, "antes")
    _cadena(result)
    m.en_posicion(result, "despues")


def _queso(result):
    return [x for x in result["days"][0]["meals"][0]["ingredients"] if "queso" in x.lower()]


def test_la_linea_fantasma_sale_recortada_por_los_topes(_catalogo, monkeypatch):
    monkeypatch.setattr(_mdc(), "ASSEMBLE_MUTATORS_BEFORE_CHAIN", True)
    r = _plan()
    _cola(r)
    q = _queso(r)
    assert len(q) == 1 and q[0].startswith(f"{go.MEAL_CHEESE_CAP_G} g"), q
    assert r.get("_phantom_ingredients_repaired"), "la telemetría del mutador sigue en el plan"


def test_knob_apagado_orden_viejo_la_linea_escapa_al_tope(_catalogo, monkeypatch):
    monkeypatch.setattr(_mdc(), "ASSEMBLE_MUTATORS_BEFORE_CHAIN", False)
    r = _plan()
    _cola(r)
    assert _queso(r) == ["250 g de Queso blanco"], "apagado: la línea nace detrás de la cadena, sin tope"


def test_en_posicion_corre_una_sola_vez(_catalogo, monkeypatch):
    m = _mdc()
    llamadas = []
    monkeypatch.setattr(m, "aplicar", lambda result, ck=None: llamadas.append(1))
    for on in (True, False):
        monkeypatch.setattr(m, "ASSEMBLE_MUTATORS_BEFORE_CHAIN", on)
        llamadas.clear()
        m.en_posicion({}, "antes")
        m.en_posicion({}, "despues")
        assert llamadas == [1]


# ─────────────── 3. el instrumento de la fila clinical_band_final ───────────────
def test_huella_de_contenido_ignora_estampados_de_fecha():
    m = _mdc()
    a = _plan()
    b = json.loads(json.dumps(a))
    b["days"][0]["date"] = "2026-09-29"
    b["days"][0]["meals"][0]["_display"] = {"en-US": {"name": "Scramble"}}
    assert m.huella_days(a["days"]) == m.huella_days(b["days"])
    b["days"][0]["meals"][0]["ingredients"][0] = "3 huevos"
    assert m.huella_days(a["days"]) != m.huella_days(b["days"])


def test_input_equals_chain_out():
    m = _mdc()
    pd = _plan()
    m.entrada_cadena(pd, {}, "assemble-tail")
    m.salida_cadena(pd, "assemble-tail")
    assert pd.get(m._CLAVE_SALIDA, {}).get("surface") == "assemble-tail"
    # sin tocar days → el pre-INSERT recibe la salida de la cadena
    m.entrada_cadena(pd, {"plan_id": "p-1"}, "pre-INSERT")
    ctx = pd[m._CLAVE_CTX]
    assert ctx["input_equals_chain_out"] is True and ctx["plan_id"] == "p-1"
    assert m._CLAVE_SALIDA not in pd, "la huella no se persiste"
    m.salida_cadena(pd, "pre-INSERT")
    assert m._CLAVE_CTX not in pd and m._CLAVE_SALIDA not in pd


def test_input_distinto_si_algo_muto_days_tras_la_cadena():
    m = _mdc()
    pd = _plan()
    m.entrada_cadena(pd, {}, "assemble-tail")
    m.salida_cadena(pd, "assemble-tail")
    pd["days"][0]["meals"][0]["ingredients"].append("30 g de Queso blanco")
    m.entrada_cadena(pd, {}, "pre-INSERT")
    assert pd[m._CLAVE_CTX]["input_equals_chain_out"] is False


def test_sin_cadena_previa_es_none_y_el_chunk_no_deja_huella():
    m = _mdc()
    pd = _plan()
    m.entrada_cadena(pd, {}, "pre-INSERT")
    assert pd[m._CLAVE_CTX]["input_equals_chain_out"] is None
    m.salida_cadena(pd, "pre-INSERT")
    m.entrada_cadena(pd, {}, "chunk-T1 semana 2")
    m.salida_cadena(pd, "chunk-T1 semana 2")
    assert m._CLAVE_SALIDA not in pd and m._CLAVE_CTX not in pd


def test_la_fila_lleva_final_score_double_plan_id_run_id_e_igualdad(monkeypatch):
    import db_core
    filas = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: filas.append(p) or True)
    m = _mdc()
    pd = _plan()
    pd["_run_id"] = "run-9"
    m.entrada_cadena(pd, {}, "assemble-tail")
    m.salida_cadena(pd, "assemble-tail")
    m.entrada_cadena(pd, {"plan_id": "plan-7"}, "pre-INSERT")
    band = {"score": 11 / 12, "cells_in_band": 11, "cells_total": 12}
    go._emit_clinical_band_final_metric(band, 11 / 12, pd, "user-1", "pre-INSERT")
    assert filas, "la fila se emitió"
    meta = json.loads(filas[0][-1])
    assert meta["final_score"] == 11 / 12, "double, no float4: 0.9166… no puede salir como 0.91666 < pre"
    assert meta["plan_id"] == "plan-7" and meta["run_id"] == "run-9"
    assert meta["input_equals_chain_out"] is True and meta["chain_out_surface"] == "assemble-tail"


def test_el_escudo_abre_y_cierra_el_contexto_y_el_relleno_pasa_el_plan_id():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("def _finalize_plan_data_for_insert(")
    j = src.index("\ndef apply_plan_quality_finalize_chain(", i)
    body = src[i:j]
    i_in = body.index("entrada_cadena(")
    assert i_in < body.index("_fpc(_pd[\"days\"]"), "la huella de entrada se toma antes del primer pase"
    assert body.rindex("salida_cadena(") > body.index("retirar_prohibidos("), "la de salida, tras el último"
    k = src.index("def fill_placeholder_meal_plan_atomic(")
    fill = src[k:src.index("\ndef ", k + 10)]
    assert 'safe["plan_id"] = plan_id' in fill and 'safe.pop("plan_id", None)' in fill


def test_el_mapa_de_tramos_no_carga_la_cadena_a_los_mutadores(monkeypatch):
    m = _mdc()
    monkeypatch.setattr(m, "aplicar", lambda result, ck=None: ck("pre_phantom_repair"))
    marcas = []
    monkeypatch.setattr(m, "ASSEMBLE_MUTATORS_BEFORE_CHAIN", True)
    m.en_posicion({}, "antes", marcas.append)
    assert marcas == ["pre_phantom_repair", "pre_assemble_tail_chain"]
    marcas.clear()
    monkeypatch.setattr(m, "ASSEMBLE_MUTATORS_BEFORE_CHAIN", False)
    m.en_posicion({}, "despues", marcas.append)
    assert marcas == ["pre_phantom_repair"]
