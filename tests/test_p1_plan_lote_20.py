# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-20 · 2026-09-12] Vigésimo lote del plan de pendientes: E6 · ARQ30-P1-03, la asignación del horizonte por
comidas VIABLES — primero medida en sombra, después el allocator mínimo tras un knob apagado.

`horizon.build_blueprint` asigna la familia de proteína del día con round-robin (`pool[d % len(pool)]`) y fija 3
candidatos por día × franja con los filtros del usuario. Una franja sin candidato no dejaba rastro: la clave no existía
en `candidates` y el modelo improvisaba. Medido con `scripts/measure_horizon_slots.py` (6 países × 25 perfiles del landing
× 4 escenarios de compra): en compra mensual SIN congelador, del día 9 en adelante Res/Cerdo/Pollo no tienen almuerzo
que aguante hasta el día, y la franja quedaba vacía — mientras otra familia del pool sí tenía plato. Eso es el
acoplamiento que ARQ30-P1-03 describe, y por eso:

  · el blueprint anota `registry.empty_slots` (día, franja, familia, cocina, `rescuable_by_family`), y la rebanada
    lleva los de sus días — sólo cuando los hay, para que un blueprint sin huecos no cambie de forma ni de hash;
  · con `MEALFIT_HORIZON_VIABLE_FAMILY` (default OFF) la familia del día se mueve, determinista y en orden rotado
    desde la propuesta, a la primera del pool con plato en TODAS las franjas del día (`family_reassignments`);
    mueve `d["protein"]`, así que candidatos, prompt, sembrador y gate de fidelidad ven la misma familia. Si ninguna
    cubre, se conserva la del round-robin. Apagado ⇒ byte-idéntico a antes. La cohorte la elige el dueño.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402

_EFF = {"diet": {"type": "balanced", "allergies": [], "exclusions": []}, "market_country": "DO",
        "recurrence": {"global_mode": "balanced"}, "shopping": {"main_cycle_days": 7}, "policy_hash": "p"}


def _registry_falso(monkeypatch, sin_plato):
    """Un registry en el que (franja, familia) ∈ `sin_plato` no tiene candidato y todo lo demás sí. `family=None`
    (cualquier familia) siempre tiene plato: la biblioteca no está vacía, el round-robin eligió mal."""
    import dish_registry as dr

    def _tc(country, slot, family=None, *, k=6, **kw):
        if family is not None and (slot, family) in sin_plato:
            return []
        return [{"template_id": f"tpl_{slot}_{family or 'any'}_{i}", "name": f"{slot} {family or 'any'} {i}"} for i in range(min(k, 3))]

    monkeypatch.setattr(dr, "template_candidates", _tc)
    monkeypatch.setattr(dr, "registry_hash", lambda c: "abcd1234")
    monkeypatch.setattr(dr, "registry_snapshot_version", lambda: "v-test")


def _bp(monkeypatch, sin_plato=(), viable="false", dias=3):
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", viable)
    _registry_falso(monkeypatch, set(sin_plato))
    return horizon.build_blueprint(dict(_EFF), total_days=dias, meals_per_day=3)


def _fam_del_dia(bp, i):
    return bp["days"][i]["protein"]


# ────────────────────────────────────────────────────────────── el diagnóstico (knob apagado)

def test_una_franja_sin_candidato_queda_anotada_y_dice_si_otra_familia_la_rescataria(monkeypatch):
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    bp = _bp(monkeypatch, sin_plato={("lunch", fam0)})
    reg = bp["registry"]
    assert "0:lunch" not in reg["candidates"] and "0:breakfast" in reg["candidates"]
    assert reg["empty_slots"] == [{"day_index": 0, "slot": "lunch", "family": fam0, "culture_country": "DO",
                                   "rescuable_by_family": True}]
    assert _fam_del_dia(bp, 0) == fam0, "apagado, el round-robin no se toca"
    assert "family_reassignments" not in reg and "viable_family" not in reg


def test_sin_huecos_el_blueprint_no_cambia_de_forma(monkeypatch):
    bp = _bp(monkeypatch)
    assert "empty_slots" not in bp["registry"] and "viable_family" not in bp["registry"]
    sl = horizon.slice_for_chunk(bp, 0, 3)
    assert "empty_slots" not in sl["registry"] and "viable_family" not in sl["registry"]


def test_la_rebanada_lleva_solo_los_huecos_de_sus_dias(monkeypatch):
    bp0 = _bp(monkeypatch, dias=4)
    fam2 = _fam_del_dia(bp0, 2)
    bp = _bp(monkeypatch, sin_plato={("dinner", fam2)}, dias=4)
    assert [e["day_index"] for e in bp["registry"]["empty_slots"]] == [2]
    assert "empty_slots" not in horizon.slice_for_chunk(bp, 0, 2)["registry"]
    assert [e["day_index"] for e in horizon.slice_for_chunk(bp, 2, 2)["registry"]["empty_slots"]] == [2]


def test_un_hueco_de_biblioteca_se_distingue_de_un_mal_reparto(monkeypatch):
    """Si NINGUNA familia tiene plato en la franja, no es el round-robin: faltan plantillas."""
    import dish_registry as dr
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", "false")
    _registry_falso(monkeypatch, set())
    _orig = dr.template_candidates

    def _tc(country, slot, family=None, *, k=6, **kw):
        return [] if slot == "dinner" else _orig(country, slot, family, k=k, **kw)

    monkeypatch.setattr(dr, "template_candidates", _tc)
    bp = horizon.build_blueprint(dict(_EFF), total_days=1, meals_per_day=3)
    e = bp["registry"]["empty_slots"]
    assert len(e) == 1 and e[0]["slot"] == "dinner" and e[0]["rescuable_by_family"] is False


# ────────────────────────────────────────────────────────────── el allocator mínimo (knob encendido)

def test_con_el_knob_la_familia_del_dia_se_mueve_a_una_con_plato_en_todas_sus_franjas(monkeypatch):
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    pool = bp0["protein_pool"]
    assert len(pool) >= 2, pool
    bp = _bp(monkeypatch, sin_plato={("lunch", fam0)}, viable="true")
    reg = bp["registry"]
    esperada = (pool[pool.index(fam0) + 1:] + pool[:pool.index(fam0)])[0]
    assert _fam_del_dia(bp, 0) == esperada, "la siguiente del pool en orden rotado desde la propuesta"
    assert reg["viable_family"] is True
    assert len(reg["family_reassignments"]) == 1
    r = reg["family_reassignments"][0]
    assert (r["day_index"], r["from"], r["to"], r["franjas_cubiertas"], r["de"]) == (0, fam0, esperada, 3, 3)
    assert "empty_slots" not in reg and "0:lunch" in reg["candidates"]
    assert all(c.endswith(f"_{esperada}_0") or esperada in c for c in reg["candidates"]["0:lunch"][:1])
    # los demás días conservan su round-robin
    assert _fam_del_dia(bp, 1) == _fam_del_dia(bp0, 1) and _fam_del_dia(bp, 2) == _fam_del_dia(bp0, 2)


def test_la_reasignacion_viaja_en_la_rebanada_y_el_dia_movido_tambien(monkeypatch):
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    bp = _bp(monkeypatch, sin_plato={("lunch", fam0)}, viable="true")
    sl = horizon.slice_for_chunk(bp, 0, 1)
    assert sl["registry"]["viable_family"] is True
    assert sl["registry"]["family_reassignments"][0]["from"] == fam0
    assert sl["days"][0]["protein"] == bp["days"][0]["protein"] != fam0
    sl2 = horizon.slice_for_chunk(bp, 1, 2)
    assert "family_reassignments" not in sl2["registry"] and sl2["registry"]["viable_family"] is True


def test_un_hueco_de_desayuno_no_condena_al_almuerzo_y_la_cena(monkeypatch):
    """Exigir «todas las franjas» dejaba 120 franjas rescatables sin rescatar (medido): si el desayuno no tiene plato en
    ninguna familia, la familia que cubre MÁS franjas gana igual, y el hueco de biblioteca queda anotado como tal."""
    import dish_registry as dr
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    pool = bp0["protein_pool"]
    siguiente = (pool[pool.index(fam0) + 1:] + pool[:pool.index(fam0)])[0]
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", "true")
    _registry_falso(monkeypatch, {("lunch", fam0), ("dinner", fam0)})
    _orig = dr.template_candidates

    def _tc(country, slot, family=None, *, k=6, **kw):
        return [] if slot == "breakfast" else _orig(country, slot, family, k=k, **kw)   # desayuno: nadie tiene plato

    monkeypatch.setattr(dr, "template_candidates", _tc)
    bp = horizon.build_blueprint(dict(_EFF), total_days=1, meals_per_day=3)
    reg = bp["registry"]
    assert _fam_del_dia(bp, 0) == siguiente
    r = reg["family_reassignments"][0]
    assert (r["from"], r["to"], r["franjas_cubiertas"], r["de"]) == (fam0, siguiente, 2, 3)
    assert [e["slot"] for e in reg["empty_slots"]] == ["breakfast"] and reg["empty_slots"][0]["rescuable_by_family"] is False
    assert "0:lunch" in reg["candidates"] and "0:dinner" in reg["candidates"]


def test_si_ninguna_familia_cubre_se_conserva_el_round_robin_y_queda_anotado(monkeypatch):
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    sin = {("lunch", f) for f in bp0["protein_pool"]}   # ninguna familia del pool tiene almuerzo
    bp = _bp(monkeypatch, sin_plato=sin, viable="true")
    assert _fam_del_dia(bp, 0) == fam0
    assert bp["registry"]["empty_slots"][0]["slot"] == "lunch"
    assert bp["registry"]["empty_slots"][0]["rescuable_by_family"] is True   # con family=None sí hay plato (fuera del pool)
    assert "family_reassignments" not in bp["registry"]


def test_apagado_es_byte_identico_salvo_el_diagnostico(monkeypatch):
    """La única diferencia entre knob off y la conducta anterior es la clave `empty_slots`, y sólo cuando hay huecos."""
    a = _bp(monkeypatch)
    b = _bp(monkeypatch)
    assert horizon.blueprint_hash(a) == horizon.blueprint_hash(b)
    assert "empty_slots" not in a["registry"]


def test_el_allocator_minimo_es_determinista(monkeypatch):
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    a = _bp(monkeypatch, sin_plato={("lunch", fam0)}, viable="true")
    b = _bp(monkeypatch, sin_plato={("lunch", fam0)}, viable="true")
    assert horizon.blueprint_hash(a) == horizon.blueprint_hash(b)


def test_el_knob_nace_apagado_y_la_regla_vive_donde_se_fijan_los_candidatos():
    # [P1-PLAN-LOTE-21 · 2026-09-12] Nació off y así se midió; el dueño lo encendió por defecto en el lote 21. El
    # docstring cuenta las dos cosas (ver test_p1_plan_lote_21.py para el default vivo).
    doc = horizon.viable_family_enabled.__doc__ or ""
    assert "Nació off" in doc and "P1-PLAN-LOTE-21" in doc and "byte-idéntico" in doc
    src = inspect.getsource(horizon._registry_block_for_country)
    assert "P1-PLAN-LOTE-20-VIABLE-FAMILY" in src and "P1-PLAN-LOTE-20-EMPTY-SLOTS" in src
    assert src.index('d["protein"] = fam = _mejor') < src.index("for slot in (d.get(\"slots\") or []):")


# ────────────────────────────────────────────────────────────── la sonda

def _sonda():
    p = _BACKEND / "scripts" / "measure_horizon_slots.py"
    spec = importlib.util.spec_from_file_location("measure_horizon_slots", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p.read_text(encoding="utf-8")


def test_la_sonda_cuenta_un_blueprint_y_da_tres_veredictos(monkeypatch):
    mod, src = _sonda()
    assert not re.search(r"\b(UPDATE|INSERT|DELETE|TRUNCATE|ALTER|DROP)\b", " ".join(re.findall(r'"([^"]*)"', src)))
    bp0 = _bp(monkeypatch)
    fam0 = _fam_del_dia(bp0, 0)
    c = mod.contar(_bp(monkeypatch, sin_plato={("lunch", fam0)}))
    assert c["franjas"] == 9 and c["vacias"] == 1 and c["rescatables_por_familia"] == 1 and c["con_3"] == 8
    base = {"blueprints": 1, "franjas": 9, "vacias": 0, "rescatables_por_familia": 0, "hueco_biblioteca": 0}
    assert mod.veredicto(base).startswith("SIN HUECOS")
    assert mod.veredicto({**base, "vacias": 2, "rescatables_por_familia": 2}).startswith("ALLOCATOR URGE")
    assert mod.veredicto({**base, "vacias": 2, "hueco_biblioteca": 2}).startswith("ALLOCATOR NO URGE")
    assert mod.veredicto({**base, "blueprints": 0}).startswith("NO CONCLUYENTE")
    assert ("semanal", "weekly", "limited", None, 7) in mod.ESCENARIOS and len(mod.PAISES) == 6


# ────────────────────────────────────────────────────────────── docs y marker

def test_docs_y_plan():
    doc = (_BACKEND / "docs" / "arq30_e5_e7_diseno_canario.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-20" in doc and "MEALFIT_HORIZON_VIABLE_FAMILY" in doc and "empty_slots" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "MEALFIT_HORIZON_VIABLE_FAMILY" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| E6 | ✅ 2026-09-12 · P1-03 medido + allocator mínimo" in plan  # [P1-PLAN-LOTE-21] el sufijo «(knob off)» cambió al encenderlo


def test_marker_bumpeado():
    """«No anterior a este lote», no «igual a hoy»: el lote siguiente vuelve a bumpear el marker (lección de LOTE-13)."""
    import app

    assert "[P1-PLAN-LOTE-20 · 2026-09-12]" in (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
