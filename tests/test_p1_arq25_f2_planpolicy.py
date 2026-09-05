"""[P1-ARQ25-F2-PLANPOLICY · 2026-09-02] Fase 2 del roadmap 2.5 — PlanPolicy en modo shadow.

Ancla los entregables de la fase:
  A. adapters formulario V1 → `requested` (stapleFoods→anclas, groceryDuration→ciclo, dietType
     vía `canonicalize_diet_type`, país vía `country_for_form_data`);
  B. compilador requested → effective con `relaxations[]` (campo, pedido, aplicado, reason code,
     evidencia) siguiendo la precedencia §6.3; decisión #4: presupuesto DURO ⇒ `waiting_user`,
     nunca modificación silenciosa; orientativo sin precios;
  C. `policy_hash` estable: misma entrada ⇒ mismo hash (orden de anclas irrelevante);
  D. `template_id` acuñado al cargar las 6 bibliotecas: 100 % cobertura, únicos, estables, alias;
  E. medición shadow: distancia política ↔ plan V1;
  F. persistencia: el run lleva requested/effective/relaxations/hash (autenticados) y TODO plan
     lleva `_plan_policy` (invitados incluidos) — parser sobre los call sites; knob off ⇒ no-op.
"""
from __future__ import annotations
import re

import json
import os
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
pp = pytest.importorskip("plan_policy")

_KNOWN = ["Huevo", "Avena", "Plátano verde", "Pollo", "Arroz", "Queso blanco"]
_CTX = {"budget_floor_dop": 12000.0, "pricing_mode": "native", "known_ingredients": _KNOWN}


def _form(**over):
    base = {
        "country": "DO", "dietType": "balanced", "allergies": ["Ninguna"], "medicalConditions": ["Ninguna"],
        "dislikes": ["Ninguno"], "stapleFoods": ["Huevo", "Avena"], "groceryDuration": "monthly",
        "budget": "medium", "budgetCurrency": "DOP", "householdSize": 1, "cookingTime": "30min",
    }
    base.update(over)
    return base


# ───────────────────────────────────────────── A. adapters
def test_adapters_map_the_v1_form():
    req = pp.policy_from_form(_form(dietType="vegetariana", groceryDuration="biweekly", householdSize="3"))
    assert req["schema_version"] == 1 and req["market_country"] == "DO"
    assert req["diet"]["type"] == "vegetarian", "dietType pasa por canonicalize_diet_type"
    assert req["shopping"]["main_cycle_days"] == 15 and req["shopping"]["fresh_topup_days"] == 7
    assert [a["ingredient_id"] for a in req["food_anchors"]] == ["huevo", "avena"]
    assert req["food_anchors"][0]["min_per_7d"] == 2 and req["food_anchors"][0]["max_per_7d"] == 7
    assert req["household_size"] == 3 and req["budget"]["mode"] == "hard" and req["budget"]["tier"] == "medium"
    assert req["recurrence"]["global_mode"] == "balanced" and set(req["recurrence"]["slot_modes"]) == set(pp.SLOTS)


def test_adapters_drop_sentinels_and_accept_snake_case_staples():
    req = pp.policy_from_form(_form(stapleFoods=None, staple_foods=["Ninguno", "Huevo", "huevo", " "], allergies=["Ninguna", "Maní"]))
    assert [a["name"] for a in req["food_anchors"]] == ["Huevo"], "sentinelas y duplicados fuera"
    assert req["diet"]["allergies"] == ["Maní"]


def test_ingredient_id_is_a_stable_slug_and_never_replaces_the_name():
    assert pp.ingredient_id_for("Plátano verde") == "platano_verde"
    assert pp.ingredient_id_for("  Huevo ") == pp.ingredient_id_for("huevo") == "huevo"
    assert pp.canonical_name_for("platano_verde", _KNOWN) == "Plátano verde"
    assert pp.canonical_name_for("no_existe", _KNOWN) is None


# ───────────────────────────────────────────── B. compilador y precedencia
def test_allergy_beats_anchor_rank_1():
    req = pp.policy_from_form(_form(allergies=["Mariscos"], stapleFoods=["Camarones", "Huevo"]))
    eff, rels = pp.compile_policy(req, context=_CTX)
    assert [a["name"] for a in eff["food_anchors"]] == ["Huevo"]
    r = [x for x in rels if x["reason_code"] == "anchor_conflicts_allergy"]
    assert r and r[0]["rank"] == 1 and r[0]["requested"] == "Camarones" and r[0]["evidence"]["allergy"] == "Mariscos"


def test_diet_beats_anchor_rank_2():
    req = pp.policy_from_form(_form(dietType="vegana", stapleFoods=["Pollo", "Huevo", "Avena"]))
    eff, rels = pp.compile_policy(req, context=_CTX)
    assert [a["name"] for a in eff["food_anchors"]] == ["Avena"]
    assert sorted(x["requested"] for x in rels if x["reason_code"] == "anchor_conflicts_diet") == ["Huevo", "Pollo"]


def test_market_availability_rank_3_only_when_catalog_is_provided():
    req = pp.policy_from_form(_form(stapleFoods=["Huevo", "Quinoa negra de los Andes"]))
    eff, rels = pp.compile_policy(req, context=_CTX)
    assert [a["name"] for a in eff["food_anchors"]] == ["Huevo"]
    assert any(x["reason_code"] == "anchor_not_in_market" and x["rank"] == 3 for x in rels)
    eff2, rels2 = pp.compile_policy(req, context={})
    assert len(eff2["food_anchors"]) == 2 and "market_check_skipped" in eff2["notes"], "sin catálogo no se inventa evidencia"


def test_hard_budget_below_floor_is_never_silently_changed_decision_4():
    req = pp.policy_from_form(_form(budget="custom", budgetAmount=500, budgetCurrency="DOP"))
    eff, rels = pp.compile_policy(req, context={**_CTX, "budget_amount_dop": 500.0})
    assert eff["budget"]["amount"] == 500 and eff["budget"]["status"] == "below_floor", "la cifra se conserva"
    r = [x for x in rels if x["reason_code"] == "budget_below_floor"]
    assert r and r[0]["action"] == "waiting_user" and r[0]["rank"] == 4
    assert r[0]["evidence"] == {"floor_dop": 12000.0, "amount_dop": 500.0}
    texto = pp.explain_relaxations(rels)
    assert any("12000" in t and "500" in t for t in texto), "la explicación lleva las dos cifras"


def test_hard_budget_ok_when_above_floor():
    req = pp.policy_from_form(_form(budget="custom", budgetAmount=20000, budgetCurrency="DOP"))
    eff, rels = pp.compile_policy(req, context={**_CTX, "budget_amount_dop": 20000.0})
    assert eff["budget"]["status"] == "ok" and not [x for x in rels if x["field"].startswith("budget")]


def test_budget_is_advisory_where_there_are_no_prices():
    req = pp.policy_from_form(_form(country="CO", budget="custom", budgetAmount=100, budgetCurrency="COP"))
    eff, rels = pp.compile_policy(req, context={"pricing_mode": "beta_no_prices", "budget_floor_dop": 12000.0, "budget_amount_dop": 5.0})
    assert eff["budget"]["mode"] == "advisory" and eff["budget"]["status"] == "advisory"
    assert any(x["reason_code"] == "budget_advisory_no_prices" for x in rels)
    assert not any(x["reason_code"] == "budget_below_floor" for x in rels), "sin precios no se bloquea"


def test_no_freezer_no_topup_shortens_the_cycle_rank_4():
    req = pp.policy_from_form(_form(groceryDuration="monthly", freezerMode="none", freshTopup="no"))
    eff, rels = pp.compile_policy(req, context=_CTX)
    assert eff["shopping"]["main_cycle_days"] == 7
    r = [x for x in rels if x["reason_code"] == "cycle_shortened_no_freezer_no_topup"]
    assert r and r[0]["requested"] == 30 and r[0]["applied"] == 7


def test_recurrence_is_clamped_and_anchors_capped_rank_5():
    req = pp.policy_from_form(_form(stapleFoods=[f"Alimento{i}" for i in range(12)]))
    req["food_anchors"][0]["min_per_7d"], req["food_anchors"][0]["max_per_7d"] = 9, 2
    eff, rels = pp.compile_policy(req, context={})
    assert len(eff["food_anchors"]) == pp.MAX_ANCHORS
    assert any(x["reason_code"] == "anchors_capped" for x in rels)
    a0 = eff["food_anchors"][0]
    assert (a0["min_per_7d"], a0["max_per_7d"]) == (2, 2), "min>max se colapsa al max acotado"
    assert any(x["reason_code"] == "recurrence_clamped" for x in rels)


def test_relaxation_records_carry_the_four_explainable_fields():
    req = pp.policy_from_form(_form(allergies=["Huevo"], stapleFoods=["Huevo"]))
    _, rels = pp.compile_policy(req, context=_CTX)
    r = rels[0]
    assert set(r) >= {"field", "requested", "applied", "reason_code", "evidence", "rank", "action"}


# ───────────────────────────────────────────── C. hash
def test_same_input_same_hash_and_anchor_order_irrelevant():
    a = pp.compile_policy(pp.policy_from_form(_form(stapleFoods=["Huevo", "Avena"])), context=_CTX)[0]
    b = pp.compile_policy(pp.policy_from_form(_form(stapleFoods=["Avena", "Huevo"])), context=_CTX)[0]
    assert a["policy_hash"] == b["policy_hash"]
    c = pp.compile_policy(pp.policy_from_form(_form(stapleFoods=["Huevo"])), context=_CTX)[0]
    assert a["policy_hash"] != c["policy_hash"]
    assert pp.policy_hash({**a, "compiled_at": "x", "notes": ["y"]}) == a["policy_hash"], "volátiles fuera del hash"


# ───────────────────────────────────────────── D. template_id
def test_template_id_is_minted_for_all_six_libraries_unique_and_stable():
    cov = pp.template_id_coverage()
    assert set(cov) == {"do", "co", "es", "mx", "pr", "us"}, cov
    for lib, c in cov.items():
        assert c["templates"] > 0 and c["with_id"] == c["templates"] == c["unique"], (lib, c)
    assert sum(c["templates"] for c in cov.values()) >= 338, "al menos las 338 plantillas del roadmap (F7-D subió la barra a ≥80 por biblioteca: 522)"


def test_template_id_golden_and_alias_keeps_id_across_rename():
    t = {"name": "Mangú de plátano verde con huevos revueltos", "base": "platano", "technique": "hervido+majado"}
    tid = pp.mint_template_id(t, "do")
    assert tid.startswith("tpl_") and len(tid) == 16
    assert tid == pp.mint_template_id(dict(t, name="  MANGÚ de plátano verde con HUEVOS revueltos "), "do"), "acentos/mayúsculas/espacios no cambian el id"
    assert tid != pp.mint_template_id(t, "us"), "la biblioteca forma parte del id"
    pp.TEMPLATE_ALIASES["Mangú clásico con huevos"] = t["name"]
    try:
        assert pp.mint_template_id(dict(t, name="Mangú clásico con huevos"), "do") == tid, "un renombre con alias conserva el id"
    finally:
        pp.TEMPLATE_ALIASES.pop("Mangú clásico con huevos", None)


def test_dish_library_attaches_ids_on_load():
    src = (BACKEND / "dish_library.py").read_text(encoding="utf-8")
    assert "attach_template_ids(loaded, library_key_for_path(_path))" in src
    dl = pytest.importorskip("dish_library")
    templates = dl.load_dish_templates()
    assert templates and all(t.get("template_id") and t.get("template_version") == 1 for t in templates)


# ───────────────────────────────────────────── E. shadow
def test_shadow_measures_anchors_exclusions_cycle_and_budget():
    eff, _ = pp.compile_policy(pp.policy_from_form(_form(stapleFoods=["Huevo"], dislikes=["Camarones"])), context=_CTX)
    plan = {"total_days_requested": 30, "budget_reconciliation": {"reference_rd": 20000, "estimated_cycle_rd": 12000},
            "days": [{"day": 1, "meals": [{"name": "x", "ingredients": ["Huevo", "Avena"]}]},
                     {"day": 2, "meals": [{"name": "y", "ingredients": [{"name": "Plátano verde"}, "Camarones"]}]},
                     {"day": 3, "meals": [{"name": "z", "ingredients": ["Huevos revueltos"]}]}]}
    m = pp.measure_plan_against_policy(plan, eff)
    assert m["days_measured"] == 3 and m["cycle_match"] is True and m["budget_over"] is False
    a = m["anchors"][0]
    assert a["ingredient_id"] == "huevo" and a["days_present"] == 2 and a["per_7d"] == 4.67 and a["ok"] is True
    assert m["exclusion_violations"] == ["Camarones"]
    assert 0.0 <= m["distance"] <= 1.0 and m["distance"] == 0.25, "1 de 4 componentes falla"


def test_shadow_without_anchors_or_budget_reports_none_not_zero():
    eff, _ = pp.compile_policy(pp.policy_from_form(_form(stapleFoods=[])), context={})
    m = pp.measure_plan_against_policy({"days": []}, eff)
    assert m["anchor_coverage"] is None and m["budget_over"] is None and m["cycle_match"] is None and m["distance"] is None


# ───────────────────────────────────────────── F. knob y persistencia
def test_knob_off_is_a_noop_and_shadow_stamps_the_plan(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "off")
    plan = {"days": []}
    assert pp.stamp_plan_policy(plan, _form()) is None and "_plan_policy" not in plan
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "shadow")
    out = pp.stamp_plan_policy(plan, _form())
    assert out and plan["_plan_policy"]["policy_hash"] and "_plan_policy_shadow" in plan
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "nonsense")
    assert pp.policy_mode() == "off", "valor inválido ⇒ off"


def test_compile_from_form_never_raises():
    out = pp.compile_from_form({"groceryDuration": object()})
    assert isinstance(out, dict) and "requested" in out and "effective" in out


def test_run_insert_persists_policy_and_router_compiles_it():
    gl = (BACKEND / "generation_lifecycle.py").read_text(encoding="utf-8")
    i = gl.find("INSERT INTO plan_generation_runs")
    win = gl[i:i + 1400]
    for col in ("requested_policy", "effective_policy", "relaxations", "policy_hash", "policy_schema_version", "engine_versions"):
        assert col in win, col
    assert "policy: Optional[dict] = None" in gl
    rt = (BACKEND / "routers" / "plans_generation.py").read_text(encoding="utf-8")
    assert "policy=_policy_for_run(data)," in rt and "def _policy_for_run(data: dict):" in rt
    assert "compile_from_form(data)" in rt and "policy_active()" in rt


def test_every_delivered_plan_is_stamped_before_persist_including_guests():
    rp = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = rp.find("def _postprocess_pipeline_result(")
    body = rp[i:i + 20000]
    k_stamp = body.find("_stamp_policy(result, data, total_days_requested=total_days_requested)")
    k_save = body.find("selected_techniques = result.pop(\"_selected_techniques\", None)")
    assert -1 not in (k_stamp, k_save) and k_stamp < k_save, "el sello va ANTES de cualquier persistencia"
    assert "emit_policy_shadow_metric" in body


def test_marker_bumped():
    """El marker se bumpeó al cerrar la fase; P-fixes posteriores lo mueven (por diseño), así que
    se ancla el FORMATO y que la fecha no retroceda por debajo del cierre de F2 (2026-09-02)."""
    src = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "(P[0-9]-[A-Z0-9-]+) · (\d{4}-\d{2}-\d{2})"', src)
    assert m, "marker con formato Pn-SLUG · YYYY-MM-DD"
    assert m.group(2) >= "2026-09-02"


def test_shadow_cycle_and_budget_from_real_prod_shapes():
    """[2026-09-02] Primer plan shadow en prod: `cycle_match=None` porque `total_days_requested`
    aún no vivía en el resultado al sellar, y el reconcile real trae `status` (no `estimated_cycle_rd`)."""
    from plan_policy import measure_plan_against_policy
    effective = {"shopping": {"main_cycle_days": 30}, "food_anchors": [], "diet": {}}
    plan = {"days": [{"meals": [{"ingredients": ["Huevo"]}]}],
            "budget_reconciliation": {"status": "excedido", "ratio": 1.141, "delta_rd": 2868, "floor_rd": 16250}}
    m = measure_plan_against_policy(plan, effective, total_days_requested=30)
    assert m["cycle_match"] is True
    assert m["budget_over"] is True
    m2 = measure_plan_against_policy(dict(plan, budget_reconciliation={"status": "dentro"}), effective, total_days_requested=7)
    assert m2["cycle_match"] is False and m2["budget_over"] is False
    assert measure_plan_against_policy(plan, effective)["cycle_match"] is None


def test_postprocess_passes_total_days_to_stamp():
    src = (Path(__file__).parents[1] / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "_stamp_policy(result, data, total_days_requested=total_days_requested)" in src

