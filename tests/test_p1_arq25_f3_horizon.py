"""[P1-ARQ25-F3-HORIZON · 2026-09-02] Fase 3 del roadmap 2.5 — Horizon allocator y superficies obedientes.

Ancla los entregables de la fase:
  A. Blueprint 7/15/30 determinista (§6.5): fronteras de chunk = `split_with_absorb` (H2), anclas
     programadas dentro de su banda, familias de proteína que respetan dieta/alergias/exclusiones,
     hash estable (misma política ⇒ mismo blueprint).
  B. Rebanada inmutable por chunk (`_blueprint_slice`) + `input_hash` = huella + hash de la rebanada.
  C. Validadores de fidelidad (ancla ausente / franja / banda / repetición exacta / ingrediente):
     las 13 golden policies CUMPLEN sus bandas con un plan construido desde su blueprint y FALLAN
     al mutarlo; en `enforce` sustituyen a los gates de repetición de V1 que contradicen la banda.
  D. Motivo neutral versionado `renewal.v1` (alias legado `variety`) en las tres superficies §6.1.
  E. Paridad: todas las superficies de §6.6 leen la política por el MISMO módulo (parser sobre los
     call sites: chunk 0 cola/legacy, chunks 2..N, renovación, swap, regen de día, shuffle, caché
     semántica, self-critique, aprendizaje, shopping).
  F. Canary: `enforce` global o por usuario (`MEALFIT_PLAN_POLICY_ENFORCE_USERS`); `MEALFIT_FIDELITY_GATE`
     warn|block, nunca rechaza en el intento final ni en `shadow`.
  G. Migración idempotente en ambos SSOT + marker.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
h = pytest.importorskip("horizon")
pp = pytest.importorskip("plan_policy")


def _src(rel: str) -> str:
    return (BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _shadow_env(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "shadow")
    monkeypatch.delenv("MEALFIT_FIDELITY_GATE", raising=False)
    monkeypatch.delenv("MEALFIT_PLAN_POLICY_ENFORCE_USERS", raising=False)
    yield


# ═══════════════════════════════════════════════════════ golden policies
def _eff(*, mode="balanced", country="DO", diet="balanced", allergies=(), exclusions=(), anchors=(),
         cycle=7, topup=None, freezer="limited", extra=None) -> dict:
    e = {
        "schema_version": 1, "market_country": country,
        "recurrence": {"global_mode": mode, "slot_modes": {s: mode for s in ("breakfast", "lunch", "dinner", "snack")}},
        "food_anchors": [
            {"ingredient_id": pp.ingredient_id_for(a[0]), "name": a[0], "slots": list(a[1] or []),
             "min_per_7d": a[2], "max_per_7d": a[3], "preparation_mode": "vary_preparation"}
            for a in anchors],
        "shopping": {"main_cycle_days": cycle, "fresh_topup_days": topup, "freezer_mode": freezer, "batch_cooking": "sometimes"},
        "diet": {"type": diet, "allergies": list(allergies), "exclusions": list(exclusions)},
        "clinical": {"conditions": []}, "budget": {"tier": "medium", "mode": "hard"},
        "household_size": 1, "culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}],
    }
    if extra:
        e.update(extra)
    e["policy_hash"] = pp.policy_hash(e)
    return e


GOLDEN = {
    "g01_routine_7d_egg_breakfast": (_eff(mode="routine", anchors=[("Huevo", ["breakfast"], 5, 7)], cycle=7), 7, 4),
    "g02_balanced_15d_egg_avena": (_eff(anchors=[("Huevo", ["breakfast"], 3, 5), ("Avena", [], 2, 7)], cycle=15, topup=7), 15, 4),
    "g03_explore_30d_no_anchors": (_eff(mode="explore", cycle=30, topup=7), 30, 4),
    "g04_vegan_15d_lentejas": (_eff(diet="vegan", anchors=[("Lentejas", ["lunch"], 2, 4)], cycle=15), 15, 3),
    "g05_fish_allergy_balanced_7d": (_eff(allergies=["pescado"], anchors=[("Pollo", [], 3, 5)], cycle=7), 7, 4),
    "g06_pescatarian_30d_topup": (_eff(diet="pescatarian", anchors=[("Atún", ["lunch"], 1, 3)], cycle=30, topup=7), 30, 4),
    "g07_routine_30d_5_meals": (_eff(mode="routine", anchors=[("Huevo", ["breakfast"], 7, 7), ("Avena", ["snack"], 4, 7)], cycle=30, topup=7), 30, 5),
    "g08_balanced_15d_freezer_none": (_eff(anchors=[("Queso", ["breakfast"], 2, 4)], cycle=15, freezer="none"), 15, 4),
    "g09_explore_7d_two_anchors": (_eff(mode="explore", anchors=[("Yogur", ["snack"], 2, 3), ("Guineo", [], 2, 4)], cycle=7), 7, 4),
    "g10_us_balanced_15d": (_eff(country="US", anchors=[("Pollo", ["dinner"], 2, 5)], cycle=15, topup=7,
                                 extra={"culture_weights": [{"profile_id": "us_everyday", "weight": 1.0}]}), 15, 3),
    "g11_routine_15d_narrow_band": (_eff(mode="routine", anchors=[("Plátano verde", ["breakfast"], 2, 3)], cycle=15), 15, 4),
    "g12_balanced_30d_eight_anchors": (_eff(anchors=[(n, [], 1, 7) for n in ("Huevo", "Avena", "Pollo", "Arroz", "Habichuelas", "Guineo", "Queso", "Yogur")], cycle=30, topup=7), 30, 4),
    "g13_vegetarian_7d_3_meals": (_eff(diet="vegetarian", exclusions=["res"], anchors=[("Huevo", ["breakfast"], 4, 7)], cycle=7, freezer="full"), 7, 3),
}


def _plan_from_blueprint(bp: dict, sl: dict) -> list:
    """Plan sintético que OBEDECE la rebanada: anclas en su franja, proteína del día, nombres de
    plato únicos salvo en rutina (donde se repite dentro del límite)."""
    mode = (sl.get("recurrence") or {}).get("global_mode")
    lim = int((sl.get("repetition_limits") or {}).get("max_exact_repeat") or 1)
    days = []
    for i, d in enumerate(sl["days"]):
        meals = []
        slots = d.get("slots") or ["breakfast", "lunch", "dinner"]
        for j, slot in enumerate(slots):
            anchors_here = [a["name"] for a in (d.get("anchors") or []) if (a.get("slot") or slot) == slot]
            ings = list(anchors_here)
            if slot in ("lunch", "dinner") and d.get("protein"):
                ings.append(d["protein"])
            ings += [f"Vegetal {i}-{j}", f"Base {i}-{j}"]
            # en rutina el desayuno se repite (hasta el límite); fuera de rutina cada plato es único
            name = f"Plato {slot} {i % max(1, lim) if mode == 'routine' and slot == 'breakfast' else i}-{j}"
            meals.append({"name": name, "type": {"breakfast": "Desayuno", "lunch": "Almuerzo", "dinner": "Cena", "snack": "Merienda"}[slot],
                          "ingredients": ings})
        days.append({"day": i + 1, "meals": meals})
    return days


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_blueprint_respects_bands_chunks_and_hash(key):
    eff, total, mpd = GOLDEN[key]
    bp = h.build_blueprint(eff, total_days=total, meals_per_day=mpd)
    # H2: fronteras = split_with_absorb; cada día en exactamente un chunk
    from constants import PLAN_CHUNK_SIZE, split_with_absorb
    sizes = split_with_absorb(total, PLAN_CHUNK_SIZE) if total > PLAN_CHUNK_SIZE else [total]
    assert [c["days_count"] for c in bp["chunks"]] == [n for n in sizes if n > 0]
    assert sum(c["days_count"] for c in bp["chunks"]) == total == len(bp["days"])
    for c in bp["chunks"]:
        for d in range(c["days_offset"], c["days_offset"] + c["days_count"]):
            assert bp["days"][d]["chunk_index"] == c["chunk_index"]
    # anclas dentro de su banda escalada al horizonte
    for a in bp["anchors"]:
        n = len(a["scheduled_days"])
        lo, hi = math.floor(a["min_per_7d"] * total / 7.0), math.ceil(a["max_per_7d"] * total / 7.0)
        assert lo <= n <= hi, (key, a["name"], n, lo, hi)
        if a["min_per_7d"] > 0:
            assert n >= 1
        assert a["scheduled_days"] == sorted(set(a["scheduled_days"]))
        assert all(0 <= d < total for d in a["scheduled_days"])
    # familias: dieta/alergias/exclusiones (nombres = SSOT, sin traducir)
    fams = bp["protein_families"]
    diet = eff["diet"]["type"]
    if diet == "vegan":
        assert not any(f in fams for f in ("Pollo", "Res", "Pescado", "Huevo", "Queso"))
    if diet == "vegetarian":
        assert not any(f in fams for f in ("Pollo", "Res", "Pescado"))
    if "pescado" in eff["diet"]["allergies"]:
        assert not any(f in fams for f in ("Pescado", "Atún", "Camarones"))
    for x in eff["diet"]["exclusions"]:
        assert not any(pp._matches(f, x) for f in fams)
    assert set(bp["protein_pool"]) <= set(fams)
    # hash estable + sensible al modo
    assert h.build_blueprint(eff, total_days=total, meals_per_day=mpd)["blueprint_hash"] == bp["blueprint_hash"]
    other = copy.deepcopy(eff)
    other["recurrence"]["global_mode"] = "explore" if eff["recurrence"]["global_mode"] != "explore" else "routine"
    assert h.build_blueprint(other, total_days=total, meals_per_day=mpd)["blueprint_hash"] != bp["blueprint_hash"]


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_slices_cover_horizon_and_plan_from_blueprint_is_faithful(key):
    eff, total, mpd = GOLDEN[key]
    bp = h.build_blueprint(eff, total_days=total, meals_per_day=mpd)
    covered, hashes = [], set()
    for c in bp["chunks"]:
        sl = h.slice_for_chunk(bp, c["days_offset"], c["days_count"])
        assert [d["day_index"] for d in sl["days"]] == list(range(c["days_offset"], c["days_offset"] + c["days_count"]))
        assert sl["slice_hash"] == h.slice_hash(sl)          # inmutable: el hash es reproducible
        assert sl["blueprint_hash"] == bp["blueprint_hash"]
        hashes.add(sl["slice_hash"])
        covered += [d["day_index"] for d in sl["days"]]
        # plan que obedece la rebanada ⇒ cero issues; mutado ⇒ ancla ausente
        days = _plan_from_blueprint(bp, sl)
        issues = h.fidelity_issues(days, sl, eff)
        assert issues == [], (key, c, [i["code"] for i in issues])
        sched = [(i, d) for i, d in enumerate(sl["days"]) if d.get("anchors")]
        if sched:
            i, d = sched[0]
            anchor = d["anchors"][0]
            mutated = copy.deepcopy(days)
            # con franja ⇒ basta quitarla de SU día; sin franja la promesa es la cuota de la
            # ventana (P1-PANTRY-KEY-VULGAR-FRACTIONS): se quita de TODOS los días del bloque
            targets = [i] if anchor.get("slot") else range(len(mutated))
            for t in targets:
                for m in mutated[t]["meals"]:
                    m["ingredients"] = [x for x in m["ingredients"] if not h.anchor_in_text(anchor["name"], x)]
                    m["name"] = m["name"] + " sin ancla"
            codes = {x["code"] for x in h.fidelity_issues(mutated, sl, eff)}
            assert codes & {"anchor_missing_day", "anchor_under_scheduled"}, (key, codes)
    assert covered == list(range(total))
    assert len(hashes) == len(bp["chunks"])


def test_exact_repeat_and_ingredient_days_validators():
    eff = GOLDEN["g03_explore_30d_no_anchors"][0]
    bp = h.build_blueprint(eff, total_days=30, meals_per_day=4)
    sl = h.slice_for_chunk(bp, 3, 4)
    assert sl["repetition_limits"]["max_exact_repeat"] == 1
    days = [{"day": i + 1, "meals": [{"name": "Pollo guisado", "type": "Almuerzo", "ingredients": ["Pollo", "Arroz"]}]} for i in range(4)]
    codes = {i["code"] for i in h.fidelity_issues(days, sl, eff)}
    assert {"exact_repeat_exceeded", "ingredient_days_exceeded"} <= codes
    # en rutina el mismo plato 3 veces por semana es CORRECTO (límite escalado ≥ 2 en 4 días)
    eff_r = GOLDEN["g01_routine_7d_egg_breakfast"][0]
    sl_r = h.slice_for_chunk(h.build_blueprint(eff_r, total_days=7, meals_per_day=4), 0, 3)
    days_r = [{"day": i + 1, "meals": [{"name": "Huevos revueltos", "type": "Desayuno", "ingredients": ["Huevo"]},
                                       {"name": f"Almuerzo {i}", "type": "Almuerzo", "ingredients": ["Pollo"]}]} for i in range(3)]
    assert not [i for i in h.fidelity_issues(days_r, sl_r, eff_r) if i["code"] == "exact_repeat_exceeded"]
    assert not [i for i in h.fidelity_issues(days_r, sl_r, eff_r) if i["code"] == "anchor_missing_day"]


def test_anchor_slot_mismatch_and_bands():
    eff = GOLDEN["g01_routine_7d_egg_breakfast"][0]
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=7, meals_per_day=3), 0, 3)
    days = [{"day": i + 1, "meals": [{"name": "Avena", "type": "Desayuno", "ingredients": ["Avena"]},
                                     {"name": "Huevo hervido", "type": "Cena", "ingredients": ["Huevo"]}]} for i in range(3)]
    codes = [i["code"] for i in h.fidelity_issues(days, sl, eff)]
    assert "anchor_slot_mismatch" in codes and "anchor_missing_day" not in codes
    over = [{"day": i + 1, "meals": [{"name": "Huevos", "type": "Desayuno", "ingredients": ["Huevo"]},
                                     {"name": "Tortilla", "type": "Cena", "ingredients": ["Huevo"]}]} for i in range(3)]
    eff_narrow = GOLDEN["g11_routine_15d_narrow_band"][0]
    sl_n = h.slice_for_chunk(h.build_blueprint(eff_narrow, total_days=15, meals_per_day=3), 0, 3)
    days_n = [{"day": i + 1, "meals": [{"name": "Mangú", "type": "Desayuno", "ingredients": ["Plátano verde"]}]} for i in range(3)]
    assert "recurrence_above_band" in {i["code"] for i in h.fidelity_issues(days_n, sl_n, eff_narrow)}
    assert h.fidelity_issues([], sl, eff) == []
    assert h.fidelity_issues(over, None, None) == []


# ═══════════════════════════════════════════ rebanada + input_hash
def test_chunk_input_hash_binds_fingerprint_to_slice():
    eff = GOLDEN["g02_balanced_15d_egg_avena"][0]
    bp = h.build_blueprint(eff, total_days=15, meals_per_day=4)
    s1, s2 = h.slice_for_chunk(bp, 0, 3), h.slice_for_chunk(bp, 3, 4)
    assert h.chunk_input_hash("fp", None) == "fp"
    assert h.chunk_input_hash("fp", s1) != h.chunk_input_hash("fp", s2) != "fp"
    assert h.chunk_input_hash("fp", s1) == h.chunk_input_hash("fp", json.loads(json.dumps(s1)))


def test_inject_policy_into_pipeline_data_and_off_mode(monkeypatch):
    form = {"country": "DO", "dietType": "balanced", "allergies": ["Ninguna"], "dislikes": ["Ninguno"],
            "stapleFoods": ["Huevo"], "groceryDuration": "biweekly", "budget": "medium", "budgetCurrency": "DOP",
            "householdSize": 1, "cookingTime": "30min", "mealOrganization": "routine", "totalDays": 15, "user_id": "u1"}
    pd = {"_days_to_generate": 3}
    bp = h.inject_policy_into_pipeline_data(pd, form_data=form, total_days=15, days_offset=0, days_count=3, user_id="u1")
    assert bp and bp["total_days"] == 15
    assert [d["day_index"] for d in pd[h.BLUEPRINT_SLICE_KEY]["days"]] == [0, 1, 2]
    assert pd[h.POLICY_EFFECTIVE_KEY]["recurrence"]["global_mode"] == "routine"
    assert pd[h.POLICY_ENFORCED_KEY] is False   # shadow global, sin canary
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_ENFORCE_USERS", "U1, otro")
    assert h.policy_enforced("u1") and not h.policy_enforced("u2")
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "off")
    pd2 = {"_days_to_generate": 3}
    assert h.inject_policy_into_pipeline_data(pd2, form_data=form, total_days=15, user_id="u1") is None
    assert pd2 == {"_days_to_generate": 3}
    assert h.policy_mode_for_user("u1") == "off"


# ═══════════════════════════════════════════ motivo neutral versionado
@pytest.mark.parametrize("reason,expected", [
    ("variety", True), ("renewal", True), ("renewal.v1", True), ("RENEWAL.V1", True), ("renewal.v2", True),
    ("time", False), ("dislike", False), ("", False), (None, False), ("similar", False),
])
def test_is_renewal_reason(reason, expected):
    assert h.is_renewal_reason(reason) is expected


def test_normalize_and_default_swap_reason(monkeypatch):
    assert h.normalize_update_reason("variety") == "renewal.v1" == h.RENEWAL_REASON_VERSIONED
    assert h.normalize_update_reason("time") == "time" and h.normalize_update_reason(None) is None
    assert h.default_swap_reason("u1") == "variety"
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "enforce")
    assert h.default_swap_reason("u1") == "renewal.v1"


def test_three_surfaces_of_6_1_use_the_neutral_reason():
    go = _src("graph_orchestrator.py")
    assert 'is_variety_regen = _is_renewal_reason_f3(form_data.get("update_reason"))' in go
    assert 'is_variety_regen = form_data.get("update_reason") == "variety"' not in go
    pg = _src("prompts/plan_generator.py")
    assert '_is_renewal_reason_f3(form_data.get("update_reason"))' in pg
    assert 'if form_data.get("update_reason") == "variety":' not in pg
    ah = _src("ai_helpers.py")
    assert "if _is_renewal_reason_f3(update_reason):" in ah
    assert "if update_reason == 'variety':" in ah   # el chain conserva su primer branch (test P3-NEWPLAN)
    assert "El usuario RENUEVA su plan. NO es una petición de más variedad" in ah
    # swap: motivo por defecto neutral bajo enforce
    rp = _src("routers/plans.py")
    assert 'swap_reason = data.get("swap_reason") or _default_swap_reason_f3(data.get("user_id"))' in rp
    assert 'data.get("swap_reason", "variety")' not in rp
    ct = _src("cron_tasks.py")
    assert ct.count("'swap:renewal.v1'") == 2
    fe = BACKEND.parent / "frontend" / "src" / "pages" / "Settings.jsx"
    if fe.exists():
        assert "reason: 'renewal.v1', isPlanExpired: false, entry_point: 'settings_renovar'" in fe.read_text(encoding="utf-8")


# ═══════════════════════════════════════════ gates: fidelidad vs variedad
def test_filter_variety_issues_for_policy_by_mode():
    rep = "MISMO PLATO REPETIDO ENTRE DÍAS (rechazo de variedad): 'huevo' en 3 días."
    rep2 = "MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): pollo."
    clash = "PAREO CHOCANTE FRUTA+SALADO (rechazo de coherencia de sabor)."
    routine = GOLDEN["g01_routine_7d_egg_breakfast"][0]
    balanced = GOLDEN["g02_balanced_15d_egg_avena"][0]
    explore = GOLDEN["g09_explore_7d_two_anchors"][0]
    assert h.filter_variety_issues_for_policy([rep, rep2, clash], routine, enforced=True) == [clash]
    assert h.filter_variety_issues_for_policy([rep, rep2, clash], balanced, enforced=True) == [rep2, clash]
    assert h.filter_variety_issues_for_policy([rep, rep2, clash], explore, enforced=True) == [rep, rep2, clash]
    assert h.filter_variety_issues_for_policy([rep, rep2, clash], routine, enforced=False) == [rep, rep2, clash]


def test_review_fidelity_gate_modes(monkeypatch):
    import horizon as hz
    monkeypatch.setattr(hz, "emit_fidelity_metric", lambda *a, **k: None)
    eff = GOLDEN["g01_routine_7d_egg_breakfast"][0]
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=7, meals_per_day=3), 0, 3)
    days = [{"day": i + 1, "meals": [{"name": "Avena", "type": "Desayuno", "ingredients": ["Avena"]}]} for i in range(3)]
    variety = ["MISMO PLATO REPETIDO ENTRE DÍAS (rechazo de variedad): 'huevo' en 3 días."]
    # shadow: mide, no filtra, no rechaza
    fd = {h.POLICY_EFFECTIVE_KEY: eff, h.BLUEPRINT_SLICE_KEY: sl, h.POLICY_ENFORCED_KEY: False, "user_id": "u"}
    plan = {"days": days}
    v, rej = h.review_fidelity_gate(plan, fd, variety, attempt=1, max_attempts=3)
    assert v == variety and rej == [] and plan[h.FIDELITY_REPORT_KEY]["codes"] == ["anchor_missing_day"]
    assert plan[h.FIDELITY_REPORT_KEY]["enforced"] is False
    # enforce + warn: filtra los gates de repetición, no rechaza
    fd[h.POLICY_ENFORCED_KEY] = True
    v, rej = h.review_fidelity_gate(plan, fd, variety, attempt=1, max_attempts=3)
    assert v == [] and rej == []
    # enforce + block: rechaza salvo en el intento final
    monkeypatch.setenv("MEALFIT_FIDELITY_GATE", "block")
    v, rej = h.review_fidelity_gate(plan, fd, variety, attempt=1, max_attempts=3)
    assert rej and rej[0].startswith("ANCLA AUSENTE")
    v, rej = h.review_fidelity_gate(plan, fd, variety, attempt=3, max_attempts=3)
    assert rej == []
    # sin política ⇒ passthrough
    assert h.review_fidelity_gate({"days": days}, {}, variety, attempt=1, max_attempts=3) == (variety, [])


def test_fidelity_gate_knob_defaults(monkeypatch):
    assert h.fidelity_gate_mode() == "warn"
    monkeypatch.setenv("MEALFIT_FIDELITY_GATE", "block")
    assert h.fidelity_gate_mode() == "block"
    monkeypatch.setenv("MEALFIT_FIDELITY_GATE", "loco")
    assert h.fidelity_gate_mode() == "warn"


# ═══════════════════════════════════════════ prompt / seeder / aprendizaje
def test_policy_prompt_block_only_under_enforce_and_names_anchors():
    eff = GOLDEN["g01_routine_7d_egg_breakfast"][0]
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=7, meals_per_day=4), 0, 3)
    assert h.policy_prompt_block(eff, sl, surface="planner_seeder", enforced=False) == ""
    assert h.policy_prompt_block(None, sl, surface="x") == ""
    blk = h.policy_prompt_block(eff, sl, surface="planner_seeder", enforced=True)
    assert "RUTINA" in blk and "Huevo" in blk and "Día 1" in blk and "desayuno" in blk
    swap = h.policy_prompt_block(eff, None, surface="swap", slot="Desayuno", enforced=True)
    assert "ancla de la franja → Huevo" in swap


def test_apply_slice_to_seeder_pools_follows_families():
    eff = GOLDEN["g03_explore_30d_no_anchors"][0]
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=30, meals_per_day=4), 3, 4)
    pool = ["Pollo", "Res", "Cerdo", "Pavo", "Pescado", "Huevo", "Atún"]
    out = h.apply_slice_to_seeder_pools(sl, ["Res", "Cerdo", "Pavo", "Pollo"], pool, days=4)
    assert len(out) == 4 and len({x.lower() for x in out}) == 4
    fams = [d["protein"] for d in sl["days"]]
    assert all(pp._matches(f, o) for f, o in zip(fams, out) if f in pool)
    assert h.apply_slice_to_seeder_pools({}, ["Res"], pool, days=2) == ["Res"]


def test_exclude_anchors_from_fatigue_and_rank_days():
    eff = GOLDEN["g02_balanced_15d_egg_avena"][0]
    assert h.exclude_anchors_from_fatigue(["Huevo", "Pollo", "Avena"], eff) == ["Pollo"]
    assert h.exclude_anchors_from_fatigue(["Huevo"], None) == ["Huevo"]
    d_no = {"meals": [{"name": "Pollo", "ingredients": ["Pollo"]}]}
    d_yes = {"meals": [{"name": "Huevos con avena", "ingredients": ["Huevo", "Avena"]}]}
    assert h.rank_days_by_policy([d_no, d_yes], eff) == [d_yes, d_no]
    assert h.rank_days_by_policy([d_no, d_yes], None) == [d_no, d_yes]


# ═══════════════════════════════════════════ shopping: ventanas 7/15/30
def test_shopping_projection_windows_and_demand_stamp():
    eff30 = GOLDEN["g03_explore_30d_no_anchors"][0]
    w = h.shopping_projection_windows(eff30, 30)
    assert [(x["kind"], x["start_day"], x["end_day"]) for x in w] == [
        ("main", 0, 30), ("fresh_topup", 7, 14), ("fresh_topup", 14, 21), ("fresh_topup", 21, 28), ("fresh_topup", 28, 30)]
    assert h.shopping_projection_windows(GOLDEN["g01_routine_7d_egg_breakfast"][0], 7) == [
        {"kind": "main", "start_day": 0, "end_day": 7, "days": 7, "cycle_days": 7, "fresh_only": False}]
    pd = {"days": [{"day": 1, "meals": []}], "total_days_requested": 15, "_plan_policy": {"effective": GOLDEN["g08_balanced_15d_freezer_none"][0]}}
    st = h.stamp_demand_windows(pd)
    assert st["freezer_mode"] == "none" and st["freeze_horizon_days"] == 0 and st["windows"][0]["end_day"] == 15
    assert pd["_ingredient_demand"]["policy_hash"] == GOLDEN["g08_balanced_15d_freezer_none"][0]["policy_hash"]
    pd_full = {"days": [], "total_days_requested": 30, "_plan_policy": {"effective": GOLDEN["g13_vegetarian_7d_3_meals"][0]}}
    assert h.stamp_demand_windows(pd_full)["freeze_horizon_days"] == 30
    assert h.stamp_demand_windows({"days": []}) is None


# ═══════════════════════════════════════════ paridad de superficies (§6.6)
def test_all_surfaces_read_the_policy_through_horizon():
    go = _src("graph_orchestrator.py")
    for key in ('"_blueprint_slice",', '"_plan_policy_effective",', '"_policy_enforced",', '"_policy_day_index",'):
        assert key in go, key                                        # whitelist del orquestador
    assert "_rep_issues, _fid_rejects = _review_fidelity_gate(" in go   # gates de variedad
    assert '_inc_semantic_cache_stat("policy_bypass")' in go            # caché semántica
    assert "{policy_block}{staples_block}{spread_block}" in go            # self-critique
    assert "fatigued_ingredients = _excl_anchors_f3(fatigued_ingredients" in go  # aprendizaje (orquestador)
    ah = _src("ai_helpers.py")
    assert "chosen_proteins = _apply_slice_f3(_bp_slice, chosen_proteins, unique_proteins, days=_dc, inject_missing=_bp_inject)" in ah
    assert 'surface="planner_seeder"' in ah                              # chunk 0..N + renovación (mismo seeder)
    ag = _src("agent.py")
    assert 'surface="swap", slot=meal_type' in ag                        # swap individual
    rp = _src("routers/plans.py")
    assert rp.count("_attach_policy_f3(") == 2                           # swap endpoint + regen de día
    assert "plan_data=plan_data, day_index=data.get(\"day_index\")" in rp
    assert rp.count("_horizon_inject(pipeline_data, data, total_days_requested, actual_user_id)") == 2  # legacy sync/SSE
    assert "_bp_f3 = _bp_for_plan(str(plan_id), data, total_days_requested)" in rp  # chunks 2..N
    assert "UPDATE plan_chunk_queue SET input_hash = %s WHERE meal_plan_id = %s" in rp
    assert "_enq_proj_f3(str(plan_id), actual_user_id" in rp             # shopping: proyección
    assert "_stamp_windows_f3(result, (_compiled_policy or {}).get(\"effective\"))" in rp
    gi = _src("generation_inputs.py")
    assert "blueprint = inject_policy_into_pipeline_data(" in gi and '"blueprint": blueprint,' in gi  # cola chunk 0
    pgn = _src("routers/plans_generation.py")
    assert 'await asyncio.to_thread(persist_run_blueprint, str(run["id"]), inputs["blueprint"])' in pgn
    gl = _src("generation_lifecycle.py")
    assert "_chunk_input_hash_f3(chunk_snapshot.get(\"form_data\") or {})" in gl
    assert gl.count('form_data["_policy_enforced"] = _policy_enforced_f3(user_id)') == 1
    ct = _src("cron_tasks.py")
    assert ct.count('form_data["_policy_enforced"] = _policy_enforced_f3(user_id)') == 1  # chunks 2..N al ejecutar
    assert "pantry_filtered_pool = _rank_days_f3(pantry_filtered_pool" in ct              # smart shuffle
    assert "_fatigued_ingredients = _excl_anchors_f3(_fatigued_ingredients" in ct         # aprendizaje (worker)
    sc = _src("shopping_calculator.py")
    assert "_stamp_windows_f3(plan_result)" in sc                                          # shopping (builder)


def test_fidelity_report_is_surface_independent():
    eff = GOLDEN["g02_balanced_15d_egg_avena"][0]
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=15, meals_per_day=4), 3, 4)
    days = _plan_from_blueprint(h.build_blueprint(eff, total_days=15, meals_per_day=4), sl)
    reports = [h.fidelity_report(days, sl, eff, surface=s) for s in ("initial", "chunk", "renew", "swap", "regen")]
    assert len({(r["score"], tuple(r["codes"]), r["slice_hash"]) for r in reports}) == 1
    assert reports[0]["score"] == 1.0


# ═══════════════════════════════════════════ migración + marker + knobs
def test_migration_exists_idempotent_and_ssot_identical():
    name = "p1_arq25_f3_horizon_blueprint_2026_09_02.sql"
    mig = BACKEND / "migrations" / name
    assert mig.exists()
    sql = mig.read_text(encoding="utf-8")
    assert sql.count("ADD COLUMN IF NOT EXISTS") == 3 and "DO $$" in sql and "RAISE EXCEPTION" in sql
    assert "plan_generation_runs" in sql and "blueprint_hash" in sql and "allocator_version" in sql
    root = BACKEND.parent / "migrations" / name
    if root.exists():
        assert root.read_bytes().replace(b"\r\n", b"\n") == mig.read_bytes().replace(b"\r\n", b"\n")


def test_marker_bumped_and_persist_uses_run_row():
    assert 'P1-ARQ25-F3-HORIZON' in _src("horizon.py")   # el marker de app.py sigue avanzando con cada P-fix
    hz = _src("horizon.py")
    assert "UPDATE plan_generation_runs SET blueprint = %s, blueprint_hash = %s, allocator_version = %s WHERE id = %s" in hz
    assert "INSERT INTO plan_jobs (job_type, plan_id, user_id, plan_revision, dedup_key, payload)" in hz


def test_meal_slot_and_chunk_boundaries_helpers():
    assert h.meal_slot({"type": "Desayuno"}) == "breakfast" and h.meal_slot({"meal": "cena"}) == "dinner"
    assert h.meal_slot({"type": "Merienda AM"}) == "snack" and h.meal_slot({}) is None
    assert [c["days_count"] for c in h.chunk_boundaries(3)] == [3]
    assert [c["days_count"] for c in h.chunk_boundaries(30)] == [3, 4, 4, 4, 4, 4, 4, 3]
    assert h.slots_for_day(3) == ["breakfast", "lunch", "dinner"] and len(h.slots_for_day(6)) == 6


def test_family_matches_by_food_class_not_word_root():
    """[canary 2026-09-03] Nevera estricta: el pool era sardinas/mozzarella/hígado y las 3 familias
    programadas (Pescado/Huevo/Res) cayeron al fallback porque `_matches` compara raíces de palabra."""
    assert h.family_matches("Pescado", "Sardinas en lata") and h.family_matches("Pescado", "Filete de pescado blanco")
    assert h.family_matches("Res", "Hígado de res") and h.family_matches("Queso", "Queso Mozzarella")
    assert h.family_matches("Cerdo", "Chuleta de cerdo") and not h.family_matches("Res", "Chuleta de cerdo")
    assert not h.family_matches("Pescado", "Queso Mozzarella")
    sl = {"days": [{"protein": "Pescado"}, {"protein": "Huevo"}, {"protein": "Res"}], "recurrence": {"global_mode": "balanced"}}
    pool = ["Queso Mozzarella", "Sardinas en lata", "Hígado de res"]
    assert h.apply_slice_to_seeder_pools(sl, pool, pool, days=3) == ["Sardinas en lata", "Queso Mozzarella", "Hígado de res"]


def test_anchor_detection_survives_vulgar_fractions_and_modifiers():
    """[P1-PANTRY-KEY-VULGAR-FRACTIONS · 2026-09-03] Primer plan del canary: «¾ cucharada de
    mantequilla de maní» dos veces en el día 2 y el validador marcó el ancla AUSENTE."""
    assert h.anchor_in_text("Mantequilla de maní", "¾ cucharada de mantequilla de maní")
    assert h.anchor_in_text("Kiwi", "½ kiwi en cubos") and h.anchor_in_text("Huevo", "2 huevos batidos")
    assert not h.anchor_in_text("Mantequilla de maní", "¾ cucharada de miel")
    assert not h.anchor_in_text("Pollo", "1 repollo")
    eff = _eff(anchors=[("Mantequilla de maní", [], 2, 7)], cycle=7)
    sl = h.slice_for_chunk(h.build_blueprint(eff, total_days=7, meals_per_day=4), 0, 3)
    sched = [d["day_index"] for d in sl["days"] if d.get("anchors")]
    days = [{"day": i + 1, "meals": [{"name": "Batido", "type": "Merienda", "ingredients": ["¾ taza de yogurt", "¾ cucharada de mantequilla de maní"]}]}
            if i in sched else {"day": i + 1, "meals": [{"name": f"Avena {i}", "type": "Desayuno", "ingredients": ["40 g de avena"]}]} for i in range(3)]
    assert h.fidelity_issues(days, sl, eff) == []
    # sin franja: ponerla OTRO día del bloque no es infidelidad (cuota cumplida)
    if len(sched) == 1 and sched[0] != 0:
        moved = [{"day": 1, "meals": days[sched[0]]["meals"]}] + [{"day": i + 1, "meals": [{"name": f"Avena {i}", "type": "Desayuno", "ingredients": ["40 g de avena"]}]} for i in range(1, 3)]
        assert h.fidelity_issues(moved, sl, eff) == []
    # y si NO aparece las veces programadas ⇒ anchor_under_scheduled
    none = [{"day": i + 1, "meals": [{"name": f"Avena {i}", "type": "Desayuno", "ingredients": ["40 g de avena"]}]} for i in range(3)]
    if sched:
        assert {x["code"] for x in h.fidelity_issues(none, sl, eff)} == {"anchor_under_scheduled"}
    # con franja: sí se exige el día y la franja
    eff_s = _eff(anchors=[("Huevo", ["breakfast"], 5, 7)], cycle=7, mode="routine")
    sl_s = h.slice_for_chunk(h.build_blueprint(eff_s, total_days=7, meals_per_day=3), 0, 3)
    bad = [{"day": i + 1, "meals": [{"name": "Avena", "type": "Desayuno", "ingredients": ["avena"]}]} for i in range(3)]
    assert "anchor_missing_day" in {x["code"] for x in h.fidelity_issues(bad, sl_s, eff_s)}
