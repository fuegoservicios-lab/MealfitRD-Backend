"""[A1-HARDEN-POOLS clase 6 · P0-2-POOL-MAIN-ARITY · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10):
49% de los rechazos del reviewer en 72h (39/80) = "MISMA PROTEÍNA REPETIDA EL MISMO DÍA". Causa
estructural: el pool de un día traía solo 2 proteínas gate-label (heavy+huevo) para 3 comidas
principales → repetición forzada por construcción, sin importar cuánto corrija el LLM después.

Clase 6 garantiza aridad mínima: si un día trae MENOS proteínas gate-label distintas que el target,
se rellena con proteínas gate-label YA vistas en pools de OTROS días del MISMO skeleton (mismo
form_data → mismo filtro de alergia/dieta/dislikes ya aplicado por el catálogo — reusar cross-día es
seguro, cero candidatos nuevos sin vetar). Nunca quita nada; solo añade cuando hay margen. Función
pura (sin DB). tooltip-anchor: A1-HARDEN-POOLS
"""
import graph_orchestrator as go


def _days_with(pool_of):
    return {"days": [{"day": i + 1, "protein_pool": list(p), "carb_pool": [], "fruit_pool": []}
                     for i, p in enumerate(pool_of)]}


def _gate_labels_in(pool):
    labels = set()
    for p in pool:
        lbl = go._gate_label_of(p)
        if lbl:
            labels.add(lbl)
    return labels


def test_main_arity_fills_gap_from_other_days(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY_TARGET", 3)
    # día 1 (el forensic real): solo Salmón (pescado) + Huevos = 2 gate-labels para 3 mains.
    # otros días del mismo skeleton ya vetaron Res/Pollo/Cerdo → candidatos seguros para tomar prestado.
    skel = _days_with([
        ["Salmón", "Huevos", "Mantequilla de maní", "Yogurt griego sin azúcar"],
        ["Res", "Chivo", "Habichuelas rojas"],
        ["Pollo", "Cerdo", "Habichuelas rojas"],
    ])
    go.harden_day_pools(skel, {}, None)
    day1_labels = _gate_labels_in(skel["days"][0]["protein_pool"])
    assert len(day1_labels) >= 3, f"día 1 debió llegar a arity>=3, quedó en {day1_labels}"
    # nunca quita lo que ya había
    assert "Mantequilla de maní" in skel["days"][0]["protein_pool"]
    assert "Yogurt griego sin azúcar" in skel["days"][0]["protein_pool"]


def test_main_arity_noop_when_already_sufficient(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY_TARGET", 3)
    skel = _days_with([
        ["Pollo", "Res", "Pescado"],
        ["Cerdo", "Atún", "Pavo"],
    ])
    before = [list(d["protein_pool"]) for d in skel["days"]]
    go.harden_day_pools(skel, {}, None)
    after = [d["protein_pool"] for d in skel["days"]]
    assert before == after, "día ya con arity suficiente no debe tocarse"


def test_main_arity_capped_by_skeleton_universe(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY_TARGET", 3)
    # SOLO 2 proteínas gate-label existen en TODO el skeleton (catálogo del plan las vetó así) →
    # no se puede alcanzar arity=3 en ningún día; no debe reventar ni inventar candidatos.
    skel = _days_with([
        ["Salmón", "Huevos"],
        ["Salmón", "Huevos"],
    ])
    go.harden_day_pools(skel, {}, None)
    for d in skel["days"]:
        assert _gate_labels_in(d["protein_pool"]) == {"pescado", "huevo"}
        assert len(d["protein_pool"]) == 2


def test_main_arity_off_leaves_pool_untouched(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", False)
    skel = _days_with([
        ["Salmón", "Huevos"],
        ["Res", "Chivo", "Pollo"],
    ])
    before = list(skel["days"][0]["protein_pool"])
    go.harden_day_pools(skel, {}, None)
    assert skel["days"][0]["protein_pool"] == before


def test_main_arity_master_switch_off(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", False)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    skel = _days_with([["Salmón", "Huevos"], ["Res", "Pollo", "Cerdo"]])
    before = list(skel["days"][0]["protein_pool"])
    counts = go.harden_day_pools(skel, {}, None)
    assert skel["days"][0]["protein_pool"] == before
    assert counts == {"condition_removed": 0, "saltcured_removed": 0, "sameday_bound": 0,
                      "crossday_capped": 0, "main_arity_added": 0}


def test_gate_label_helper_matches_known_aliases():
    assert go._gate_label_of("Pechuga de pollo") == "pollo"
    assert go._gate_label_of("Filete de Tilapia") == "pescado"
    assert go._gate_label_of("Huevos") == "huevo"
    assert go._gate_label_of("Habichuelas rojas") is None  # exenta del gate, no cuenta para arity
    assert go._gate_label_of("Yogurt griego sin azúcar") is None
    assert go._gate_label_of("Mantequilla de maní") is None
