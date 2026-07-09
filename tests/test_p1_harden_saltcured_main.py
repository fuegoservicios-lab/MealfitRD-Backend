"""[A1-HARDEN-POOLS clase 5 · 2026-07-09] Exclusión salado-como-principal.

Hoy P1-SODIUM-BOMB-POOL solo baja el PESO x0.1 de las proteínas curadas en sal (bacalao/salami/
tocino...) en el sorteo → pueden salir igual como principal del día. Esta garantía las excluye
DURAMENTE del protein_pool (slot principal), universal (el budget de sodio OMS es goal-independiente).
Nunca vacía el pool (el peso x0.1 y los backstops post-hoc siguen).

Función pura (sin DB). tooltip-anchor: A1-HARDEN-POOLS
"""
import graph_orchestrator as go


def test_saltcured_excluded_from_protein_pool(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Salami Dominicano", "Pollo"],
                      "carb_pool": ["Arroz integral"], "fruit_pool": ["Fresa"]}]}
    counts = go.harden_day_pools(skel, {}, None)
    pool = [p.lower() for p in skel["days"][0]["protein_pool"]]
    assert not any("bacalao" in p or "salami" in p for p in pool)
    assert any("pollo" in p for p in pool)
    assert counts["saltcured_removed"] >= 2


def test_saltcured_off_leaves_pool(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", False)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Pollo"],
                      "carb_pool": [], "fruit_pool": []}]}
    go.harden_day_pools(skel, {}, None)
    assert any("bacalao" in p.lower() for p in skel["days"][0]["protein_pool"])


def test_saltcured_graceful_when_all_salt(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Salami Dominicano"],
                      "carb_pool": [], "fruit_pool": []}]}
    go.harden_day_pools(skel, {}, None)
    assert len(skel["days"][0]["protein_pool"]) >= 1  # nunca vacío


def test_saltcured_universal_all_goals(monkeypatch):
    # el budget de sodio no depende del goal → excluye igual con o sin goal
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Tocino", "Pollo"], "carb_pool": [], "fruit_pool": []}]}
    go.harden_day_pools(skel, {"main_goal": "gain_muscle"}, None)
    assert not any("tocino" in p.lower() for p in skel["days"][0]["protein_pool"])
