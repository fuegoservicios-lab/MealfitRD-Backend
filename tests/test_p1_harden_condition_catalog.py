"""[A1-HARDEN-POOLS clase 3 · 2026-07-09] Filtro de pool por condición médica.

El hueco arquitectónico: _get_fast_filtered_catalogs (constants.py) filtra alergias/dislikes/dieta
pero NUNCA condiciones médicas → un DM2 tenía toronja/arroz-blanco en su pool y solo se corregía
post-hoc. Esta garantía los quita del pool ANTES del day-gen, reusando el SSOT de tokens de
condition_rules.collect_substitutions (solo food-identity, preserve_qty=True). Nunca vacía un pool
(el backstop post-hoc collect_substitutions caza el residuo de la canasta sobre-restringida).

Función pura (sin DB). tooltip-anchor: A1-HARDEN-POOLS
"""
import graph_orchestrator as go


def _skel():
    return {"days": [{"day": 1,
        "protein_pool": ["Salami Dominicano", "Pollo", "Bacalao salado"],
        "carb_pool": ["Arroz blanco", "Arroz integral"],
        "fruit_pool": ["Toronja", "Fresa"]}]}


def test_dm2_removes_contraindicated_identities(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = _skel()
    counts = go.harden_day_pools(skel, {"medicalConditions": ["dm2"]}, None)
    d = skel["days"][0]
    assert "Toronja" not in d["fruit_pool"]        # toronja→CYP3A4 fuera del pool DM2
    assert "Arroz blanco" not in d["carb_pool"]    # IG alto fuera
    assert "Fresa" in d["fruit_pool"]              # el resto intacto
    assert "Arroz integral" in d["carb_pool"]
    assert counts["condition_removed"] >= 2


def test_hta_removes_embutidos_and_bacalao(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": ["hipertension"]}, None)
    pool = [p.lower() for p in skel["days"][0]["protein_pool"]]
    assert not any("salami" in p for p in pool)
    assert not any("bacalao" in p for p in pool)
    assert any("pollo" in p for p in pool)         # proteína legítima sobrevive


def test_graceful_never_empties_protein_pool(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    # pool 100% contraindicado para HTA → NO se vacía (post-hoc collect_substitutions lo swapea)
    skel = {"days": [{"day": 1, "protein_pool": ["Salami Dominicano", "Bacalao salado"],
                      "carb_pool": ["Arroz integral"], "fruit_pool": ["Fresa"]}]}
    go.harden_day_pools(skel, {"medicalConditions": ["hipertension"]}, None)
    assert len(skel["days"][0]["protein_pool"]) >= 1


def test_condition_catalog_off_leaves_pool_untouched(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", False)  # clase 3 OFF
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": ["dm2"]}, None)
    assert "Toronja" in skel["days"][0]["fruit_pool"]


def test_no_conditions_is_noop(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": []}, None)
    # sin condiciones activas → nada que filtrar
    assert "Toronja" in skel["days"][0]["fruit_pool"]
    assert "Arroz blanco" in skel["days"][0]["carb_pool"]
