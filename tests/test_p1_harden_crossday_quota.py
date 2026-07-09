"""[A1-HARDEN-POOLS clase 2 · 2026-07-09] Cuota round-robin cross-día.

Generaliza el cap max-1-día de _SKELETON_RESTRICTED a TODAS las proteínas pesadas: ninguna proteína
puede estar en el pool de más de ceil(num_days / distinct_available) días → la monotonía cross-día del
principal se vuelve estructuralmente imposible (el día beyond-quota literalmente no puede suministrarla).
Legumbres/yogurt/huevo exentas (repetibles). Graceful: nunca vacía un pool; si solo hay 1 proteína la
cuota = num_days (no puede diversificar).

Función pura (sin DB). tooltip-anchor: A1-HARDEN-POOLS
"""
import graph_orchestrator as go


def _days_with(pool_of):
    return {"days": [{"day": i + 1, "protein_pool": list(p), "carb_pool": [], "fruit_pool": []}
                     for i, p in enumerate(pool_of)]}


def _count_days_containing(skel, token):
    return sum(1 for d in skel["days"]
               if any(token in p.lower() for p in d.get("protein_pool", [])))


def test_crossday_caps_overused_protein(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CROSSDAY_QUOTA", True)
    # pollo en los 4 días; 5 proteínas distintas disponibles → quota = ceil(4/5) = 1
    skel = _days_with([["Pollo", "Res"], ["Pollo", "Cerdo"], ["Pollo", "Pescado"], ["Pollo", "Atún"]])
    go.harden_day_pools(skel, {}, None)
    assert _count_days_containing(skel, "pollo") <= 1   # ya no en los 4 días
    for d in skel["days"]:
        assert len(d["protein_pool"]) >= 1               # nunca vacío


def test_crossday_off_leaves_repetition(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CROSSDAY_QUOTA", False)
    skel = _days_with([["Pollo", "Res"], ["Pollo", "Cerdo"], ["Pollo", "Pescado"], ["Pollo", "Atún"]])
    go.harden_day_pools(skel, {}, None)
    assert _count_days_containing(skel, "pollo") == 4     # sin cap


def test_crossday_graceful_single_protein(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CROSSDAY_QUOTA", True)
    # solo 1 proteína disponible → quota = num_days → no puede diversificar, no vacía
    skel = _days_with([["Pollo"], ["Pollo"], ["Pollo"]])
    go.harden_day_pools(skel, {}, None)
    assert _count_days_containing(skel, "pollo") == 3
    for d in skel["days"]:
        assert len(d["protein_pool"]) >= 1


def test_crossday_exempts_legumes(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CROSSDAY_QUOTA", True)
    # legumbres repetibles: habichuelas en todos los días NO se capea
    skel = _days_with([["Habichuelas rojas", "Pollo"], ["Habichuelas rojas", "Res"],
                       ["Habichuelas rojas", "Cerdo"], ["Habichuelas rojas", "Pescado"]])
    go.harden_day_pools(skel, {}, None)
    assert _count_days_containing(skel, "habichuela") == 4  # exenta
