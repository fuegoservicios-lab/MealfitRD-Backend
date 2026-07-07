"""[P1-PROTEIN-FLOOR-POST-CAPS · 2026-06-27] (mejora arquitectónica — auditoría workflow + crítico adversario)
Raíz del DÉFICIT DE PROTEÍNA recurrente (visto en 4+ runs: 82/100, 76/90, 69/80, 64/80): el closer del motor
corre PRE-cap; los caps clínicos (DM2/bariátrica) recortan lácteos (queso=proteína) DESPUÉS y recuperan kcal
escalando carbos/veg → el % de proteína cae bajo el piso. `_repair_protein_floor_post_caps` re-cierra el gap
DESPUÉS de los caps con proteína ANIMAL DENSA NO-LÁCTEA + re-cuadra C/F. SEGURIDAD: RENAL → skip total (KDIGO).
Reusa el closer + reconcile probados. Idempotente (fill-to-target + flag `_final_protein_close`).
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent
_NUT = {"macros": {"protein_g": 80, "carbs_g": 180, "fats_g": 53}}


def _wire(monkeypatch, capture=None):
    """Mockea closer/reconcile/candidatos para probar la ORQUESTACIÓN sin DB real."""
    import graph_orchestrator as g

    def fake_closer(m, target, db, cands, allergies=None, fill_pct=0.92, max_add_g=300, slot_cal_target=0.0, enforce_min_threshold=True):
        if capture is not None:
            capture["cands"] = cands
            capture["max_add_g"] = max_add_g
        cur = g._meal_macro_num(m.get("protein"))
        if cur >= target * 0.9:
            return 0
        m["protein"] = round(target * 0.92)
        return round(target * 0.92 - cur)

    monkeypatch.setattr(g, "_close_protein_gap_for_meal", fake_closer)
    monkeypatch.setattr(g, "_macro_aware_day_reconcile",
                        lambda ms, c, f, db: (capture and capture.__setitem__("reconcile", capture.get("reconcile", 0) + 1)) or True)
    monkeypatch.setattr(g, "_safe_high_density_proteins",
                        lambda al, db, min_protein=18.0: [(0.30, "Queso ricotta", None), (0.28, "Yogurt griego", None),
                                                          (0.25, "Pechuga de pollo", None), (0.22, "Filete de pescado blanco", None)])
    return g


def test_closes_deficit_post_caps(monkeypatch):
    g = _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 10, "cals": 400},
                                 {"meal": "Cena", "protein": 8, "cals": 350}]}]
    added = g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object())
    assert added > 0
    assert days[0]["meals"][0]["protein"] > 10  # se re-cerró
    assert all(m.get("_final_protein_close") for m in days[0]["meals"])  # idempotencia marcada


def test_renal_skips_injection(monkeypatch):
    g = _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 5, "cals": 400}]}]
    added = g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Enfermedad renal crónica"]}, db=object())
    assert added == 0, "RENAL: NUNCA inyectar proteína (techo KDIGO)"
    assert not days[0]["meals"][0].get("_final_protein_close")


def test_no_deficit_no_change(monkeypatch):
    g = _wire(monkeypatch)
    # proteína ya cumple (>= 90% del slot target = 0.9 * 80 * 0.5 = 36)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 45, "cals": 400},
                                 {"meal": "Cena", "protein": 45, "cals": 350}]}]
    added = g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object())
    assert added == 0


def test_candidates_exclude_cheeses_allow_yogurt(monkeypatch):
    # [P3-PROTEIN-FLOOR-SWEET-DAIRY · 2026-06-28] Los QUESOS/leche (cap 30g → 90g los excedería) NO son candidatos;
    # pero el YOGUR SÍ (cap 120g, add ≤90g seguro) — es la proteína coherente para cerrar el piso en platos dulces.
    cap = {}
    g = _wire(monkeypatch, capture=cap)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 5, "cals": 400}]}]
    g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object())
    names = [str(c[1]).lower() for c in cap.get("cands", [])]
    assert names, "el closer debió recibir candidatos"
    assert not any(("queso" in n or "ricotta" in n or "cottage" in n or "leche" in n) for n in names), \
        f"los QUESOS/leche NO deben ser candidatos (cap 30g): {names}"
    assert any("yogur" in n for n in names), f"el yogur SÍ debe entrar (proteína coherente para dulces): {names}"
    assert any("pollo" in n or "pescado" in n for n in names)


def test_knob_off_disables(monkeypatch):
    g = _wire(monkeypatch)
    monkeypatch.setattr(g, "REPAIR_PROTEIN_POST_CAPS", False)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 5, "cals": 400}]}]
    assert g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object()) == 0


def test_anchors():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-PROTEIN-FLOOR-POST-CAPS" in go and "def _repair_protein_floor_post_caps" in go
    assert "MEALFIT_REPAIR_PROTEIN_POST_CAPS" in go
    # corre en assemble DESPUÉS de los caps clínicos
    assert "_repair_protein_floor_post_caps(days, nutrition, form_data)" in go
