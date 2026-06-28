"""[P3-PROTEIN-CLOSER-SCALE-FIRST · 2026-06-28] Fix 4 (el que quedó) de la auditoría de calidad de recetas: antes de
PEGAR una proteína nueva como ingrediente extra (bolt-on incoherente), el closer intenta CRECER la proteína de mayor
densidad YA presente en el plato (subir la pechuga/pescado/queso fresco existente) — el cierre más "chef". Clamp ≤2×,
respeta headroom calórico, solo densidad ≥12 (yogur/ricotta no cierran el gap dentro del clamp → caen al append).
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


class _DB:
    _DENS = {"pechuga": 31, "pollo": 31, "pescado": 20, "queso fresco": 24, "ricotta": 11, "yogur": 10}

    def macros_from_ingredient_string(self, s):
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*g", s)
        if not m:
            return {}
        grams = float(m.group(1))
        d = next((v for k, v in self._DENS.items() if k in s.lower()), 8)
        return {"grams": grams, "protein": grams * d / 100.0, "carbs": grams * 0.02,
                "fats": grams * 0.03, "kcal": grams * (d * 4 / 100.0 + 0.3)}


def _grp(ing, db):
    low = str(ing).lower()
    return "protein" if any(k in low for k in ("pechuga", "pollo", "pescado", "queso", "ricotta", "yogur", "huevo")) else "other"


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_ingredient_macro_group", _grp)


def _meal(ings, prot, cals=200):
    return {"name": "Almuerzo", "protein": prot, "carbs": 20, "fats": 5, "cals": cals, "ingredients": list(ings)}


def test_scales_existing_dense_protein(monkeypatch):
    _wire(monkeypatch)
    m = _meal(["80g de pechuga de pollo", "100g de arroz integral"], prot=25)
    added = g._try_scale_existing_protein(m, 35, _DB(), _sa)
    assert added > 0
    assert "pechuga" in m["ingredients"][0].lower() and m["ingredients"][0] != "80g de pechuga de pollo"
    assert m["protein"] >= 34  # cerró el gap creciendo la pechuga
    assert m.get("_protein_closed") is True
    assert len(m["ingredients"]) == 2  # NO añadió un ingrediente nuevo


def test_skips_low_density_protein(monkeypatch):
    _wire(monkeypatch)
    m = _meal(["120g de yogur griego", "95g de lechosa"], prot=12)
    assert g._try_scale_existing_protein(m, 25, _DB(), _sa) == 0  # yogur denso<12 → cae al append


def test_skips_when_no_protein_line(monkeypatch):
    _wire(monkeypatch)
    m = _meal(["100g de arroz integral", "1 guineo"], prot=4)
    assert g._try_scale_existing_protein(m, 20, _DB(), _sa) == 0


def test_respects_calorie_headroom(monkeypatch):
    _wire(monkeypatch)
    m = _meal(["80g de pechuga de pollo"], prot=25, cals=480)
    assert g._try_scale_existing_protein(m, 40, _DB(), _sa, slot_cal_target=480) == 0  # sin headroom → no crece


def test_clamp_max_2x(monkeypatch):
    _wire(monkeypatch)
    m = _meal(["50g de pechuga de pollo"], prot=15)  # 50g=15.5g prot
    g._try_scale_existing_protein(m, 100, _DB(), _sa)  # gap enorme
    import re
    grams = float(re.search(r"(\d+(?:\.\d+)?)\s*g", m["ingredients"][0]).group(1))
    assert grams <= 100.1, f"no debe crecer más de 2× (50→100): {grams}"


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P3-PROTEIN-CLOSER-SCALE-FIRST" in src and "def _try_scale_existing_protein" in src
    assert "PROTEIN_CLOSER_SCALE_FIRST" in src
