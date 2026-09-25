# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-254 · 2026-09-25] La política dice los días de fresco reales, y la pasta «caracol» no es un marisco.

· `plan_policy` declaraba `evidence={"fresh_days": 7}` para la compra única sin congelador (batería rd252, owner_like);
  desde el lote 218 son 3 (`MEALFIT_SINGLE_TRIP_NO_FREEZER_FREE_DAYS`).
· «caracol» (lote 250) casaba con «1 taza de pasta caracol».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _relajacion(monkeypatch=None):
    import plan_policy as pp
    out = pp.compile_from_form({"groceryDuration": "monthly", "freezerMode": "none", "freshTopup": "no",
                                "mealOrganization": "balanced"})
    return [x for x in out.get("relaxations") or [] if x.get("reason_code") == "pantry_proteins_after_first_week"]


def test_la_evidencia_dice_los_dias_reales(monkeypatch):
    monkeypatch.delenv("MEALFIT_SINGLE_TRIP_NO_FREEZER_FREE_DAYS", raising=False)
    r = _relajacion()
    assert r and r[0]["evidence"]["fresh_days"] == 3, r
    import plan_policy as pp
    assert "primeros 3 días" in pp.explain_relaxations(r)[0]
    monkeypatch.setenv("MEALFIT_SINGLE_TRIP_NO_FREEZER_FREE_DAYS", "7")
    assert _relajacion()[0]["evidence"]["fresh_days"] == 7


def test_pasta_caracol_no_es_marisco():
    import graph_orchestrator as go
    import vocabulario_mar as vm
    assert "caracol" not in vm.MARISCOS_EXTRA
    plan = {"days": [{"meals": [{"name": "x", "ingredients": ["1 taza de pasta caracol", "150 g de jaiba"]}]}]}
    v = go._scan_allergen_violations(plan, ["Mariscos"])
    assert [x[1] for x in v] == ["150 g de jaiba"], v


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 254
