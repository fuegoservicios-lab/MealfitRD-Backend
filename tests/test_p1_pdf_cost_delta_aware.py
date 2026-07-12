"""[P1-PDF-COST-DELTA-AWARE · 2026-07-12] El recuadro de costos del PDF respeta el delta de la Nevera.

Vivo (owner, 09:3xZ): PDF con 8 ítems visibles (36 excluidos por Nevera) que suman
~RD$1,270 mostraba "Esta compra RD$5,989" y "Ciclo RD$16,771" — la preferencia SSOT del
backend (P1-BUDGET-COST-SSOT, `shopping_cost_summary`) describe el plan COMPLETO, no la
compra real tras excluir lo que ya tienes.

Fix (frontend, handleDownloadShoppingList): cuando `deltaItemsRemoved > 0`:
  - "Esta compra" = suma LOCAL de lo impreso (delta).
  - "Ciclo real" = delta de hoy + perecederos COMPLETOS × (semanas − 1) — la Nevera solo
    ahorra la semana 1; los frescos de las semanas 2..N se recompran completos (el
    `perishable_rd` full lo aporta el resumen backend, con fallback local).
Sin exclusiones (delta=0) el SSOT del backend sigue mandando (paridad con la
reconciliación de presupuesto). tooltip-anchor: P1-PDF-COST-DELTA-AWARE
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

with open(os.path.join(_ROOT, "frontend", "src", "pages", "Dashboard.jsx"),
          encoding="utf-8") as f:
    _DASH = f.read()


def test_delta_aware_gate_present():
    i = _DASH.find("P1-PDF-COST-DELTA-AWARE")
    assert i != -1, "el bloque delta-aware desapareció del builder del PDF"
    win = _DASH[i:i + 2600]
    assert "_deltaAware = (deltaItemsRemoved || 0) > 0" in win, \
        "el gate es el MISMO contador del banner 'ya están en tu Nevera'"
    assert "!_deltaAware && _backendCostSummary" in win, \
        "'Esta compra' solo usa el SSOT backend cuando NO hubo exclusiones"


def test_cycle_formula_charges_future_weeks_full():
    win = _DASH[_DASH.find("P1-PDF-COST-DELTA-AWARE"):][:2600]
    assert re.search(
        r"_stableCost \+ _perishableCost\s*\+ _fullPerishableRd \* Math\.max\(0, _cycleCostMultiplier - 1\)",
        win), (
        "ciclo delta-aware = delta hoy + perecederos FULL × (semanas−1): lo de la Nevera "
        "solo ahorra la semana 1, las siguientes recompran completo")
    assert "perishable_rd" in win, "el full de perecederos viene del resumen backend (fallback local)"


def test_ssot_preserved_when_no_delta():
    win = _DASH[_DASH.find("P1-PDF-COST-DELTA-AWARE"):][:2600]
    assert "cycle_total_rd" in win and "trip_total_rd" in win, \
        "sin exclusiones, la paridad con budget_reconciliation (P1-BUDGET-COST-SSOT) se conserva"
