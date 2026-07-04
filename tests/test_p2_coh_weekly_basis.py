"""[P2-COH-WEEKLY-BASIS · 2026-07-04] El banner "(71 detalles)" era ruido de BASE.

Caso vivo (plan c5d800fd, renovación 2026-07-04): `run_shopping_coherence_guard`
comparaba lo esperado de recetas (base SEMANAL: los días del plan × household)
contra `aggregated_shopping_list` = la lista ACTIVA del usuario — que para
quincenal/mensual es la HÍBRIDA (estables ×2/×4 semanas). Resultado: TODOS los
estables divergían ~100-300% + una fila `delta=inf` por split de unidad →
71 divergencias (38 unknown + 33 unit_mismatch), 0 accionables. El block-mode de
assemble ya las trataba como no-accionables (action=not_applicable), pero la capa
post-swap (P1-SWAP-COHERENCE-ESCALATE) contaba `inf > 0.30` como crítica →
"Revisa tu lista de compras (71 detalles)" al usuario, con summary de
`{ingredient: "?", hypothesis: unknown}` (leía la key equivocada).

Fix en 3 piezas:
  A. El guard usa la lista SEMANAL como base canónica (fallback a la activa para
     fixtures/planes legacy sin la key).
  B. `critical_count` post-swap solo cuenta deltas FINITOS (>0.30).
  C. El summary lee `food` (la key real del detector).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_SC = _read("shopping_calculator.py")
_GO = _read("graph_orchestrator.py")


# ---------------------------------------------------------------------------
# A · base semanal canónica (source + funcional)
# ---------------------------------------------------------------------------

def test_guard_prefers_weekly_list_source():
    i = _SC.index("[P2-COH-WEEKLY-BASIS")
    blk = _SC[i:i + 1400]
    assert 'plan_result.get("aggregated_shopping_list_weekly")' in blk
    assert 'or plan_result.get("aggregated_shopping_list") or []' in blk, \
        "fallback a la lista activa para fixtures/planes legacy"


def test_guard_uses_weekly_not_active_functional(monkeypatch):
    import shopping_calculator as sc

    # expected fijo (sin parsear recetas reales): 100 g de arroz blanco.
    monkeypatch.setattr(sc, "expected_sum_from_recipes",
                        lambda plan, apply_yield=False, multiplier=1.0: {"Arroz blanco": {"g": 100.0}})
    monkeypatch.setattr(sc, "_is_verified_for_shopping", lambda name: True, raising=False)

    def _mk_item(qty):
        return {"name": "Arroz blanco", "market_qty_numeric": qty, "market_unit": "g",
                "category": "Despensa", "is_staple": False}

    plan = {
        "days": [{"day": 1, "meals": []}],
        # SEMANAL coincide con lo esperado (100 g) → cero divergencias.
        "aggregated_shopping_list_weekly": [_mk_item(100.0)],
        # ACTIVA mensual ×4 (400 g) → si el guard usara esta base, divergiría.
        "aggregated_shopping_list": [_mk_item(400.0)],
    }
    div = sc.run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)
    _mags = [d for d in div if d.get("magnitude")]
    assert not _mags, f"el guard usó la lista ACTIVA (mensual ×4) como base: {_mags[:3]}"

    # Legacy: sin key semanal → fallback a la activa (comportamiento previo intacto).
    plan_legacy = {
        "days": [{"day": 1, "meals": []}],
        "aggregated_shopping_list": [_mk_item(400.0)],
    }
    div2 = sc.run_shopping_coherence_guard(plan_legacy, mode_override="warn", multiplier=1.0)
    assert any(d.get("magnitude") for d in div2), \
        "sin key semanal debe seguir comparando contra la activa (fallback)"


# ---------------------------------------------------------------------------
# B · criticidad post-swap solo con deltas finitos
# ---------------------------------------------------------------------------

def test_postswap_critical_count_ignores_nonfinite():
    i = _GO.index("def _coh_finite_delta(")
    blk = _GO[i - 800:i + 900]
    assert "inf > 0.30" in blk or "delta=inf" in blk or "no-accionable" in blk, \
        "el porqué (inf contaba como crítico) debe quedar documentado"
    assert "abs(_coh_finite_delta(d)) > 0.30" in _GO, \
        "critical_magnitudes debe usar el helper finito"
    # la selección del summary usa la MISMA vara.
    assert "abs(_coh_finite_delta(_d)) > 0.30" in _GO


# ---------------------------------------------------------------------------
# C · summary con la key real del detector
# ---------------------------------------------------------------------------

def test_summary_reads_food_key():
    i = _GO.index('plan_result["_swap_coherence_warnings"] = {')
    blk = _GO[max(0, i - 2500):i]
    assert '_d.get("food")' in blk, 'el summary debe leer `food` (key real del guard)'


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------

def test_marker_bumped():
    app = _read("app.py")
    assert '_LAST_KNOWN_PFIX = "P2-COH-WEEKLY-BASIS · 2026-07-04"' in app
