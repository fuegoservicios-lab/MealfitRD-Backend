"""[P1-MICRO-BUDGET-NONFAT-FIRST · 2026-07-07] El presupuesto kcal COMPARTIDO del micro-closer va
primero a los micros SIN backstop.

Bug de raíz (audit solver+seeder 2026-07-07): `_floors_ordered` ponía TODOS los seedables
(omega3/vitE/vitA) primero (P1-CLOSER-SEED-PRIORITY — correcto ANTES del fatswap). Post-fatswap,
omega3/vitE tienen su PROPIO presupuesto (fatswap para escalar + una reserva para sembrar) → quedaban
doble-servidos, mientras hierro/folato ("de mayor consecuencia", sin backstop) iban al FINAL y se
quedaban sin budget en el worst-day típico (corto en ambos fat-micros).

Fix: nuevo orden vía `_micro_budget_rank` (vitA seedable-sin-backstop → no-grasa-no-seedable
hierro/folato → grasa-basados omega3/vitE al final) + una reserva dedicada para el SEED de los
grasa-basados (el fatswap solo ESCALA portadores existentes, no siembra) para que ponerlos al final
NUNCA mate su siembra en el caso día-sin-portador + el guard del `break` de budget hecho reserve-aware +
el guard clínico del fatswap (ahora path principal de escalado de fat-micros).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ─────────────────────────── Parser anchors ───────────────────────────

def test_marker_in_source():
    assert "P1-MICRO-BUDGET-NONFAT-FIRST" in _GO


def test_reorder_gated_and_uses_rank():
    assert "if MICRO_BUDGET_NONFAT_FIRST:" in _GO
    assert "_floors_ordered = sorted(floors.items(), key=lambda kv: _micro_budget_rank(kv[0]))" in _GO
    # rollback preservado en el else.
    assert "_floors_ordered = sorted(floors.items(), key=lambda kv: kv[0] not in _MICRO_SEED_SOURCES)" in _GO


def test_break_is_fat_micro_aware():
    """El `break` universal de budget agotado se volvió un `continue` selectivo: solo salta los micros
    NO-grasa; los grasa-basados (al final del orden) SIGUEN para llegar a su backstop (fatswap/reserva).
    Sin esto, ponerlos al final los dejaría sin cierre alguno."""
    assert "if kcal_budget_left <= 1.0 and k not in _FAT_BASED_MICROS:" in _GO
    j = _GO.index("if kcal_budget_left <= 1.0 and k not in _FAT_BASED_MICROS:")
    assert "continue" in _GO[j:j + 900]  # continue (no break) → sigue iterando hasta los fat-micros


def test_seed_from_reserve_gate_and_deduction():
    # la reserva SOLO rescata el caso no-carrier (el fatswap cubre el has-carrier escalando).
    assert "_seed_from_reserve = (k in _FAT_BASED_MICROS and not _had_carriers" in _GO
    assert "and (kcal_budget_left > 25.0 or _seed_from_reserve):" in _GO
    assert "_fat_seed_reserve -= _kc_seed" in _GO


def test_fatswap_honors_clinical_guard():
    """El reorder hace del fatswap el path PRINCIPAL de escalado de fat-micros → debe honrar el mismo
    `_ceiling_risky_contributor` que el loop principal (no escalar queso/mantequilla en dislip/HTA)."""
    # ventana desde el loop del fatswap.
    i = _GO.index("for _fc, _fm, _fi, _fing in _fs_cands:")
    win = _GO[i:i + 1000]
    assert "_ceiling_risky_contributor(k, _sa_fs(str(_fing).lower()))" in win


def test_knobs_defined():
    assert 'MICRO_BUDGET_NONFAT_FIRST = _env_bool("MEALFIT_MICRO_BUDGET_NONFAT_FIRST", True)' in _GO
    assert 'MICRO_FAT_SEED_RESERVE_KCAL = _env_int("MEALFIT_MICRO_FAT_SEED_RESERVE_KCAL", 120' in _GO


# ─────────────────────────── Unit: _micro_budget_rank ───────────────────────────

def test_budget_rank_orders_nonfat_before_fat():
    import graph_orchestrator as g
    # grasa-basados (backstop propio) al final.
    assert g._micro_budget_rank("omega3_g") == 2
    assert g._micro_budget_rank("vit_e_mg") == 2
    # seedable sin backstop graso → primero.
    assert g._micro_budget_rank("vit_a_mcg") == 0
    # no-grasa no-seedable (mayor consecuencia, solo budget compartido) → medio, ANTES que los grasa.
    for k in ("iron_mg", "folate_mcg", "vit_c_mg", "potassium_mg", "calcium_mg", "magnesium_mg", "fiber_g"):
        assert g._micro_budget_rank(k) == 1, k
        assert g._micro_budget_rank(k) < g._micro_budget_rank("omega3_g")


def test_knob_defaults():
    import graph_orchestrator as g
    assert g.MICRO_BUDGET_NONFAT_FIRST is True
    assert g.MICRO_FAT_SEED_RESERVE_KCAL == 120


# ─────────────────────────── Funcional: reserva ───────────────────────────

class _FakeDB:
    """avena = portador de fibra escalable (drena el budget compartido); omega3 sin portador (seed)."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        # NB: el closer busca la clave de `_MICRO_CLOSER_INGREDIENT_KEY` → fibra es "fiber" (no "fiber_g").
        if "avena" in low:
            return {"fiber": 4.0, "omega3_g": 0.0}
        if "linaza" in low or "chia" in low or "chía" in low or "nuez" in low or "nueces" in low:
            return {"omega3_g": 0.6, "fiber": 1.0}
        return {"fiber": 0.0, "omega3_g": 0.0}

    def macros_from_ingredient_string(self, s):
        # kcal alto → el kcal-cap del escalado consume el budget compartido COMPLETO en una pasada
        # (factor = 1 + budget/kcal ⇒ consumo = budget), drenándolo deterministamente antes del fat-micro.
        return {"kcal": 250.0, "protein": 3.0, "carbs": 15.0, "fats": 2.0}

    def grams_from_ingredient_string(self, s):
        return 40.0


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_MAX_SCALE", 5.0)  # deja que el budget compartido drene
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _mk_report():
    # fibra (no-grasa, rank 1) con piso ENORME → escala la avena y drena el budget compartido;
    # omega-3 (grasa, rank 2) sin portador → necesita sembrar. Con el reorder, fibra va primero.
    return {"panel": [
        {"key": "fiber_g", "piso": 9999.0, "valor": 4.0, "status": "bajo"},
        {"key": "omega3_g", "piso": 1.6, "valor": 0.0, "status": "bajo"},
    ], "gaps": [
        {"key": "fiber_g", "piso": 9999.0, "status": "bajo"},
        {"key": "omega3_g", "piso": 1.6, "status": "bajo"},
    ], "coverage": 1.0, "per_day_floors": {"flagged": False}}


def _mk_plan():
    return {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Avena", "ingredients": ["40 g de avena"],
         "ingredients_raw": ["40 g de avena"], "recipe": ["Cocina."]},
        {"meal": "Merienda", "name": "Fruta", "ingredients": ["1 manzana"],
         "ingredients_raw": ["1 manzana"], "recipe": ["Sirve."]},
    ]}]}


def _has_omega3_seed(plan):
    _all = " | ".join(i for d in plan["days"] for m in d["meals"]
                      for i in m["ingredients"]).lower()
    return any(t in _all for t in ("linaza", "chia", "chía", "nuez", "nueces"))


def test_fat_micro_seed_survives_shared_budget_drain(go, monkeypatch):
    """Con el reorder, la fibra (rank 1) va primero y drena el budget compartido; omega-3 (rank 2, sin
    portador) llega sin budget compartido pero SIEMBRA desde la reserva dedicada → NO regresión."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: _mk_report())
    plan = _mk_plan()
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    assert _has_omega3_seed(plan), "el seed del fat-micro debe sobrevivir vía la reserva dedicada"


def test_without_reserve_fat_micro_seed_starves(go, monkeypatch):
    """Prueba de que es la RESERVA lo que salva la siembra: con reserva=0 y el budget compartido
    agotado por la fibra, el fat-micro (al final del orden) no puede sembrar (regresión reproducida)."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: _mk_report())
    monkeypatch.setattr(go, "MICRO_FAT_SEED_RESERVE_KCAL", 0)
    plan = _mk_plan()
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    assert not _has_omega3_seed(plan), "sin reserva y con budget compartido agotado, el seed no entra"
