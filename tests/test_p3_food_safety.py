"""[P3-FOOD-SAFETY · 2026-06-13] Guard DETERMINISTA de seguridad alimentaria: huevo (TCS)
crudo/poco cocido = vector de Salmonella. Hallazgo CRÍTICO de la auditoría clínica multi-agente
(plan 11d17452: ½ huevo crudo licuado en un batido + 1¼ huevos sin paso de cocción en un wrap).

Cubre: (1) detección por preparación licuada (blended), (2) detección por ausencia de cocción
(no_cook), (3) NO-falsos-positivos cuando el huevo se cocina (tortilla/revoltillo/receta), (4)
auto-fix macro-preservante + idempotente, (5) top-up nunca añade huevo crudo a un batido.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import (
    _scan_raw_egg_violations,
    _apply_food_safety_fixes,
    _protein_topup_meal,
    _RAW_EGG_TERMS,
)


def _plan(*meals):
    return {"days": [{"day": 1, "meals": list(meals)}]}


# ---------- Detección ----------

def test_detecta_huevo_crudo_en_batido():
    plan = _plan({"name": "Batido Cremoso de Mango", "ingredients": ["1 mango (200g)", "½ huevo"]})
    viol = _scan_raw_egg_violations(plan)
    assert len(viol) == 1
    assert viol[0][3] == "blended"


def test_detecta_huevo_en_jugo_y_licuado():
    plan = _plan(
        {"name": "Jugo Verde Energético", "ingredients": ["espinaca", "1 huevo"]},
        {"name": "Licuado de Avena", "ingredients": ["avena", "2 claras de huevo"]},
    )
    viol = _scan_raw_egg_violations(plan)
    assert len(viol) == 2
    assert all(v[3] == "blended" for v in viol)


def test_detecta_huevo_sin_paso_de_coccion():
    # Wrap frío con huevo y sin indicador de cocción → no_cook.
    plan = _plan({
        "name": "Wrap de Batata y Ricotta",
        "ingredients": ["tortilla integral", "0.8 batata (160g)", "1¼ huevos"],
        "recipe": ["Rellena la tortilla con la batata y el huevo.", "Enrolla y sirve."],
    })
    viol = _scan_raw_egg_violations(plan)
    assert len(viol) == 1
    assert viol[0][3] == "no_cook"


# ---------- NO falsos positivos (huevo cocido) ----------

def test_no_flag_tortilla_por_nombre():
    plan = _plan({"name": "Tortilla de Huevos con Batata", "ingredients": ["2 huevos", "batata"]})
    assert _scan_raw_egg_violations(plan) == []


def test_no_flag_revoltillo_por_nombre():
    plan = _plan({"name": "Revoltillo Cremoso de Huevos con Queso", "ingredients": ["2 huevos", "queso"]})
    assert _scan_raw_egg_violations(plan) == []


def test_no_flag_cuando_receta_indica_coccion():
    plan = _plan({
        "name": "Desayuno Energético",
        "ingredients": ["2 huevos", "pan integral"],
        "recipe": ["Cocina los huevos en sartén hasta que cuajen.", "Sirve con el pan."],
    })
    assert _scan_raw_egg_violations(plan) == []


def test_no_flag_sin_huevo():
    plan = _plan({"name": "Batido de Mango con Yogur", "ingredients": ["mango", "yogur griego"]})
    assert _scan_raw_egg_violations(plan) == []


# ---------- Auto-fix ----------

def test_autofix_inyecta_nota_batido_y_es_macro_preservante():
    meal = {"name": "Batido de Mango", "ingredients": ["1 mango (200g)", "½ huevo"],
            "protein": 10, "carbs": 45, "fats": 13, "cals": 335, "recipe": ["Licúa todo."]}
    plan = _plan(meal)
    snapshot = {k: meal[k] for k in ("ingredients", "protein", "carbs", "fats", "cals")}
    n = _apply_food_safety_fixes(plan)
    assert n == 1
    # nota inyectada
    assert any("Seguridad alimentaria" in s for s in meal["recipe"])
    assert any("pasteurizado" in s.lower() or "proteína en polvo" in s.lower() for s in meal["recipe"])
    # macro-preservante: cantidades, macros y token del ingrediente intactos
    assert meal["ingredients"] == snapshot["ingredients"]
    for k in ("protein", "carbs", "fats", "cals"):
        assert meal[k] == snapshot[k]


def test_autofix_nota_nocook_distinta():
    meal = {"name": "Wrap Frío", "ingredients": ["tortilla", "1 huevo"], "recipe": ["Arma el wrap."]}
    plan = _plan(meal)
    _apply_food_safety_fixes(plan)
    nota = " ".join(meal["recipe"])
    assert "Seguridad alimentaria" in nota
    assert "71" in nota  # instrucción de cocción completa


def test_autofix_idempotente():
    meal = {"name": "Batido de Fresa", "ingredients": ["fresa", "1 huevo"], "recipe": ["Licúa."]}
    plan = _plan(meal)
    assert _apply_food_safety_fixes(plan) == 1
    largo = len(meal["recipe"])
    assert _apply_food_safety_fixes(plan) == 0  # segunda pasada no duplica
    assert len(meal["recipe"]) == largo


# ---------- Top-up nunca añade huevo crudo a un batido ----------

class _FakeInfo:
    def __init__(self, name, protein, carbs, fats, kcal):
        self.name = name
        self.protein = protein
        self.carbs = carbs
        self.fats = fats
        self.kcal = kcal


class _FakeDB:
    _MAP = {
        "claras de huevo": _FakeInfo("Claras de huevo", 11, 1, 0, 52),   # más magra (P/kcal)
        "yogur griego": _FakeInfo("Yogur griego", 10, 4, 0, 59),
    }

    def lookup(self, name):
        return self._MAP.get(str(name).strip().lower())


def test_topup_evita_huevo_crudo_en_batido():
    # Pool con claras (más magra) + yogur. En un batido NO debe elegir claras (crudo).
    meal = {"name": "Batido Proteico de Mango", "ingredients": ["mango"],
            "protein": 3, "carbs": 40, "fats": 1, "cals": 180}
    added = _protein_topup_meal(meal, 350.0, _FakeDB(), ["claras de huevo", "yogur griego"])
    assert added > 0
    nuevo = " ".join(meal["ingredients"]).lower()
    assert not any(t in nuevo for t in _RAW_EGG_TERMS), f"top-up metió huevo crudo: {nuevo}"
    assert "yogur" in nuevo


def test_topup_si_permite_huevo_en_comida_cocida():
    # En una comida normal (no batido) el huevo SÍ es candidato válido (se cocina).
    meal = {"name": "Tortilla Ligera", "ingredients": ["vegetales"],
            "protein": 3, "carbs": 5, "fats": 2, "cals": 60}
    added = _protein_topup_meal(meal, 300.0, _FakeDB(), ["claras de huevo"])
    assert added > 0
    assert "huevo" in " ".join(meal["ingredients"]).lower()
