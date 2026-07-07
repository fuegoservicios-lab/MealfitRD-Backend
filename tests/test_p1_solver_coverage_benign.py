"""[P1-SOLVER-COVERAGE-BENIGN · 2026-07-07] Refinamiento forense del coverage-gate del solver.

Forense post-deploy (60 planes → 50 meals): 32% caía a "parcial" (cap 2.0x) SOLO por tener "½ taza
de agua" o "1 cda de cilantro" entre sus líneas — el agua no aporta macros y las hierbas puras son
guarnición despreciable, así que NO son "masa oculta" pero arrastraban la cobertura hacia abajo.
Fix: excluir agua + hierbas del DENOMINADOR de cobertura (como los condimentos "al gusto"), con
word-boundary (\bagua\b NO matchea "aguacate"). Vegetales/frutas VAGAS ("ensalada verde"/"frutas
variadas") NO se excluyen — esas sí son masa no-resuelta legítima que debe abstener/capear.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _FakeDB:
    _MAC = {
        "pollo": {"protein": 30.0, "carbs": 0.0, "fats": 3.0, "kcal": 150.0},
        "arroz": {"protein": 3.0, "carbs": 40.0, "fats": 0.5, "kcal": 180.0},
    }

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        for tok, mac in self._MAC.items():
            if tok in low:
                return dict(mac)
        return None  # agua / cilantro / ensalada → no-resuelto


# ─────────────────────────── Parser + marker ───────────────────────────

def test_marker_and_regex_wired_in_gate():
    assert "P1-SOLVER-COVERAGE-BENIGN" in _GO
    assert "_SOLVER_COV_BENIGN_RE = _re.compile(" in _GO
    i = _GO.index("def _apply_macro_solver_to_meal")
    win = _GO[i:i + 3800]
    assert "_SOLVER_COV_BENIGN_RE.search(_cs_low)" in win  # usado en el denominador de cobertura


# ─────────────────────────── Regex word-boundary (substring-safe) ───────────────────────────

def test_benign_regex_is_word_bounded():
    import graph_orchestrator as g
    r = g._SOLVER_COV_BENIGN_RE
    # agua/hierbas → excluidos
    assert r.search("agua fria")
    assert r.search("½ taza de agua (125ml)")
    assert r.search("1 cda de cilantro fresco picado")
    assert r.search("1 ramita de perejil")
    # 'res'↔'fresas': agua NO debe matchear aguacate; menta NO debe colarse por pimienta
    assert not r.search("150 g de aguacate")
    assert not r.search("1 cdta de pimienta")
    # comida vaga NO-benigna → SIGUE contando como masa no-resuelta
    assert not r.search("ensalada verde")
    assert not r.search("frutas variadas de temporada")


# ─────────────────────────── Funcional: agua/hierbas no arrastran cobertura ───────────────────────────

def test_water_and_herbs_do_not_drag_coverage():
    """Sin la exclusión: 2 comida + 2 agua/hierba = 2/4 = 0.5 < 0.6 → abstendría. CON la exclusión:
    agua+cilantro fuera → 2/2 = 1.0 → el solver aplica y escala (trata la comida como full-coverage)."""
    import graph_orchestrator as g
    meal = {
        "name": "Pollo con arroz",
        "ingredients": ["100 g de pollo", "100 g de arroz",
                        "½ taza de agua (125ml)", "1 cda de cilantro fresco picado"],
        "ingredients_raw": ["100 g de pollo", "100 g de arroz",
                            "½ taza de agua (125ml)", "1 cda de cilantro fresco picado"],
        "protein": 33, "carbs": 40, "fats": 4, "cals": 330,
    }
    changed = g._apply_macro_solver_to_meal(
        meal, {"kcal": 600.0, "protein": 55.0, "carbs": 60.0, "fats": 15.0}, _FakeDB())
    assert changed is True, "agua/cilantro no deben arrastrar la comida a la abstención"
    assert meal["protein"] > 33  # escaló como full-coverage (no abstuvo)


def test_genuine_unresolved_bulk_still_abstains():
    """Contraprueba: 'frutas variadas' (comida vaga, NO benigna) SÍ arrastra la cobertura → abstiene,
    aunque el agua acompañante se excluya. 1 comida + 1 fruta-vaga = 1/2 = 0.5 < 0.6."""
    import graph_orchestrator as g
    meal = {
        "name": "Frutas con Semillas",
        "ingredients": ["frutas variadas de temporada", "½ taza de agua (125ml)", "100 g de arroz"],
        "ingredients_raw": ["frutas variadas de temporada", "½ taza de agua (125ml)", "100 g de arroz"],
        "protein": 4, "carbs": 45, "fats": 2, "cals": 210,
    }
    before = list(meal["ingredients"])
    changed = g._apply_macro_solver_to_meal(
        meal, {"kcal": 600.0, "protein": 40.0, "carbs": 60.0, "fats": 20.0}, _FakeDB())
    assert changed is False  # 'frutas variadas' (bulk vago) mantiene la cobertura baja → abstiene
    assert meal["ingredients"] == before
