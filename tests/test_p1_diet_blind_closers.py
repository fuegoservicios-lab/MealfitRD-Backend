# [P1-DIET-BLIND-CLOSERS · 2026-08-08] CAPA 2 del cierre veg* (hermano de
# P1-DIET-BLIND-DIRECTIVES, que arregló los PROMPTS): los closers DETERMINISTAS del piso de
# proteína elegían candidatos filtrando por ALERGIAS pero no por DIETA.
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_GO_SRC = None


def _go_src():
    global _GO_SRC
    if _GO_SRC is None:
        p = os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py")
        _GO_SRC = open(p, encoding="utf-8").read()
    return _GO_SRC


# ---------------------------------------------------------------------------
# 7. CAPA 2 — closers deterministas: candidatos de proteína por dieta
#    (verificación post-deploy 2026-08-08 04:09: el LLM ya generaba conceptos veg* limpios
#    —«Yogur de Soya», «Tofu», «Locrio de lentejas»— y el closer del piso de proteína les
#    atornillaba '230g de atún'/'225g de camarones'/'2 tazas de Yogurt' con el nombre
#    reflejado (', Atún en Agua'). _safe_high_density_proteins filtraba por ALERGIAS pero
#    no por DIETA.)
# ---------------------------------------------------------------------------

class _FakeInfo:
    protein = 20.0
    kcal = 100.0


class _FakeNutDB:
    def lookup(self, name):
        return _FakeInfo()


def _viola_dieta(nombre, canon):
    import graph_orchestrator as go
    mini = {"days": [{"meals": [{"name": str(nombre), "ingredients": [str(nombre)]}]}]}
    return bool(go._scan_diet_violations(mini, canon))


def test_closer_candidates_vegan_sin_animal_ni_lacteo():
    import graph_orchestrator as go
    cands = go._safe_high_density_proteins(["Ninguna"], _FakeNutDB(), diet="vegana")
    malos = [n for _, n, _ in cands if _viola_dieta(n, "vegan")]
    assert malos == [], f"el closer aún ofrecería a un vegano: {malos}"


def test_closer_candidates_vegetarian_sin_carne_pescado():
    import graph_orchestrator as go
    cands = go._safe_high_density_proteins(["Ninguna"], _FakeNutDB(), diet="vegetariana")
    malos = [n for _, n, _ in cands if _viola_dieta(n, "vegetarian")]
    assert malos == [], f"el closer aún ofrecería a un vegetariano: {malos}"
    nombres = " ".join(n for _, n, _ in cands).lower()
    assert ("queso" in nombres) or ("huevo" in nombres) or ("yogurt" in nombres), (
        "vegetarian debe conservar lácteo/huevo como candidatos")


def test_closer_candidates_balanced_intactos():
    import graph_orchestrator as go
    a = [n for _, n, _ in go._safe_high_density_proteins(["Ninguna"], _FakeNutDB())]
    b = [n for _, n, _ in go._safe_high_density_proteins(["Ninguna"], _FakeNutDB(), diet=None)]
    c = [n for _, n, _ in go._safe_high_density_proteins(["Ninguna"], _FakeNutDB(), diet="balanced")]
    assert a == b == c and a, "balanced debe quedar byte-idéntico (y no vacío)"


def test_wiring_closer_callsites_pasan_dieta():
    # Los 4 builders de candidatos + el sweet-dairy interno deben pasar la dieta. Sin este
    # wiring el filtro existe pero nadie lo llama (código inerte — clase P1-G).
    src = _go_src()
    # ventana DOTALL acotada: la llamada abarca varias líneas y contiene parens anidados
    # (form_data.get(...)), así que [^)]* se cortaría antes de diet=.
    n_diet_calls = len(re.findall(
        r"_safe_high_density_proteins\(.{0,240}?diet\s*=", src, re.DOTALL))
    assert n_diet_calls >= 4, (
        f"solo {n_diet_calls} call sites de _safe_high_density_proteins pasan diet= (esperados ≥4)")


