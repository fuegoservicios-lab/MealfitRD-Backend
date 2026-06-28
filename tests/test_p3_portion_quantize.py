"""[P3-PORTION-QUANTIZE · 2026-06-13] Cuantización de porciones a unidades de cocina medibles.

Hallazgo HIGH de la auditoría clínica: el solver/cal-reconcile (y el propio LLM) producen
fracciones decimales no medibles ('0.66 huevos', '3.87 papas', '0.53 taza', '3.74 rebanadas')
que el usuario no puede pesar/medir → matan la adherencia. El paso de cuantización las redondea
a incrementos de cocina (¼ taza, ¼ cda, ½ unidad discreta, 5 g) AJUSTANDO los macros del meal
por el delta exacto del aporte de cada ingrediente → receta↔macro↔lista quedan consistentes.

Cubre: (1) redondeo por tipo de unidad, (2) 'al gusto' limpia el número espurio, (3) hint de
gramos integerizado, (4) no-op cuando ya es medible, (5) consistencia macro vía delta en
_apply_portion_quantization, (6) ingredients_raw en lockstep.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import quantize_ingredient_string as q
from graph_orchestrator import _apply_portion_quantization


# ---------- Helper offline: redondeo por tipo de unidad ----------

def test_huevo_discreto_a_entero():
    # [P3-HUMAN-WHOLE-DISCRETE · 2026-06-28] huevo es indivisible → ENTERO (antes 0.5; medio huevo no es cocinable).
    out, fac = q("0.66 huevos enteros")
    assert out == "1 huevos enteros"
    assert fac > 1.0


def test_papa_redondea_a_entero():
    out, fac = q("3.87 papas medianas (580.23g)")
    assert out.startswith("4 papas")
    assert "(600g)" in out  # hint escalado + integerizado


def test_rebanada_a_entero():
    # [P3-HUMAN-WHOLE-DISCRETE · 2026-06-28] rebanada de pan → ENTERO (antes 3.5; media rebanada no es práctica).
    out, _ = q("3.74 rebanadas de pan integral (112.08g)")
    assert out.startswith("4 rebanadas")


def test_taza_a_cuarto():
    out, _ = q("0.53 taza de avena (52.68g)")
    assert out.startswith("0.5 taza")
    assert "(50g)" in out


def test_cucharada_a_cuarto():
    out, _ = q("1.27 cucharada de aceite de oliva (19.12ml)")
    assert out.startswith("1.25 cucharada")
    assert "(19ml)" in out


def test_gramos_se_redondean_a_5():
    out, _ = q("113.85g de lomo de cerdo")
    assert out == "115g de lomo de cerdo"


def test_al_gusto_quita_numero_espurio():
    out, fac = q("0.91 sal y pimienta al gusto")
    assert out == "Sal y pimienta al gusto"
    assert fac == 1.0  # no afecta macros


def test_fraccion_minima_sube_a_medio():
    # Porción absurda del LLM/solver → ½ unidad medible (trade-off aceptado).
    out, fac = q("0.15 platano maduro grande (29.91g)")
    assert out.startswith("0.5 platano")
    assert fac > 1.0


def test_ya_medible_no_op():
    out, fac = q("1 taza de arroz integral (200g)")
    assert fac == 1.0
    assert out.startswith("1 taza")


def test_sin_cantidad_lider_intacto():
    out, fac = q("Sal al gusto")
    assert fac == 1.0


def test_unicode_fraction_medible_se_preserva():
    # ½ ya es una medida de cocina válida → se preserva tal cual (no-op).
    out, fac = q("½ taza de leche (120ml)")
    assert fac == 1.0
    assert out.startswith("½ taza")


def test_tercio_de_taza_se_preserva():
    # ⅓ taza existe como taza medidora → NO debe redondearse a ¼.
    out, fac = q("⅓ taza de arroz (65g)")
    assert fac == 1.0  # 0.333 ya es una fracción permitida


def test_fraccion_no_estandar_snap_a_tercio():
    # 0.30 taza → snap a ⅓ (0.33), la fracción permitida más cercana.
    out, _ = q("0.30 taza de arroz (60g)")
    assert out.startswith("0.33 taza")


def test_numero_mixto_unicode_se_parsea():
    # '1¼ huevos' (=1.25) debe parsearse y cuantizar a unidad medible, NO quedar intacto.
    out, fac = q("1¼ huevos")
    assert not out.startswith("1¼")
    assert out.startswith("1 huevos") or out.startswith("1.5 huevos")
    assert fac != 1.0


# ---------- _apply_portion_quantization: consistencia macro vía delta ----------

class _FakeInfo:
    def __init__(self, name, protein, carbs, fats, kcal, dpu=None, dpc=None):
        self.name = name
        self.protein, self.carbs, self.fats, self.kcal = protein, carbs, fats, kcal
        self.density_g_per_unit = dpu
        self.density_g_per_cup = dpc
        self.fiber = self.sodium_mg = None
        self.source = "test"
        self.fdc_id = None
        self.is_dominican = False
        self.container_weight_g = None


class _FakeDB:
    """Implementa solo macros_from_ingredient_string usando el hint de gramos."""
    import re as _re
    _HINT = _re.compile(r"\((\d+(?:\.\d+)?)\s*(g|ml)\b[^)]*\)", _re.I)
    _MAP = {"huevo": _FakeInfo("Huevos", 13, 1, 11, 155)}

    def macros_from_ingredient_string(self, s):
        m = self._HINT.search(s)
        if not m:
            # huevo discreto sin hint: ~50g por unidad
            if "huevo" in s.lower():
                # cantidad líder
                import re
                lead = re.match(r"\s*([\d.]+)", s)
                qty = float(lead.group(1)) if lead else 1.0
                grams = qty * 50.0
            else:
                return None
        else:
            grams = float(m.group(1))
        info = self._MAP["huevo"]
        f = grams / 100.0
        return {"name": info.name, "grams": grams, "kcal": info.kcal * f,
                "protein": info.protein * f, "carbs": info.carbs * f, "fats": info.fats * f}


def test_apply_quantization_ajusta_macros_por_delta():
    # [P3-HUMAN-WHOLE-DISCRETE · 2026-06-28] 0.66 huevos → 1 huevo entero (sube): el delta de macros se SUMA
    # consistentemente (antes el huevo iba a 0.5 y bajaba; ahora va a entero y sube — el mecanismo de delta es el mismo).
    meal = {"name": "Tortilla", "ingredients": ["0.66 huevos enteros"],
            "protein": 30, "carbs": 5, "fats": 10, "cals": 250}
    plan = {"days": [{"day": 1, "meals": [meal]}]}
    n = _apply_portion_quantization(plan, _FakeDB())
    assert n == 1
    assert meal["ingredients"][0] == "1 huevos enteros"
    # macros subieron (porción mayor) pero siguen consistentes
    assert meal["protein"] > 30
    assert meal["cals"] > 250
    assert meal["macros"][0] == f"P:{meal['protein']}g"


def test_apply_quantization_no_op_cuando_todo_medible():
    meal = {"name": "Arroz", "ingredients": ["1 taza de arroz integral (200g)"],
            "protein": 5, "carbs": 45, "fats": 1, "cals": 210}
    plan = {"days": [{"day": 1, "meals": [meal]}]}
    assert _apply_portion_quantization(plan, _FakeDB()) == 0
    assert meal["protein"] == 5  # intacto


def test_ingredients_raw_en_lockstep():
    meal = {"name": "Avena", "ingredients": ["0.53 taza de avena (52.68g)"],
            "ingredients_raw": ["0.53 taza de avena (52.68g)"],
            "protein": 8, "carbs": 30, "fats": 4, "cals": 180}
    plan = {"days": [{"day": 1, "meals": [meal]}]}
    _apply_portion_quantization(plan, _FakeDB())
    assert meal["ingredients"][0].startswith("0.5 taza")
    assert meal["ingredients_raw"][0].startswith("0.5 taza")  # mismo factor
