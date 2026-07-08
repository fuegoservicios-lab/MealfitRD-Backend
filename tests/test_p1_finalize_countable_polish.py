"""[P1-FINALIZE-COUNTABLE-POLISH + P1-REWRITE-DORADO-HOMONYM + P2-CLOSER-STEP-BOUNDARY-DEDUP
+ P2-PACK-UNITS-MATCH + P2-SPICE-MOLIDO-DEFAULT · 2026-07-06] Batch "implementa" del
review #13 (plan fcc7a9f0).

1. CORRUPCIÓN del rewriter del autofix de proteína repetida (forense: bollitos día 3,
   `_protein_autofix_applied='pescado->pollo'`): (a) el alias "dorado" (pez) matcheaba el
   participio culinario → "hasta que estén pechuga de pollo y crujientes"; (b) los pasos
   no recibían los COMPUESTOS → "filete de pechuga de pollo blanco"; (c) las notas
   "se reemplazó X por Y" se reescribían perdiendo la procedencia.
2. Pulido de frontera del display: decimales re-introducidos por el re-trim de banda
   post-quantize ("2.97 limones", "1.49 dientes"), dup unidad-alimento ("1 filete de
   Filete de pescado blanco"), marca del súper en receta ("(Campos)"), cap cítrico/comida.
3. Paso 💪 "Incorpora X…" redundante cuando otros pasos ya trabajan X (piña + cottage).
4. Paquetes por UNIDADES reales del envase ("Burrito 5 unid"): density master 48g/u vs
   SKU 71g/u sub-compraba 4 paquetes = 20 tortillas para ~30.
5. Especias: default MOLIDO (no "Comino Entero" para receta que pide molido).
"""
import re
from pathlib import Path

import pytest

import graph_orchestrator as go
import shopping_calculator as sc


class _StubDB:
    def macros_from_ingredient_string(self, s):
        return None


# ───────────── 1. rewriter del autofix: dorado + compuestos + provenance ─────────────

def test_dorado_participle_survives_protein_swap(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_REPEAT_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "VARIETY_GATE_SAME_DAY_PROTEIN", True)
    day = {"day": 1, "meals": [
        {"name": "Pescado al Vapor con Hierbas",
         "ingredients": ["150 g de filete de pescado blanco"],
         "ingredients_raw": ["150 g de filete de pescado blanco"],
         "recipe": ["Montaje: sirve el pescado al vapor."]},
        {"name": "Bollitos de Yautía Rellenos de Queso",
         "ingredients": ["1 taza de yautía", "150 g de filete de pescado blanco"],
         "ingredients_raw": ["1 taza de yautia", "150 g de filete de pescado blanco"],
         "recipe": [
             "El Toque de Fuego: hornea 15 minutos, volteando a la mitad, hasta que "
             "estén dorados y crujientes por fuera.",
             "💪 Cocina filete de pescado blanco a la plancha o hervido y sírvelo "
             "como proteína del plato.",
             "Montaje: sirve los bollitos en un plato.",
         ]},
    ]}
    n = go._protein_repeat_autofix([day], form_data=None, db=_StubDB())
    assert n == 1, "pescado repetido el mismo día → una comida reescrita"
    blob = " ".join(str(s) for s in day["meals"][1]["recipe"])
    assert "dorados y crujientes" in blob, (
        f"'dorados' es participio culinario, NO el pez — jamás sustituirlo: {blob}"
    )
    assert "pechuga de pollo y crujientes" not in blob, f"corrupción fcc7a9f0: {blob}"
    assert "pechuga de pollo blanco" not in blob, (
        f"el compuesto 'filete de pescado blanco' se sustituye COMPLETO en pasos: {blob}"
    )
    assert "pechuga de pollo" in blob, "la nota 💪 sigue al swap (nombra la proteína nueva)"


def test_steps_rewriter_exempts_provenance_note():
    meal = {"recipe": [
        "⚠ Nota: se reemplazó camarón por pechuga de pollo por tu alergia a mariscos.",
        "Montaje: sirve el camarón con arroz.",
    ]}
    go._rewrite_recipe_steps_after_subs(meal, [(["camarón", "camaron"], "pechuga de pollo")])
    assert "se reemplazó camarón" in meal["recipe"][0], (
        "las notas de procedencia citan el alimento ORIGINAL a propósito"
    )
    assert "camarón" not in meal["recipe"][1].lower(), "los pasos reales SÍ se reescriben"


# ───────────── 2. pulido de frontera del display ─────────────

def _meal_days(ings, raw=None, recipe=None):
    return [{"day": 1, "meals": [{
        "name": "X", "ingredients": list(ings),
        "ingredients_raw": list(raw if raw is not None else ings),
        "recipe": list(recipe or ["Montaje: sirve."]),
    }]}]


def test_messy_decimals_snapped_display_only():
    days = _meal_days(["1.49 dientes de ajo", "3.39 dientes de ajo picado"],
                      raw=["1.49 dientes de ajo", "3.39 dientes de ajo picado"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert not any("1.49" in s or "3.39" in s for s in ings), (
        f"decimales del re-trim de banda → fracción de cocina: {ings}"
    )
    assert any("1.49" in s for s in raw) and any("3.39" in s for s in raw), (
        f"el RAW no se toca (fuente de macros/lista): {raw}"
    )


def test_citrus_meal_cap_display_and_raw():
    days = _meal_days(["2.97 limones"], raw=["2.97 limón"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert not any("2.97" in s for s in ings + raw), (
        f"3 limones para UN filete es inflación del LLM — cap {go.CITRUS_MEAL_CAP_UNITS:g}: "
        f"{ings} / {raw}"
    )
    assert any(re.match(r"^2 lim", s) for s in ings), f"cap a 2 en display: {ings}"
    assert any(re.match(r"^2 lim", s) for s in raw), (
        f"cap TAMBIÉN en raw — la lista deja de sobre-demandar contra P6-CITRUS-CAP: {raw}"
    )


def test_citrus_meal_cap_unicode_fraction():
    """[P1-CITRUS-UNICODE-FRAC · 2026-07-08] vivo: "2½ limones" (Plátano+queso+pescado) quedó SIN
    capear pese al cap por-comida — el regex exigía espacio tras el dígito y la fracción unicode
    pegada ("2½", sin espacio) lo evadía. Mismo modo de fallo que P1-COUNT-UNICODE-FRAC (aguacate)."""
    days = _meal_days(["2½ limones"], raw=["2½ limón"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert not any("2½" in s for s in ings + raw), (
        f"2.5 limones para un plato pequeño es inflación — cap {go.CITRUS_MEAL_CAP_UNITS:g}: "
        f"{ings} / {raw}"
    )
    assert any(re.match(r"^2 lim", s) for s in ings), f"cap a 2 en display: {ings}"
    assert any(re.match(r"^2 lim", s) for s in raw), f"cap TAMBIÉN en raw: {raw}"


def test_citrus_bare_fraction_under_cap_untouched():
    """"½ limón" (sin lead entero, fracción sola) no debe tocarse — 0.5 < cap 2."""
    days = _meal_days(["½ limón (jugo)"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert any(s.startswith("½ limón") for s in ings), f"no debe tocar bajo el cap: {ings}"


def test_unit_food_dup_collapsed():
    days = _meal_days(["1 filete de Filete de pescado blanco (150g)"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    raw = days[0]["meals"][0]["ingredients_raw"]
    for lst in (ings, raw):
        assert not any("filete de filete" in s.lower() for s in lst), f"dup fuera: {lst}"
        assert any("filete de pescado blanco" in s.lower() for s in lst), lst


def test_brand_paren_stripped_annotations_kept():
    days = _meal_days(["¼ taza de arroz blanco (Campos) (37 g crudo)",
                       "½ taza de queso ricotta (105 g)",
                       "1 limón (jugo)"])
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert not any("Campos" in s for s in ings), f"la marca vive en lista/Nevera: {ings}"
    assert any("(37 g crudo)" in s for s in ings), "anotaciones con dígitos se quedan"
    assert any("(105 g)" in s for s in ings), "anotación de gramos se queda"
    assert any("(jugo)" in s for s in ings), "anotación en minúscula se queda"


def test_polish_knob_off(monkeypatch):
    monkeypatch.setattr(go, "FINALIZE_DISPLAY_POLISH", False)
    days = _meal_days(["2.97 limones"])
    assert go._polish_finalize_display(days) == 0


# ───────────── 3. dedup boundary del paso 💪 ─────────────

def test_redundant_incorpora_step_removed():
    days = _meal_days(
        ["2 tazas de piña", "140 g de queso"],
        recipe=["Mise en place: pela la piña y córtala en cubos.",
                "💪 Incorpora queso a la preparación y mézclalo antes de servir.",
                "Montaje: coloca la piña en un bowl, añade el queso cottage y sirve."])
    n = go._dedup_redundant_closer_steps(days)
    rec = days[0]["meals"][0]["recipe"]
    assert n == 1 and not any("💪" in s for s in rec), (
        f"el Montaje YA añade el queso — el 💪 'Incorpora queso' sobra: {rec}"
    )


def test_incorpora_step_kept_when_unique():
    days = _meal_days(
        ["¼ taza de yogurt"],
        recipe=["Mise en place: tuesta el pan.",
                "💪 Incorpora yogurt a la preparación y mézclalo antes de servir.",
                "Montaje: unta la ricotta en las tostadas."])
    assert go._dedup_redundant_closer_steps(days) == 0, (
        "ningún otro paso trabaja el yogurt → el 💪 es su única instrucción, se queda"
    )


def test_cocina_step_never_removed():
    days = _meal_days(
        ["3 huevos"],
        recipe=["💪 Cocina huevo a la plancha o hervido y sírvelo como proteína del plato.",
                "Montaje: sirve con el huevo y el jugo de naranja al lado."])
    assert go._dedup_redundant_closer_steps(days) == 0, (
        "'Cocina X' lleva la ÚNICA instrucción de cocción — mencionar el alimento al "
        "servir no la reemplaza"
    )


# ───────────── 4. paquetes por unidades reales del envase ─────────────

def test_pack_count_by_declared_units():
    master = {"market_container": "paquete", "container_weight_g": 356,
              "density_g_per_unit": 48.0, "category": "Despensa",
              "market_packages": [
                  {"grams": 356, "price": 95, "label": "Burrito 5 unid 356 gr · Wala"}]}
    need_units = 29.7
    lbs = need_units * 48.0 / 453.592
    obj = sc.apply_smart_market_units("Tortilla de trigo", lbs, "lb", 0.0, master)
    assert obj["market_qty"] == 6, (
        f"~30 tortillas / 5 por paquete = 6 paquetes (por gramos del master salían "
        f"{obj}): la density master (48g/u) no es la del SKU (356/5=71g/u)"
    )


def test_pack_count_gram_path_unchanged_without_unid():
    master = {"market_container": "paquete", "container_weight_g": 907,
              "density_g_per_unit": None, "category": "Despensa",
              "market_packages": [
                  {"grams": 907, "price": 165, "label": "Organics Grano Largo 2 Lb · Goya"}]}
    obj = sc.apply_smart_market_units("Arroz integral", 1.9, "lb", 0.0, master)
    assert obj["market_qty"] == 1, f"sin 'unid' en el label, el path por gramos manda: {obj}"


# ───────────── 5. especias: default molido ─────────────

def test_spice_default_prefers_molido():
    defaults = {"comino": [
        {"grams": 28.35, "price": 50, "label": "Entero 1 Oz · Badia", "unit": "sobre"},
        {"grams": 56.7, "price": 85, "label": "Molido 2 Oz · Badia", "unit": "sobre"},
    ]}
    got = sc._resolve_brand_default("Comino", defaults)
    assert got and len(got) == 1 and "Molido" in got[0]["label"], (
        f"la receta pide comino MOLIDO — el entero no sustituye sin molinillo: {got}"
    )


def test_spice_fail_open_when_only_entero():
    defaults = {"comino": [
        {"grams": 28.35, "price": 50, "label": "Entero 1 Oz · Badia", "unit": "sobre"},
    ]}
    got = sc._resolve_brand_default("Comino", defaults)
    assert got and "Entero" in got[0]["label"], "solo entero en catálogo → fail-open"


def test_non_spice_molido_still_excluded():
    defaults = {"mani": [
        {"grams": 800, "price": 173, "label": "800 gr · Wala", "unit": "funda"},
        {"grams": 400, "price": 120, "label": "Molido 400 gr · Wala", "unit": "funda"},
    ]}
    got = sc._resolve_brand_default("Maní", defaults)
    assert got and all("Molido" not in p["label"] for p in got), (
        f"maní molido (mantequilla artesanal) NO es maní a secas: {got}"
    )


# ───────────── 6. la marca jamás en la receta (prompt) ─────────────

def test_brand_context_prompt_forbids_brand_in_recipe():
    src = (Path(go.__file__).resolve().parent / "brand_personalization.py").read_text(
        encoding="utf-8")
    i = src.index("P2-BRAND-NOT-IN-RECIPE")
    win = src[max(0, i - 1200):i + 1200]
    assert "JAMÁS escribas la marca" in win, (
        "el day-gen copiaba 'arroz blanco (Campos)' del contexto de marcas a la receta"
    )
