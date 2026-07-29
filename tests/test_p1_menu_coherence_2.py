"""[P1-MENU-COHERENCE-2 · 2026-07-29] "Las combinaciones deben ser coherentes para que
apetezca" (owner, plan vivo c8fa78c5). Cinco cierres:

1. [CLARAS-DE-RES] "Batido Refrescante de Sandía y Claras de RES": el sinónimo SUELTO
   "molida" de res matcheó "linaza MOLIDA" → el namefix creyó que había res real y
   sustituyó el fantasma (12ª mordida, esta vez de sinónimo). Solo frases inequívocas
   ("carne molida"/"res molida"); y las CLARAS entran como frase completa a los syns de
   huevo (sustituir solo el sustantivo fabrica "Claras de <X>").
2. [EGG-SWAP-SYNC] El swap huevo-crudo→yogur cambiaba la COMPOSICIÓN pero nombre y pasos
   seguían vendiendo claras ("Mide… las claras pasteurizadas") → sincronía total.
3. [YOGURT-SAVORY] ¾ taza de yogurt + "Sirve yogurt al lado" en Ropa Vieja con Tostones:
   en plato SALADO y COCINADO el lácteo dulce del closer queda como último recurso.
4. [RAW-FLOUR-BOLT] "20 g de harina de trigo" suelta en TOSTADAS de pan + paso fabricado:
   harina sin contexto de masa se remueve (el pool de harinas del seeder queda intacto —
   P1-FLOURS-POOLS es pedido del owner; las arepitas que SÍ amasan no se tocan).
5. [NAMED-DAIRY-BUMP] "Ropa Vieja de QUESO de Freír" con 15 g de queso: lácteo del nombre
   presente-pero-mínimo se bumpea al piso del knob (30 g), sin insertar línea nueva.

tooltip-anchor: P1-MENU-COHERENCE-2
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_GO = (Path(__file__).resolve().parents[1] / "graph_orchestrator.py").read_text(encoding="utf-8")


# ─────────────── 1. claras de res ───────────────

def test_molida_alone_is_not_res():
    assert "molida" not in go._PHANTOM_PROTEIN_SYNS["res"], "solo frases (carne molida/res molida)"
    assert "carne molida" in go._PHANTOM_PROTEIN_SYNS["res"]


def test_batido_claras_no_longer_renamed_to_res():
    meal = {"name": "Batido Refrescante de Sandía y Claras de Huevo",
            "ingredients": ["1½ tazas de sandía en cubos", "½ taza de Yogurt griego sin azúcar",
                            "1¼ cdtas de linaza molida", "Sal al gusto"],
            "recipe": ["Licúa todo."]}
    renamed = go._fix_phantom_protein_in_name(meal, go.strip_accents)
    assert renamed is False, "sin carne real (la linaza molida NO es res) → no renombrar"
    assert meal["name"] == "Batido Refrescante de Sandía y Claras de Huevo"
    assert meal.get("_name_honesty_degraded") is True, "flag honesto en vez de mentira"


def test_claras_phrase_replaced_whole_when_real_meat_exists():
    meal = {"name": "Wrap de Claras de Huevo con Vegetales",
            "ingredients": ["120 g de pechuga de pollo", "2 tortillas"],
            "recipe": ["Arma el wrap."]}
    assert go._fix_phantom_protein_in_name(meal, go.strip_accents) is True
    assert "claras" not in meal["name"].lower()
    assert "de Pollo" in meal["name"] or "Pollo" in meal["name"]


# ─────────────── 2. swap huevo→yogur sincroniza ───────────────

def test_egg_swap_syncs_name_and_steps():
    meal = {"name": "Batido Refrescante de Sandía y Claras de Huevo",
            "ingredients": ["1½ tazas de sandía en cubos", "3 claras de huevo",
                            "1¼ cdtas de linaza molida"],
            "ingredients_raw": ["1½ tazas de sandía en cubos", "3 claras de huevo",
                                "1¼ cdtas de linaza molida"],
            "recipe": ["Mise en place: mide la sandía en cubos, las claras pasteurizadas y la linaza.",
                       "Montaje: licúa la sandía, las claras y la linaza. Sirve."],
            "protein": 15, "carbs": 20, "fats": 5, "cals": 190}
    assert go._substitute_blended_raw_egg(meal, None) is True
    blob_i = " ".join(meal["ingredients"]).lower()
    assert "clara" not in blob_i and "huevo" not in blob_i
    blob_r = " ".join(meal["recipe"]).lower()
    assert "clara" not in blob_r, f"pasos siguen vendiendo claras: {meal['recipe']}"
    assert "yogur griego" in blob_r
    assert "Claras" not in meal["name"] and "Yogurt Griego" in meal["name"]


# ─────────────── 3. yogurt en salado caliente ───────────────

def test_closer_savory_hot_filters_sweet_dairy_anchor():
    i = _GO.index("def _close_protein_gap_for_meal")
    j = _GO.index("\ndef ", i + 10)
    body = _GO[i:j]
    assert "_meal_is_hot_cooked(meal, _sa)" in body, (
        "el filtro de lácteo dulce en plato salado caliente debe vivir en el closer"
    )
    assert body.count("P1-MENU-COHERENCE-2") >= 1
    assert "_pool_no_sd" in body and "_SWEET_DAIRY_TOKENS" in body


class _MC2Info:
    def __init__(self, name, protein, carbs=2.0, fats=1.0, kcal=95.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


def _mc2_savory_hot_meal():
    # Merienda salada COCINADA ("Saltea 5 min" en los pasos → _meal_is_hot_cooked=True),
    # sin proteína animal principal aún (pool no se vacía por P1-CLOSER-NO-DOUBLE-MAIN).
    return {
        "meal": "Merienda", "name": "Sándwich integral de vegetales",
        "protein": 5, "carbs": 30, "fats": 8, "cals": 212,
        "ingredients": ["2 rebanadas de pan integral", "1/2 taza de vegetales salteados"],
        "ingredients_raw": ["2 rebanadas de pan integral", "1/2 taza de vegetales salteados"],
        "recipe": ["MISE EN PLACE: Prepara los vegetales.",
                   "EL TOQUE DE FUEGO: Saltea 5 min.",
                   "MONTAJE: Arma el sándwich."],
    }


def test_closer_savory_hot_filters_sweet_dairy_functional(monkeypatch):
    """[P1-MENU-COHERENCE-2 finding #3] La versión parser-based de arriba no puede detectar
    un closure inerte (código presente, efecto ausente — sabotaje verificado: reemplazar
    `_pool = _pool_no_sd` por `pass` deja el test de arriba en verde). Llamando a la función
    REAL: con una alternativa NO-láctea-dulce disponible, el yogurt debe quedar filtrado del
    plato salado caliente (mismo patrón que test_p1_closer_sweet_guard.py, sin DB)."""
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [
        (1.0, "Yogurt griego", _MC2Info("Yogurt griego", protein=19.0, kcal=97.0)),
        (2.0, "Pechuga de pollo", _MC2Info("Pechuga de pollo", protein=31.0, kcal=165.0)),
    ]
    meal = _mc2_savory_hot_meal()
    g = go._close_protein_gap_for_meal(meal, 22.0, None, candidates, enforce_min_threshold=False)
    assert g > 0
    blob = " ".join(meal["ingredients"]).lower()
    assert "yogurt" not in blob and "pollo" in blob, (
        f"plato salado+caliente con alternativa disponible debe filtrar el lácteo dulce; "
        f"ingredients={meal['ingredients']}"
    )


def test_closer_savory_hot_keeps_sweet_dairy_as_last_resort(monkeypatch):
    """Contraparte del test anterior: SIN alternativa no-láctea-dulce, el yogurt se conserva
    (piso de proteína manda — 'pool quedaría vacío ⇒ se conserva', docstring del closer)."""
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [(1.0, "Yogurt griego", _MC2Info("Yogurt griego", protein=19.0, kcal=97.0))]
    meal = _mc2_savory_hot_meal()
    g = go._close_protein_gap_for_meal(meal, 22.0, None, candidates, enforce_min_threshold=False)
    assert g > 0
    blob = " ".join(meal["ingredients"]).lower()
    assert "yogurt" in blob, f"sin alternativa, el yogurt es último recurso; ingredients={meal['ingredients']}"


# ─────────────── 4. harina cruda de cumplimiento ───────────────

def test_raw_flour_bolt_removed_without_dough_context():
    days = [{"meals": [
        {"name": "Tostadas de Harina de Trigo con Mango",
         "ingredients": ["2 rebanadas de pan integral", "20g de harina de trigo",
                         "½ mango mediano"],
         "ingredients_raw": ["2 rebanadas de pan integral", "20g de harina de trigo",
                             "½ mango mediano"],
         "recipe": ["Mise en place: tuesta las rebanadas de pan.",
                    "Incorpora también harina de trigo durante la preparación.",
                    "Montaje: unta la mantequilla de maní."]},
        {"name": "Arepitas de Harina con Hígado",
         "ingredients": ["½ taza de harina de trigo (60 g)", "150 g de hígado de cerdo"],
         "ingredients_raw": ["½ taza de harina de trigo (60 g)", "150 g de hígado de cerdo"],
         "recipe": ["Mezcla la harina de trigo con sal; amasa hasta obtener una masa suave.",
                    "Cocina las arepas 3-4 minutos por lado."]},
    ]}]
    assert go._strip_raw_flour_compliance_bolt(days) == 1
    m0, m1 = days[0]["meals"]
    assert not any("harina" in str(s).lower() for s in m0["ingredients"])
    assert not any("harina" in str(s).lower() for s in m0["recipe"]), "paso-bolt fuera"
    assert any("harina" in str(s).lower() for s in m1["ingredients"]), "las arepitas amasan → intactas"
    assert go._strip_raw_flour_compliance_bolt(days) == 0, "idempotente"


# ─────────────── 5. lácteo del nombre presente-pero-mínimo ───────────────

def test_named_dairy_present_but_tiny_bumped():
    days = [{"day": 1, "meals": [{
        "name": "Ropa Vieja de Queso de Freír con Champiñones",
        "ingredients": ["15 g de queso de freír", "2½ tazas de champiñones"],
        "ingredients_raw": ["15 g de queso de freír", "2½ tazas de champiñones"],
        "recipe": ["Desmenuza el queso."]}]}]
    out = go._repair_name_phantom_dairy(days)
    assert any(x.get("bumped_from_g") == 15.0 for x in out), out
    assert days[0]["meals"][0]["ingredients"][0].startswith(f"{go.NAME_PHANTOM_DAIRY_G} g de queso")


def test_named_dairy_honest_grams_untouched():
    days = [{"day": 1, "meals": [{
        "name": "Ensalada con Queso Blanco",
        "ingredients": ["40 g de queso blanco fresco"],
        "ingredients_raw": ["40 g de queso blanco fresco"],
        "recipe": ["Mezcla."]}]}]
    assert go._repair_name_phantom_dairy(days) == []
    assert days[0]["meals"][0]["ingredients"][0].startswith("40 g")


def test_named_dairy_bump_no_false_hit_on_requeson():
    """[P1-MENU-COHERENCE-2-BUMP-BOUNDARY] 'queso' ⊂ 'requeson' (strip_accents('requesón')) es
    substring, no palabra. Con una línea de requesón ANTES de la línea real de queso, el bare
    `in` del bump loop la tomaba como el lácteo del nombre — sin \b se creía 'ya honesto'
    (>=30g) y la línea real de 15g de queso quedaba sin tocar. Debe bumpear la línea REAL."""
    days = [{"day": 1, "meals": [{
        "name": "Ropa Vieja de Queso de Freír con Tostones",
        "ingredients": ["50 g de requesón", "15 g de queso de freír", "2 tostones"],
        "ingredients_raw": ["50 g de requesón", "15 g de queso de freír", "2 tostones"],
        "recipe": ["Desmenuza el queso."]}]}]
    out = go._repair_name_phantom_dairy(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0] == "50 g de requesón", "el requesón (35g por encima del piso) no se toca"
    assert ings[1].startswith(f"{go.NAME_PHANTOM_DAIRY_G} g de queso"), (
        f"la línea REAL de queso (15g) debía bumpearse al piso; ingredients={ings}")
    assert any(x.get("bumped_from_g") == 15.0 and x.get("food") == "queso" for x in out), out
