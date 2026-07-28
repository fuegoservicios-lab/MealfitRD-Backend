"""[P1-VISUAL-AUDIT-3 · 2026-07-28] 3ª revisión visual (plan 9af221fb — el del arco
minor→high→quirúrgico con banda 1.00). Presupuesto DENTRO por primera vez; 7 defectos
nuevos → 7 reparaciones. Cada caso ancla el texto VIVO exacto.

1. [P1-FRESH-CANNED-PROSE] "Escurre filete de pescado blanco…, reservando el líquido
   de la lata" con FILETE FRESCO (y pechuga "escurrida" y desmenuzada CRUDA): verbos de
   enlatado sobre proteína fresca → prosa de fresco/batch. El atún real queda intacto.
2. [P1-BAKING-POWDER-CAP] "1¼ cdas de polvo de hornear" (~18 g) → 1 cdta.
3. [P1-RICE-SIDE-MIN-COOKABLE] "15 g de arroz blanco crudo" (2ª aparición) → piso
   cocinable CLOSER_COOKABLE_MIN_G (40 g).
4. [P1-HERB-COUNT-GENDER] "3 perejil fresco"→"3 ramitas de perejil…" + "Pescado blanco
   Guisadas"→"Guisado" (femeninos legítimos intactos: Habichuelas/Papas Guisadas).
5. [P1-HOT-DAIRY-ASIDE] yogurt del closer sobre revoltillo CALIENTE → "al lado"
   (en bowls fríos sigue el mezclar; el guiso-de-yogurt del LLM no pasa por aquí).
6. [P1-OVERCOVER-LABEL] cottage 16 Oz cover ~5× en sección semanal sin letrero →
   "alcanza ~14 días — no recompres cada semana" (capado por vida útil).
7. mandarinas al half-count range ("2½ mandarinas" → "2–3 mandarinas").

tooltip-anchor: P1-VISUAL-AUDIT-3
"""
from __future__ import annotations

import graph_orchestrator as go
import humanize_ingredients as hi
import shopping_calculator as sc
from constants import strip_accents


def test_fresh_canned_prose_fix():
    days = [{"meals": [
        {"name": "Puré con Pescado", "ingredients": ["1 filete de pescado"],
         "recipe": ["Escurre filete de pescado blanco y desmenúzalas ligeramente con un tenedor, reservando el líquido de la lata.",
                    "Incorpora el tomate y el líquido reservado de filete de pescado blanco; cocina 5 minutos."]},
        {"name": "Tortilla de Pechuga", "ingredients": ["1 pechuga de pollo (≈200 g)"],
         "recipe": ["Escurre pechuga de pollo y desmenúzalas con un tenedor."]},
        {"name": "Ensalada de atún", "ingredients": ["1 lata de atún en agua"],
         "recipe": ["Escurre atún en agua y desmenúzalo, reservando el líquido de la lata."]},
    ]}]
    assert go._fresh_protein_canned_prose_fix(days) == 2
    r0 = days[0]["meals"][0]["recipe"]
    assert "lata" not in r0[0] and "Corta filete de pescado blanco en trozos" in r0[0]
    assert "chorrito de agua" in r0[1]
    assert "ya cocida" in days[0]["meals"][1]["recipe"][0]
    assert "líquido de la lata" in days[0]["meals"][2]["recipe"][0]  # atún real intacto


def test_baking_powder_cap():
    d = [{"meals": [{"ingredients": ["1¼ cdas de polvo de hornear", "½ cdta de polvo de hornear"]}]}]
    assert go._baking_powder_cap_pass(d) == 1
    assert d[0]["meals"][0]["ingredients"] == ["1 cdta de polvo de hornear", "½ cdta de polvo de hornear"]


def test_rice_side_min_cookable():
    d = [{"meals": [{"ingredients": ["15 g de arroz blanco crudo", "100 g de arroz blanco crudo"],
                     "protein": 5, "carbs": 30, "fats": 2, "cals": 160}]}]
    assert go._floor_subservible_portions(d) >= 1
    assert d[0]["meals"][0]["ingredients"][0].startswith("40 g de arroz")
    assert d[0]["meals"][0]["ingredients"][1].startswith("100 g")


def test_herb_count_and_gender():
    d = [{"meals": [{"name": "Pescado blanco Guisadas al Caribe",
                     "ingredients": ["3 perejil fresco para decorar", "5 cilantro fresco",
                                     "3 ramitas de tomillo"],
                     "recipe": ["Corona con pollo guisadas."]},
                    {"name": "Habichuelas Guisadas", "ingredients": [], "recipe": []},
                    {"name": "Papas Guisadas con Res", "ingredients": [], "recipe": []}]}]
    assert go._herb_count_and_gender_polish(d) == 1
    m = d[0]["meals"][0]
    assert m["name"] == "Pescado blanco Guisado al Caribe"
    assert m["ingredients"][0].startswith("3 ramitas de perejil")
    assert m["ingredients"][1].startswith("5 ramitas de cilantro")
    assert m["ingredients"][2] == "3 ramitas de tomillo"
    assert "pollo guisado" in m["recipe"][0]
    assert d[0]["meals"][1]["name"] == "Habichuelas Guisadas"
    assert d[0]["meals"][2]["name"] == "Papas Guisadas con Res"


def test_hot_dairy_aside():
    rev = {"name": "Revoltillo Ligero con Papas y Yogurt",
           "recipe": ["El Toque de Fuego: sofríe y cuaja el huevo."]}
    bowl = {"name": "Bowl de Frutas con Yogurt", "recipe": ["Montaje: coloca la fruta en un bowl."]}
    assert go._meal_is_hot_cooked(rev, strip_accents) is True
    assert go._meal_is_hot_cooked(bowl, strip_accents) is False
    # "guisa" ⊂ "GUISAntes" (10ª mordida de subcadena del maratón): el plato frío
    # "Batata con Guisantes" NO es caliente; la familia guis- culinaria SÍ.
    assert go._meal_is_hot_cooked(
        {"name": "Batata con Guisantes", "recipe": ["Montaje: sirve."]}, strip_accents) is False
    assert go._meal_is_hot_cooked(
        {"name": "Pollo Guisado", "recipe": []}, strip_accents) is True
    assert go._meal_is_hot_cooked(
        {"name": "X", "recipe": ["Guisa el pollo 10 min."]}, strip_accents) is True
    assert go._closer_protein_step_text("yogurt", True, baked=True) == \
        "Sirve yogurt al lado para acompañar."
    assert "mézclalo" in go._closer_protein_step_text("yogurt", True, baked=False)


def test_overcover_label_in_hybrid_stage():
    weekly = [
        {"name": "Queso cottage", "display_qty": "1 tarro (16 Oz · Westby)",
         "pkg_cover_ratio": 4.977, "shelf_life_days": 14, "category": "Lácteos"},
        {"name": "Pulpo", "display_qty": "¾ lb", "pkg_cover_ratio": 0.97,
         "shelf_life_days": 2, "category": "Proteínas"},
    ]
    out = sc._build_hybrid_shopping_list(weekly, [], {}, None, None)
    cot = next(i for i in out if i["name"] == "Queso cottage")
    pul = next(i for i in out if i["name"] == "Pulpo")
    assert "alcanza ~14 días — no recompres cada semana" in cot["display_qty"]
    assert "alcanza" not in pul["display_qty"]


def test_mandarina_half_count_range():
    assert hi.half_count_range("2½ mandarinas medianas") == "2–3 mandarinas medianas"
