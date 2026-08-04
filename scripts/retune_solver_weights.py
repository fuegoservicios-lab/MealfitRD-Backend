#!/usr/bin/env python
"""[P3-SOLVER-W-RETUNE · 2026-08-04] Harness OFFLINE de re-tune de los CUATRO pesos
`SOLVER_W_*` del LSQ de `portion_solver.py`.

Por qué existe: el comentario de `P1-SOLVER-KCAL-ROW-REDUNDANT` declara el residuo —
«con 0.1 la fila kcal SIGUE dominando (share 81.7-84.0%) … devolverles su significado
exige re-tunear los CUATRO simultáneamente contra un harness de comidas vivas». Ese
harness era un script efímero que nunca se commiteó, así que cada tune posterior
arrancaba de cero y el comentario era la única memoria. Este archivo ES el harness.

Uso:
    conda activate mealfit
    python scripts/retune_solver_weights.py --meals <ruta>/meals_harness.json
    python scripts/retune_solver_weights.py --meals ... --mode baseline
    python scripts/retune_solver_weights.py --meals ... --mode grid --top 20

El JSON de comidas NO se versiona (sale de `plan_data` de producción: dato de usuario).
Se pasa por CLI. Forma esperada — lista de objetos con al menos:
    {"plan": str, "meal_name": str, "ingredients_raw": [str, ...],
     "meal_macros": {"cals"|"calories": n, "protein": n, "carbs": n, "fats": n}}

DE DÓNDE SALEN LOS MACROS POR ALIMENTO (decisión documentada)
-------------------------------------------------------------
El solver necesita `A` (vector de macros por línea), y eso vive en `master_ingredients`
(Neon). Un harness que exija DB no es reproducible en CI ni en un worktree sin `.env`:
`IngredientNutritionDB()` sale VACÍO y el tune mediría el vacío, no el sistema.

Por eso el catálogo es un STUB EMBEBIDO (`_STUB_CATALOG`): la LISTA de alimentos se
derivó del propio JSON de comidas (296 nombres canónicos distintos tras
`shopping_calculator._parse_quantity`, cubiertos por frecuencia de aparición) y los
valores por 100 g son tabla nutricional estándar (USDA/SR-Legacy, la misma fuente que
`scripts/populate_nutrition_db.py` usa para poblar el catálogo real). Es una RÉPLICA
aproximada del catálogo, no el catálogo.

Consecuencia honesta: los números absolutos de este harness NO son los de producción
—un alimento que aquí no resuelve sí resuelve allá—. Lo que el harness mide bien es la
DIFERENCIA entre dos vectores de pesos sobre el MISMO conjunto de casos, que es
exactamente la pregunta del tune. `--mode baseline` imprime la calibración (cobertura
del stub + reproducción del 77.6% publicado) para que quien lea la tabla sepa cuánta fe
tenerle. tooltip-anchor: P3-SOLVER-W-RETUNE

TARGET POR COMIDA: los `meal_macros` estampados del plato (es lo que el sistema declaró
que esa comida entrega). Riesgo asumido y MEDIDO por `--mode baseline`: un plato que ya
pasó por el solver tiene su target alcanzable por construcción — la métrica
`ya_clavado_sin_solver` reporta qué fracción de las comidas está en banda ANTES de
resolver, o sea el headroom real que el tune puede mover.

[P2-LOGGER-EXEMPT: script CLI one-shot, salida a stdout intencional]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # consola Windows cp1252 rompe en acentos/emoji
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Stub-catálogo offline. Tupla: (kcal, protein_g, carbs_g, fats_g, g/taza, g/unidad,
# (alias, ...)) por 100 g. `None` en densidad = ese alimento no se mide así.
# Los alias existen porque el tier-2 de `IngredientNutritionDB` es word-boundary
# exacto: "Tomates" NO matchea "Tomate" (`\btomate\b` falla contra "tomates").
# ---------------------------------------------------------------------------
_STUB_CATALOG: dict = {
    # --- condimentos / aromáticos (casi siempre "al gusto" ⇒ 0 g ⇒ fuera del solver)
    "Sal": (0, 0, 0, 0, None, None, ("sal y pimienta", "sal marina")),
    "Pimienta negra": (255, 10.4, 64, 3.3, 100, None, ("pimienta",)),
    "Orégano dominicano": (265, 9, 69, 4.3, 90, None, ("oregano",)),
    "Comino": (375, 17.8, 44, 22.3, 100, None, ("comino en polvo",)),
    "Canela en polvo": (247, 4, 81, 1.2, 120, None, ("canela",)),
    "Cúrcuma": (312, 9.7, 67, 3.2, 120, None, ("curcuma en polvo",)),
    "Ajo": (149, 6.4, 33, 0.5, 136, 3.0, ("ajo picado", "dientes de ajo", "diente de ajo")),
    "Ajo en polvo": (331, 16.6, 72.7, 0.7, 130, None, ()),
    "Cebolla en polvo": (341, 10.4, 79, 1.0, 130, None, ()),
    "Albahaca seca": (233, 23, 48, 4.1, 21, None, ("albahaca",)),
    "Laurel": (313, 7.6, 75, 8.4, 20, 0.2, ("hojas de laurel", "hoja de laurel")),
    "Tomillo": (101, 5.6, 24.5, 1.7, 40, None, ()),
    "Cilantro": (23, 2.1, 3.7, 0.5, 16, None, ("cilantro fresco", "cilantro picado")),
    "Perejil": (36, 3.0, 6.3, 0.8, 60, None, ("perejil fresco", "perejil picado")),
    "Jengibre": (80, 1.8, 17.8, 0.8, 96, None, ()),
    "Hojas menta": (70, 3.8, 14.9, 0.9, 30, None, ("menta", "hojas de menta")),
    "Vinagre blanco": (18, 0.0, 0.04, 0.0, 239, None, ("vinagre",)),
    "Polvo de hornear": (53, 0.0, 27.7, 0.0, 220, None, ()),
    "Vainilla": (288, 0.1, 12.6, 0.1, 208, None, ("extracto de vainilla",)),
    "Agua": (0, 0, 0, 0, 240, None, ("agua fria", "agua potable", "hielo", "ola")),
    # --- grasas
    "Aceite de oliva": (884, 0, 0, 100, 216, None, ("aceite oliva",)),
    "Aceite vegetal": (884, 0, 0, 100, 218, None, ("aceite de maiz", "aceite")),
    "Aceite de coco": (892, 0, 0, 99, 218, None, ()),
    "Aguacate": (160, 2.0, 8.5, 14.7, 150, 200.0, ("aguacates",)),
    "Aceitunas": (145, 1.0, 3.8, 15.3, 135, 4.0, ("aceitunas rebanadas", "aceitunas verdes")),
    "Mantequilla de maní": (588, 25.1, 20.0, 50.4, 258, None, ("mantequilla de mani natural",)),
    "Mantequilla de almendras": (614, 21.0, 18.8, 55.5, 256, None, ()),
    # --- proteínas animales
    "Huevo": (143, 12.6, 0.7, 9.5, None, 50.0, ("huevos",)),
    "Claras de huevo": (52, 10.9, 0.7, 0.2, 243, 33.0, ("clara de huevo", "claras")),
    "Pollo": (165, 31.0, 0, 3.6, None, None, ("pechuga de pollo", "pechuga", "muslo de pollo")),
    "Carne de res": (200, 26.0, 0, 10.0, None, None, ("res", "carne molida de res", "carne molida")),
    "Cerdo": (242, 27.0, 0, 14.0, None, None, ("lomo de cerdo",)),
    "Chivo": (143, 27.1, 0, 3.0, None, None, ()),
    "Conejo": (173, 33.0, 0, 3.5, None, None, ()),
    "Pavo molido": (203, 27.0, 0, 10.4, None, None, ("pavo",)),
    "Hígado de res": (135, 20.4, 3.9, 3.6, None, None, ()),
    "Pescado": (105, 21.0, 0, 2.0, None, None, ("filete de pescado", "mero", "chillo")),
    "Camarones": (99, 24.0, 0.2, 0.3, None, None, ("camaron",)),
    "Atún en agua": (116, 25.5, 0, 0.8, None, None, ("atun",)),
    "Calamar limpio": (92, 15.6, 3.1, 1.4, None, None, ("calamar",)),
    "Mejillones": (86, 11.9, 3.7, 2.2, None, None, ("mejillon",)),
    "Pulpo": (82, 14.9, 2.2, 1.0, None, None, ()),
    "Cangrejo": (87, 18.1, 0, 1.1, None, None, ()),
    # --- lácteos
    "Yogurt": (61, 3.5, 4.7, 3.3, 245, 170.0, ("yogur",)),
    "Yogurt griego sin azúcar": (59, 10.2, 3.6, 0.4, 245, 170.0,
                                 ("yogurt griego", "yogur griego", "yogurt griego natural")),
    "Leche": (61, 3.2, 4.8, 3.3, 244, None, ("leche entera", "leche descremada")),
    "Queso blanco": (280, 18.0, 3.0, 22.0, 132, None, ("queso blanco dominicano",)),
    "Queso": (350, 25.0, 2.0, 28.0, 132, None, ()),
    "Queso mozzarella": (280, 22.2, 3.1, 17.0, 112, None, ("mozzarella",)),
    "Queso ricotta": (174, 11.3, 3.0, 13.0, 246, None, ("ricotta",)),
    "Queso cottage": (98, 11.1, 3.4, 4.3, 226, None, ("cottage",)),
    "Queso parmesano": (392, 35.8, 3.2, 25.8, 100, None, ("parmesano",)),
    "Queso gouda": (356, 24.9, 2.2, 27.4, 132, None, ("gouda",)),
    "Queso cheddar": (403, 22.9, 3.4, 33.1, 113, None, ("cheddar",)),
    "Queso de hoja": (300, 20.0, 3.0, 23.0, 132, None, ()),
    "Queso de freír": (330, 22.0, 3.0, 26.0, 132, None, ("queso de freir",)),
    # --- proteína vegetal / legumbres (CRUDAS, igual que el catálogo real)
    "Soya texturizada": (336, 52.0, 30.0, 1.2, 90, None, ("proteina de soya",)),
    "Edamame": (121, 12.0, 8.9, 5.2, 155, None, ()),
    "Lentejas": (352, 24.6, 63.4, 1.1, 192, None, ("lenteja",)),
    "Habichuelas rojas": (333, 23.6, 60.0, 0.8, 184, None, ("habichuelas", "habichuela roja")),
    "Frijoles pintos": (347, 21.4, 62.6, 1.2, 193, None, ("frijoles",)),
    "Garbanzos": (364, 19.3, 60.7, 6.0, 200, None, ("garbanzo",)),
    "Habas": (341, 26.1, 58.3, 1.5, 150, None, ("haba",)),
    "Guisantes secos": (341, 24.6, 60.4, 1.2, 197, None, ("guisantes",)),
    # --- cereales / almidones
    "Avena": (389, 16.9, 66.3, 6.9, 80, None, ("avena en hojuelas", "hojuelas de avena")),
    "Arroz blanco": (365, 7.1, 80.0, 0.7, 185, None, ("arroz",)),
    "Arroz integral": (370, 7.9, 77.2, 2.9, 185, None, ()),
    "Quinoa": (368, 14.1, 64.2, 6.1, 170, None, ("quinoa seca", "quinua")),
    "Cebada": (352, 9.9, 73.5, 1.2, 200, None, ("cebada seca",)),
    "Bulgur seco": (342, 12.3, 75.9, 1.3, 140, None, ("bulgur",)),
    "Pasta integral seca": (348, 14.6, 71.7, 2.5, 105, None, ("pasta integral", "pasta")),
    "Harina de maíz precocida": (362, 7.0, 78.0, 1.5, 120, None, ("harina de maiz",)),
    "Harina de trigo": (364, 10.3, 76.3, 1.0, 125, None, ()),
    "Pan integral familiar": (247, 13.0, 41.0, 3.4, None, 28.0,
                              ("pan integral", "pan de trigo integral")),
    "Tortilla integral": (310, 9.0, 50.0, 7.0, None, 45.0, ("tortillas integrales",)),
    "Tortilla de trigo": (306, 8.0, 51.0, 7.6, None, 45.0, ("tortillas de trigo",)),
    "Galletas de soda": (421, 9.0, 72.0, 10.0, None, 3.0, ("galleta de soda",)),
    "Granola": (471, 10.0, 64.0, 20.0, 112, None, ()),
    "Torta de casabe": (348, 1.4, 84.0, 0.6, None, 60.0,
                        ("casabe", "tortas de casabe", "laminas de casabe", "lamina de casabe")),
    "Maíz dulce en granos": (86, 3.2, 19.0, 1.2, 165, None, ("maiz dulce", "maiz")),
    # --- viveres
    "Papa": (77, 2.0, 17.5, 0.1, 150, 173.0, ("papas",)),
    "Batata": (86, 1.6, 20.1, 0.1, 133, 130.0, ("batatas",)),
    "Yuca": (160, 1.4, 38.1, 0.3, 206, None, ("pedazo de yuca",)),
    "Ñame": (118, 1.5, 27.9, 0.2, 150, None, ("name", "pedazo de name")),
    "Yautía": (132, 1.5, 31.0, 0.4, 150, None, ("yautia", "pedazo de yautia")),
    "Mapuey": (118, 1.5, 27.9, 0.2, 150, None, ()),
    "Plátano verde": (122, 1.3, 31.9, 0.4, 148, 180.0,
                      ("platano verde", "platano", "platanos verde", "platanos verdes")),
    "Guineo": (89, 1.1, 22.8, 0.3, 150, 118.0, ("guineos", "guineo verde", "guineito verde",
                                                "guineitos verdes", "guineito")),
    "Auyama": (26, 1.0, 6.5, 0.1, 116, None, ()),
    # --- vegetales
    "Cebolla": (40, 1.1, 9.3, 0.1, 160, 110.0,
                ("cebolla morada", "cebolla en plumas", "cebolla picada", "cebollas")),
    "Cebollín": (32, 1.8, 7.3, 0.7, 100, 15.0,
                 ("cebollin", "cebollines", "tallo de cebollin", "tallos de cebollin")),
    "Puerro": (61, 1.5, 14.2, 0.3, 104, 89.0, ()),
    "Tomate": (18, 0.9, 3.9, 0.2, 180, 123.0, ("tomates",)),
    "Ají cubanela": (20, 0.9, 4.6, 0.2, 149, 60.0, ("aji cubanela", "aji")),
    "Ají morrón": (31, 1.0, 6.0, 0.3, 149, 119.0, ("aji morron", "pimiento")),
    "Pimentón rojo": (31, 1.0, 6.0, 0.3, 149, 119.0,
                      ("pimenton rojo", "pimenton amarillo", "pimenton")),
    "Zanahoria": (41, 0.9, 9.6, 0.2, 128, 61.0, ("zanahorias",)),
    "Lechuga": (15, 1.4, 2.9, 0.2, 36, None, ()),
    "Kale": (49, 4.3, 8.8, 0.9, 67, None, ()),
    "Espinacas": (23, 2.9, 3.6, 0.4, 30, None, ("espinaca",)),
    "Berro": (11, 2.3, 1.3, 0.1, 34, None, ("berros",)),
    "Bok choy": (13, 1.5, 2.2, 0.2, 70, None, ()),
    "Rúcula": (25, 2.6, 3.7, 0.7, 20, None, ("rucula",)),
    "Repollo": (25, 1.3, 5.8, 0.1, 89, None, ()),
    "Repollo morado": (31, 1.4, 7.4, 0.2, 89, None, ()),
    "Pepino": (15, 0.7, 3.6, 0.1, 119, 300.0,
               ("pepinos", "pepino en medias lunas", "pepino en rodajas")),
    "Calabacín": (17, 1.2, 3.1, 0.3, 124, 196.0, ("calabacin",)),
    "Berenjena": (25, 1.0, 5.9, 0.2, 82, 458.0, ()),
    "Tayota": (19, 0.8, 4.5, 0.1, 132, 203.0, ()),
    "Molondrones": (33, 1.9, 7.5, 0.2, 100, None, ("molondron",)),
    "Vainitas": (31, 1.8, 7.0, 0.1, 110, None, ("vainita",)),
    "Brócoli": (34, 2.8, 6.6, 0.4, 91, None, ("brocoli", "brocoli en floretes")),
    "Coliflor": (25, 1.9, 5.0, 0.3, 107, None, ()),
    "Coles de bruselas": (43, 3.4, 9.0, 0.3, 88, None, ()),
    "Espárragos": (20, 2.2, 3.9, 0.1, 134, None, ("esparragos",)),
    "Nabo": (28, 0.9, 6.4, 0.1, 130, 122.0, ()),
    "Rábano": (16, 0.7, 3.4, 0.1, 116, 4.5, ("rabano", "rabanos")),
    "Remolacha": (43, 1.6, 9.6, 0.2, 136, 82.0, ()),
    "Apio": (16, 0.7, 3.0, 0.2, 101, 40.0, ("tallo de apio", "tallos de apio")),
    "Champiñones rebanados": (22, 3.1, 3.3, 0.3, 70, None, ("champinones", "champinon", "setas")),
    "Palmito": (28, 2.5, 4.6, 0.6, 146, None, ("palmitos",)),
    "Corazones de alcachofa": (47, 3.3, 10.5, 0.2, 168, None,
                               ("corazones de alcachofa escurridos", "alcachofa")),
    "Cundeamor": (19, 1.0, 4.3, 0.2, 124, None, ()),
    # --- frutas
    "Limón": (29, 1.1, 9.3, 0.3, 244, 58.0, ("limones", "limon", "jugo de limon")),
    "Lechosa": (43, 0.5, 10.8, 0.3, 145, 500.0, ("papaya",)),
    "Mango": (60, 0.8, 15.0, 0.4, 165, 200.0, ("mangos",)),
    "Manzana": (52, 0.3, 13.8, 0.2, 125, 182.0, ("manzanas",)),
    "Piña": (50, 0.5, 13.1, 0.1, 165, None, ("pina",)),
    "Melón": (34, 0.8, 8.2, 0.2, 160, None, ("melon",)),
    "Sandía": (30, 0.6, 7.6, 0.2, 152, None, ("sandia",)),
    "Fresas": (32, 0.7, 7.7, 0.3, 152, None, ("fresa",)),
    "Arándanos": (57, 0.7, 14.5, 0.3, 148, None, ("arandanos",)),
    "Uvas": (69, 0.7, 18.1, 0.2, 151, None, ("uva",)),
    "Guayaba": (68, 2.6, 14.3, 1.0, 165, 55.0, ("guayabas",)),
    "Pera": (57, 0.4, 15.2, 0.1, 140, 178.0, ("peras",)),
    "Ciruelas": (46, 0.7, 11.4, 0.3, 165, 66.0, ("ciruela",)),
    "Kiwi": (61, 1.1, 14.7, 0.5, 180, 76.0, ()),
    "Toronja": (42, 0.8, 10.7, 0.1, 230, 246.0, ()),
    "Níspero": (47, 0.4, 12.1, 0.2, 149, 40.0, ("nispero",)),
    "Guanábana": (66, 1.0, 16.8, 0.3, 225, None, ("guanabana", "guanabana sin semillas")),
    "Pulpa de chinola": (97, 2.2, 23.4, 0.7, 236, None, ("chinola", "maracuya")),
    "Miel": (304, 0.3, 82.4, 0.0, 339, None, ("miel de abeja",)),
    # --- frutos secos / semillas
    "Semillas de linaza": (534, 18.3, 28.9, 42.2, 168, None, ("linaza", "linaza molida")),
    "Semillas de girasol": (584, 20.8, 20.0, 51.5, 140, None, ("girasol",)),
    "Semillas de calabaza": (559, 30.2, 10.7, 49.1, 129, None, ("pepitas",)),
    "Semillas de chía": (486, 16.5, 42.1, 30.7, 170, None, ("chia", "semillas de chia")),
    "Almendras fileteadas": (579, 21.2, 21.6, 49.9, 92, None, ("almendras", "almendra")),
    "Maní": (567, 25.8, 16.1, 49.2, 146, None, ("mani", "mani tostado", "cacahuate")),
    "Merey": (553, 18.2, 30.2, 43.9, 137, None, ("mereyes", "maranon", "cajuil")),
    "Pistachos": (560, 20.2, 27.2, 45.3, 123, None, ("pistacho",)),
    "Nueces mixtas": (607, 20.0, 21.0, 54.0, 130, None, ("nueces", "nuez")),
    "Ajonjolí": (573, 17.7, 23.4, 49.7, 144, None, ("ajonjoli", "sesamo")),
}

_ROW_KEYS = ("kcal_per_100g", "protein_g_per_100g", "carbs_g_per_100g", "fats_g_per_100g")
_MACROS = ("kcal", "protein", "carbs", "fats")


def build_stub_rows() -> list:
    """`_STUB_CATALOG` → filas con la forma de `master_ingredients` (lo que
    `IngredientNutritionDB(rows=...)` indexa). Determinista, sin DB."""
    rows = []
    for name, t in _STUB_CATALOG.items():
        kcal, prot, carbs, fats, cup, unit, aliases = t
        row = {"name": name, "aliases": list(aliases),
               "density_g_per_cup": cup, "density_g_per_unit": unit}
        for k, v in zip(_ROW_KEYS, (kcal, prot, carbs, fats)):
            row[k] = float(v)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Carga de casos
# ---------------------------------------------------------------------------
def load_cases(path: str) -> list:
    """JSON de comidas vivas → [(nombre, [líneas], target)]. Descarta las comidas
    sin líneas o sin macros estampados (no hay target ⇒ no hay caso)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    cases = []
    for m in raw:
        # `ingredients` (display) y NO `ingredients_raw`: es la lista que el callsite de
        # producción le pasa al solver (`graph_orchestrator._apply_macro_solver_to_meal`
        # arma `_ing_strs` desde `meal["ingredients"]`). El raw se reescribe DESPUÉS, por
        # el sync por-alimento — medir sobre él sería medir otro camino.
        lines = m.get("ingredients") or m.get("ingredients_raw") or []
        mm = m.get("meal_macros") or {}
        kcal = mm.get("cals", mm.get("calories"))
        if not lines or not kcal:
            continue
        tgt = {"kcal": float(kcal), "protein": float(mm.get("protein") or 0),
               "carbs": float(mm.get("carbs") or 0), "fats": float(mm.get("fats") or 0)}
        if tgt["protein"] <= 0 and tgt["carbs"] <= 0 and tgt["fats"] <= 0:
            continue
        cases.append((m.get("plan", "?"), m.get("meal_name") or "?", list(lines), tgt))
    return cases


class CachingDB:
    """Memoiza `macros_from_ingredient_string` sobre el stub. El catálogo no cambia entre
    puntos del tune, así que resolver 2.700 líneas × ~100 puntos es trabajo repetido puro
    (el grid tardaba ~25 min por eso). Delega TODO lo demás al db real — no reimplementa
    nada del resolvedor."""

    def __init__(self, inner):
        self._inner = inner
        self._cache: dict = {}

    def macros_from_ingredient_string(self, s):
        key = str(s)
        if key not in self._cache:
            self._cache[key] = self._inner.macros_from_ingredient_string(s)
        return self._cache[key]

    def __getattr__(self, item):
        return getattr(self._inner, item)


def meal_coverage(lines: list, db) -> float:
    """Cobertura de resolución con el MISMO criterio que el gate de producción
    (`P1-SOLVER-COVERAGE-GATE`): líneas resueltas / líneas-con-cantidad, excluyendo
    condimentos «al gusto» y benignos no-bulk. Se importa el SSOT de
    `graph_orchestrator` en vez de re-escribirlo: una segunda tabla de hints derivaría."""
    from graph_orchestrator import _TRUTHUP_CONDIMENT_HINTS, _SOLVER_COV_BENIGN_RE
    from constants import strip_accents as _sa
    n_q = n_r = 0
    for s in lines:
        low = _sa(str(s).lower())
        if any(h in low for h in _TRUTHUP_CONDIMENT_HINTS) or _SOLVER_COV_BENIGN_RE.search(low):
            continue
        n_q += 1
        if db.macros_from_ingredient_string(s):
            n_r += 1
    return (n_r / n_q) if n_q else 1.0


def attach_entries(cases: list, db, apply_gate: bool = True) -> list:
    """4-tupla → 5-tupla `(plan, nombre, líneas, target, (entries, max_scale))`.

    Espeja el callsite de producción: bajo `SOLVER_MIN_COVERAGE` el solver SE ABSTIENE
    (la comida no es un caso del tune — en prod nunca la resuelve), y con cobertura
    parcial el clamp baja a `SOLVER_PARTIAL_MAX_SCALE`. Sin esto el harness le pediría al
    solver que compense con las líneas resueltas la masa que al STUB le falta, y el tune
    premiaría pesos que sobre-escalan."""
    from graph_orchestrator import SOLVER_MIN_COVERAGE, SOLVER_PARTIAL_MAX_SCALE
    out, abst = [], 0
    for p, n, lines, tgt in cases:
        cov = meal_coverage(lines, db) if apply_gate else 1.0
        if apply_gate and cov < SOLVER_MIN_COVERAGE:
            abst += 1
            continue
        mx = SOLVER_PARTIAL_MAX_SCALE if (apply_gate and cov < 0.999) else None
        out.append((p, n, lines, tgt, (solver_entries(lines, db), mx)))
    if apply_gate:
        print(f"gate de cobertura (espejo de P1-SOLVER-COVERAGE-GATE): {abst} comidas "
              f"abstenidas de {len(cases)} — el solver tampoco las resuelve en producción")
    return out


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def _in_band(achieved: dict, tgt: dict, tol: float, macros) -> bool:
    for m in macros:
        t = tgt.get(m) or 0.0
        if t <= 0:
            continue
        if abs((achieved.get(m) or 0.0) / t - 1.0) > tol:
            return False
    return True


def solver_entries(lines: list, db) -> list:
    """Macros por línea de las que ENTRAN al LSQ — mismo filtro que
    `_compute_scale_factors` (`macros` y `group` no nulos). Se precomputa una vez por
    comida y se reusa en todas las evaluaciones: el stub-catálogo no cambia con los pesos."""
    import portion_solver as ps
    out = []
    for s in lines:
        mc = db.macros_from_ingredient_string(s)
        if not mc:
            continue
        contrib = {m: mc[m] * ps._KCAL_PER_G[m] for m in ps._KCAL_PER_G}
        if any(contrib.values()):
            out.append(mc)
    return out


def rows_for_case(entries: list, tgt: dict, weights: tuple) -> tuple:
    """(A_rows, b, w, labels) tal como `_compute_scale_factors` los arma — para
    alimentar `effective_row_shares` sin tocar el solver."""
    if not entries:
        return [], [], [], []
    wmap = dict(zip(_MACROS, weights))
    A, b, w, labels = [], [], [], []
    for m in _MACROS:
        if (tgt.get(m) or 0) > 0:
            A.append([e[m] for e in entries])
            b.append(float(tgt[m]))
            w.append(wmap[m])
            labels.append(m)
    return A, b, w, labels


def evaluate(cases: list, weights: tuple, db, tol: float = 0.10) -> dict:
    """Resuelve las N comidas con `weights` = (kcal, protein, carbs, fats) y
    devuelve las métricas del tune. Muta los globales del módulo del solver (que es
    de donde `_compute_scale_factors` los lee) y los restaura al salir."""
    import portion_solver as ps
    prev = (ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS)
    ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS = weights
    # ⚠️ La banda CLÍNICA de producción no es ±10% simétrico: es [0.90, 1.12] para P/C/F y la
    # ESTRECHA [0.95, 1.05] para kcal (`SOLVER_BAND_*`). Es justo el argumento con el que
    # P1-SOLVER-KCAL-ROW-REDUNDANT decidió NO bajar el peso kcal a 0 ("el MAPE de kcal salta a
    # 2.3% y la banda kcal es la ESTRECHA → el all-4 empeora"). Un tune que solo mire ±10%
    # simétrico puede "ganar" pp que la banda real no paga. Se mide con el SSOT del solver
    # (`_converged_report` con la semántica de banda encendida); no altera ningún factor porque
    # el `converged` que `solve_meal_macros` computa internamente no se consume aquí.
    prev_band = ps.SOLVER_CONVERGED_USES_BAND
    ps.SOLVER_CONVERGED_USES_BAND = True
    n_all4 = n_pcf = n_band = 0
    ape = {m: [] for m in _MACROS}
    shares = {m: [] for m in _MACROS}
    try:
        for _plan, _name, lines, tgt, (entries, mx) in cases:
            res = (ps.solve_meal_macros(lines, tgt, db=db, max_scale=mx, max_scale_protein=mx)
                   if mx else ps.solve_meal_macros(lines, tgt, db=db))
            ach = res["achieved"]
            if _in_band(ach, tgt, tol, _MACROS):
                n_all4 += 1
            if _in_band(ach, tgt, tol, ("protein", "carbs", "fats")):
                n_pcf += 1
            if ps._converged_report(ach, tgt, tol)[0]:
                n_band += 1
            for m in _MACROS:
                t = tgt.get(m) or 0.0
                if t > 0:
                    ape[m].append(abs((ach.get(m) or 0.0) / t - 1.0))
            # share efectivo de cada fila con ESTOS pesos (dato del helper puro)
            _A, _b, _w, _labels = rows_for_case(entries, tgt, weights)
            if _A:
                for lab, sh in zip(_labels, ps.effective_row_shares(_A, _b, _w)):
                    shares[lab].append(sh)
    finally:
        ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS = prev
        ps.SOLVER_CONVERGED_USES_BAND = prev_band
    n = len(cases) or 1
    return {
        "weights": tuple(weights),
        "all4": n_all4 / n,
        "pcf": n_pcf / n,
        "band": n_band / n,
        "mape": {m: (sum(v) / len(v) if v else 0.0) for m, v in ape.items()},
        "share": {m: (sum(v) / len(v) if v else 0.0) for m, v in shares.items()},
        "n": len(cases),
    }


def baseline_report(cases: list, db) -> dict:
    """Calibración del harness: cobertura del stub-catálogo y cuántas comidas ya
    están en banda ANTES de resolver (headroom honesto)."""
    n_lines = n_res = 0
    n_pre_all4 = n_pre_pcf = 0
    kcal_ratio = []
    for _plan, _name, lines, tgt, _ents in cases:
        entries = []
        for s in lines:
            mc = db.macros_from_ingredient_string(s)
            n_lines += 1
            if mc:
                n_res += 1
                entries.append(mc)
        base = {m: sum(e[m] for e in entries) for m in _MACROS}
        if _in_band(base, tgt, 0.10, _MACROS):
            n_pre_all4 += 1
        if _in_band(base, tgt, 0.10, ("protein", "carbs", "fats")):
            n_pre_pcf += 1
        if tgt["kcal"] > 0:
            kcal_ratio.append(base["kcal"] / tgt["kcal"])
    kcal_ratio.sort()
    n = len(cases) or 1
    return {
        "n_cases": len(cases),
        "cobertura_lineas": n_res / (n_lines or 1),
        "ya_clavado_sin_solver_all4": n_pre_all4 / n,
        "ya_clavado_sin_solver_pcf": n_pre_pcf / n,
        "kcal_ratio_p10": kcal_ratio[int(0.10 * len(kcal_ratio))] if kcal_ratio else 0,
        "kcal_ratio_p50": kcal_ratio[len(kcal_ratio) // 2] if kcal_ratio else 0,
        "kcal_ratio_p90": kcal_ratio[int(0.90 * len(kcal_ratio))] if kcal_ratio else 0,
    }


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
# Rejilla por eje. Incluye a propósito valores AGRESIVOS: kcal muy bajo (la fila
# redundante) y macros altos — la vecindad del punto actual ya se sabe que mueve
# décimas, así que una rejilla fina alrededor de 1.5/1.1/1.4 mediría ruido.
_GRID = {
    "kcal": (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
    "protein": (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
    "carbs": (0.5, 1.0, 1.1, 2.0, 3.0, 5.0, 8.0),
    "fats": (0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0),
}
_AXES = ("kcal", "protein", "carbs", "fats")


def _score(r: dict) -> tuple:
    """Criterio del tune: all-4-en-banda MÁXIMO; desempate por MAPE de proteína MÍNIMO.
    Tupla ordenable (mayor es mejor).

    Las dos métricas se REDONDEAN a 0.1 pp a propósito. Sin eso, la fase fina aceptaba
    movimientos que mejoraban el MAPE en la 6ª cifra decimal (0.0085 gana a 0.01 por
    0.002 pp) y el tune escupía defaults con 4 decimales que no significan nada: sería
    exactamente el «un operador que los toque mide ruido» que este fix vino a cerrar,
    ahora cometido por el propio harness. Empate ⇒ gana el incumbente (`>` estricto)."""
    return (round(r["all4"], 3), -round(r["mape"]["protein"], 3))


def coordinate_descent(cases: list, db, start: tuple, passes: int = 3,
                       log=print) -> tuple:
    """Descenso por coordenadas sobre `_GRID`. Determinista: orden de ejes fijo,
    orden de valores fijo, y el punto de partida decide los empates (`>` estricto)."""
    cur = list(start)
    seen: dict = {}

    def ev(w):
        key = tuple(w)
        if key not in seen:
            seen[key] = evaluate(cases, key, db)
        return seen[key]

    best = ev(cur)
    log(f"  inicio {tuple(cur)} → all4={best['all4']:.3%} mape_P={best['mape']['protein']:.3%}")
    for p in range(passes):
        moved = False
        for ax_i, ax in enumerate(_AXES):
            for v in _GRID[ax]:
                cand = list(cur)
                cand[ax_i] = v
                r = ev(cand)
                if _score(r) > _score(best):
                    best, cur, moved = r, cand, True
            log(f"  pase {p + 1} · eje {ax:<7} → {tuple(cur)} "
                f"all4={best['all4']:.3%} mape_P={best['mape']['protein']:.3%}")
        if not moved:
            log(f"  pase {p + 1}: sin movimiento — óptimo local de la rejilla.")
            break
    return tuple(cur), best, seen


def refine(cases: list, db, start: tuple, seen: dict, log=print) -> tuple:
    """Segunda fase FINA: vecindario multiplicativo alrededor del ganador de la rejilla
    gruesa. Los factores están clampeados al rango que el validador de los knobs acepta
    (`0.0 < v <= 10.0`) — un punto que el knob rechazaría no es un punto elegible."""
    mult = (0.5, 0.7, 0.85, 1.2, 1.5, 2.0)
    cur = list(start)

    def ev(w):
        key = tuple(round(v, 4) for v in w)
        if key not in seen:
            seen[key] = evaluate(cases, key, db)
        return seen[key]

    best = ev(cur)
    for ax_i, ax in enumerate(_AXES):
        for f in mult:
            v = round(cur[ax_i] * f, 4)
            if not (0.0 < v <= 10.0):
                continue
            cand = list(cur)
            cand[ax_i] = v
            r = ev(cand)
            if _score(r) > _score(best):
                best, cur = r, cand
        log(f"  fino · eje {ax:<7} → {tuple(cur)} all4={best['all4']:.3%}")
    return tuple(cur), best


def split_half(cases: list, db, weights: tuple) -> tuple:
    """Anti-overfitting: parte los casos por PLAN (no por índice — dos comidas del mismo
    plan comparten target y catálogo, así que partir por índice filtraría información de
    una mitad a la otra) y evalúa el punto en cada mitad."""
    plans = sorted({c[0] for c in cases})
    a_plans = set(plans[::2])
    a = [c for c in cases if c[0] in a_plans]
    b = [c for c in cases if c[0] not in a_plans]
    return evaluate(a, weights, db), evaluate(b, weights, db)


def high_fidelity(cases: list, db, tol: float = 0.10) -> list:
    """Sub-conjunto donde la reconstrucción del stub COINCIDE con los macros estampados
    (kcal a ±tol sin resolver). Sirve para separar «el punto gana» de «el punto gana
    porque a mi catálogo le faltan líneas»: si el ganador también gana aquí, el
    resultado no es un artefacto de la cobertura del stub."""
    out = []
    for c in cases:
        _p, _n, lines, tgt, _e = c
        base = 0.0
        for s in lines:
            mc = db.macros_from_ingredient_string(s)
            if mc:
                base += mc["kcal"]
        if tgt["kcal"] > 0 and abs(base / tgt["kcal"] - 1.0) <= tol:
            out.append(c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-tune offline de los 4 pesos SOLVER_W_*")
    ap.add_argument("--meals", required=True, help="JSON de comidas vivas (NO versionado)")
    ap.add_argument("--mode", default="tune", choices=("baseline", "tune", "confirm", "points"))
    ap.add_argument("--point", default="", help="confirm: 'kcal,prot,carb,fats' a contrastar")
    ap.add_argument("--points", default="",
                    help="points: lista 'k,p,c,f;k,p,c,f;…' a evaluar tal cual (candidatos "
                         "REDONDOS: un default que un operador no puede leer no es un default)")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--top", type=int, default=10, help="filas de la tabla final")
    ap.add_argument("--limit", type=int, default=0, help="usar solo las primeras N comidas")
    ap.add_argument("--no-gate", action="store_true",
                    help="NO espejar el gate de cobertura de producción (mide el sesgo del stub)")
    ap.add_argument("--quiet-logs", action="store_true", default=True)
    args = ap.parse_args(argv)

    if args.quiet_logs:
        logging.disable(logging.CRITICAL)

    from nutrition_db import IngredientNutritionDB
    import portion_solver as ps

    db = CachingDB(IngredientNutritionDB(rows=build_stub_rows()))
    cases = load_cases(args.meals)
    if args.limit:
        cases = cases[: args.limit]
    cases = attach_entries(cases, db, apply_gate=not args.no_gate)
    actual = (ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS)

    print(f"=== [P3-SOLVER-W-RETUNE] harness offline · {len(cases)} comidas · "
          f"stub-catálogo de {len(_STUB_CATALOG)} alimentos ===")
    b = baseline_report(cases, db)
    print(f"cobertura de líneas resueltas por el stub : {b['cobertura_lineas']:.1%}")
    print(f"comidas YA en banda ±10% SIN resolver     : all4 {b['ya_clavado_sin_solver_all4']:.1%} "
          f"· P/C/F {b['ya_clavado_sin_solver_pcf']:.1%}   ← headroom del tune")
    print(f"ratio kcal reconstruido/estampado         : p10 {b['kcal_ratio_p10']:.2f} · "
          f"p50 {b['kcal_ratio_p50']:.2f} · p90 {b['kcal_ratio_p90']:.2f}")

    r0 = evaluate(cases, actual, db)
    print(f"\npunto ACTUAL {actual}: all4 {r0['all4']:.1%} · P/C/F {r0['pcf']:.1%} · "
          f"banda clínica {r0['band']:.1%} · MAPE P {r0['mape']['protein']:.1%} "
          f"C {r0['mape']['carbs']:.1%} F {r0['mape']['fats']:.1%} kcal {r0['mape']['kcal']:.1%}")
    print(f"  share efectivo medio (w·b²): " +
          " · ".join(f"{m} {r0['share'][m]:.1%}" for m in _MACROS))
    if args.mode == "baseline":
        return 0

    if args.mode == "points":
        pts = [tuple(float(x) for x in p.split(",")) for p in args.points.split(";") if p.strip()]
        rows = [(p, evaluate(cases, p, db)) for p in pts]
        head = (f"{'kcal':>7} {'prot':>6} {'carb':>6} {'fats':>6} | {'all-4':>6} {'banda':>6} | "
                f"{'MAPE P':>6} {'MAPE C':>6} {'MAPE F':>6} {'MAPEkcal':>7} | {'share kcal':>9}")
        print("\n" + head)
        for w, r in rows:
            print(f"{w[0]:>7.4f} {w[1]:>6.2f} {w[2]:>6.2f} {w[3]:>6.2f} | {r['all4']:>6.1%} "
                  f"{r['band']:>6.1%} | {r['mape']['protein']:>6.1%} {r['mape']['carbs']:>6.1%} "
                  f"{r['mape']['fats']:>6.1%} {r['mape']['kcal']:>7.1%} | {r['share']['kcal']:>9.1%}")
        return 0

    if args.mode == "confirm":
        cand = tuple(float(x) for x in args.point.split(","))
        hi = high_fidelity(cases, db)
        print(f"\nsub-conjunto ALTA FIDELIDAD (kcal reconstruida a ±10% de la estampada): "
              f"{len(hi)} de {len(cases)} comidas")
        for label, subset in (("TODAS", cases), ("ALTA-FID", hi)):
            ra = evaluate(subset, actual, db)
            rc = evaluate(subset, cand, db)
            print(f"  {label:<9} n={ra['n']:<4} actual all4 {ra['all4']:.1%} / banda "
                  f"{ra['band']:.1%} / MAPE P {ra['mape']['protein']:.1%}  →  "
                  f"cand all4 {rc['all4']:.1%} / banda {rc['band']:.1%} / MAPE P "
                  f"{rc['mape']['protein']:.1%}   (Δall4 {(rc['all4'] - ra['all4']) * 100:+.1f} pp, "
                  f"Δbanda {(rc['band'] - ra['band']) * 100:+.1f} pp)")
        sa, sb = split_half(cases, db, cand)
        ra, rb = split_half(cases, db, actual)
        print(f"  mitades por plan: A {ra['all4']:.1%}→{sa['all4']:.1%} "
              f"(Δ {(sa['all4'] - ra['all4']) * 100:+.1f} pp, n={sa['n']}) · "
              f"B {rb['all4']:.1%}→{sb['all4']:.1%} "
              f"(Δ {(sb['all4'] - rb['all4']) * 100:+.1f} pp, n={sb['n']})")
        return 0

    print("\n--- fase 1: descenso por coordenadas sobre rejilla gruesa ---")
    best_w, best_r, seen = coordinate_descent(cases, db, actual, passes=args.passes)
    print("--- fase 2: refinamiento fino alrededor del ganador ---")
    best_w, best_r = refine(cases, db, best_w, seen)

    # ⚠️ 4 decimales en TODAS las columnas de peso: con `:.2f` la fase fina imprimía seis
    # filas distintas como «0.01 8.00 0.50 8.00» con métricas diferentes — el formato
    # colapsaba puntos que NO son el mismo punto.
    def _fmt(w, r):
        return (f"{w[0]:>7.4f} {w[1]:>6.2f} {w[2]:>6.2f} {w[3]:>6.2f} | "
                f"{r['all4']:>6.1%} {r['pcf']:>6.1%} {r['band']:>6.1%} | "
                f"{r['mape']['protein']:>6.1%} {r['mape']['carbs']:>6.1%} "
                f"{r['mape']['fats']:>6.1%} {r['mape']['kcal']:>7.1%} | {r['share']['kcal']:>9.1%}")

    head = (f"{'kcal':>7} {'prot':>6} {'carb':>6} {'fats':>6} | {'all-4':>6} {'P/C/F':>6} "
            f"{'banda':>6} | {'MAPE P':>6} {'MAPE C':>6} {'MAPE F':>6} {'MAPEkcal':>7} | "
            f"{'share kcal':>9}")
    print(f"\n=== TABLA (top {args.top} de {len(seen)} puntos evaluados) ===")
    print(head)
    for w, r in sorted(seen.items(), key=lambda kv: _score(kv[1]), reverse=True)[: args.top]:
        print(_fmt(w, r))
    print("-" * len(head))
    print(_fmt(actual, r0) + "   ← ACTUAL")

    delta = (best_r["all4"] - r0["all4"]) * 100.0
    dband = (best_r["band"] - r0["band"]) * 100.0
    print(f"\nMEJOR: {best_w} → all-4 {best_r['all4']:.1%} "
          f"(actual {r0['all4']:.1%}, Δ {delta:+.1f} pp) · banda clínica "
          f"{best_r['band']:.1%} (actual {r0['band']:.1%}, Δ {dband:+.1f} pp)")

    sa, sb = split_half(cases, db, best_w)
    ra, rb = split_half(cases, db, actual)
    print(f"anti-overfitting (mitades por plan): mitad A {ra['all4']:.1%} → {sa['all4']:.1%} "
          f"(Δ {(sa['all4'] - ra['all4']) * 100:+.1f} pp, n={sa['n']}) · "
          f"mitad B {rb['all4']:.1%} → {sb['all4']:.1%} "
          f"(Δ {(sb['all4'] - rb['all4']) * 100:+.1f} pp, n={sb['n']})")

    material = delta >= 1.5 and dband >= 0.0 and (sa["all4"] > ra["all4"]) and (sb["all4"] > rb["all4"])
    print("VEREDICTO: " + ("CAMBIAR defaults (Δ ≥ +1.5 pp, la banda clínica no regresa y "
                           "las DOS mitades mejoran)" if material else
                           "CONFIRMAR defaults (la mejora no es material, regresa la banda "
                           "clínica, o no se sostiene en las dos mitades)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
