"""[P1-CULINARY-GOLDEN · 2026-07-31] Generador ONE-SHOT del golden set culinario.

Se corre UNA vez (los fixtures resultantes se commitean y quedan estáticos).
Determinista: el contenido de los 5 planes "buenos" y sus 5 mutaciones está
CURADO A MANO (recetas reales de `data/dominican_dish_recipes.json`, nombres
de alimentos EXACTOS del catálogo `master_ingredients`) — no hay `random.*` en
el camino de construcción, así que regenerar es byte-idéntico por construcción
(más fuerte que "misma seed": cero fuentes de entropía). `_RNG` se deja
definido por si un futuro regenerado necesita variar algo — hoy no se usa.
Solo SELECT contra la DB (prod) — jamás escribe.

Uso (desde backend/):
    "C:/Users/angel/miniconda3/envs/mealfit/python.exe" scripts/build_culinary_golden_set.py
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import db_core
db_core.connection_pool.open()          # ⚠️ sin esto el catálogo sale vacío
from shopping_calculator import get_master_ingredients

_RNG = random.Random(20260731)
_OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "culinary_golden"
_DATA = Path(__file__).resolve().parent.parent / "data"

RECIPES = json.loads((_DATA / "dominican_dish_recipes.json").read_text(encoding="utf-8"))["recipes"]
TEMPLATES = json.loads((_DATA / "dish_templates.json").read_text(encoding="utf-8"))["templates"]
CATALOG_NAMES = {str(r["name"]).strip() for r in get_master_ingredients()}

_PASOS_POR_METODO = {
    # verbo inicial coherente con el método declarado de la receta canónica
    "hervido": "Hierve", "guisado": "Guisa", "plancha": "Cocina a la plancha",
    "horneado": "Hornea", "frito": "Fríe", "salteado": "Saltea", "tostado": "Tuesta",
}


def _meal(slot: str, nombre: str, ingredientes: list, pasos: list, extra=None):
    m = {"meal": slot, "name": nombre, "ingredients": ingredientes, "recipe": pasos}
    if extra:
        m.update(extra)
    return m


def _find_meal(plan: dict, day: int, name: str) -> dict:
    for d in plan["days"]:
        if d["day"] == day:
            for m in d["meals"]:
                if m["name"] == name:
                    return m
    raise KeyError(f"meal no encontrada: day={day} name={name!r}")


# ---------------------------------------------------------------------------
# Los 5 planes "buenos" — construidos desde recetas reales de
# dominican_dish_recipes.json (ingredientes = nombre EXACTO del catálogo).
# ---------------------------------------------------------------------------

def _plan_01() -> dict:
    """3 meals/día (Desayuno, Almuerzo, Cena). Sin _meta especial."""
    return {
        "days": [
            {"day": 1, "meals": [
                _meal("Desayuno", "Avena cocida con canela", [
                    "50 g Avena", "200 g Leche", "0.5 g Canela en polvo",
                ], [
                    "Hierve la leche a fuego medio.",
                    "Agrega la avena y la canela en polvo, revuelve.",
                    "Cocina 5 minutos hasta espesar.",
                    "Sirve caliente.",
                ]),
                _meal("Almuerzo", "Locrío de pollo", [
                    "60 g Arroz blanco", "65 g Pechuga de pollo", "8 g Aceite vegetal",
                    "12 g Cebolla", "15 g Tomate",
                ], [
                    "Sofríe la cebolla y el tomate en el aceite vegetal.",
                    "Agrega la pechuga de pollo en trozos y dora.",
                    "Incorpora el arroz y el agua, guisa a fuego bajo hasta que el arroz esté cocido.",
                    "Sirve caliente.",
                ]),
                _meal("Cena", "Pescado guisado a la criolla", [
                    "160 g Filete de pescado blanco", "7 g Aceite vegetal", "18 g Cebolla",
                    "12 g Ají cubanela", "20 g Tomate", "15 g Salsa de tomate", "4 g Ajo",
                    "8 g Limón", "1 g Sal",
                ], [
                    "Sazona el filete de pescado blanco con limón, ajo y sal.",
                    "Sofríe la cebolla y el ají cubanela en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora el pescado y guisa a fuego bajo 8 minutos.",
                    "Sirve caliente.",
                ]),
            ]},
            {"day": 2, "meals": [
                _meal("Desayuno", "Huevos revueltos con cebolla", [
                    "100 g Huevo", "7 g Aceite vegetal", "25 g Cebolla", "1 g Sal", "0.3 g Pimienta negra",
                ], [
                    "Bate los huevos con sal y pimienta negra.",
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Vierte el huevo batido y revuelve hasta cuajar.",
                    "Sirve caliente.",
                ]),
                _meal("Almuerzo", "Moro de habichuelas", [
                    "60 g Arroz blanco", "35 g Habichuelas rojas", "6 g Aceite vegetal",
                    "12 g Cebolla", "8 g Ají cubanela",
                ], [
                    "Sofríe la cebolla y el ají cubanela en el aceite vegetal.",
                    "Agrega las habichuelas rojas y el arroz, mezcla bien.",
                    "Hierve con agua hasta que el arroz absorba el líquido.",
                    "Sirve caliente.",
                ]),
                _meal("Cena", "Pollo guisado dominicano", [
                    "140 g Pechuga de pollo", "7 g Aceite vegetal", "18 g Cebolla",
                    "12 g Ají cubanela", "18 g Tomate", "15 g Salsa de tomate", "4 g Ajo",
                    "0.5 g Orégano dominicano", "1 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ajo en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora la pechuga de pollo y el orégano dominicano, guisa a fuego bajo 15 minutos.",
                    "Sazona con sal y sirve caliente.",
                ]),
            ]},
        ],
    }


def _plan_02() -> dict:
    """4 meals/día (Desayuno, Merienda, Almuerzo, Cena)."""
    return {
        "days": [
            {"day": 1, "meals": [
                _meal("Desayuno", "Mangú de plátano verde con huevos revueltos", [
                    "180 g Plátano verde", "15 g Aceite vegetal", "15 g Cebolla",
                    "100 g Huevo", "1 g Sal", "0.3 g Pimienta negra",
                ], [
                    "Hierve el plátano verde hasta ablandar.",
                    "Maja el plátano con la cebolla y un poco del aceite vegetal.",
                    "Bate los huevos con sal y pimienta negra y sofríelos en el resto del aceite.",
                    "Sirve el mangú acompañado de los huevos revueltos.",
                ]),
                _meal("Merienda", "Yogurt griego con fresas y mango", [
                    "170 g Yogurt griego sin azúcar", "60 g Fresas", "60 g Mango", "8 g Miel",
                ], [
                    "Sirve el yogurt griego en un bol.",
                    "Cubre con las fresas y el mango picados.",
                    "Añade un hilo de miel por encima.",
                ]),
                _meal("Almuerzo", "Sancocho dominicano", [
                    "70 g Carne de res", "50 g Yuca", "40 g Plátano verde", "30 g Auyama",
                    "30 g Ñame", "10 g Cebolla", "5 g Aceite vegetal",
                ], [
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Agrega la carne de res y dora por todos lados.",
                    "Incorpora la yuca, el plátano verde, la auyama y el ñame con agua suficiente.",
                    "Hierve a fuego bajo 40 minutos hasta que los víveres estén tiernos.",
                    "Sirve caliente en un tazón hondo.",
                ]),
                _meal("Cena", "Pollo al horno con especias", [
                    "160 g Pechuga de pollo", "6 g Aceite de oliva", "5 g Ajo", "10 g Naranja",
                    "0.5 g Orégano dominicano", "12 g Cebolla", "1.2 g Sal", "0.3 g Pimienta negra",
                ], [
                    "Mezcla el aceite de oliva con el ajo, la naranja, el orégano dominicano, la sal y la pimienta negra.",
                    "Unta la pechuga de pollo con el adobo y añade la cebolla en rodajas.",
                    "Hornea a 200°C por 25 minutos hasta dorar.",
                    "Sirve caliente.",
                ]),
            ]},
            {"day": 2, "meals": [
                _meal("Desayuno", "Pan integral con mantequilla de maní", [
                    "56 g Pan integral personal", "32 g Mantequilla de maní",
                ], [
                    "Tuesta ligeramente las rebanadas de pan integral.",
                    "Unta con la mantequilla de maní.",
                    "Sirve.",
                ]),
                _meal("Merienda", "Batida de guineo", [
                    "110 g Guineo", "240 g Leche", "10 g Miel",
                ], [
                    "Licúa el guineo con la leche y la miel hasta obtener una bebida cremosa.",
                    "Sirve fría.",
                ]),
                _meal("Almuerzo", "Moro de habichuelas negras", [
                    "70 g Arroz blanco", "55 g Habichuelas negras", "9 g Aceite vegetal",
                    "18 g Cebolla", "12 g Ají cubanela", "15 g Salsa de tomate", "4 g Ajo",
                    "4 g Cilantro", "1.5 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ajo en el aceite vegetal.",
                    "Agrega la salsa de tomate y cocina 2 minutos.",
                    "Incorpora las habichuelas negras y el arroz con agua suficiente.",
                    "Hierve a fuego bajo hasta que el arroz absorba el líquido.",
                    "Espolvorea cilantro y sirve caliente.",
                ]),
                _meal("Cena", "Carne de res guisada", [
                    "150 g Carne de res", "7 g Aceite vegetal", "18 g Cebolla", "12 g Ají cubanela",
                    "18 g Tomate", "15 g Salsa de tomate", "4 g Ajo", "10 g Ají morrón", "1 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ají morrón en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora la carne de res y el ajo, guisa a fuego bajo 25 minutos hasta ablandar.",
                    "Sazona con sal y sirve caliente.",
                ]),
            ]},
        ],
    }


def _plan_03() -> dict:
    """5 meals/día (Desayuno, Merienda, Almuerzo, Merienda, Cena)."""
    return {
        "days": [
            {"day": 1, "meals": [
                _meal("Desayuno", "Huevos revueltos con cebolla", [
                    "100 g Huevo", "7 g Aceite vegetal", "25 g Cebolla", "1 g Sal", "0.3 g Pimienta negra",
                ], [
                    "Bate los huevos con sal y pimienta negra.",
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Vierte el huevo batido y revuelve hasta cuajar.",
                    "Sirve caliente.",
                ]),
                _meal("Merienda", "Casabe con queso", [
                    "50 g Casabe", "60 g Queso blanco",
                ], [
                    "Corta el casabe en trozos.",
                    "Cubre con el queso blanco en láminas.",
                    "Sirve a temperatura ambiente.",
                ]),
                _meal("Almuerzo", "Garbanzos guisados", [
                    "60 g Garbanzos", "20 g Cebolla", "15 g Ají cubanela", "20 g Tomate",
                    "18 g Salsa de tomate", "5 g Ajo", "4 g Cilantro", "8 g Aceite vegetal", "2 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ajo en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora los garbanzos cocidos con un poco de caldo y guisa hasta espesar.",
                    "Espolvorea cilantro y sirve caliente.",
                ]),
                _meal("Merienda", "Batida de guineo", [
                    "110 g Guineo", "240 g Leche", "10 g Miel",
                ], [
                    "Licúa el guineo con la leche y la miel hasta obtener una bebida cremosa.",
                    "Sirve fría.",
                ]),
                _meal("Cena", "Pescado guisado a la criolla", [
                    "160 g Filete de pescado blanco", "7 g Aceite vegetal", "18 g Cebolla",
                    "12 g Ají cubanela", "20 g Tomate", "15 g Salsa de tomate", "4 g Ajo",
                    "8 g Limón", "1 g Sal",
                ], [
                    "Sazona el filete de pescado blanco con limón, ajo y sal.",
                    "Sofríe la cebolla y el ají cubanela en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora el pescado y guisa a fuego bajo 8 minutos.",
                    "Sirve caliente.",
                ]),
            ]},
            {"day": 2, "meals": [
                _meal("Desayuno", "Avena cocida con canela", [
                    "50 g Avena", "200 g Leche", "0.5 g Canela en polvo",
                ], [
                    "Hierve la leche a fuego medio.",
                    "Agrega la avena y la canela en polvo, revuelve.",
                    "Cocina 5 minutos hasta espesar.",
                    "Sirve caliente.",
                ]),
                _meal("Merienda", "Yogurt griego con fresas y mango", [
                    "170 g Yogurt griego sin azúcar", "60 g Fresas", "60 g Mango", "8 g Miel",
                ], [
                    "Sirve el yogurt griego en un bol.",
                    "Cubre con las fresas y el mango picados.",
                    "Añade un hilo de miel por encima.",
                ]),
                _meal("Almuerzo", "Locrío de pollo", [
                    "60 g Arroz blanco", "65 g Pechuga de pollo", "8 g Aceite vegetal",
                    "12 g Cebolla", "15 g Tomate",
                ], [
                    "Sofríe la cebolla y el tomate en el aceite vegetal.",
                    "Agrega la pechuga de pollo en trozos y dora.",
                    "Incorpora el arroz y el agua, guisa a fuego bajo hasta que el arroz esté cocido.",
                    "Sirve caliente.",
                ]),
                _meal("Merienda", "Pan integral con mantequilla de maní", [
                    "56 g Pan integral personal", "32 g Mantequilla de maní",
                ], [
                    "Tuesta ligeramente las rebanadas de pan integral.",
                    "Unta con la mantequilla de maní.",
                    "Sirve.",
                ]),
                _meal("Cena", "Bacalao guisado con papa", [
                    "110 g Bacalao", "120 g Papa", "8 g Aceite de oliva", "25 g Cebolla",
                    "12 g Ají cubanela", "25 g Tomate", "20 g Salsa de tomate", "4 g Ajo",
                    "10 g Aceitunas", "12 g Ají morrón",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ají morrón en el aceite de oliva.",
                    "Agrega el tomate, la salsa de tomate y el ajo, cocina 3 minutos.",
                    "Incorpora el bacalao desalado y la papa en trozos con un poco de agua.",
                    "Guisa a fuego bajo 15 minutos hasta que la papa esté tierna.",
                    "Añade las aceitunas y sirve caliente.",
                ]),
            ]},
        ],
    }


def _plan_04() -> dict:
    """3 meals/día, 100% vegetariano (ovo-lacto): sin carne/pescado/mariscos."""
    plan = {
        "days": [
            {"day": 1, "meals": [
                _meal("Desayuno", "Avena cocida con canela", [
                    "50 g Avena", "200 g Leche", "0.5 g Canela en polvo",
                ], [
                    "Hierve la leche a fuego medio.",
                    "Agrega la avena y la canela en polvo, revuelve.",
                    "Cocina 5 minutos hasta espesar.",
                    "Sirve caliente.",
                ]),
                _meal("Almuerzo", "Moro de habichuelas", [
                    "60 g Arroz blanco", "35 g Habichuelas rojas", "6 g Aceite vegetal",
                    "12 g Cebolla", "8 g Ají cubanela",
                ], [
                    "Sofríe la cebolla y el ají cubanela en el aceite vegetal.",
                    "Agrega las habichuelas rojas y el arroz, mezcla bien.",
                    "Hierve con agua hasta que el arroz absorba el líquido.",
                    "Sirve caliente.",
                ]),
                _meal("Cena", "Berenjena guisada a la criolla", [
                    "180 g Berenjena", "30 g Cebolla", "25 g Ají cubanela", "20 g Ají morrón",
                    "40 g Tomate", "25 g Salsa de tomate", "4 g Ajo", "10 g Aceite vegetal", "1.5 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ají morrón en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora la berenjena en cubos y el ajo, guisa a fuego bajo 15 minutos.",
                    "Sazona con sal y sirve caliente.",
                ]),
            ]},
            {"day": 2, "meals": [
                _meal("Desayuno", "Yogurt griego con fresas y mango", [
                    "170 g Yogurt griego sin azúcar", "60 g Fresas", "60 g Mango", "8 g Miel",
                ], [
                    "Sirve el yogurt griego en un bol.",
                    "Cubre con las fresas y el mango picados.",
                    "Añade un hilo de miel por encima.",
                ]),
                _meal("Almuerzo", "Lentejas guisadas", [
                    "55 g Lentejas", "25 g Zanahoria", "15 g Cebolla", "12 g Ají cubanela",
                    "4 g Ajo", "3 g Cilantro", "18 g Salsa de tomate", "6 g Aceite vegetal", "2 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ajo en el aceite vegetal.",
                    "Agrega la salsa de tomate y cocina 2 minutos.",
                    "Incorpora las lentejas cocidas y la zanahoria en rodajas, guisa a fuego bajo hasta espesar.",
                    "Espolvorea cilantro y sirve caliente.",
                ]),
                _meal("Cena", "Tortilla de huevo con cebolla y queso", [
                    "100 g Huevo", "20 g Cebolla", "25 g Queso blanco", "6 g Aceite vegetal", "1 g Sal",
                ], [
                    "Bate los huevos con sal.",
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Vierte el huevo batido, agrega el queso blanco y cuaja a fuego medio.",
                    "Sirve caliente.",
                ]),
            ]},
        ],
    }
    plan["_meta"] = {"vegetariano": True}
    return plan


def _plan_05() -> dict:
    """3 meals/día. Incluye la trampa FP plural(ingrediente)↔singular(paso):
    '2 tomates' en ingredients vs 'Ralla el tomate' en el paso — el falso
    positivo real del dry-run 2026-07-31 contra el plan 7c545d59."""
    plan = {
        "days": [
            {"day": 1, "meals": [
                _meal("Desayuno", "Huevos revueltos con cebolla", [
                    "100 g Huevo", "7 g Aceite vegetal", "25 g Cebolla", "1 g Sal", "0.3 g Pimienta negra",
                ], [
                    "Bate los huevos con sal y pimienta negra.",
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Vierte el huevo batido y revuelve hasta cuajar.",
                    "Sirve caliente.",
                ]),
                _meal("Almuerzo", "Locrío de pollo con tomate rallado", [
                    "60 g Arroz blanco", "65 g Pechuga de pollo", "8 g Aceite vegetal",
                    "12 g Cebolla", "2 tomates",
                ], [
                    "Ralla el tomate y resérvalo.",
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Agrega la pechuga de pollo en trozos y dora.",
                    "Incorpora el arroz, el tomate rallado y el agua; guisa a fuego bajo hasta que el arroz esté cocido.",
                    "Sirve caliente.",
                ]),
                _meal("Cena", "Carne de res guisada", [
                    "150 g Carne de res", "7 g Aceite vegetal", "18 g Cebolla", "12 g Ají cubanela",
                    "18 g Tomate", "15 g Salsa de tomate", "4 g Ajo", "10 g Ají morrón", "1 g Sal",
                ], [
                    "Sofríe la cebolla, el ají cubanela y el ají morrón en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora la carne de res y el ajo, guisa a fuego bajo 25 minutos hasta ablandar.",
                    "Sazona con sal y sirve caliente.",
                ]),
            ]},
            {"day": 2, "meals": [
                _meal("Desayuno", "Avena cocida con canela", [
                    "50 g Avena", "200 g Leche", "0.5 g Canela en polvo",
                ], [
                    "Hierve la leche a fuego medio.",
                    "Agrega la avena y la canela en polvo, revuelve.",
                    "Cocina 5 minutos hasta espesar.",
                    "Sirve caliente.",
                ]),
                _meal("Almuerzo", "Sancocho dominicano", [
                    "70 g Carne de res", "50 g Yuca", "40 g Plátano verde", "30 g Auyama",
                    "30 g Ñame", "10 g Cebolla", "5 g Aceite vegetal",
                ], [
                    "Sofríe la cebolla en el aceite vegetal.",
                    "Agrega la carne de res y dora por todos lados.",
                    "Incorpora la yuca, el plátano verde, la auyama y el ñame con agua suficiente.",
                    "Hierve a fuego bajo 40 minutos hasta que los víveres estén tiernos.",
                    "Sirve caliente en un tazón hondo.",
                ]),
                _meal("Cena", "Pescado guisado a la criolla", [
                    "160 g Filete de pescado blanco", "7 g Aceite vegetal", "18 g Cebolla",
                    "12 g Ají cubanela", "20 g Tomate", "15 g Salsa de tomate", "4 g Ajo",
                    "8 g Limón", "1 g Sal",
                ], [
                    "Sazona el filete de pescado blanco con limón, ajo y sal.",
                    "Sofríe la cebolla y el ají cubanela en el aceite vegetal.",
                    "Agrega el tomate y la salsa de tomate, cocina 3 minutos.",
                    "Incorpora el pescado y guisa a fuego bajo 8 minutos.",
                    "Sirve caliente.",
                ]),
            ]},
        ],
    }
    plan["_meta"] = {"trampa_plural": True}
    return plan


_PLAN_BUILDERS = {1: _plan_01, 2: _plan_02, 3: _plan_03, 4: _plan_04, 5: _plan_05}


def _build_bueno(idx: int) -> dict:
    """Plan válido construido desde recetas canónicas de RECIPES. golden_01:
    3 meals/día (Desayuno/Almuerzo/Cena). golden_02: 4 meals/día (+Merienda).
    golden_03: 5 meals/día (2 Meriendas). golden_04: vegetariano
    (_meta.vegetariano=true). golden_05: trampa plural↔singular
    (_meta.trampa_plural=true: '2 tomates' en ingredients + 'Ralla el tomate'
    en el paso)."""
    return _PLAN_BUILDERS[idx]()


# ---------------------------------------------------------------------------
# Mutaciones — cada función inyecta el defecto en un meal existente (mutado
# in-place) y devuelve el `detail` legible del ground truth.
# ---------------------------------------------------------------------------

def _mut_verbo_imposible(meal: dict) -> str:
    """[capa1:V1] Verbo de cocción aplicado a un alimento ready-to-eat
    (Casabe). Si el meal no lo tiene ya, lo añade; siempre añade el paso."""
    if not any("Casabe" in ing for ing in meal["ingredients"]):
        meal["ingredients"].append("50 g Casabe")
    meal["recipe"].append("Cuece el Casabe según las instrucciones del paquete.")
    return ("Casabe (ready_to_eat) recibe la instrucción de cocción 'Cuece' — "
            "verbo imposible sobre un alimento que ya viene listo para comer.")


def _mut_estado_imposible(meal: dict, protein: str) -> str:
    """[capa1:V2] Frase '(ya viene cocido)' sobre una proteína fresca que la
    receta SÍ cocina en sus otros pasos."""
    meal["recipe"].append(f"{protein} (ya viene cocido/a) — solo recaliéntalo/a antes de servir.")
    return (f"'{protein}' (proteína fresca; la receta la cocina en otros pasos) "
            f"marcada como '(ya viene cocido)'.")


def _mut_ingrediente_huerfano(meal: dict, food: str, grams: int = 80) -> str:
    """[capa1:V3] Ingrediente añadido a la lista que ningún paso menciona."""
    meal["ingredients"].append(f"{grams} g {food}")
    return f"Se añadió '{grams} g {food}' a ingredients; ningún paso de la receta lo menciona."


def _mut_combo_absurdo(meal: dict, new_name: str, add_food: str, step: str, add_grams: int = 50) -> str:
    """[juez] Combinación culturalmente inválida (p.ej. dulce de desayuno +
    embutido frito)."""
    original = meal["name"]
    meal["name"] = new_name
    meal["ingredients"].append(f"{add_grams} g {add_food}")
    meal["recipe"].append(step)
    return (f"Renombrado '{original}' → '{new_name}'; se añadió '{add_food}' "
            f"(combinación culturalmente incoherente para el slot).")


def _mut_tecnica_impropia(meal: dict, variant: str, step: str) -> str:
    """[capa1:V1 variante 'v1' / juez variante 'juez'] Técnica de cocción
    incoherente con el alimento. La variante 'v1' usa el verbo 'licúa' sobre
    un alimento sólido sin 'licuar' en prep_methods (atrapable por regex de
    verbo); la variante 'juez' describe una técnica rara sin un verbo de
    cocción canónico limpio (solo un LLM la juzga como culinariamente rara)."""
    meal["recipe"].append(step)
    if variant == "v1":
        tag = "verbo 'licúa' sobre alimento sólido sin 'licuar' en prep_methods"
    else:
        tag = "técnica incoherente sin verbo de cocción canónico — requiere juicio LLM"
    return f"Paso añadido: '{step}' ({tag})."


def _mut_nombre_no_corresponde(meal: dict, new_name: str = "Moro de guandules") -> str:
    """[juez] El nombre del plato no corresponde a sus ingredientes reales."""
    original = meal["name"]
    meal["name"] = new_name
    return f"Renombrado '{original}' → '{new_name}' sin gandules entre los ingredientes."


_MUTACIONES = {
    # class → función(meal, **kwargs) -> detail str. Cada una inyecta el
    # defecto en un meal existente y devuelve la descripción ground-truth.
    "verbo_imposible": _mut_verbo_imposible,
    "estado_imposible": _mut_estado_imposible,
    "ingrediente_huerfano": _mut_ingrediente_huerfano,
    "combo_absurdo": _mut_combo_absurdo,
    "tecnica_impropia": _mut_tecnica_impropia,
    "nombre_no_corresponde": _mut_nombre_no_corresponde,
}

_EXPECTED_BY = {
    "verbo_imposible": "capa1:V1",
    "estado_imposible": "capa1:V2",
    "ingrediente_huerfano": "capa1:V3",
    "combo_absurdo": "juez",
    "nombre_no_corresponde": "juez",
    # tecnica_impropia depende de la variante (v1 → capa1:V1, juez → juez)
}

# Spec por plan: (day, meal_name_en_el_bueno, clase, kwargs). tecnica_impropia
# aparece EXACTAMENTE 2 veces en todo el set (plan 1 variante v1, plan 3
# variante juez) — el resto de las clases cubre las 5 restantes reparticiones
# para que las 6 clases queden presentes con 4-6 defectos por mutado.
_MUTATION_SPECS = {
    1: [
        (1, "Avena cocida con canela", "combo_absurdo", {
            "new_name": "Avena cremosa con salami frito",
            "add_food": "Salami", "add_grams": 50,
            "step": "Fríe el salami y mézclalo con la avena.",
        }),
        (1, "Locrío de pollo", "estado_imposible", {"protein": "Pechuga de pollo"}),
        (1, "Pescado guisado a la criolla", "ingrediente_huerfano", {"food": "Zanahoria"}),
        (2, "Moro de habichuelas", "verbo_imposible", {}),
        (2, "Pollo guisado dominicano", "tecnica_impropia", {
            "variant": "v1", "step": "Licúa la Pechuga de pollo hasta obtener una crema.",
        }),
    ],
    2: [
        (2, "Pan integral con mantequilla de maní", "combo_absurdo", {
            "new_name": "Pan integral con mantequilla de maní y salami frito",
            "add_food": "Salami", "add_grams": 40,
            "step": "Fríe el salami y sírvelo junto al pan.",
        }),
        (1, "Pollo al horno con especias", "estado_imposible", {"protein": "Pechuga de pollo"}),
        (1, "Sancocho dominicano", "ingrediente_huerfano", {"food": "Ají morrón"}),
        (1, "Yogurt griego con fresas y mango", "verbo_imposible", {}),
        (2, "Moro de habichuelas negras", "nombre_no_corresponde", {}),
    ],
    3: [
        (1, "Casabe con queso", "verbo_imposible", {}),
        (2, "Locrío de pollo", "estado_imposible", {"protein": "Pechuga de pollo"}),
        (1, "Garbanzos guisados", "ingrediente_huerfano", {"food": "Zanahoria"}),
        (2, "Yogurt griego con fresas y mango", "tecnica_impropia", {
            "variant": "juez",
            "step": "Deja el yogurt griego sobre la plancha caliente unos minutos para intensificar su sabor.",
        }),
        (2, "Bacalao guisado con papa", "nombre_no_corresponde", {}),
    ],
    4: [
        (1, "Avena cocida con canela", "combo_absurdo", {
            "new_name": "Avena cremosa con salami frito",
            "add_food": "Salami", "add_grams": 50,
            "step": "Fríe el salami y mézclalo con la avena.",
        }),
        (2, "Tortilla de huevo con cebolla y queso", "estado_imposible", {"protein": "Huevo"}),
        (1, "Moro de habichuelas", "ingrediente_huerfano", {"food": "Ají morrón"}),
        (2, "Lentejas guisadas", "verbo_imposible", {}),
        (1, "Berenjena guisada a la criolla", "nombre_no_corresponde", {}),
    ],
    5: [
        (2, "Avena cocida con canela", "combo_absurdo", {
            "new_name": "Avena cremosa con salami frito",
            "add_food": "Salami", "add_grams": 50,
            "step": "Fríe el salami y mézclalo con la avena.",
        }),
        (1, "Carne de res guisada", "estado_imposible", {"protein": "Carne de res"}),
        (2, "Sancocho dominicano", "ingrediente_huerfano", {"food": "Ají morrón"}),
        (2, "Pescado guisado a la criolla", "verbo_imposible", {}),
        (1, "Huevos revueltos con cebolla", "nombre_no_corresponde", {}),
    ],
}


def _mutar(bueno: dict, idx: int) -> tuple[dict, list]:
    mutado = copy.deepcopy(bueno)
    mutado.pop("_meta", None)  # los mutados no heredan la etiqueta de "bueno creativo"
    defects = []
    for day, meal_name, cls, kwargs in _MUTATION_SPECS[idx]:
        meal = _find_meal(mutado, day, meal_name)
        fn = _MUTACIONES[cls]
        detail = fn(meal, **kwargs)
        if cls == "tecnica_impropia":
            expected_by = "capa1:V1" if kwargs.get("variant") == "v1" else "juez"
        else:
            expected_by = _EXPECTED_BY[cls]
        defects.append({
            "day": day, "meal": meal["meal"], "class": cls,
            "detail": detail, "expected_by": expected_by,
        })
    return mutado, defects


def _foods_de(plan: dict) -> set:
    """Alimentos EXACTOS del catálogo referenciados en los ingredients del
    plan. Matching por sufijo con frontera de palabra (nunca substring
    crudo — lección 'pollo⊂repollo'): un ingrediente '<qty> <Nombre>' hace
    match si termina en ' <catalogo>' (o su plural '<catalogo>s'/'<catalogo>es'
    — cierra la trampa plural↔singular sobre el propio extractor). Cuando
    varios catálogos matchean, gana el más largo (alias más largo gana,
    mismo criterio que usará el validador determinista real).

    Fail-LOUD (no fail-open) si un ingrediente no resuelve a ningún nombre de
    catálogo: la primera versión de este extractor fallaba en silencio (el
    ingrediente simplemente no entraba a `found`), lo que dejaba pasar 3
    nombres YA CADUCADOS del catálogo ('Estevia', 'Pan integral', 'Pimiento
    morrón', 'Salami dominicano' — el catálogo real usa 'Ají morrón'/'Salami'
    y no tiene stevia) sin que el abort de `main()` los viera, porque ese
    abort solo mira lo que este extractor SÍ encontró. Un extractor que
    calla ante lo que no reconoce es indistinguible de uno que nunca corrió."""
    found: set = set()
    sin_match: list = []
    for day in plan["days"]:
        for m in day["meals"]:
            for ing in m["ingredients"]:
                low = ing.lower().strip()
                best = None
                for cname in CATALOG_NAMES:
                    cname_low = cname.lower()
                    for suf in (cname_low, cname_low + "s", cname_low + "es"):
                        if low == suf or low.endswith(" " + suf):
                            if best is None or len(cname) > len(best):
                                best = cname
                            break
                if best:
                    found.add(best)
                else:
                    sin_match.append(f"day={day['day']} meal={m['name']!r} ing={ing!r}")
    if sin_match:
        raise SystemExit(
            "_foods_de: ingrediente(s) sin match de catálogo (nombre caducado o "
            f"typo, arreglar en el script — NO relajar este check): {sin_match}")
    return found


def main():
    _OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"mutados": {}, "catalog_foods_used": []}
    usados = set()
    for i in range(1, 6):
        bueno = _build_bueno(i)
        (_OUT / f"golden_{i:02d}_bueno.json").write_text(
            json.dumps(bueno, ensure_ascii=False, indent=2), encoding="utf-8")
        mutado, defects = _mutar(bueno, i)   # 4-6 defectos, clases repartidas
        (_OUT / f"golden_{i:02d}_mutado.json").write_text(
            json.dumps(mutado, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["mutados"][f"golden_{i:02d}_mutado"] = {
            "base": f"golden_{i:02d}_bueno", "defects": defects}
        usados |= _foods_de(bueno)
    desconocidos = sorted(f for f in usados if f not in CATALOG_NAMES)
    if desconocidos:
        raise SystemExit(f"alimentos fuera del catálogo: {desconocidos}")
    manifest["catalog_foods_used"] = sorted(usados)
    (_OUT / "golden_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: 11 archivos en {_OUT}")


if __name__ == "__main__":
    main()
