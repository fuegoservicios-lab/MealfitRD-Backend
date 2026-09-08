# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY · 2026-09-08] Un día armado sin llamar al modelo.

## De dónde sale

El dueño puso el objetivo en números: llevar el determinismo **de 2 a 10**. Medido el 08-sep, el
sistema estaba en 2 — los guards, la lista de compras y el descuento de la Nevera ya eran
deterministas, pero **qué comes** lo decidía el LLM en cada generación, y toda la maquinaria para
no depender de él (PlanPolicy, blueprint, Dish Registry, CandidateSet, biblioteca de 140 recetas)
estaba construida, probada, desplegada y **nunca había corrido en producción**.

## La medición que reordenó el trabajo

Iba a empezar enchufando la biblioteca de recetas. Contra 808 comidas de 60 planes vivos: **0
recibirían receta congelada**. Mientras el modelo invente el nombre del plato no hay coincidencia
posible. *El cuello de botella no era el texto: era quién elige el plato.*

## Lo medido con este módulo, 14 días × 3 perfiles clínicos

| perfil | días | calorías | proteína ±15 % | idénticos al re-armar | sucias |
|---|---|---|---|---|---|
| mantenimiento 2000 | 14/14 | ±0,0 % | 14/14 | 14/14 | 0/56 |
| pérdida de grasa 1700 alta prot. | 14/14 | ±0,0 % | 12/14 | 14/14 | 0/56 |
| ganancia muscular 2600 | 14/14 | −0,3 % | 14/14 | 14/14 | 0/56 |

Dos palancas hicieron falta, y la segunda sólo después de medir que la primera no bastaba:
**(A)** pedir 25 candidatos en vez de 3 y elegir por MACROS (con 3, la proteína se iba a −39 % en
pérdida de grasa); **(B)** inclinar constituyentes —más proteico, menos carbohidratado, con las
calorías clavadas— que subió ese perfil de 1/14 a 12/14.

## Lo que este test NO puede decir

Que el plato sea bueno. Eso lo dice el paladar del dueño, y este mismo día vimos dos veces que un
escáner en verde convive con un defecto que él rechaza. Aquí se ancla lo que sí es verificable:
que sea **determinista**, que **cuadre**, y que **no invente**.
"""
import json
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REG = _BACKEND / "data" / "registry" / "dish_registry_do_v1.json"
_LIB = _BACKEND / "data" / "registry" / "recipe_library_do_v1.json"


def _sa(s):
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


@pytest.fixture(scope="module")
def dd():
    import deterministic_day
    return deterministic_day


@pytest.fixture(scope="module")
def plantillas():
    if not _REG.exists():
        pytest.skip("sin snapshot del registry")
    return {t["template_id"]: t
            for t in json.loads(_REG.read_text(encoding="utf-8")).get("templates") or []}


@pytest.fixture(scope="module")
def catalogo_falso(plantillas):
    """Catálogo sintético: cada alimento de las plantillas con macros plausibles y ESTABLES.

    A propósito no se toca la base: un test que depende de Neon mide la red. Lo que se prueba aquí
    es la aritmética y el determinismo, y para eso el catálogo sólo tiene que ser consistente.
    """
    fuera = {}
    for t in plantillas.values():
        for c in (t.get("constituents") or []):
            n = c.get("name")
            if not n or n in fuera:
                continue
            h = sum(ord(ch) for ch in n)
            fuera[n] = {"kcal_per_100g": 80 + h % 300,
                        "protein_g_per_100g": h % 25,
                        "carbs_g_per_100g": (h * 3) % 60,
                        "fats_g_per_100g": (h * 7) % 20}
    return fuera


def test_el_knob_nace_apagado(dd, monkeypatch):
    """Encender esto cambia QUÉ come el usuario. Un default sembrado es indistinguible de una
    elección — la lección que este repo ya pagó con el país del formulario."""
    monkeypatch.delenv("MEALFIT_DETERMINISTIC_DAY", raising=False)
    assert dd.deterministic_day_enabled() is False


def test_la_banda_de_escala_es_por_franja(dd):
    """Media merienda se entiende sola; medio locrio no. Con una banda única, 2 de 14 días se
    quedaban sin merienda (mediana del registro 321 kcal contra un objetivo de 200)."""
    assert dd._banda("merienda")[0] < dd._banda("almuerzo")[0]
    assert dd._banda("cena") == dd._banda("almuerzo")
    assert dd._banda("lo que sea") == dd._BANDA_DEFECTO


def test_el_condimento_no_escala(dd):
    """Un locrio para dos no lleva el doble de orégano, y multiplicar la sal por 1,5 es un problema
    clínico, no de sabor."""
    for n in ("Sal", "Orégano dominicano", "Ajo", "Pimienta negra"):
        assert dd._no_escala(n), n
    for n in ("Pechuga de pollo", "Arroz blanco", "Batata"):
        assert not dd._no_escala(n), n


def test_la_clase_se_decide_por_densidad_no_por_nombre(dd):
    """Clasificar por nombre es cómo «res» acabó dentro de «fresas» y «pollo» dentro de «repollo».
    Aquí manda la densidad de macros."""
    assert dd._clase({"kcal_per_100g": 120, "protein_g_per_100g": 23, "carbs_g_per_100g": 0}) == "proteico"
    assert dd._clase({"kcal_per_100g": 360, "protein_g_per_100g": 7, "carbs_g_per_100g": 80}) == "carbo"
    assert dd._clase({"kcal_per_100g": 884, "protein_g_per_100g": 0, "carbs_g_per_100g": 0}) == "otro"
    assert dd._clase({"kcal_per_100g": 0}) == "otro"


def test_una_plantilla_sin_macros_NO_es_candidata(dd, plantillas, catalogo_falso):
    """Un nutriente ausente no es cero (`P1-ARQ27-F1`): 7 de las 144 plantillas tienen algún
    constituyente sin macros y se descartan en vez de mentir con un total corto."""
    tid = next(iter(plantillas))
    roto = dict(catalogo_falso)
    for c in (plantillas[tid].get("constituents") or []):
        roto[c["name"]] = {"kcal_per_100g": None}
        break
    obj = {"kcal": 500, "protein_g": 30, "carbs_g": 60, "fats_g": 15}
    assert dd.elegir_plantilla([tid], obj, roto, plantillas, "almuerzo") is None


def test_elegir_es_DETERMINISTA_y_desempata_por_id(dd, plantillas, catalogo_falso):
    """Desempatar por el orden de la lista dependería de cómo vino la lista; por id no depende de
    nada. Se comprueba invirtiendo el orden de entrada."""
    tids = list(plantillas)[:20]
    obj = {"kcal": 600, "protein_g": 35, "carbs_g": 70, "fats_g": 18}
    a = dd.elegir_plantilla(tids, obj, catalogo_falso, plantillas, "almuerzo")
    b = dd.elegir_plantilla(list(reversed(tids)), obj, catalogo_falso, plantillas, "almuerzo")
    if a is None:
        pytest.skip("ningún candidato entra en banda con el catálogo sintético")
    assert b is not None
    assert a[0]["template_id"] == b[0]["template_id"]
    assert round(a[1], 6) == round(b[1], 6)


def test_construir_deja_RASTRO(dd, plantillas, catalogo_falso, monkeypatch):
    """Sin `_meal_source`, un día determinista y uno del modelo se ven IGUAL en la base y nadie
    puede medir cuántos hay. Es la lección del día sobre lo que es inerte sin dejar huella."""
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    lib = json.loads(_LIB.read_text(encoding="utf-8"))["por_id"] if _LIB.exists() else {}
    tid = next((k for k in plantillas if k in lib), None)
    if not tid:
        pytest.skip("ninguna plantilla con receta congelada")
    m = dd.construir_comida(plantillas[tid], 1.0, catalogo_falso, "almuerzo", "DO")
    if m is None:
        pytest.skip("el catálogo sintético no cubre esta plantilla")
    assert m["_meal_source"] == "deterministic"
    assert m["_template_id"] == tid
    assert m["_recipe_source"] == "library"
    assert m["recipe"] and len(m["recipe"]) >= 3


def test_sin_receta_congelada_NO_se_construye(dd, plantillas, catalogo_falso, monkeypatch):
    """El determinismo del texto es parte del contrato: si no hay receta escrita, que la escriba el
    LLM. Devolver `None` es seguro — el llamador cae al camino de siempre."""
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    import recipe_library as rl
    monkeypatch.setattr(rl, "recipe_for_dish_name", lambda *a, **k: None)
    tid = next(iter(plantillas))
    assert dd.construir_comida(plantillas[tid], 1.0, catalogo_falso, "almuerzo", "DO") is None


def test_inclinar_sube_proteina_SIN_mover_calorias(dd, catalogo_falso):
    """La inclinación existe porque el escalado uniforme no cambia PROPORCIONES: multiplicar por
    1,2 sube los tres macros a la vez. Medido: pérdida de grasa pasó de 1/14 a 12/14 días en banda.
    Y las calorías tienen que quedarse donde estaban, o se pierde el ±0,2 % ya ganado."""
    cat = {
        "Pechuga de pollo": {"kcal_per_100g": 107, "protein_g_per_100g": 22.5,
                             "carbs_g_per_100g": 0, "fats_g_per_100g": 2},
        "Arroz blanco": {"kcal_per_100g": 359, "protein_g_per_100g": 7,
                         "carbs_g_per_100g": 80, "fats_g_per_100g": 0.5},
    }
    lineas = [[150.0, "Pechuga de pollo", "proteico"], [100.0, "Arroz blanco", "carbo"]]
    antes = dd._macros([(l[0], l[1]) for l in lineas], cat)
    fuera = dd._inclinar([list(x) for x in lineas], cat, obj_p=60.0)
    despues = dd._macros([(l[0], l[1]) for l in fuera], cat)
    assert despues["protein_g"] > antes["protein_g"], "la inclinación no subió la proteína"
    assert abs(despues["kcal"] - antes["kcal"]) / antes["kcal"] < 0.05, (
        "la inclinación movió las calorías: el ±0,2 % que ya estaba clavado se pierde")


def test_inclinar_NUNCA_borra_un_ingrediente(dd):
    """Quitar del todo un ingrediente cambia el plato Y lo deja comprado sin usar en la receta —
    el huérfano V3 que el escáner ya caza. De ahí el suelo del 30 %."""
    cat = {
        "Pechuga de pollo": {"kcal_per_100g": 107, "protein_g_per_100g": 22.5,
                             "carbs_g_per_100g": 0, "fats_g_per_100g": 2},
        "Arroz blanco": {"kcal_per_100g": 359, "protein_g_per_100g": 7,
                         "carbs_g_per_100g": 80, "fats_g_per_100g": 0.5},
    }
    lineas = [[100.0, "Pechuga de pollo", "proteico"], [200.0, "Arroz blanco", "carbo"]]
    fuera = dd._inclinar(lineas, cat, obj_p=9999.0)     # objetivo imposible a propósito
    arroz = next(l for l in fuera if l[1] == "Arroz blanco")
    assert arroz[0] >= 200.0 * dd._TILT_MIN_FRAC - 0.05, (
        f"el arroz bajó a {arroz[0]} g: por debajo del suelo el plato deja de ser su nombre")
    assert arroz[0] > 0


def test_la_costura_de_la_biblioteca_exige_el_MISMO_conjunto(monkeypatch, plantillas):
    """En la medición del 08-sep hubo exactamente un caso —«Huevos revueltos con cebolla y
    casabe»— donde el nombre casaba y el conjunto de alimentos no. Sin este guard le habríamos
    servido una receta que habla de comida que no está en el plato."""
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    import recipe_library as rl
    if not _LIB.exists():
        pytest.skip("sin biblioteca")
    lib = json.loads(_LIB.read_text(encoding="utf-8"))["por_id"]
    tid = next((k for k in plantillas if k in lib), None)
    if not tid:
        pytest.skip("ninguna plantilla con receta")
    t = plantillas[tid]
    bueno = {"name": t.get("name"), "recipe": ["del modelo"],
             "ingredients": [f"{c['grams']:g} g de {c['name']}"
                             for c in (t.get("constituents") or [])]}
    assert rl.apply_library_recipe(bueno) is True
    assert bueno["_recipe_source"] == "library"
    assert rl.apply_library_recipe(bueno) is False, "no es idempotente"

    impostor = {"name": t.get("name"), "recipe": ["del modelo"],
                "ingredients": ["999 g de Un alimento que no lleva"]}
    assert rl.apply_library_recipe(impostor) is False
    assert impostor["recipe"] == ["del modelo"]


def test_la_costura_calla_con_el_knob_apagado(monkeypatch, plantillas):
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "0")
    import recipe_library as rl
    tid = next(iter(plantillas))
    t = plantillas[tid]
    meal = {"name": t.get("name"), "recipe": ["del modelo"],
            "ingredients": [f"{c['grams']:g} g de {c['name']}"
                            for c in (t.get("constituents") or [])]}
    assert rl.apply_library_recipe(meal) is False
    assert rl.apply_library_recipes_to_days([{"meals": [meal]}]) == 0
