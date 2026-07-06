"""[P1-RECIPE-COHERENCE-LIVE · 2026-06-30] Fixes de coherencia de recetas detectados en output REAL de producción
(5 recetas que el owner mostró post-deploy P1+P2):

  P1-CLOSER-PRECOOKED-WORDING — el paso "💪" del closer de proteína decía "Cocina sardinas en lata a la plancha o
     hervido" (absurdo: enlatado YA viene cocido). Helper SSOT `_closer_protein_step_text`: lácteo blando→'Incorpora';
     enlatado/pre-cocido→'Escurre e incorpora (ya viene cocido)'; resto→'Cocina'. Idem el sufijo " cocido" del ingrediente.
  P1-RECIPE-OFFCATALOG-CONDIMENT — el LLM usó "salsa de soya" (off-catálogo) en los pasos → dropeada de la lista pero
     dejada en el texto (receta rota). Prompt rule 5 ahora la prohíbe explícitamente.
  P1-DISH-PALATABILITY — combos disparate (avena salteada salada con guisantes) + proteína incongruente pegada
     (sardinas en un revoltillo). Prompt bloque de creatividad ahora lo desalienta.
"""
from pathlib import Path
import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DAYGEN = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")


# ───────────────────────── P1-CLOSER-PRECOOKED-WORDING ─────────────────────────
def test_precooked_protein_step_wording():
    f = g._closer_protein_step_text
    # enlatados / pre-cocidos → NO "cocina a la plancha", sí "escurre e incorpora (ya viene cocido)"
    for nm in ("sardinas en lata", "Atún en agua", "Salmón ahumado"):
        s = f(nm, False)
        assert "escurre e incorpora" in s.lower(), f"'{nm}' debería ser escurrir-e-incorporar, dio: {s}"
        assert "ya viene cocido" in s.lower()
        assert "a la plancha o hervido" not in s.lower(), f"'{nm}' NO debe decir cocina a la plancha"
    # lácteo blando → incorpora (no cocina)
    assert "incorpora" in f("Queso cottage", False).lower()
    assert "a la plancha" not in f("Queso ricotta", False).lower()
    # carne real → sí cocina
    assert "cocina" in f("Pechuga de pollo", False).lower()
    assert "a la plancha o hervido" in f("Filete de pescado blanco", False).lower()


def test_precooked_hint_and_helper_wired():
    assert hasattr(g, "_PRECOOKED_PROTEIN_HINT")
    assert "sardina" in g._PRECOOKED_PROTEIN_HINT and "en lata" in g._PRECOOKED_PROTEIN_HINT
    # [P1-CLOSER-HYGIENE · 2026-07-06] re-anclado: los DOS callsites ahora pasan por el SSOT
    # anti-dup/anti-redundante (_append_closer_protein_step), que internamente usa el builder.
    assert _GRAPH.count("_append_closer_protein_step(") >= 3, "def + 2 callsites"
    assert "_closer_protein_step_text(nm, no_cook)" in _GRAPH, "el builder sigue siendo el SSOT del texto"
    # ya no debe quedar el branch hardcodeado "Cocina {nm} a la plancha o hervido" inline en los call sites
    assert "P1-CLOSER-PRECOOKED-WORDING" in _GRAPH


def test_precooked_ingredient_no_redundant_cocido():
    # el sufijo " cocido" no se añade a un enlatado (evita "sardinas en lata cocido")
    # [P1-CLOSER-HYGIENE · 2026-07-06] re-anclado: el guard creció (lácteos + nombres con "cocid"
    # tampoco llevan el sufijo — "½ taza de yogurt cocido" vivo en da7bb310).
    assert 'cook = "" if (no_cook or _pre_cooked or _dairy_nm or "cocid" in _nm_strip) else " cocido"' in _GRAPH


# ───────────────────────── P1-RECIPE-OFFCATALOG-CONDIMENT (prompt) ─────────────────────────
def test_offcatalog_condiments_forbidden_in_prompt():
    low = _DAYGEN.lower()
    assert "salsa de soya" in low, "rule 5 debe prohibir explícitamente salsa de soya"
    assert "p1-recipe-offcatalog-condiment" in low
    assert "salsa inglesa" in low or "worcestershire" in low


# ───────────────────────── P1-DISH-PALATABILITY (prompt) ─────────────────────────
def test_palatability_guard_in_prompt():
    low = _DAYGEN.lower()
    assert "p1-dish-palatability" in low
    assert "disparate" in low
    assert "salteado salado" in low or "avena con guisantes" in low


# ───────────────────────── P1-CLOSER-NO-DUP-CHEESE (#1) ─────────────────────────
def test_closer_no_dup_cheese_wired():
    assert g.CLOSER_NO_DUP_PROTEIN is True
    assert hasattr(g, "_CLOSER_CHEESE_HINT") and "ricotta" in g._CLOSER_CHEESE_HINT and "cottage" in g._CLOSER_CHEESE_HINT
    assert "P1-CLOSER-NO-DUP-CHEESE" in _GRAPH
    # el guard prefiere un alt no-queso de la categoría del slot
    assert "_pref2 = _DAIRY_EGG_PROTEIN_HINT if light else _MEAT_PROTEIN_HINT" in _GRAPH


# ───────────────────────── P1-RECIPE-OFFCATALOG-CONDIMENT strip (#2a) ─────────────────────────
def test_offcatalog_condiment_stripped_from_recipe_text():
    assert g.RECIPE_OFFCATALOG_STRIP_ENABLED is True
    meal = {"name": "Salteado de Guisantes y Avena",
            "ingredients": ["100 g de Guisantes", "20 g de Avena", "Sal al gusto"],  # salsa de soya NO está
            "recipe": ["Mise en place: pica la cebolla.",
                       "El Toque de Fuego: saltea los guisantes y la avena. Añade la salsa de soya, sal y pimienta y saltea 1 minuto."]}
    n = g._strip_offcatalog_condiments_from_recipe(meal)
    assert n == 1, "debe limpiar el paso con salsa de soya"
    joined = " ".join(meal["recipe"]).lower()
    assert "salsa de soya" not in joined, "la salsa de soya off-catálogo debe salir del texto"
    assert "sal y pimienta" in joined, "el resto del paso (sal, pimienta) se preserva"
    # idempotente
    assert g._strip_offcatalog_condiments_from_recipe(meal) == 0


def test_offcatalog_strip_noop_when_in_ingredients():
    # si el ingrediente SÍ está listado (caso hipotético), NO se borra del texto
    meal = {"name": "X", "ingredients": ["10 ml de Salsa de soya"],
            "recipe": ["El Toque de Fuego: añade la salsa de soya."]}
    assert g._strip_offcatalog_condiments_from_recipe(meal) == 0
