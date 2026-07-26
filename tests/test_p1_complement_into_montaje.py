"""[P1-COMPLEMENT-INTO-MONTAJE · 2026-07-26] El paso pegado que delataba al corrector automático.

Medido sobre 72 comidas de 6 planes vivos: **18 (25%)** traían un paso del tipo

    2. Incorpora también queso cottage a la preparación y mézclalo antes de servir.
    3. Montaje: Sirve lechosa junto a las almendras… formando una merienda fresca.

y el Montaje ni mencionaba el ingrediente. Casi todas son meriendas frías y casi siempre el olvidado
es el queso cottage. Se lee como lo que es: un parche encima de una receta que se olvidó de algo.

El corrector (`P2-RECIPE-REVERSE-COHERENCE`) hace lo CORRECTO — sin él el usuario compra un
ingrediente que ninguna instrucción usa, y por eso el matcher de producción reporta 0 sin usar. Lo
que falla es la **prosa**. Fusionarlo en el Montaje da:

    2. Montaje: Sirve lechosa… formando una merienda fresca. Termina con queso cottage.

## La condición de seguridad, y por qué NO basta `_meal_is_no_cook`

Mover un ingrediente al emplatado de un plato que se cocina puede saltarse su cocción. Mi primera
versión se apoyaba en `_meal_is_no_cook` y **dejaba pasar platos cocinados**. La comprobación se hace
ahora aquí, sobre los propios pasos, con un regex de verbos de cocción deliberadamente AMPLIO: un
falso positivo sólo cuesta conservar el paso suelto (correcto, sólo peor escrito); un falso negativo
manda a emplatar algo que había que cocinar. La asimetría manda.
"""
import copy

import pytest

import graph_orchestrator as go


_FRIA = {
    "name": "Lechosa Fresca con Almendras y Tostada Integral y Queso cottage",
    "ingredients": ["100 g de lechosa en cubos", "25 g de almendras fileteadas",
                    "1 rebanada de pan integral familiar", "35 g de queso cottage"],
    "recipe": ["Mise en place: Pela y corta lechosa en cubos; mide las almendras fileteadas.",
               "Montaje: Sirve lechosa junto a las almendras fileteadas y la rebanada de pan "
               "integral familiar, formando una merienda fresca."],
}
_COCINADA = {
    "name": "Filete de pescado blanco al Limón sobre Yuca Majada",
    "ingredients": ["1 filete de pescado", "1 pedazo de yuca", "1 cda de aceite de oliva"],
    "recipe": ["Mise en place: Pela y corta la yuca en trozos medianos.",
               "El Toque de Fuego: Hierve la yuca a fuego medio durante 18-22 minutos.",
               "Montaje: Extiende la yuca majada en el plato y coloca el filete encima."],
}


# ───────────── 1. el caso frío: se fusiona ─────────────

def test_la_merienda_fria_se_fusiona_sin_paso_suelto():
    m = copy.deepcopy(_FRIA)
    n_antes = len(m["recipe"])
    assert go._ensure_ingredients_used_in_recipe(m) >= 1
    assert len(m["recipe"]) == n_antes, "no debe añadir paso: va dentro del Montaje"
    montaje = m["recipe"][-1]
    assert montaje.lower().startswith("montaje")
    assert "queso cottage" in montaje.lower()
    assert "Termina con" in montaje


def test_el_ingrediente_NUNCA_se_queda_sin_usar():
    """Lo que este guard existe para impedir: que el usuario compre algo que ninguna instrucción
    menciona. La fusión cambia el DÓNDE, jamás el SI."""
    m = copy.deepcopy(_FRIA)
    go._ensure_ingredients_used_in_recipe(m)
    texto = " ".join(m["recipe"]).lower()
    assert "queso cottage" in texto


# ───────────── 2. el caso cocinado: NO se toca ─────────────

def test_el_plato_cocinado_conserva_su_paso_separado():
    """Fusionar aquí podría saltarse la cocción del ingrediente."""
    m = copy.deepcopy(_COCINADA)
    n_antes = len(m["recipe"])
    if go._ensure_ingredients_used_in_recipe(m):
        assert len(m["recipe"]) > n_antes, "en plato cocinado debe ir como paso propio"


@pytest.mark.parametrize("paso", [
    "El Toque de Fuego: Hierve la yuca 18 minutos.",
    "Sofríe el ajo por 1 minuto.",
    "Hornea durante 25-30 minutos hasta que estén dorados.",
    "Saltea el bok choy 3-4 minutos.",
    "Cocina el arroz tapado a fuego bajo.",
    "Tuesta el pan a fuego medio.",
    "Asa las tortitas en la parrilla.",
    "Cocina en el airfryer 12 minutos.",
])
def test_el_regex_reconoce_la_coccion(paso):
    from constants import strip_accents as sa
    assert go._COOKING_VERB_RE.search(sa(paso.lower())), paso


def test_los_tres_platos_que_mi_test_MALO_dio_por_frios():
    """Al medir borré por error el paso «El Toque de Fuego» de tres platos (mi filtro casaba la
    palabra «incorpora», que ese paso contiene) y parecían fríos. Sobre la receta real los tres
    cocinan. Queda anclado para que la próxima versión del guard no los pierda."""
    from constants import strip_accents as sa
    for pasos in (
        ["El Toque de Fuego: Hierve la yuca a fuego medio; incorpora el jugo de limón."],
        ["El Toque de Fuego: estofa a fuego bajo 8 minutos e incorpora la coliflor."],
        ["El Toque de Fuego: maja la batata cocida a fuego lento e incorpora el edamame."],
    ):
        assert go._COOKING_VERB_RE.search(sa(" ".join(pasos).lower()))


# ───────────── 3. no duplicar, no romper ─────────────

def test_si_el_montaje_ya_lo_nombra_no_se_toca():
    m = copy.deepcopy(_FRIA)
    m["recipe"][-1] = "Montaje: Sirve lechosa con el queso cottage por encima."
    n_antes = len(m["recipe"])
    go._ensure_ingredients_used_in_recipe(m)
    assert m["recipe"][-1].lower().count("queso cottage") == 1
    assert len(m["recipe"]) == n_antes


def test_sin_montaje_cae_al_paso_suelto():
    m = copy.deepcopy(_FRIA)
    m["recipe"] = ["Mise en place: pela y corta la lechosa en cubos."]
    n_antes = len(m["recipe"])
    if go._ensure_ingredients_used_in_recipe(m):
        assert len(m["recipe"]) > n_antes


def test_puntuacion_limpia():
    m = copy.deepcopy(_FRIA)
    m["recipe"][-1] = "Montaje: Sirve lechosa con almendras"      # sin punto final
    go._ensure_ingredients_used_in_recipe(m)
    s = m["recipe"][-1]
    assert ".." not in s and " ." not in s and s.endswith(".")


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "COMPLEMENT_INTO_MONTAJE", False)
    m = copy.deepcopy(_FRIA)
    n_antes = len(m["recipe"])
    go._ensure_ingredients_used_in_recipe(m)
    assert len(m["recipe"]) > n_antes, "con el knob apagado vuelve el paso suelto"


def test_fail_safe():
    assert go._merge_complement_into_montaje([], ["x"]) is False
    assert go._merge_complement_into_montaje(["Montaje: sirve."], []) is False
    assert go._merge_complement_into_montaje(None, ["x"]) is False
