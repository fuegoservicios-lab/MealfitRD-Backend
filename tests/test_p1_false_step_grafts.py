"""[P1-FALSE-STEP-GRAFTS · 2026-07-25] Pasos de receta que describen un plato que no es.

Revisión visual de las 12 recetas del plan vivo `1d3c6643` (banda 1.00, entregado). Los macros
estaban bien; los PASOS mentían. Cuatro causas distintas, todas del mismo tipo: un texto genérico
se injerta en un plato que no lo admite.

    Revoltillo de huevos:  "Agrega atún en agua a la licuadora y licúa hasta integrar."
    Bollitos al vapor:     "Añade huevo al GUISO en los últimos minutos"      ← no hay guiso
    Canoas de batata:      "Cocina camarones a la plancha o hervidos"         ← ya son "cocido"

El del atún es el peor: quien siga el paso hace puré de atún.

## La cascada del atún

    "Vierta el huevo batido"        → `batido` matchea la técnica LICUADO
    → el anotador escribe "cocine por 1-2 min de licuado a velocidad alta" en un paso de SARTÉN
    → el closer busca "licua" en los pasos y lo encuentra dentro de "licuado"
    → declara el plato batido → "Agrega atún a la licuadora"

⚠️ La defensa que existía era el ORDEN de las tuplas de técnica: `revuelto`/`revuelve` van antes
que `licua`, así que un revoltillo matchearía primero. Falló porque esta receta conjuga
*"revuelva"* y *"sin revolver"* — ninguna contiene `revuelve`. **Depender de que el LLM elija una
conjugación concreta no es una defensa.** Neutralizar el participio sí: "huevo batido" es
siempre alimento, nunca técnica.

Es la misma clase de bug que `"mora"`⊂`"morado"` y `"pollo"`⊂`"repollo"`: substring sin anclar
sobre nombres de alimentos en español.
"""
import pytest

import graph_orchestrator as go
from constants import strip_accents


# ───────────── 1. alimento que parece técnica ─────────────

@pytest.mark.parametrize("texto,tecnica_prohibida", [
    ("vierta el huevo batido sobre las papas y cocine", "licuado"),
    ("bata los huevos batidos con sal", "licuado"),
    ("incorpora las claras batidas con movimientos envolventes", "licuado"),
    ("agrega la crema batida por encima", "licuado"),
    ("ensalada fria de guisantes con vinagreta", "tapado"),
])
def test_palabra_alimento_no_dispara_tecnica(texto, tecnica_prohibida):
    limpio = go._strip_food_words_for_technique(texto)
    default = go._TIMETEMP_FALLBACK_DEFAULT
    for tokens, td in go._TIMETEMP_TECHNIQUE_DEFAULTS:
        if any(t in limpio for t in tokens):
            default = td
            break
    assert tecnica_prohibida not in default, f"{texto!r} → {default!r}"


def test_el_batido_DE_VERDAD_sigue_siendo_licuado():
    """La neutralización no puede desactivar el caso que P2-LICUADO-TIMETEMP vino a cerrar."""
    for texto in ("batido de lechosa con mantequilla de mani",
                  "coloca todo en la licuadora y licua a alta velocidad",
                  "procesa hasta obtener una mezcla homogenea"):
        limpio = go._strip_food_words_for_technique(texto)
        default = go._TIMETEMP_FALLBACK_DEFAULT
        for tokens, td in go._TIMETEMP_TECHNIQUE_DEFAULTS:
            if any(t in limpio for t in tokens):
                default = td
                break
        assert "licuado" in default, f"{texto!r} → {default!r}"


def test_no_muta_el_paso_que_ve_el_usuario():
    """`_strip_food_words_for_technique` es sólo para emparejar; el texto es intocable."""
    meal = {"name": "Revoltillo de Huevos", "recipe": [
        "Mise en place: bate el huevo con una pizca de sal.",
        "El Toque de Fuego: vierte el huevo batido sobre las papas y revuelve."]}
    antes = list(meal["recipe"])
    go._clamp_recipe_time_temp_outliers(meal)
    assert "huevo batido" in " ".join(meal["recipe"]), meal["recipe"]
    assert len(meal["recipe"]) == len(antes)


# ───────────── 2. vapor no es guiso ─────────────

def test_bollitos_al_vapor_no_son_guiso():
    """Caso vivo: "tapa y cocina al vapor" casaba y producía "Añade huevo al GUISO"."""
    meal = {"name": "Bollitos de Harina de Negrito Rellenos de Queso Mozzarella", "recipe": [
        "Coloca los bollitos en una vaporera o cesta de bambú sobre agua hirviendo; "
        "tapa y cocina al vapor durante 12-15 minutos hasta que estén cocidos."]}
    assert go._meal_is_stewy(meal, strip_accents) is False


@pytest.mark.parametrize("paso", [
    "coloca la bandeja en el horno, tapa con papel aluminio y cocina 25 minutos",
    "tapa la freidora de aire y cocina por 12 minutos",
])
def test_horno_y_airfryer_tampoco(paso):
    assert go._meal_is_stewy({"name": "Plato", "recipe": [paso]}, strip_accents) is False


def test_el_guiso_DE_VERDAD_sigue_siendo_guiso():
    """No vale cerrar el falso positivo desactivando el detector."""
    assert go._meal_is_stewy(
        {"name": "Bowl Tibio", "recipe": [
            "Cubre con agua, tapa y cocina a fuego bajo por 15 minutos hasta que espese."]},
        strip_accents) is True
    assert go._meal_is_stewy(
        {"name": "Frijoles Pintos Guisados", "recipe": ["Sofríe y sirve."]}, strip_accents) is True


def test_tapa_y_cocina_LEJOS_no_cuentan():
    """El `.*` original recorría el blob entero: un 'tapa' del paso 2 casaba con un 'cocina' del
    paso 6. La ventana acotada es lo que lo impide."""
    meal = {"name": "Ensalada", "recipe": [
        "Tapa el bowl y refrigera.",
        "Pica el tomate en cubos. Lava la lechuga. Escurre y reserva aparte en un colador grande.",
        "Sirve frío. Si sobra, cocina el resto al día siguiente."]}
    assert go._meal_is_stewy(meal, strip_accents) is False


# ───────────── 3. lo que ya viene cocido no se cocina ─────────────

def test_camarones_cocido_en_la_linea():
    """El closer sólo miraba el nombre de catálogo ("Camarones"); la línea del plato decía
    "150g camarones cocido"."""
    txt = go._closer_protein_step_text("camarones", no_cook=False, precooked=True)
    assert "Cocina" not in txt, txt
    assert "Incorpora camarones" in txt


def test_concordancia_plural_en_incorpora():
    """Plan vivo: "Incorpora camarones … mézclalo"."""
    assert "mézclalos" in go._closer_protein_step_text("camarones", no_cook=False, precooked=True)
    assert "mézclalo " in go._closer_protein_step_text("queso cottage", no_cook=True) + " "


def test_precooked_se_detecta_desde_la_linea_del_plato():
    meal = {"name": "Canoas de Batata", "ingredients": ["150g camarones cocido", "½ batata"],
            "recipe": ["Hornea las batatas 25 minutos.", "Montaje: sirve caliente."]}
    assert go._append_closer_protein_step(meal, "camarones", no_cook=False) is True
    paso = next(s for s in meal["recipe"] if "💪" in s)
    assert "Cocina camarones" not in paso, paso


# ───────────── 4. la cascada completa, extremo a extremo ─────────────

def test_revoltillo_no_manda_el_atun_a_la_licuadora():
    """El caso reportado por el owner, tal cual salió del plan vivo."""
    meal = {"name": "Revoltillo de Huevos con Papas y Aguacate",
            "ingredients": ["1 huevo", "3 papas medianas", "110g de atún en agua"],
            "recipe": [
                "Mise en place: bata el huevo con una pizca de sal.",
                "El Toque de Fuego: caliente el aceite en una sartén antiadherente. Vierta el "
                "huevo batido sobre las papas, reduzca el fuego y cocine sin revolver por 1 "
                "minuto, luego revuelva suavemente con una espátula.",
                "Montaje: sirva el revoltillo caliente en un plato."]}
    go._clamp_recipe_time_temp_outliers(meal)
    assert "licuado" not in " ".join(meal["recipe"]).lower(), meal["recipe"]
    go._append_closer_protein_step(meal, "atún en agua", no_cook=False)
    paso = next((s for s in meal["recipe"] if "💪" in s), "")
    assert "licuadora" not in paso.lower(), paso
