"""[P1-SUBST-STALE-STEP · 2026-08-01, ronda 4b] Consolidación del marker: los 5 tests
originales (defecto real, plan de producción 5f4bb17e-14cb-4db3-8d97-79933af690cf, score 97.2)
nacieron en `test_p1_swap_prose_honest.py` (ronda 1 del fix) porque ese archivo ya tenía el
fixture del swap huevo→yogur — se mueven aquí para que el slug del marker (`P1-SUBST-STALE-STEP`
→ `p1_subst_stale_step`) matchee un archivo `test_<slug>*.py`, contrato que
`test_p2_hist_audit_14_marker_test_link.py` enforza sobre `_LAST_KNOWN_PFIX`
(`backend/app.py`). Los tests en sí NO cambiaron de comportamiento esperado — moved clean, sin
imports rotos.

Contexto del defecto (ronda 1, sin cambios): `_substitute_blended_raw_egg`
(`graph_orchestrator.py`) renombra la MENCIÓN del alimento dentro del paso ("el huevo" → "el
yogur griego") pero dejaba la INSTRUCCIÓN DE COCCIÓN del huevo original intacta:

  ANTES  (huevo real): "El Toque de Fuego: Hierve el huevo en agua durante 8 minutos
                         hasta que estén firmes; pélalos y córtalos en trozos."
  POST-SWAP (bug):      "El Toque de Fuego: Hierve el yogur griego en agua durante 8
                         minutos hasta que estén firmes; pélalos y córtalos en trozos."

— huevo duro aplicado a un lácteo listo-para-comer, servido tal cual al usuario.

Ronda 4b (finding de review, demostrado por EJECUCIÓN — ninguna evidencia en prod aún): la
ronda 1 reescribía el PASO ENTERO en cuanto detectaba el verbo fósil colgando cerca de la
mención del sustituto. Un paso que mezcle esa cocción fósil con una acción legítima de OTRO
alimento en una cláusula vecina ("...; mientras tanto, tuesta el pan integral hasta que esté
dorado.") perdía esa acción también — la misma clase paso-vs-cláusula que las rondas 1-3 de
`culinary_coherence.py` (`_clause_bounds`/`_occurrence_resolves`) ya cerraron para el scan V1.
`_rewrite_stale_cooking_step` (nuevo en esta ronda) acota la reescritura a la(s) cláusula(s)
ofensora(s) via `culinary_coherence.clause_bounds` (export público nuevo de esta ronda) — las
demás cláusulas del paso sobreviven.

tooltip-anchor: P1-SUBST-STALE-STEP
"""
from __future__ import annotations

import culinary_coherence as cc
import graph_orchestrator as go


def _meal_egg_step(step_text, name="Batido de prueba", desc=None):
    meal = {
        "name": name,
        "ingredients": ["2 huevos crudos"],
        "ingredients_raw": ["2 huevos crudos"],
        "recipe": [step_text],
    }
    if desc is not None:
        meal["desc"] = desc
    return meal


def _meal_real_batido_caribeno():
    """Reconstrucción del meal real (antes del swap) a partir del texto post-swap observado
    en producción — la ÚNICA diferencia estructural es 'huevo'/'clara' donde el plan vivo ya
    dice 'yogur griego' (efecto del bug), reconstruido hacia atrás para poder ejercer
    `_substitute_blended_raw_egg` end-to-end como el resto de este archivo."""
    return {
        "name": "Batido Caribeño de Mango, Avena y Chía",
        "ingredients": [
            "65 g de mango en cubos", "15 g de avena", "75 ml de leche descremada",
            "10 g de semillas de chía", "1¾ cdtas de mantequilla de maní",
            "1 huevo entero", "2 claras de huevo",
        ],
        "ingredients_raw": [
            "65 g de mango en cubos", "15 g de avena", "75 ml de leche descremada",
            "10 g de semillas de chía", "1¾ cdtas de mantequilla de maní",
            "1 huevo entero", "2 claras de huevo",
        ],
        "recipe": [
            "Mise en place: Pela y corta 65 g de mango en cubos; mide 15 g de avena, 10 g de "
            "semillas de chía y 1¾ cdta de mantequilla de maní (5 g); mide 75 ml de leche "
            "descremada; separa 1 huevo y 2 claras.",
            "El Toque de Fuego: Hierve el huevo en agua durante 8 minutos hasta que estén "
            "firmes; pélalos y córtalos en trozos.",
            "Montaje: Coloca el mango, la avena, las semillas de chía, la mantequilla de "
            "maní, el huevo, la clara y la leche descremada en la licuadora; procesa durante "
            "1-2 minutos hasta obtener un batido homogéneo y sírvelo bien frío.",
        ],
        "protein": 15, "carbs": 30, "fats": 5, "cals": 220,
    }


# ─────────────────────────── ronda 1 (movidos, sin cambios) ───────────────────────────

def test_p1_subst_stale_step_hierve_el_huevo_se_reescribe_a_incorpora():
    """El defecto real: el paso 'El Toque de Fuego' quedaba diciendo 'Hierve el yogur
    griego...pélalos...' — huevo duro sobre un lácteo listo-para-comer. Debe reescribirse a
    'Incorpora el yogur griego.', preservando el prefijo de sección.

    [ronda 4b] También ancla que un paso 100% fósil (sin ninguna cláusula vecina que mencione
    otro alimento) sigue colapsando COMPLETO — la reescritura por cláusula no deja un remanente
    "; pélalos y córtalos en trozos." colgando: esa cláusula no nombra ningún alimento propio
    (son pronombres del huevo ya hervido) y el barrido de `_rewrite_stale_cooking_step` la funde
    en la misma reescritura."""
    meal = _meal_real_batido_caribeno()
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    assert meal["recipe"][1] == "El Toque de Fuego: Incorpora el yogur griego.", meal["recipe"][1]
    # nunca sobrevive un verbo de cocción térmica aplicado al sustituto
    blob = meal["recipe"][1].lower()
    assert "hierve" not in blob and "pélalos" not in blob and "pelalos" not in blob


def test_p1_subst_stale_step_montaje_dedup_comma_connector():
    """Bonus del mismo meal: 'el yogur griego, el yogur griego' (dos swaps independientes
    adyacentes, separados por coma) colapsa a una sola mención — el Montaje SIGUE listando el
    resto de ingredientes intacto (no se reescribe el paso entero, solo se deduplica)."""
    meal = _meal_real_batido_caribeno()
    go._substitute_blended_raw_egg(meal, None)
    montaje = meal["recipe"][2]
    assert "yogur griego, el yogur griego" not in montaje.lower(), montaje
    assert montaje == (
        "Montaje: Coloca el mango, la avena, las semillas de chía, la mantequilla de maní, "
        "el yogur griego y la leche descremada en la licuadora; procesa durante 1-2 minutos "
        "hasta obtener un batido homogéneo y sírvelo bien frío."
    ), montaje


def test_p1_subst_stale_step_guard_licuar_del_montaje_no_dispara():
    """[Guard 1, encontrado corriendo la suite ANTES de fijar el fix] 'licúa'/'licuadora' es
    el verbo ESPERADO en TODA preparación 'blended' (única `kind` que invoca esta función) —
    licuar el sustituto junto con el resto es correcto por definición, no un residuo de
    cocción térmica. El Montaje ('...en la licuadora; procesa...') NO debe reescribirse a
    'Incorpora...' — sigue mencionando el resto de los ingredientes."""
    meal = _meal_real_batido_caribeno()
    go._substitute_blended_raw_egg(meal, None)
    montaje = meal["recipe"][2]
    assert montaje.startswith("Montaje: Coloca el mango"), (
        f"'licúa'/'licuadora' disparó un rewrite espurio del Montaje: {montaje!r}")
    assert "sírvelo bien frío" in montaje


def test_p1_subst_stale_step_guard_verbo_de_otro_alimento_no_dispara():
    """[Guard 2, encontrado corriendo la suite ANTES de fijar el fix — rompía
    `test_p1_dangling_adverb_plus_participle_both_dropped` en test_p1_swap_prose_honest.py] Un
    verbo de cocción legítimo aplicado a OTRO alimento del MISMO paso ('Hornea...la avena...')
    no debe disparar el rewrite solo porque el sustituto aparece más adelante en la misma
    oración — la ventana de atribución es la vecindad INMEDIATA del sustituto, no 'el paso
    completo'."""
    meal = _meal_egg_step(
        "Hornea hasta que la avena horneada esté firme y las claras completamente cocidas.",
        name="Avena horneada con clara")
    meal["ingredients"] = ["2 claras de huevo"]
    meal["ingredients_raw"] = ["2 claras de huevo"]
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    step = meal["recipe"][0]
    assert step.lower() == "hornea hasta que la avena horneada esté firme y el yogur griego.", step
    assert not step.lower().startswith("incorpora"), (
        f"el verbo 'Hornea' cocina la AVENA, no el sustituto — no debía disparar: {step!r}")


def test_p1_subst_stale_step_idempotent_after_rewrite():
    """El paso ya reescrito a 'Incorpora el yogur griego.' no vuelve a mutarse en una segunda
    pasada (sin términos de huevo/clara/yema restantes en `ingredients`, `changed` es False y
    la función retorna temprano sin tocar `recipe`)."""
    meal = _meal_real_batido_caribeno()
    go._substitute_blended_raw_egg(meal, None)
    recipe_after_first = list(meal["recipe"])
    changed_again = go._substitute_blended_raw_egg(meal, None)
    assert changed_again is False
    assert meal["recipe"] == recipe_after_first


# ═══════════════ ronda 4b (2026-08-01) — rewrite por CLÁUSULA, no por paso ═══════════════
#
# Finding del review sobre la ronda 1 (demostrado por ejecución — sin evidencia en prod aún):
# la reescritura operaba sobre el PASO ENTERO, así que un paso que mezclara la cocción fósil
# del huevo con una acción legítima de OTRO alimento en una cláusula vecina perdía esa acción
# también. Caso sintético construido para ejercer exactamente ese finding.

def _meal_sibling_action_across_semicolon():
    """Caso SINTÉTICO (no observado en prod) que reproduce el finding del review: la cláusula
    del huevo y una acción legítima sobre pan integral conviven en el MISMO paso, separadas por
    ';' — estructuralmente IDÉNTICO al caso real de `_meal_real_batido_caribeno` (también un
    ';' entre dos cláusulas tras 'El Toque de Fuego'), pero semánticamente distinto: la 2ª
    cláusula aquí SÍ nombra un alimento propio (pan integral) y trae su propio verbo de cocción
    (tostar), a diferencia de 'pélalos y córtalos en trozos' (pronombres del huevo, sin verbo
    de cocción ni alimento propio)."""
    return {
        "name": "Desayuno de prueba",
        "ingredients": ["2 huevos crudos", "2 rebanadas de pan integral"],
        "ingredients_raw": ["2 huevos crudos", "2 rebanadas de pan integral"],
        "recipe": [
            "El Toque de Fuego: Hierve el huevo en agua durante 8 minutos; mientras tanto, "
            "tuesta el pan integral en la tostadora hasta que esté dorado.",
        ],
    }


def test_p1_subst_stale_step_sibling_action_across_semicolon_survives():
    """El finding del review: la cláusula del huevo se reescribe a 'Incorpora el yogur griego'
    pero la acción legítima sobre OTRO alimento ('tuesta el pan integral...'), separada por
    ';', SOBREVIVE — antes de la ronda 4b el paso entero se reemplazaba y esa acción se perdía
    por completo."""
    meal = _meal_sibling_action_across_semicolon()
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    step = meal["recipe"][0]
    assert step == (
        "El Toque de Fuego: Incorpora el yogur griego; mientras tanto, tuesta el pan integral "
        "en la tostadora hasta que esté dorado."
    ), step
    blob = step.lower()
    # la cocción fósil del huevo desaparece de su propia cláusula...
    assert "hierve" not in blob
    # ...pero la acción legítima sobre el pan sigue intacta, palabra por palabra
    assert "tuesta el pan integral en la tostadora hasta que esté dorado" in blob


def test_p1_subst_stale_step_full_fossil_step_still_collapses_round4b():
    """Contraste directo con el test anterior usando el MISMO meal real de la ronda 1 (Batido
    Caribeño): sin una cláusula vecina que nombre otro alimento, el paso 100% fósil sigue
    dando 'El Toque de Fuego: Incorpora el yogur griego.' completo — la ronda 4b no regresiona
    el caso real ya cerrado por acotar de más."""
    meal = _meal_real_batido_caribeno()
    go._substitute_blended_raw_egg(meal, None)
    assert meal["recipe"][1] == "El Toque de Fuego: Incorpora el yogur griego."


def test_p1_subst_stale_step_clause_bounds_exported_public_and_consistent():
    """[P1-SUBST-STALE-STEP · ronda 4b] `culinary_coherence.clause_bounds` (público, nuevo en
    esta ronda) enumera TODAS las cláusulas de un texto; `_clause_bounds` (privado, ronda 3)
    delega en él para resolver la cláusula de UNA sola posición. Ancla la equivalencia para que
    un futuro refactor de cualquiera de las dos no rompa la otra sin que un test lo note."""
    texto = "Hierve el yogur griego en agua durante 8 minutos; mientras tanto, tuesta el pan."
    semicolon = texto.index(";")
    final_dot = texto.rindex(".")
    bounds = cc.clause_bounds(texto)
    assert bounds == [(0, semicolon), (semicolon + 1, final_dot)], bounds
    # `_clause_bounds` (privado) resuelve al mismo span que su lista pública para una posición
    # dentro de cada cláusula.
    assert cc._clause_bounds(texto, texto.index("Hierve")) == bounds[0]
    assert cc._clause_bounds(texto, texto.index("tuesta")) == bounds[1]
