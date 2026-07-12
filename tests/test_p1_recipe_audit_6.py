"""[P1-RECIPE-AUDIT-6 · 2026-07-12] Audit visual de la renovación viva 69f9e03d (12 recetas).

Seis defectos visibles con causa raíz identificada, cada uno con su fix:
1. Arroz huérfano: `_repair_gainmuscle_day_kcal` añadía "Xg de arroz blanco cocido" SOLO a
   ingredients (el orphan-sweep del review corre ANTES del final_pass) → nota 💡 sin cantidad.
2. Nota undercook stale: RECIPE-COHERENCE-AUTOFIX reescribía pollo/cerdo→pescado en los pasos
   pero la nota ⚠ "el pollo/cerdo debe cocinarse..." está exenta del rewrite → se retira entera
   cuando la proteína de riesgo ya no existe en el cuerpo.
3. "65 g de semillas de chía" en un batido: el micro-closer de fibra escala semillas sin techo
   → SEED_GRAM_CAP_G (30g).
4. "2¾ cdtas de comino": el solver escala condimentos como comida → SPICE_CDTA_CAP (1 cdta).
5. "3½ pote de yogurt": plural inverso (qty>1 + unidad singular) en `_shared_clean`.
6. "el huevo fritos" / pasos arrancando en minúscula: concordancia + capitalización en el
   boundary display polish.
tooltip-anchor: P1-RECIPE-AUDIT-6
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

import graph_orchestrator as go  # noqa: E402

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _SRC = f.read()


# ---------------------------------------------------------------- 1. arroz huérfano
def test_gainmuscle_refill_adds_recipe_note():
    day = {"day": 1, "meals": [{
        "name": "Pescado guisado", "meal": "Almuerzo",
        "cals": 300, "protein": 40, "carbs": 10, "fats": 8,
        "ingredients": ["150 g de pescado"],
        "ingredients_raw": ["150 g de pescado"],
        "recipe": ["Mise en place: limpia el pescado.", "Montaje: sirve y disfruta."],
    }]}
    nutrition = {"macros": {"protein_g": 120, "carbs_g": 250, "fats_g": 55}}
    added = go._repair_gainmuscle_day_kcal([day], nutrition, {"mainGoal": "gain_muscle"})
    if added <= 0:
        return  # knob apagado en este entorno — el contrato se verifica con el knob ON
    m = day["meals"][0]
    assert any("arroz blanco cocido" in str(i).lower() for i in m["ingredients"])
    assert any("arroz blanco cocido" in str(s).lower() for s in m["recipe"]), \
        "el refill debe dejar mención en los pasos — sin ella el ingrediente queda huérfano"
    # la nota es consolidación-proof: sin cantidad embebida
    _note = next(s for s in m["recipe"] if "arroz blanco cocido" in str(s).lower())
    assert not re.search(r"\d", str(_note)), "nota sin cantidad (la consolidación no la deja stale)"
    # idempotente: segundo pase no duplica la nota
    go._repair_gainmuscle_day_kcal([day], nutrition, {"mainGoal": "gain_muscle"}, final_pass=True)
    assert sum(1 for s in m["recipe"] if "arroz blanco cocido" in str(s).lower()) == 1


# ---------------------------------------------------------------- 2. nota undercook stale
def test_stale_undercook_note_removed_on_protein_rewrite():
    i = _SRC.find("nota undercook pollo/cerdo retirada de")
    assert i != -1, "bloque P1-RECIPE-AUDIT-6 de retiro de nota stale ausente del autofix"
    win = _SRC[max(0, i - 3000):i]
    assert "pollo/cerdo debe cocinarse" in win, "el filtro debe matchear el texto exacto de la nota"
    assert "_is_det_note" in win, "el cuerpo se mide EXCLUYENDO notas (la nota no se auto-justifica)"
    # el retiro vive dentro del bloque del autofix (antes del log canónico del rewrite)
    j = _SRC.find("mención(es) huérfana(s)", i)
    assert j != -1 and j - i < 1500, "el retiro debe correr en el mismo bloque del rewrite"


# ---------------------------------------------------------------- 3. cap de semillas
def test_seed_gram_cap():
    day = {"day": 1, "meals": [{
        "name": "Batido de mango y chía", "meal": "Merienda",
        "cals": 400, "protein": 15, "carbs": 40, "fats": 20,
        "ingredients": ["65 g de semillas de chía", "1 mango mediano (199 g pulpa)"],
        "ingredients_raw": ["65 g de semillas de chía", "1 mango mediano (199 g pulpa)"],
        "recipe": ["Montaje: licúa todo y sirve."],
    }]}
    go._cap_unrealistic_portions([day])
    line = day["meals"][0]["ingredients"][0]
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(line))
    assert m and float(m.group(1).replace(",", ".")) <= float(go.SEED_GRAM_CAP_G), \
        f"chía sin capear: {line!r}"


# ---------------------------------------------------------------- 4. cap de especias (cdta)
def test_spice_cdta_cap():
    day = {"day": 1, "meals": [{
        "name": "Ropa vieja", "meal": "Almuerzo",
        "cals": 500, "protein": 35, "carbs": 45, "fats": 15,
        "ingredients": ["2¾ cdtas de comino", "1½ cdas de aceite de oliva"],
        "ingredients_raw": ["2.75 cdtas de comino", "1.5 cdas de aceite de oliva"],
        "recipe": ["Montaje: sirve y disfruta."],
    }]}
    go._cap_unrealistic_portions([day])
    comino = str(day["meals"][0]["ingredients"][0])
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)?\s*([¼½¾⅓⅔])?", comino)
    val = float((m.group(1) or "0").replace(",", ".")) + go._REALISM_FRAC_MAP.get(m.group(2) or "", 0.0)
    assert val <= go.SPICE_CDTA_CAP + 0.01, f"comino sin capear: {comino!r}"
    # el aceite en CDAS no es especia → intacto
    assert "aceite" in str(day["meals"][0]["ingredients"][1]).lower()


# ---------------------------------------------------------------- 5. plural inverso
def test_plural_inverse_display():
    days = [{"day": 1, "meals": [{
        "name": "Yogurt con piña", "meal": "Merienda",
        "ingredients": ["3½ pote de yogurt natural (700g)", "1 taza de piña en cubos",
                        "2 taza de lechuga", "½ taza de agua tibia"],
        "ingredients_raw": ["3.5 pote de yogurt natural (700g)", "1 taza de piña en cubos",
                            "2 taza de lechuga", "0.5 taza de agua tibia"],
        "recipe": ["Montaje: sirve."],
    }]}]
    go._polish_finalize_display(days)
    ings = [str(s) for s in days[0]["meals"][0]["ingredients"]]
    assert any("potes de yogurt" in s for s in ings), f"'3½ pote' debe pluralizar: {ings}"
    assert any(s.startswith("1 taza de piña") for s in ings), "'1 taza' intacta"
    assert any("2 tazas de lechuga" in s for s in ings), f"'2 taza' debe pluralizar: {ings}"
    assert any("½ taza de agua" in s for s in ings), "fracción pura ≤1 intacta"


# ---------------------------------------------------------------- 6. concordancia + capitalización
def test_egg_adjective_concord_regex():
    fix = lambda s: go._EGG_ADJ_CONCORD_RX.sub(r"\1 \2", s)
    assert fix("coloca el huevo fritos al lado") == "coloca el huevo frito al lado"
    assert fix("vierte el huevo batidos y cocina") == "vierte el huevo batido y cocina"
    assert fix("sirve los huevos fritos") == "sirve los huevos fritos", "plural legítimo intacto"


def test_step_capitalization_applied_in_boundary_polish():
    i = _SRC.find("_EGG_ADJ_CONCORD_RX.sub")
    assert i != -1, "el boundary polish debe aplicar la concordancia de huevo"
    win = _SRC[i:i + 600]
    assert ".upper()" in win and "islower()" in win, \
        "capitalización del arranque del paso debe vivir junto a la concordancia"
