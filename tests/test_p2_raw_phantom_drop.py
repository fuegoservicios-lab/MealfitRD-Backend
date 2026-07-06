"""[P2-RAW-PHANTOM-DROP · 2026-07-06] Cierre del forense del plan fcc7a9f0:
"Bollitos de Yautía y Queso" traía en ingredients_raw "50g de camarones cocido"
que NI el display NI los pasos usan — un fantasma de COMPRA (la lista lo
compraría para una receta que no lo toca). Dentro de
_restore_display_from_raw_orphans: fila raw-huérfana sin uso en display ni
pasos Y con cantidad cuantificada (arranca con dígito) → se remueve del raw.
Los "al gusto" sin cantidad (sal/pimienta) se quedan (despensa implícita), y
las filas que los pasos SÍ usan se restauran al display, jamás se dropean.
"""
import graph_orchestrator as go


def _bollitos():
    return {"days": [{"day": 1, "meals": [{
        "name": "Bollitos de Yautía y Queso",
        "ingredients": ["300 g de yautía", "80 g de queso frito"],
        "ingredients_raw": ["300 g de yautia", "80 g de queso frito",
                            "50g de camarones cocido", "Sal al gusto"],
        "recipe": [
            "Mise en place: hierve la yautía y hazla puré con una pizca de sal.",
            "Montaje: rellena los bollitos con el queso y fríelos.",
        ],
    }]}]}


def test_quantified_phantom_removed_from_raw():
    plan = _bollitos()
    meal = plan["days"][0]["meals"][0]
    go._restore_display_from_raw_orphans(plan["days"])
    raw_blob = " ".join(str(x).lower() for x in meal["ingredients_raw"])
    assert "camaron" not in raw_blob, (
        f"'50g de camarones cocido' sin uso en display NI pasos = fantasma de "
        f"compra → fuera del raw: {meal['ingredients_raw']}"
    )
    assert "camaron" not in " ".join(str(x).lower() for x in meal["ingredients"]), (
        "y por supuesto tampoco se restaura al display"
    )


def test_unquantified_unused_row_kept():
    plan = _bollitos()
    meal = plan["days"][0]["meals"][0]
    meal["recipe"] = ["Mise en place: hierve la yautía.",
                      "Montaje: rellena con el queso."]  # sin sal en pasos
    go._restore_display_from_raw_orphans(plan["days"])
    assert any("sal al gusto" in str(x).lower() for x in meal["ingredients_raw"]), (
        "'Sal al gusto' (sin cantidad) NO es fantasma — despensa implícita, se queda"
    )


def test_steps_used_row_restored_never_dropped():
    plan = _bollitos()
    meal = plan["days"][0]["meals"][0]
    meal["ingredients_raw"].append("30 g de cilantro fresco")
    meal["recipe"].append("El Toque Final: espolvorea el cilantro picado.")
    go._restore_display_from_raw_orphans(plan["days"])
    assert any("cilantro" in str(x).lower() for x in meal["ingredients_raw"]), (
        "fila usada en pasos JAMÁS se dropea del raw"
    )
    assert any("cilantro" in str(x).lower() for x in meal["ingredients"]), (
        "…y además se restaura al display (rama P1-DISPLAY-RESTORE-FROM-RAW)"
    )
