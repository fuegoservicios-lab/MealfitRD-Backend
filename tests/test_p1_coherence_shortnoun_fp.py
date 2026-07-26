"""[P1-COHERENCE-SHORTNOUN-FP · 2026-07-26] "No pude comprobarlo" no es "está mal".

## El defecto

El detector de coherencia receta↔ingredientes tenía dos líneas que se contradecían:

    for cn in core_nouns:
        if len(cn) < 4:      # ← los núcleos cortos NO se comprueban…
            continue
    ...
    err_noun = next((cn for cn in core_nouns if len(cn) >= 4), core_nouns[0])
                                                              # ← …pero sí sirven de reserva

Un ingrediente cuyo ÚNICO núcleo fuera corto no podía ser verificado jamás y por tanto
**siempre** se reportaba. Y en es-DO esos son justo los que más aparecen:

    "Sal al gusto"      → core_nouns=['sal']   ('gusto' es stopword)   → 3 letras
    "2 dientes de ajo"  → core_nouns=['ajo']   ('dientes' es stopword) → 3 letras

## El daño real (no era el ruido)

Plan `0afa0ed5`, 12 comidas: **9 errores emitidos, 7 sobre `'sal'` y uno sobre `'ajo'`**. Uno
más sobre `'magro'` — que ni es un alimento, es el adjetivo de "res magra".

El auto-patch de aguas abajo consume estos errores y **retira el ingrediente de la lista**. Se
llevó 16, de los cuales diez eran "Sal al gusto" y cuatro el ajo. Pero **no reescribió los
pasos**, así que quedaron dos recetas de doce diciéndole al usuario:

    Día 1, "Res Salteada al Wok":  "Pica el ajo finamente."      ← sin ajo en la lista
    Día 2, "Edamame Salteado":     "sofríe el ajo y la cebolla"  ← sin ajo en la lista

El usuario lee un paso que le pide algo que nunca compró. La maquinaria que existe para
impedir recetas incoherentes las estaba causando.

## El arreglo

Si no hay ningún núcleo verificable, no se afirma la violación. Más un descarte de
sazonadores vía el SSOT `_is_seasoning_name` (P1-SEASONING-WORD-BOUNDARY), que usa límite de
palabra para no tragarse Salmón/Salami/Salsa.

tooltip-anchor: P1-COHERENCE-SHORTNOUN-FP
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g


def _plan(nombre, ingredientes, pasos):
    return {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": nombre, "ingredients": ingredientes, "recipe": pasos}
    ]}]}


def _errores(nombre, ingredientes, pasos):
    r = _plan(nombre, ingredientes, pasos)
    g._run_assembly_validations(r, {}, set())
    return r.get("_recipe_coherence_errors") or []


# ───────────── 1. los falsos positivos medidos ─────────────

def test_sal_al_gusto_ya_no_se_reporta():
    """7 de los 9 errores del plan 0afa0ed5 eran este. "Sal al gusto" no tiene por qué
    aparecer en los pasos."""
    errs = _errores("Pollo Guisado",
                    ["150 g de pollo", "Sal al gusto"],
                    ["Cocina el pollo 20 minutos."])
    assert not any("'sal'" in e for e in errs), errs


def test_dientes_de_ajo_ya_no_se_reporta():
    """'dientes' es stopword, así que el único núcleo era 'ajo' (3 letras) → inverificable."""
    errs = _errores("Pollo Guisado",
                    ["150 g de pollo", "2 dientes de ajo"],
                    ["Cocina el pollo 20 minutos."])
    assert not any("'ajo'" in e for e in errs), errs


# ───────────── 2. lo que SÍ debe seguir detectando ─────────────

def test_un_alimento_real_ausente_sigue_reportandose():
    """El detector no se desactiva: un ingrediente de peso que la receta ignora es un
    defecto real y debe seguir saliendo."""
    errs = _errores("Bowl de pollo",
                    ["150 g de pollo", "200 g de calabacin"],
                    ["Cocina el pollo 20 minutos y sirve."])
    assert any("calabacin" in e.lower() for e in errs), errs


def test_si_la_receta_lo_menciona_no_hay_error():
    errs = _errores("Bowl de pollo",
                    ["150 g de pollo", "200 g de calabacin"],
                    ["Cocina el pollo. Saltea el calabacin 5 minutos."])
    assert errs == []


def test_multi_palabra_basta_con_un_nucleo(  ):
    """Contrato heredado de P6-AUTO-PATCH-1: "lomo de cerdo" está usado si la receta dice
    'cerdo'. Si esto se rompe vuelve la cascada que borraba el ingrediente entero."""
    errs = _errores("Cerdo en salsa",
                    ["200 g de lomo de cerdo"],
                    ["Dora el cerdo por ambos lados."])
    assert errs == []


# ───────────── 3. el caso vivo completo ─────────────

def test_la_receta_del_wok_no_pierde_el_ajo():
    """Reproduce Día 1 de 0afa0ed5: los pasos dicen "Pica el ajo finamente" y el ajo estaba
    listado. Antes el detector lo marcaba inverificable→error, y el auto-patch lo borraba
    dejando el paso huérfano."""
    errs = _errores(
        "Res Salteada al Wok con Pasta Integral y Vegetales",
        ["120 g de res en tiras", "100g de champiñones", "2 dientes de ajo", "Sal al gusto"],
        ["Corta la res en tiras. Pica el ajo finamente.",
         "Saltea la res, el ajo y los champiñones."],
    )
    assert errs == [], f"ningún error debería salir de esta receta: {errs}"


def test_la_enye_y_las_tildes_cuentan_como_mencion():
    """`[a-z]*` excluía la ñ: `\\bchamp[a-z]*\\b` no casaba con "champiñones". Afectaba a todo
    alimento con ñ o tilde pasada la 5ª letra."""
    for ing, paso in (
        ("100 g de champiñones", "Saltea los champiñones 3 minutos."),
        ("200 g de ñame rallado", "Ralla el ñame y reserva."),
        ("1 plátano maduro", "Corta el plátano en trozos."),
    ):
        assert _errores("Plato", [ing], [paso]) == [], ing


@pytest.mark.xfail(
    strict=True,
    reason=(
        "[P1-COHERENCE-SHORTNOUN-FP · 2026-07-26] RESIDUO VIVO, distinto de este fix: un "
        "ADJETIVO se trata como núcleo principal. 'magro' (de \"filete de res magro en tiras\") "
        "salió como 'ingrediente principal' en el plan real 0afa0ed5. 'res' queda fuera por "
        "corto y 'filete' es stopword, así que sobreviven 'magro' y 'tiras' — ninguno aparece "
        "en los pasos. Cerrarlo pide añadir descriptores (magro/magra/entero/fresco/picado/…) "
        "a RECIPE_INGREDIENT_STOPWORDS, que es una lista COMPARTIDA: cambiarla mueve otros "
        "detectores y merece su propia medición. strict=True: cuando alguien lo arregle, este "
        "test se pone verde y la suite falla para obligar a borrar el xfail."
    ),
)
def test_un_adjetivo_no_es_ingrediente_principal():
    errs = _errores("Filetes de Res",
                    ["75 g de filete de res magro en tiras"],
                    ["Marina los filetes de res y cocina en airfryer."])
    assert errs == []


# ───────────── 4. estructura ─────────────

def test_los_inverificables_no_generan_error_nunca():
    """El ancla de la CLASE: si mañana alguien vuelve a usar un núcleo corto como err_noun de
    reserva, este test cae."""
    import inspect
    cuerpo = inspect.getsource(g._run_assembly_validations)
    assert "_verificables" in cuerpo
    assert "if not _verificables:" in cuerpo
    assert "core_nouns[0]," not in cuerpo, \
        "el fallback a un núcleo corto es exactamente el bug: reportaba lo que no podía comprobar"


def test_usa_el_SSOT_de_sazonadores():
    import inspect
    cuerpo = inspect.getsource(g._run_assembly_validations)
    assert "_is_seasoning_name" in cuerpo, \
        "debe reusar el matcher con límite de palabra, no un `in` que traga Salmón/Salsa"
