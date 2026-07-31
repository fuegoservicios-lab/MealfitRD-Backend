"""[P1-CLOSER-NOTE-FUSED-FRESHCOCIDO · 2026-07-31] Pescado FRESCO declarado "(ya viene cocido)".

Encontrado en producción, plan 93d6cd70, comida "Revoltillo de Huevo con Tomate":

    ingrediente: "15 g de filete de pescado blanco"
    paso: "...Escurre e incorpora filete de pescado blanco (ya viene cocido)
           a la preparación antes de servir."

Al usuario se le dice que un filete de pescado CRUDO ya viene cocido y que lo incorpore antes de
servir. Con sardinas en lata la misma frase es correcta — el defecto es aplicarla sin mirar si el
alimento es fresco.

El guard existe (`P1-CLOSER-FRESH-COCIDO`, dentro de `_align_closer_note_food_names`) y está MUERTO
para este caso. Su regex `_CLOSER_NOTE_FOOD_RE` está anclado a `^(Escurre e incorpora|Incorpora|
Cocina|…)`, pero `_integrate_complement_steps` FUSIONA la nota al final del párrafo "El Toque de
Fuego", así que el paso ya no empieza por el verbo, `.match()` falla y la comida entera se salta.

Es la segunda vez que este mismo regex deja huérfano al guard de food-safety: `P2-CLOSER-NOTE-RE-
UNIVERSE` (v5) lo arregló ampliando los VERBOS, sin ver que el ancla `^` es igual de frágil cuando
otro pase le antepone texto. Un ancla al INICIO de la cadena es una suposición sobre lo que hará el
resto del pipeline.

⚠️ SUTILEZA QUE HACE PELIGROSO EL FIX INGENUO: la rama de reparación sustituye el paso ENTERO por la
nota reconstruida. Con la nota fusionada, eso BORRARÍA las instrucciones de cocción del plato. En un
paso fusionado hay que sustituir SOLO el segmento de la nota.

Anchor de producción: P1-CLOSER-NOTE-FUSED-FRESHCOCIDO.
"""
import pytest


TDF = ("El Toque de Fuego: Calienta el aceite en una sartén a fuego medio. Fríe las rodajas de "
       "plátano 3-4 minutos por lado. En otra sartén cocina los huevos batidos hasta obtener un "
       "revuelto cremoso. Mezcla el revuelto con el pepino y el jugo de limón.")

NOTA_FRESCO = "Escurre e incorpora filete de pescado blanco (ya viene cocido) a la preparación antes de servir."
NOTA_LATA = "Escurre e incorpora sardinas en lata (ya viene cocido) a la preparación antes de servir."


def _meal(pasos, ings):
    return {"name": "Revoltillo de Huevo con Tomate", "ingredients": list(ings),
            "recipe": list(pasos)}


# --------------------------------------------------------------- el caso de producción

def test_pescado_fresco_fusionado_deja_de_declararse_cocido():
    """El caso vivo: la nota va fusionada al TdF y el guard nunca la veía."""
    from graph_orchestrator import _align_closer_note_food_names

    m = _meal([f"{TDF} {NOTA_FRESCO}", "Montaje: Sirve caliente."],
              ["1 huevo", "15 g de filete de pescado blanco"])
    _align_closer_note_food_names(m)
    paso = m["recipe"][0]

    assert "(ya viene cocido)" not in paso, (
        f"sigue diciendo que el pescado fresco ya viene cocido: {paso!r}"
    )


def test_el_fix_no_borra_las_instrucciones_de_coccion():
    """La reparación sustituye el paso ENTERO cuando la nota está suelta; fusionada NO puede."""
    from graph_orchestrator import _align_closer_note_food_names

    m = _meal([f"{TDF} {NOTA_FRESCO}", "Montaje: Sirve caliente."],
              ["1 huevo", "15 g de filete de pescado blanco"])
    _align_closer_note_food_names(m)
    paso = m["recipe"][0]

    for pista in ("Calienta el aceite", "Fríe las rodajas", "revuelto cremoso"):
        assert pista in paso, (
            f"la reparación se llevó por delante la cocción del plato ({pista!r} desapareció): {paso!r}"
        )


def test_el_pescado_fresco_acaba_con_instruccion_de_cocinarlo():
    """No basta con quitar la mentira: hay que decirle al usuario que lo cocine."""
    from graph_orchestrator import _align_closer_note_food_names

    m = _meal([f"{TDF} {NOTA_FRESCO}", "Montaje: Sirve caliente."],
              ["1 huevo", "15 g de filete de pescado blanco"])
    _align_closer_note_food_names(m)
    paso = m["recipe"][0]
    # El TdF ya contiene "cocina los huevos", así que buscar el verbo en TODO el paso pasaría por la
    # razón equivocada. Se mira SOLO el segmento nuevo, el que viene después del texto original.
    nuevo = paso[len(TDF):].strip() if paso.startswith(TDF) else paso
    assert "pescado" in nuevo.lower(), f"el segmento de la nota perdió el alimento: {nuevo!r}"
    assert any(v in nuevo.lower() for v in ("cocina", "hervi", "plancha")), (
        f"el paso ya no miente, pero tampoco manda cocer el pescado: {nuevo!r}"
    )


# --------------------------------------------------------------- controles negativos

def test_las_sardinas_en_lata_conservan_su_wording():
    """Control: con enlatado la frase es CORRECTA — el fix no puede volverse un sobre-filtro."""
    from graph_orchestrator import _align_closer_note_food_names

    original = f"{TDF} {NOTA_LATA}"
    m = _meal([original, "Montaje: Sirve caliente."],
              ["3 huevos", "40g de sardinas en lata"])
    _align_closer_note_food_names(m)
    assert m["recipe"][0] == original, (
        f"las sardinas en lata SÍ vienen cocidas; el paso no debía cambiar: {m['recipe'][0]!r}"
    )


def test_la_nota_suelta_sigue_reparandose_como_antes():
    """Regresión: el camino que ya funcionaba (nota como paso propio) no puede romperse."""
    from graph_orchestrator import _align_closer_note_food_names

    m = _meal(["Mise en place: Prepara los ingredientes.",
               f"💪 {NOTA_FRESCO}",
               "Montaje: Sirve caliente."],
              ["1 huevo", "15 g de filete de pescado blanco"])
    _align_closer_note_food_names(m)
    assert "(ya viene cocido)" not in m["recipe"][1], m["recipe"][1]


def test_sin_nota_no_toca_nada():
    from graph_orchestrator import _align_closer_note_food_names

    pasos = [TDF, "Montaje: Sirve caliente."]
    m = _meal(pasos, ["1 huevo", "1 tomate"])
    assert _align_closer_note_food_names(m) == 0
    assert m["recipe"] == pasos


def test_es_idempotente():
    """Correrlo dos veces no puede seguir cambiando el texto."""
    from graph_orchestrator import _align_closer_note_food_names

    m = _meal([f"{TDF} {NOTA_FRESCO}", "Montaje: Sirve caliente."],
              ["1 huevo", "15 g de filete de pescado blanco"])
    _align_closer_note_food_names(m)
    primera = list(m["recipe"])
    _align_closer_note_food_names(m)
    assert m["recipe"] == primera, "el pase no es idempotente"


# --------------------------------------------------------------- anclaje estructural

def test_el_regex_no_exige_estar_al_inicio_del_paso():
    """tooltip-anchor de producción: P1-CLOSER-NOTE-FUSED-FRESHCOCIDO

    El ancla `^` es una suposición sobre lo que hará el resto del pipeline, y el pipeline ya la
    rompió una vez (la fusión al Toque de Fuego). Que el guard encuentre su nota esté donde esté.
    """
    from graph_orchestrator import _CLOSER_NOTE_FOOD_RE

    fusionado = f"{TDF} {NOTA_FRESCO}"
    assert _CLOSER_NOTE_FOOD_RE.search(fusionado), (
        "el regex no encuentra la nota del closer cuando va fusionada al Toque de Fuego"
    )
    m = _CLOSER_NOTE_FOOD_RE.search(fusionado)
    assert m.group("food").strip() == "filete de pescado blanco", m.group("food")
