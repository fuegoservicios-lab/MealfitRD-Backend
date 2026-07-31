"""[P3-REFILL-STEP-VERB-MATCHES-FOOD · 2026-07-31] "🍚 Cuece el Casabe según el paquete".

Caso real, plan af58ec2e, "Tortitas de Harina de Negrito":

    ING:  "½ torta pequeña de casabe"
    PASO: "🍚 Cuece el Casabe de tus ingredientes según el paquete y sírvelo como acompañante."

El casabe es una torta de yuca YA HORNEADA, lista para comer: no se cuece, y desde luego no "según
el paquete". El productor de ese paso (`_gainmuscle_kcal_floor`) escribe **"arroz blanco" fijo** —lo
verifiqué, es un literal—, así que un pase posterior renombró el alimento y dejó el verbo del arroz.

No conseguí identificar QUÉ pase lo renombra, y eso mismo decide la forma del fix: en vez de perseguir
al productor, se cierra en el consumidor, que es donde el defecto es observable. El repo ya tiene el
precedente exacto — `P2-SPECIES-VERB-CLEANUP` nació porque una sustitución camarón→pescado renombró el
alimento y dejó "desvena" y "hasta que estén rosados".

⚠️ El vocabulario NO se escribe a mano: se reusa `_COOKED_GRAIN_REF_KCAL`, que ya es el universo
canónico de lo que se cuece desde seco (arroz, quinoa, pasta, habichuelas, lentejas…). Una lista
nueva por incidente garantiza el próximo incidente.

Y el paso NO se borra: en la comida real el montaje tampoco menciona el casabe, así que borrarlo
dejaría el alimento sin ninguna instrucción. Se corrige el verbo.

Anchor de producción: P3-REFILL-STEP-VERB-MATCHES-FOOD.
"""
import pytest


PASO_ARROZ = "🍚 Cuece el arroz blanco de tus ingredientes según el paquete y sírvelo como acompañante."
PASO_CASABE = "🍚 Cuece el Casabe de tus ingredientes según el paquete y sírvelo como acompañante."


def _meal(paso, ings):
    return {"name": "Tortitas de Harina de Negrito", "ingredients": list(ings),
            "recipe": ["Mise en place: Ralla el calabacín.", paso, "Montaje: Sirve las tortitas."]}


# --------------------------------------------------------------- el caso real

def test_el_casabe_deja_de_cocerse():
    from graph_orchestrator import _fix_refill_step_verb

    m = _meal(PASO_CASABE, ["1 huevo", "½ torta pequeña de casabe"])
    assert _fix_refill_step_verb(m) is True
    paso = m["recipe"][1]
    assert "cuece" not in paso.lower(), f"el casabe sigue cociéndose: {paso!r}"
    assert "según el paquete" not in paso.lower(), f"sigue citando un paquete: {paso!r}"


def test_el_casabe_no_desaparece_del_paso():
    """Borrar el paso dejaría el alimento sin instrucción: el montaje real tampoco lo menciona."""
    from graph_orchestrator import _fix_refill_step_verb

    m = _meal(PASO_CASABE, ["1 huevo", "½ torta pequeña de casabe"])
    _fix_refill_step_verb(m)
    assert "casabe" in m["recipe"][1].lower(), f"el alimento se perdió: {m['recipe'][1]!r}"
    assert len(m["recipe"]) == 3, "no se borra ningún paso"


# --------------------------------------------------------------- controles negativos

def test_el_arroz_conserva_su_verbo():
    """Control: el arroz SÍ se cuece según el paquete — el fix no puede volverse un sobre-filtro."""
    from graph_orchestrator import _fix_refill_step_verb

    m = _meal(PASO_ARROZ, ["1 huevo", "40 g de arroz blanco crudo"])
    assert _fix_refill_step_verb(m) is False
    assert m["recipe"][1] == PASO_ARROZ


@pytest.mark.parametrize("grano", ["arroz integral", "quinoa", "pasta", "lentejas", "garbanzos"])
def test_los_granos_del_universo_canonico_se_respetan(grano):
    """El vocabulario sale de `_COOKED_GRAIN_REF_KCAL`, no de una lista escrita a mano aquí."""
    from graph_orchestrator import _fix_refill_step_verb

    paso = f"🍚 Cuece el {grano} de tus ingredientes según el paquete y sírvelo como acompañante."
    m = _meal(paso, ["1 huevo", f"40 g de {grano}"])
    assert _fix_refill_step_verb(m) is False, f"{grano} sí se cuece desde seco: {m['recipe'][1]!r}"


def test_no_toca_pasos_que_no_son_del_refill():
    from graph_orchestrator import _fix_refill_step_verb

    m = _meal("El Toque de Fuego: Cuece el arroz 15 minutos.", ["1 huevo"])
    antes = list(m["recipe"])
    assert _fix_refill_step_verb(m) is False
    assert m["recipe"] == antes


def test_es_idempotente():
    from graph_orchestrator import _fix_refill_step_verb

    m = _meal(PASO_CASABE, ["1 huevo", "½ torta pequeña de casabe"])
    _fix_refill_step_verb(m)
    primera = list(m["recipe"])
    assert _fix_refill_step_verb(m) is False
    assert m["recipe"] == primera


def test_tolera_basura():
    from graph_orchestrator import _fix_refill_step_verb

    assert _fix_refill_step_verb(None) is False
    assert _fix_refill_step_verb({}) is False
    assert _fix_refill_step_verb({"recipe": None}) is False
