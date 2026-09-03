"""[P2-QUALITY-INDEX-COHERENCE · 2026-08-21] El índice de calidad cobraba por ruido que el propio
guard etiqueta como ruido.

`_sub_coherencia` penaliza **15 puntos por cada divergencia de presencia** y 5 por cada una de
magnitud, leyendo `presence_count`/`magnitude_count` de la última entrada de
`_shopping_coherence_block_history`. Lo que NO mira es el campo que va al lado: `hypotheses`, donde
el guard ya clasificó CADA divergencia por su causa.

Y esa clasificación existe justamente para separar señal de artefacto. El comentario de
`_classify_divergence_hypothesis` lo dice con todas las letras al describir el bucket
`recipe_unquantified`: *«separar este bucket es el prerequisito para que los umbrales del cron
midan señal en vez de **ruido conocido**»*. El cron ya lo usa; el índice no.

MEDIDO SOBRE LOS PLANES VIVOS:

    6a4321f5 (ES)  presence=5  magnitude=1  hypotheses={'unknown': 5, 'recipe_unquantified': 1}
    d2f2dbc6 (DO)  presence=0  magnitude=3  hypotheses={'recipe_unquantified': 3}
    f474b4ea (DO)  presence=0  magnitude=3  hypotheses={'yield_uncovered': 1, 'recipe_unquantified': 2}
    e2094da6 (DO)  presence=0  magnitude=1  hypotheses={'recipe_unquantified': 1}

Tres de esos cuatro planes pierden puntos ENTEROS por divergencias que el guard ya atribuyó a la
redacción de la receta o al modelo de yield. No son defectos del plan: son carencias del
instrumento, cobradas al plan.

EL PRINCIPIO, que es lo que decide qué se excluye: **el índice mide EL PLAN**. Una divergencia
atribuida a una causa SISTÉMICA —la receta no cuantificó, el modelo de yield no cubre ese alimento,
la nevera dedujo— no es un defecto de ese plan concreto. Una atribuida al motor —el cap se comió un
modificador, se compró menos de la mitad— sí lo es.

`unknown` SIGUE COBRANDO, y esa decisión es la mitad importante. «No sé por qué diverge» es
exactamente como se ve un defecto real antes de tener nombre; perdonarlo convertiría el índice en
un medidor que sólo ve lo que ya sabemos buscar. Es la lección de `_sub_coherencia` escrita en su
propio docstring: *«un medidor inerte es peor que ninguno: da confianza falsa»*.

Lo que este P-fix NO arregla: los fantasmas de presencia de los planes beta ya persistidos. Esos
venían del espejo del guard y los cerró `P1-COHERENCE-MIRROR-KEEP` para los planes NUEVOS; el
historial viejo se queda como está, porque reescribir telemetría pasada es peor que leerla con
criterio.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pqi():
    import plan_quality_index as _p
    return _p


def _plan(hypotheses=None, presence=0, magnitude=0):
    entrada = {"presence_count": presence, "magnitude_count": magnitude,
               "action_taken": "warn_only_recalc"}
    if hypotheses is not None:
        entrada["hypotheses"] = dict(hypotheses)
    return {"_shopping_coherence_block_history": [entrada]}


def _score(pqi, plan):
    return pqi._sub_coherencia(plan)["score"]


# ── Lo que NO se le cobra al plan ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hipotesis", ["recipe_unquantified", "unit_mismatch",
                                       "yield_uncovered", "pantry_overdeduct"])
def test_no_se_cobra_una_causa_sistemica(pqi, hipotesis):
    """El guard ya atribuyó la divergencia a algo que no es este plan: la receta no cuantificó, el
    modelo de yield no cubre ese alimento, la nevera dedujo. Cobrárselo al plan mide el
    instrumento, no el motor."""
    assert _score(pqi, _plan({hipotesis: 3}, presence=3)) == 100.0


def test_el_caso_medido_de_un_plan_dominicano(pqi):
    """`d2f2dbc6`: magnitude=3 con `{'recipe_unquantified': 3}`. Perdía 15 puntos por la redacción
    de sus propias recetas."""
    assert _score(pqi, _plan({"recipe_unquantified": 3}, magnitude=3)) == 100.0


def test_el_caso_medido_con_dos_causas_sistemicas(pqi):
    """`f474b4ea`: `{'yield_uncovered': 1, 'recipe_unquantified': 2}`."""
    assert _score(pqi, _plan({"yield_uncovered": 1, "recipe_unquantified": 2}, magnitude=3)) == 100.0


# ── Lo que SÍ se le cobra ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hipotesis", ["cap_swallowed_modifier", "magnitude_undersupply"])
def test_se_sigue_cobrando_un_defecto_del_motor(pqi, hipotesis):
    """`cap_swallowed_modifier` es «el pollo estaba en la receta y no en la lista»: el modo de
    fallo que el guard entero existe para cazar. Perdonarlo sería vaciar el medidor."""
    assert _score(pqi, _plan({hipotesis: 2}, presence=2)) < 100.0


def test_unknown_sigue_cobrando(pqi):
    """La mitad importante de la decisión. «No sé por qué diverge» es exactamente como se ve un
    defecto real antes de tener nombre; perdonarlo dejaría al índice viendo sólo lo que ya sabemos
    buscar — y su propio docstring avisa de que un medidor inerte da confianza falsa."""
    assert _score(pqi, _plan({"unknown": 4}, presence=4)) < 100.0


def test_una_mezcla_cobra_solo_la_parte_real(pqi):
    """`6a4321f5` (el plan español): `{'unknown': 5, 'recipe_unquantified': 1}`. Los 5 unknown
    cuentan; el `recipe_unquantified` no."""
    mezcla = _score(pqi, _plan({"unknown": 5, "recipe_unquantified": 1}, presence=5, magnitude=1))
    solo_reales = _score(pqi, _plan({"unknown": 5}, presence=5))
    assert mezcla == solo_reales


# ── Compatibilidad hacia atrás ──────────────────────────────────────────────────────────────────

def test_sin_hipotesis_se_conserva_la_conducta_de_siempre(pqi):
    """Entradas viejas del historial no traen `hypotheses`. Ahí no hay nada que discriminar, así
    que se cobra como antes — byte-idéntico. Inventar una atribución que el guard no hizo sería
    peor que cobrar de más."""
    assert _score(pqi, _plan(None, presence=2, magnitude=1)) == 100.0 - (15 * 2) - (5 * 1)


def test_un_historial_deforme_no_revienta_el_indice(pqi):
    """Corre sobre datos persistidos de meses: la robustez no es opcional."""
    for basura in ({"hypotheses": "no-soy-un-dict", "presence_count": 1},
                   {"hypotheses": {"unknown": "dos"}, "presence_count": 1},
                   {"hypotheses": None, "presence_count": 1}):
        r = pqi._sub_coherencia({"_shopping_coherence_block_history": [basura]})
        assert isinstance(r.get("score"), float)


# ── Lo excluido queda a la vista ────────────────────────────────────────────────────────────────

def test_lo_perdonado_se_registra_no_se_esconde(pqi):
    """Un descuento invisible es el siguiente medidor mentiroso. El detalle tiene que decir cuánto
    se dejó de cobrar y por qué, o dentro de tres meses nadie sabrá si el índice subió porque el
    motor mejoró o porque alguien amplió la lista de perdones."""
    d = pqi._sub_coherencia(_plan({"unknown": 2, "recipe_unquantified": 3}, presence=5))["defectos"]
    assert d.get("coherence_no_atribuible") == 3, (
        f"el descuento no queda registrado en el detalle: {d}"
    )
