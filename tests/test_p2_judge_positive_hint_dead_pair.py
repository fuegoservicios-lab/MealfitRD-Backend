"""[P2-JUDGE-POSITIVE-HINT-DEAD-PAIR · 2026-08-23] El par de reemplazo de la guía positiva del juez
culinario estaba MUERTO, y los otros tres slots ni siquiera tenían par.

Medido antes del arreglo:

    'El desayuno dominicano va: mangú/tubérculos' in SLOT_POSITIVE_HINT['desayuno']  →  False
    El literal SSOT dice 'El desayuno dominicano va: mangú/víveres, avena/cereales calientes, …'

    Render de _culinary_judge_rubric_for_country('ES'):
      - Desayuno: El desayuno de España va: mangú/tubérculos, avena/…
      - Almuerzo: El almuerzo es el plato fuerte: arroz+habichuela+proteína+ensalada, locrio, moro,
                  asopao, pasta criolla, …
      - Cena:     …tortilla/revoltillo de cena… (batata/yuca/casabe)
      - Merienda: …casabe/galleta integral con queso…

Su autor escribió el par contra el texto YA neutralizado, pero el bucle de pares corre ANTES de
`neutralize_do_lexicon`: disparaba el par genérico ('desayuno dominicano' → 'desayuno de España'),
el neutralizador convertía víveres→tubérculos, y el juez español acababa leyendo «El desayuno de
España va: mangú/tubérculos». Un par muerto no se nota: el render sale distinto igualmente.

El arreglo deja de parchear con `.replace()` y RECONSTRUYE el bloque «GUÍA POSITIVA POR HORARIO»
desde `constants._SLOT_POSITIVE_HINT_NEUTRAL`, el espejo neutro que ya existía desde F1-Task-4 y
que nadie estaba usando aquí. Los dos lados salen del MISMO constructor
(`_culinary_judge_hints_block`), así que no pueden volver a desincronizarse.

Matiz que este test NO exagera: el guard culinario de producción está en `warn`, no en `block`, así
que esto no forzaba retries pagados — degradaba el juicio y el texto que el usuario lee.
"""
from __future__ import annotations

import pytest

import graph_orchestrator as go
from constants import SLOT_POSITIVE_HINT, _SLOT_POSITIVE_HINT_NEUTRAL

_BETA = ["ES", "MX", "US", "PR", "CO"]
# Mandatos de plato dominicano que la guía positiva NO debe darle a un juez beta. Se comparan
# contra el BLOQUE de guía positiva, no contra la rúbrica entera (los ejemplos curados del país y
# la tolerancia a la creatividad viven en otras secciones).
_MANDATOS_DO = ("mangú", "locrio", "moro", "asopao", "casabe", "criolla", "habichuela", "víveres")


@pytest.fixture(autouse=True)
def _cache_limpia():
    """La caché por país es de módulo: sin vaciarla, el primer país fija el resultado del resto."""
    go._CULINARY_JUDGE_RUBRIC_CACHE.clear()
    go._CULINARY_JUDGE_RUBRIC_CACHE["DO"] = go._CULINARY_JUDGE_RUBRIC
    yield
    go._CULINARY_JUDGE_RUBRIC_CACHE.clear()
    go._CULINARY_JUDGE_RUBRIC_CACHE["DO"] = go._CULINARY_JUDGE_RUBRIC


def _guia_positiva(rubrica: str) -> str:
    i = rubrica.find("GUÍA POSITIVA POR HORARIO")
    assert i >= 0, "la rúbrica perdió su bloque de guía positiva — mueve el guard con él"
    j = rubrica.find("ACLARACIÓN IMPORTANTE", i)
    assert j > i
    return rubrica[i:j]


def test_do_devuelve_la_rubrica_intacta():
    """Byte-identidad DO: es la invariante que sostiene todo el diseño F1."""
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC
    bloque = _guia_positiva(go._CULINARY_JUDGE_RUBRIC)
    for hint in SLOT_POSITIVE_HINT.values():
        assert hint in bloque, "la guía positiva de DO dejó de ser el literal es-DO de constants"


def test_la_rubrica_do_se_construye_con_el_mismo_constructor():
    """Si el constructor deja de producir el bloque de DO, la sustitución beta sería un no-op."""
    assert go._culinary_judge_hints_block(SLOT_POSITIVE_HINT) in go._CULINARY_JUDGE_RUBRIC


@pytest.mark.parametrize("cc", _BETA)
def test_la_guia_positiva_beta_es_el_espejo_neutro(cc):
    bloque = _guia_positiva(go._culinary_judge_rubric_for_country(cc))
    for slot, hint in _SLOT_POSITIVE_HINT_NEUTRAL.items():
        assert hint in bloque, (
            f"[{cc}] el slot '{slot}' de la guía positiva no es el espejo neutro; el bloque dice:\n"
            f"{bloque}"
        )


@pytest.mark.parametrize("cc", _BETA)
def test_la_guia_positiva_beta_no_ordena_platos_dominicanos(cc):
    bloque = _guia_positiva(go._culinary_judge_rubric_for_country(cc)).lower()
    presentes = [t for t in _MANDATOS_DO if t in bloque]
    assert not presentes, (
        f"[{cc}] la guía positiva le sigue diciendo al juez que espere {presentes} — es justo el "
        "par muerto que este P-fix cierra (medido: «El desayuno de España va: mangú/tubérculos»)."
    )


@pytest.mark.parametrize("cc", _BETA)
def test_los_cuatro_slots_siguen_presentes(cc):
    """PERDER LA GUÍA ENTERA sería peor que tenerla en dominicano: el juez se queda sin criterio."""
    bloque = _guia_positiva(go._culinary_judge_rubric_for_country(cc))
    for etiqueta in ("- Desayuno:", "- Almuerzo:", "- Cena:", "- Merienda:"):
        assert etiqueta in bloque, f"[{cc}] falta {etiqueta} en la guía positiva"


def test_no_queda_ningun_par_anclado_a_la_guia_positiva():
    """El defecto de fondo: un par escrito contra el texto YA neutralizado nunca dispara.

    Cada `_frase_do` del bucle de pares debe existir en la rúbrica CRUDA (que es sobre la que el
    bucle corre). Si no existe, el par está muerto y su slot queda sin traducir en silencio.
    """
    import ast
    import inspect

    src = inspect.getsource(go._culinary_judge_rubric_for_country)
    árbol = ast.parse(src.lstrip())
    pares = []
    for node in ast.walk(árbol):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            for elt in node.iter.elts:
                if isinstance(elt, ast.Tuple) and elt.elts:
                    izq = elt.elts[0]
                    if isinstance(izq, ast.Constant) and isinstance(izq.value, str):
                        pares.append(izq.value)
    assert pares, "no se encontró el bucle de pares — mueve el guard con él"
    muertos = [p for p in pares if p not in go._CULINARY_JUDGE_RUBRIC]
    assert not muertos, (
        f"pares de reemplazo que NO existen en la rúbrica cruda (nunca disparan): {muertos}. "
        "Escribirlos contra el texto ya neutralizado es exactamente el defecto de este P-fix."
    )
