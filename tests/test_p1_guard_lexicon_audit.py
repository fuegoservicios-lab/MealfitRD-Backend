# -*- coding: utf-8 -*-
"""[P1-GUARD-LEXICON-AUDIT · 2026-09-05] Trabajo 5 del plan corto de la 2.6, el que el roadmap no cubría.

Las tablas de sustitución comparan sus tokens por SUBCADENA. Con tokens largos eso es lo correcto —«jamon» tiene
que pescar «jamón serrano»— pero con tokens cortos muerde comida inocente, y van **trece** veces en este repo:

    «sal»   ⊂ salsa · salmón · salami
    «pollo» ⊂ repollo
    «res»   ⊂ queso fresco · fresas · ciruelas frescas
    «mero»  ⊂ número

Ninguna se detectó leyendo la tabla: todas aparecieron en un plan de un usuario. Este test invierte el orden —
cruza CADA token de CADA tabla contra los 255 constituyentes reales de las seis bibliotecas compiladas, y falla
si un token muerde un alimento que no es de su familia.

No mide texto libre del LLM; mide el catálogo, que es donde el daño se materializa en una lista de compras."""
from __future__ import annotations

import glob
import json
import re as _re
import sys
import unicodedata
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _n(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def _corpus() -> set[str]:
    """Los constituyentes reales de las seis bibliotecas compiladas. Comida que un plan sirve de verdad."""
    fuera = set()
    for f in glob.glob(str(_BACKEND / "data" / "registry" / "dish_registry_*_v1.json")):
        for t in (json.load(open(f, encoding="utf-8")).get("templates") or []):
            for c in (t.get("constituents") or []):
                nombre = (c.get("canonical") or c.get("name")) if isinstance(c, dict) else c
                if nombre:
                    fuera.add(str(nombre))
    return fuera


_CORPUS = _corpus()
pytestmark = pytest.mark.skipif(not _CORPUS, reason="snapshots no compilados")

# Un token puede aparecer dentro de un alimento de SU MISMA familia sin que sea un error: «pollo» dentro de
# «Pechuga de pollo» es exactamente lo que debe pasar. Lo que se persigue es la mordida a un alimento AJENO.
_MISMA_FAMILIA = {
    "pollo": ("pollo", "gallina"), "res": ("res", "carne"), "cerdo": ("cerdo", "chuleta", "lomo"),
    # «Pavochón» es pavo de verdad (el plato puertorriqueño), así que la mordida es correcta: misma familia.
    "pavo": ("pavo", "pavochon"), "atun": ("atun",), "salmon": ("salmon",), "mero": ("mero",),
    "jamon": ("jamon",), "salami": ("salami",), "pescado": ("pescado",), "camaron": ("camaron",),
    "bacalao": ("bacalao",), "sardina": ("sardina",), "anchoa": ("anchoa",), "arenque": ("arenque",),
    "tilapia": ("tilapia",), "chivo": ("chivo",), "conejo": ("conejo",), "higado": ("higado",),
    "bistec": ("bistec",), "costilla": ("costilla",), "pernil": ("pernil",), "tocineta": ("tocineta",),
    "longaniza": ("longaniza",), "salchicha": ("salchicha",), "pepperoni": ("pepperoni",),
    "calamar": ("calamar",), "pulpo": ("pulpo",), "cangrejo": ("cangrejo",), "langosta": ("langosta",),
    "almejas": ("almeja",), "mejillones": ("mejillon",), "gambas": ("gamba",), "lambi": ("lambi",),
    "muslo": ("muslo",), "pechuga": ("pechuga",), "trucha": ("trucha",), "chillo": ("chillo",),
    "vieira": ("vieira",), "percebes": ("percebe",), "boquerones": ("boqueron",),
}


def _palabra_en(termino: str, texto: str) -> bool:
    """`termino` como PALABRA dentro de `texto` (con plural tolerado). Ojo: la pertenencia a la familia no se
    puede comprobar por subcadena, que es el bug que este test persigue — «res» «pertenece» a «fresas» solo si
    se compara mal, y así el detector se exculpaba a sí mismo y devolvía cero mordidas."""
    return bool(_re.search(r"(?<![a-z0-9])" + _re.escape(termino) + r"(?:e?s)?(?![a-z0-9])", texto))


def _mordidas(token: str) -> list[str]:
    """Alimentos del corpus donde `token` casa por SUBCADENA sin ser de su familia."""
    t = _n(token)
    if not t:
        return []
    familia = _MISMA_FAMILIA.get(t, (t,))
    fuera = []
    for alimento in _CORPUS:
        a = _n(alimento)
        if t in a and not any(_palabra_en(f, a) for f in familia):
            fuera.append(alimento)
    return sorted(fuera)


def _culpables() -> dict:
    fuera = {}
    for tokens, _repl, _label, _neg in go._DIET_SUB_TARGETS:
        for tok in tokens:
            m = _mordidas(tok)
            if m:
                fuera[tok] = m[:4]
    return fuera


def test_los_tokens_que_muerden_obligan_a_comparar_por_palabra():
    """La tabla de dieta TIENE tokens que muerden comida ajena en el catálogo real:

        pollo ⊂ Repollo      res ⊂ Fresas

    Eso no es un bug por sí solo — lo es si además se comparan por subcadena, que es como llegó a producción y
    cambió el queso fresco de un vegetariano por soya texturizada. Mientras existan mordidas, `word_match` es
    obligatorio; el día que no quede ninguna, este test lo dirá y se podrá relajar a sabiendas."""
    culpables = _culpables()
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_diet_substitutions")
    palabra_completa = '"word_match": True' in src[i:i + 2500]
    assert not culpables or palabra_completa, (
        f"tokens que muerden comida ajena SIN comparación por palabra completa: {culpables}")
    assert culpables, ("si ya no muerde ninguno, la tabla o el catálogo cambiaron: revisa si `word_match` "
                       "sigue haciendo falta antes de borrarlo")


def test_la_comida_mordida_sobrevive_de_verdad():
    """Lo anterior lee el código; esto ejecuta la sustitución sobre los tres alimentos y comprueba el efecto."""
    for inocente in ("1 taza de Repollo", "200 g de Fresas", "150 g de Queso fresco"):
        plan = {"days": [{"meals": [{"name": "Plato", "ingredients": [inocente],
                                     "ingredients_raw": [inocente]}]}]}
        go._apply_diet_substitutions(plan, {"dietType": "vegetariana"})
        assert plan["days"][0]["meals"][0]["ingredients"] == [inocente], inocente


def test_la_dieta_compara_por_palabra_completa():
    """La defensa de verdad no es una lista de excepciones: es no comparar por subcadena."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_diet_substitutions")
    assert '"word_match": True' in src[i:i + 2500], "sin esto, la tabla vuelve a morder"


def test_ningun_token_corto_nuevo_sin_palabra_completa():
    """Regla del repo desde hoy: un token de 4 caracteres o menos NO se compara por subcadena.

    Se aplica a la tabla de dieta, que es la que los tiene. Las de condición y alérgenos usan tokens largos a
    propósito —ahí sobre-detectar es lo seguro— y por eso conservan la subcadena."""
    cortos = [t for tokens, _r, _l, _n2 in go._DIET_SUB_TARGETS for t in tokens if len(_n(t)) <= 4]
    assert cortos, "si la tabla deja de tener tokens cortos, este test sobra — bórralo con esa razón"
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_diet_substitutions")
    assert '"word_match": True' in src[i:i + 2500], f"tokens de ≤4 caracteres sin palabra completa: {cortos}"


def test_el_corpus_es_de_verdad():
    """Un corpus vacío haría pasar los tres tests de arriba sin comprobar nada."""
    assert len(_CORPUS) > 150, len(_CORPUS)
    # «Queso fresco» no está en los snapshots (sí «Queso blanco»), pero «Fresas» y «Repollo» sí: son los dos
    # alimentos del corpus que reproducen las mordidas históricas de «res» y «pollo».
    assert any(_n(x) == "fresas" for x in _CORPUS), "el alimento que muerde «res» tiene que estar"
    assert any(_n(x) == "repollo" for x in _CORPUS), "el que muerde «pollo», también"


def test_el_detector_encuentra_las_mordidas_conocidas():
    """Un detector que no puede fallar no informa: se le enseñan los tres casos históricos."""
    assert _mordidas("res"), "«res» muerde «Fresas» — si esto sale vacío, el detector está roto"
    assert any("repollo" in _n(x) for x in _mordidas("pollo"))
    assert _mordidas("sal"), "«sal» muerde salsa y salmón"
