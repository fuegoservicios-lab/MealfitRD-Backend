# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-199 · 2026-09-24] Los cerradores eligen de la Nevera cuando el revisor la va a exigir.

Producción, 72 h: 6 de 8 entregas fallaron la revisión, TODAS del único usuario con relleno semanal ligado a su Nevera;
en todos los intentos, «edamame cocido» (35-250 g) que no estaba en su Nevera, metido por el cerrador de proteína, y
semillas de girasol/linaza del cerrador de micronutrientes. Su «Compra Urgente» terminó pidiéndole 250 g de edamame y
una porción de whey. Ver `nevera_exigida.py`.
"""
from __future__ import annotations

import inspect
import re
import sys
import textwrap
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import nevera_exigida as ne  # noqa: E402

# Forma real de `get_user_inventory_net`: «<q> <nombre>» en unidades, «<q> <unidad> de <nombre>» (g, lb/lbs…) y la
# anotación de caducidad entre corchetes. Una línea con «gramos»: la lectura es la del validador, que lo entiende.
_NEVERA = ["20 Huevo", "1960 g de Yogurt", "5 Tilapia", "800 g de Habichuelas blancas",
           "200 gramos de Soya texturizada",
           "0.5 lb de Costilla de cerdo [⚠️ ATENCIÓN: Caduca en 2 días - IA: Prioriza su uso en las recetas de esta semana]",
           "0.25 lb de Queso blanco", "453.6 g de Sal"]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- paridad
class _Log:
    def __getattr__(self, _n):
        return lambda *a, **k: None


def _lo_que_decide_el_revisor(fd: dict):
    """Ejecuta el bloque REAL de `review_plan_node` (desde `is_rotation = …` hasta antes de `if clean_pantry:`)."""
    src = _src("graph_orchestrator.py")
    ini = src.index('        is_rotation = bool(form_data.get("_is_rotation_reroll", False))')
    fin = src.index("            if clean_pantry:", ini)
    ns = {"form_data": fd, "logger": _Log()}
    exec(compile(textwrap.dedent(src[ini:fin]), "<review_plan_node:despensa>", "exec"), ns)
    if not ns["needs_pantry_validation"]:
        return None
    return ns.get("clean_pantry") or None


_CASOS = [
    {},
    {"current_pantry_ingredients": list(_NEVERA)},
    {"current_pantry_ingredients": list(_NEVERA), "_pantry_advisory_only": True},
    {"current_pantry_ingredients": list(_NEVERA), "update_reason": "variety"},
    {"current_pantry_ingredients": list(_NEVERA), "update_reason": "renewal.v1"},
    {"current_pantry_ingredients": [], "current_shopping_list": ["1 lb de Pollo"]},
    {"current_pantry_ingredients": [], "current_shopping_list": ["1 lb de Pollo"], "_is_rotation_reroll": True},
    {"current_shopping_list": ["1 lb de Pollo"]},
    {"_strict_pantry_required": True},
    {"current_pantry_ingredients": ["ab", "   ", None, "Huevo", 7]},
    {"current_pantry_ingredients": "no-es-lista"},
]


@pytest.mark.parametrize("fd", _CASOS, ids=[str(i) for i in range(len(_CASOS))])
def test_la_regla_es_la_del_revisor(fd):
    assert ne.lista(dict(fd)) == _lo_que_decide_el_revisor(dict(fd))


# ---------------------------------------------------------------------------------------------------- la sonda
def test_la_sonda_no_busca_vectores_ni_avisa(monkeypatch, caplog):
    import constants
    llamadas = []
    monkeypatch.setattr(constants, "get_embedding", lambda t: llamadas.append(t) or None)
    with caplog.at_level("WARNING"):
        r = constants.validate_ingredients_against_pantry(["100 g de edamame cocido"], _NEVERA, strict_quantities=False,
                                                          probe_only=True)
    assert r is not True and llamadas == [], llamadas
    assert "RECHAZO" not in caplog.text
    constants.validate_ingredients_against_pantry(["100 g de edamame cocido"], _NEVERA, strict_quantities=False)
    assert llamadas, "sin sonda, el revisor sigue intentando el paso vectorial"


def test_admite_lo_que_hay_y_no_lo_que_falta():
    fd = {"current_pantry_ingredients": list(_NEVERA)}
    assert ne.admite("Huevo entero", fd) and ne.admite("Tilapia", fd) and ne.admite("Soya texturizada", fd)
    assert ne.admite("Costilla de cerdo", fd), "la anotación de caducidad no tapa el alimento"
    assert not ne.admite("Edamame cocido", fd) and not ne.admite("Proteína whey", fd)
    assert not ne.admite_linea("10 g de semillas de girasol", fd)
    assert not ne.admite("Queso cottage", fd), "tener queso blanco no es tener cottage"
    assert ne.admite("Edamame cocido", {}), "sin Nevera exigida no se filtra nada"


def test_la_subcadena_del_revisor_no_se_hereda():
    """El revisor casa por subcadena («sal» ⊂ «salmón», «pollo» ⊂ «repollo»). El filtro exige el token completo."""
    fd = {"current_pantry_ingredients": ["453.6 g de Sal", "1 Pollo"]}
    assert not ne.admite("Salmón", fd)
    assert ne.admite("Pechuga de pollo", fd), "«pollo» en la Nevera cubre la pechuga (tokens contenidos)"
    assert not ne.admite("Repollo", fd)


# ---------------------------------------------------------------------------------------------------- el cerrador
class _Info:
    def __init__(self, name):
        self.name = name
        self.protein = 20.0
        self.kcal = 100.0


class _DB:
    def lookup(self, name):
        return _Info(name)


def _pool_con(fd):
    import graph_orchestrator as go
    tok = ne.fijar(fd)
    try:
        return [n for _, n, _ in go._safe_high_density_proteins(["Ninguna"], _DB(), min_protein=10.0, country="DO")]
    finally:
        ne.soltar(tok)


def test_el_cerrador_de_proteina_elige_de_la_nevera():
    sin = _pool_con({})
    con = _pool_con({"current_pantry_ingredients": list(_NEVERA)})
    assert "Edamame" in sin, "el pool del país trae edamame (el caso real)"
    assert "Edamame" not in con and con, con
    assert {"Tilapia", "Soya texturizada"} <= set(con), con
    assert all(ne.admite(n, {"current_pantry_ingredients": list(_NEVERA)}) for n in con)


def test_sin_proteina_en_la_nevera_conserva_el_catalogo():
    assert _pool_con({"current_pantry_ingredients": ["453.6 g de Sal", "1 paquete de Ajo"]}) == _pool_con({})


def test_advisory_y_knob_apagado_no_filtran(monkeypatch):
    assert _pool_con({"current_pantry_ingredients": list(_NEVERA), "_pantry_advisory_only": True}) == _pool_con({})
    monkeypatch.setenv("MEALFIT_CLOSERS_RESPECT_PANTRY", "false")
    assert _pool_con({"current_pantry_ingredients": list(_NEVERA)}) == _pool_con({})


def test_las_rotaciones_prefieren_la_nevera():
    import graph_orchestrator as go
    rot = go._NIGHT_RICE_SUB_ROTATION
    assert [ne.preferir(rot, i) for i in range(7)] == [rot[i % len(rot)] for i in range(7)], "sin Nevera: idéntica"
    tok = ne.fijar({"current_pantry_ingredients": ["35 g de Casabe", "1 Plátano", "20 Huevo"]})
    try:
        assert {ne.preferir(rot, i) for i in range(7)} == {"Casabe"}
        assert ne.preferir(go._BREAKFAST_RICE_SUB_ROTATION, 1) == "Puré de plátano"
        assert ne.preferir(("Ñame", "Yautía"), 1) == "Yautía", "si la Nevera no tiene ninguna, la rotación de siempre"
    finally:
        ne.soltar(tok)


# ---------------------------------------------------------------------------------------------------- cableado
def test_el_pipeline_fija_y_suelta_la_nevera():
    import graph_orchestrator as go
    arun = inspect.getsource(go.arun_plan_pipeline)
    assert '__import__("nevera_exigida").fijar(actual_form_data)' in arun
    fin = arun[arun.rfind("finally:"):]
    assert '__import__("nevera_exigida").soltar(' in fin


def test_semillas_y_candidatos_consultan_la_nevera():
    src = _src("graph_orchestrator.py")
    assert 'return __import__("nevera_exigida").filtrar_proteinas(out)' in src
    assert re.search(r'_seed_dislikes\).*__import__\("nevera_exigida"\)\.admite_linea\(_cand\)', src)
    assert src.count('__import__("nevera_exigida").preferir(') == 2, "arroz de noche y de desayuno"
    assert 'return __import__("nevera_exigida").admite(cand)' in src, "fruta dulce+salado → aguacate/batata"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 199 and m.group(2) >= "2026-09-24"
