# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-33 · 2026-09-13] F8: el paso del gate con el perfil de knobs de PRODUCCIÓN.

La suite corre con cinco gates apagados por `conftest` y otros knobs en su default de código: mide un producto distinto
del que se entrega. Medido: bajo `prod_profile.perfil_completo()` exportado, la suite da fallos concentrados en harnesses
que construyen planes con alimentos sintéticos (el filtro de verificados los descarta). No se «arreglan» a ciegas:
`scripts/prod_profile_gate.py` corre la suite CON el perfil menos `tests/prod_profile_excluded.txt` (cada línea con el knob
que lo explica y por qué) menos la cuarentena, y después la batería con el entorno normal. La CI lo corre en una pata
paralela (`matrix.perfil`).

Lo que este test fija: la lista es exactamente la de ficheros que fallaron bajo el perfil en el artefacto medido (ni uno
de más: excluir sin medir es esconder), cada línea lleva motivo, nada se excluye dos veces (lista + cuarentena), el entorno
del paso ES el perfil, y la CI corre el paso.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("prod_profile_gate", _BACKEND / "scripts" / "prod_profile_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def artefacto():
    fs = sorted((_BACKEND / "scripts" / "data").glob("f8_prod_profile_*.json"))
    assert fs, "falta el artefacto de la medición (scripts/data/f8_prod_profile_<fecha>.json)"
    return json.loads(fs[-1].read_text(encoding="utf-8"))


def test_cada_excluido_existe_y_dice_por_que(runner):
    exc = runner.excluidos()
    assert exc, "sin lista de exclusión el paso mediría también los harnesses sintéticos"
    for f, motivo in exc.items():
        assert f.startswith("tests/test_") and f.endswith(".py"), f
        assert (_BACKEND / f).exists(), f"{f} ya no existe: sácalo de la lista"
        assert len(motivo) >= 25, f"{f}: el motivo es la razón por la que no se mide bajo producción"


def test_se_excluye_exactamente_lo_que_fallo_bajo_el_perfil(runner, artefacto):
    con_fallos = {f for f, d in artefacto["ficheros"].items() if d["fallos"] > 0}
    exc = set(runner.excluidos())
    assert exc - con_fallos == set(), f"excluidos sin fallo medido (esconder sin medir): {sorted(exc - con_fallos)}"
    assert con_fallos - exc == set(), f"fallaron bajo el perfil y no están excluidos: {sorted(con_fallos - exc)}"


def test_cada_fallo_medido_tiene_su_atribucion(artefacto):
    for f, d in artefacto["ficheros"].items():
        if d["fallos"] > 0:
            a = d.get("atribucion") or {}
            assert a.get("veredicto") in {"knob", "combinacion", "no_reproduce_aislado", "falla_sin_perfil"}, f
            assert a.get("veredicto") != "falla_sin_perfil", f"{f} falla también sin el perfil: es un rojo de la suite"


def test_el_artefacto_declara_metodo_denominador_y_perfil(artefacto):
    import prod_profile
    assert artefacto["corrida"]["tests"] > 20_000, "la medición es sobre la suite entera, no una muestra"
    assert artefacto["corrida"]["fallos"] == sum(d["fallos"] for d in artefacto["ficheros"].values())
    assert "atribución" in artefacto["metodo"]
    assert artefacto["perfil"]["leido"] == prod_profile.PROFILE_READ_AT
    assert artefacto["perfil"]["knobs"] == prod_profile.perfil_completo(), "el perfil cambió: re-medir"


def test_ningun_excluido_esta_tambien_en_cuarentena(runner):
    assert not set(runner.excluidos()) & set(runner.cuarentena())


def test_el_entorno_del_paso_es_el_perfil_de_produccion(runner):
    import prod_profile
    env = runner.entorno_perfil({"PATH": "x", "MEALFIT_COUNTRY_SYSTEM": "False"})
    for k, v in prod_profile.perfil_completo().items():
        assert env[k] == v, k
    assert env["PATH"] == "x"
    sin = runner.entorno_perfil({}, sin=("MEALFIT_VERIFIED_INGREDIENTS_ONLY",))
    assert "MEALFIT_VERIFIED_INGREDIENTS_ONLY" not in sin, "quitar un knob = volver al valor de la suite"


def test_el_comando_ignora_cuarentena_y_excluidos_y_no_corre_e2e(runner):
    cmd = runner.comando_pytest(["tests/test_x.py"], 3)
    assert "not e2e" in cmd and cmd[cmd.index("--ignore") + 1] == "tests/test_x.py"
    assert "-n" in cmd and "loadfile" in cmd


def test_la_bateria_corre_aparte_con_el_entorno_normal(runner):
    """La batería aplica el perfil por dentro y afirma que la suite diverge: bajo el perfil exportado fallaría por diseño."""
    import inspect
    src = inspect.getsource(runner.paso)
    assert "BATERIA" in src and "env=entorno_perfil()" in src
    i = src.index("BATERIA")
    assert "env=" not in src[i:i + 200], "la batería no debe heredar el perfil exportado"
    assert runner.BATERIA in runner.excluidos(), "y por eso tampoco entra al subconjunto"


def test_la_ci_corre_el_paso_de_produccion():
    ci = (_BACKEND / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"perfil:\s*\[suite,\s*produccion\]", ci)
    assert "python scripts/prod_profile_gate.py" in ci
    assert "P1-PLAN-LOTE-33" in ci


def test_docs_plan_marker():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-33" in doc and "prod_profile_gate.py" in doc and "prod_profile_excluded.txt" in doc
    assert "P1-PLAN-LOTE-33" in (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 33


# ─────────────── (bis 4) la lista que SÓLO aplica sin base de datos ───────────────
# La pata de producción de la CI dio 76 fallos que el gate local nunca vio: el gate local tiene base (el `.env` del
# dueño) y la CI no. Bajo el perfil, esos ficheros leen algo de la base; sin ella fallan por construcción. Se miden en
# un árbol sin `.env` (su propio artefacto) y salen SÓLO cuando no hay base — con base corren y pasan.


@pytest.fixture(scope="module")
def artefacto_sin_base():
    fs = sorted((_BACKEND / "scripts" / "data").glob("f8_sin_base_*.json"))
    assert fs, "falta el artefacto de la medición sin base (scripts/data/f8_sin_base_<fecha>.json)"
    return json.loads(fs[-1].read_text(encoding="utf-8"))


def test_la_medicion_sin_base_se_hizo_sin_base_y_con_el_perfil_vigente(artefacto_sin_base):
    import prod_profile
    assert artefacto_sin_base["con_base"] is False, "medida con base no dice nada de la CI"
    assert artefacto_sin_base["corrida"]["tests"] > 20_000, "la medición es sobre la suite entera, no una muestra"
    assert artefacto_sin_base["perfil"]["knobs"] == prod_profile.perfil_completo(), "el perfil cambió: re-medir"


def test_sin_base_sale_exactamente_lo_que_fallo_sin_base_y_no_estaba_ya_fuera(runner, artefacto_sin_base):
    con_fallos = {f for f, d in artefacto_sin_base["ficheros"].items() if d["fallos"] > 0}
    esperado = con_fallos - set(runner.excluidos()) - set(runner.cuarentena())
    sb = set(runner.excluidos_sin_base())
    assert sb - esperado == set(), f"en la lista sin base sin fallo medido sin base: {sorted(sb - esperado)}"
    assert esperado - sb == set(), f"fallaron sin base y no salen por ninguna lista: {sorted(esperado - sb)}"


def test_cada_fallo_sin_base_tiene_su_atribucion_y_ninguno_es_un_rojo_de_la_suite(runner, artefacto_sin_base):
    for f, d in artefacto_sin_base["ficheros"].items():
        if d["fallos"] > 0 and f in runner.excluidos_sin_base():
            a = d.get("atribucion") or {}
            assert a.get("veredicto") in {"knob", "combinacion", "no_reproduce_aislado"}, (f, a)


def test_cada_linea_sin_base_existe_dice_por_que_y_no_repite(runner):
    sb = runner.excluidos_sin_base()
    assert sb, "sin la lista, la pata de producción de la CI da rojos que el gate local no puede ver"
    ya = set(runner.excluidos()) | set(runner.cuarentena())
    for f, motivo in sb.items():
        assert f.startswith("tests/test_") and f.endswith(".py"), f
        assert (_BACKEND / f).exists(), f"{f} ya no existe: sácalo de la lista"
        assert len(motivo) >= 25, f"{f}: el motivo es la razón por la que sin base no se mide"
        assert f not in ya, f"{f} ya sale por la lista general o la cuarentena"


def test_la_lista_sin_base_solo_sale_cuando_no_hay_base(runner):
    sb = set(runner.excluidos_sin_base())
    assert not sb & set(runner.ignorados(con_base=True)), "con base esos ficheros corren bajo el perfil (y pasan)"
    assert sb <= set(runner.ignorados(con_base=False))
    assert set(runner.ignorados(sin_exclusiones=True, con_base=False)) == set(runner.cuarentena()),         "la medición no excluye nada"


def test_la_senal_de_base_es_la_misma_que_la_de_conftest(runner):
    """Dos señales distintas para «hay base» harían que el gate excluyera lo que conftest corre, o al revés."""
    import inspect
    conftest = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")
    cuerpo = conftest[conftest.index("def _db_available"):][:700]
    src = inspect.getsource(runner.hay_base)
    fichero = '/ "' + "." + 'env").exists()'
    for senal in (fichero, 'os.environ.get("MEALFIT_TESTS_HAVE_DB") == "1"'):
        assert senal in cuerpo and senal in src, senal


def test_la_ci_nombra_los_tests_lentos():
    ci = (_BACKEND / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'PYTEST_ADDOPTS: "--durations=30"' in ci

