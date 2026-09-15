# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-52 · 2026-09-15] Dos defectos de medida, ninguno del plato.

  1. El punto ENTRE DOS CIFRAS partía la oración. `culinary_coherence.clause_bounds` cortaba en cada «.», así que V4 y el
     reparador leían «0.23 g de Sal» como 23 g y «57.2 g» como 2 g. `dish_structure` y la plantilla «a la plancha» del
     cerrador usaban la misma frontera. Medido sobre los 11 planes de los últimos 21 días: inerte hoy (0 cambios en el
     scan y en el reparador), no mañana: `formatear_cantidad` escribe los decimales con punto.
  2. Veinte ficheros de test legacy suplantan al importarse módulos del backend con módulos VACÍOS. En una batería
     dirigida (`test_p0_b_synthesis_per_user_circuit_breaker` → `test_p1_plan_display_i18n` → `test_p1_plan_lote_15`),
     los dos últimos daban 15 fallos y 2 errores. El conftest ya pre-importaba `graph_orchestrator`; ahora pre-importa
     todo módulo del backend que algún test suplante, y este fichero vigila que la lista no se quede corta.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_TESTS = _BACKEND / "tests"

import culinary_coherence as cc  # noqa: E402
import dish_structure as ds  # noqa: E402
import recipe_contract as rc  # noqa: E402

_CAT = [{"name": n, "aliases": a, "category": c, "prep_methods": ["ninguno"]} for n, a, c in (
    ("Sal", [], "Condimentos"), ("Soya", ["soja"], "Legumbres"),
    ("Mantequilla de maní", ["mantequilla de mani"], "Frutos secos"), ("Repollo", [], "Vegetales"))]
_IDX = cc.build_culinary_index(_CAT)
_FRONTERA_DE_ANTES = re.compile(r"[.;]")


def _gramos(lineas) -> list:
    return sorted(v for k, v in rc._cantidades_lista(lineas, _IDX).items() if k[1] == "g")


# ─────────────── 1. el punto entre cifras es un decimal ───────────────
def test_el_punto_entre_cifras_no_parte_la_oracion():
    t = "anade 0.23 g de sal. mezcla 57.2 g de soya; sirve"
    assert [t[a:b].strip() for a, b in cc.clause_bounds(t)] == ["anade 0.23 g de sal", "mezcla 57.2 g de soya", "sirve"]


def test_el_punto_con_una_sola_cifra_al_lado_sigue_partiendo():
    t = "hierve 5 min. anade 2. luego sirve.fin"
    assert [t[a:b].strip() for a, b in cc.clause_bounds(t)] == ["hierve 5 min", "anade 2", "luego sirve", "fin"]


def test_v4_y_el_reparador_leen_el_decimal_entero():
    assert _gramos(["0.23 g de Sal", "57.2 g de Soya", "2.69 g de Mantequilla de maní"]) == pytest.approx([0.23, 2.69, 57.2])
    assert _gramos(["57,2 g de Soya"]) == pytest.approx([57.2]), "la coma ya se leía bien"


def test_con_la_frontera_de_antes_se_leia_la_fraccion(monkeypatch):
    """El defecto, fijado: y la prueba de que el instrumento de la medición (el mismo cambio de frontera) ve la diferencia."""
    monkeypatch.setattr(cc, "_SENTENCE_BOUNDARY_RE", _FRONTERA_DE_ANTES)
    assert _gramos(["0.23 g de Sal"]) == [23.0]
    assert _gramos(["57.2 g de Soya"]) == [2.0]


def test_dish_structure_usa_la_misma_frontera(monkeypatch):
    """«Sirve con 1.5 tazas de repollo rallado»: el repollo va AL LADO. Partiendo en el decimal, la cláusula «5 tazas de
    repollo rallado» perdía el «sirve con» y el repollo contaba como de dentro del plato."""
    pasos = "sirve con 1.5 tazas de repollo rallado"
    assert ds._veg_va_dentro("repollo", pasos) is False
    monkeypatch.setattr(cc, "_SENTENCE_BOUNDARY_RE", _FRONTERA_DE_ANTES)
    assert ds._veg_va_dentro("repollo", pasos) is True, "con la frontera de antes el decimal lo metía dentro"


def test_la_plantilla_de_la_plancha_no_se_corta_en_el_decimal():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    patron = r"\s+a\s+la\s+plancha(?:[^.!?]|(?<=\d)\.(?=\d))*[.!?]?"
    assert patron in src and r"\s+a\s+la\s+plancha[^.!?]*[.!?]?" not in src
    frase = "Cocina el pollo a la plancha 2.5 min por lado. Sirve."
    m = re.search(r"\bcocina\s+el pollo" + patron, frase, re.IGNORECASE)
    assert m and m.group(0).endswith("por lado."), "se comía sólo hasta el «2» y dejaba «.5 min por lado.» colgando"


def test_ningun_modulo_de_produccion_parte_oraciones_en_cualquier_punto():
    """La familia: `r"[.;]"` pelado corta decimales. Las formas con `(?<=[.;])\\s+` exigen un espacio y no los cortan."""
    culpables = [p.name for p in _BACKEND.glob("*.py") if 'r"[.;]"' in p.read_text(encoding="utf-8")]
    assert not culpables, culpables


# ─────────────── 2. ningún test deja un módulo vacío del backend ───────────────
def _modulos_del_backend() -> set:
    nombres = {p.stem for p in _BACKEND.glob("*.py")}
    return nombres | {f"routers.{p.stem}" for p in (_BACKEND / "routers").glob("*.py")}


def _es_sys_modules(nodo) -> bool:
    return (isinstance(nodo, ast.Attribute) and nodo.attr == "modules"
            and isinstance(nodo.value, ast.Name) and nodo.value.id == "sys")


def _nodos_al_importar(nodo):
    """Todo lo que corre al IMPORTAR el fichero: no entra en funciones, clases ni lambdas."""
    for hijo in ast.iter_child_nodes(nodo):
        if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield hijo
        yield from _nodos_al_importar(hijo)


def _suplantados_al_importar(tree) -> set:
    """Nombres que el fichero mete en `sys.modules` al importarse: `sys.modules["x"] = …`, `sys.modules.setdefault("x",
    …)` o un ayudante suyo que lo haga (`_install_stub("x", …)`)."""
    ayudantes = {f.name for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
        isinstance(a, ast.Assign) and any(isinstance(t, ast.Subscript) and _es_sys_modules(t.value) for t in a.targets)
        for a in ast.walk(f))}
    out = set()
    for n in _nodos_al_importar(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and _es_sys_modules(t.value) and isinstance(t.slice, ast.Constant):
                    out.add(t.slice.value)
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == "setdefault" and _es_sys_modules(f.value)) or (
                    isinstance(f, ast.Name) and f.id in ayudantes):
                out.add(n.args[0].value)
    return {x for x in out if isinstance(x, str)}


def _eager_del_conftest() -> tuple:
    tree = ast.parse((_TESTS / "conftest.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_EAGER_BACKEND_MODULES" for t in n.targets):
            return tuple(e.value for e in n.value.elts)
    return ()


def test_el_censo_ve_a_los_suplantadores_conocidos():
    """Una sonda que no ve lo que sí está es una coartada: el fichero que daba los 15 fallos tiene que salir."""
    tree = ast.parse((_TESTS / "test_p0_b_synthesis_per_user_circuit_breaker.py").read_text(encoding="utf-8"))
    assert {"agent", "memory_manager", "db", "db_core"} <= _suplantados_al_importar(tree)


def test_el_conftest_preimporta_todo_modulo_del_backend_que_un_test_suplanta_al_importarse():
    backend, eager = _modulos_del_backend(), set(_eager_del_conftest())
    assert eager, "no encuentro `_EAGER_BACKEND_MODULES` en el conftest"
    faltan = {}
    for f in sorted(_TESTS.glob("test_*.py")):
        src = f.read_text(encoding="utf-8", errors="replace")
        if "sys.modules" not in src:
            continue
        try:
            sup = _suplantados_al_importar(ast.parse(src)) & backend
        except SyntaxError:
            continue
        if sup - eager:
            faltan[f.name] = sorted(sup - eager)
    assert not faltan, (f"estos tests suplantan al importarse módulos del backend que el conftest no pre-importa; "
                        f"añádelos a `_EAGER_BACKEND_MODULES` o mueve el stub a un fixture con restauración: {faltan}")


def test_los_que_daban_los_15_fallos_son_los_reales():
    import agent
    import memory_manager
    assert isinstance(memory_manager, types.ModuleType) and getattr(memory_manager, "__file__", None)
    assert hasattr(memory_manager, "summarize_and_prune")
    assert hasattr(agent, "swap_meal") and hasattr(agent, "_prune_plan_for_chat")


#: Paquetes que algún test suplanta al importarse pero que el backend YA NO importa: Supabase salió con la migración a
#: Neon y la visión dejó el LangChain de Google. Su stub no puede sustituir a nada vivo. Si el backend vuelve a
#: importarlos, el test de abajo lo acusa y hay que pre-importarlos en el conftest.
_TERCEROS_SIN_USO = ("supabase", "langchain_google_genai")


def test_lo_de_terceros_que_un_test_suplanta_al_importarse_esta_cargado_de_verdad():
    """`cron_tasks` importa `apscheduler.triggers.cron` y 19 tests lo dejaban con `CronTrigger=object`. Con
    `test_chunked_learning_propagation` delante, `test_p0_a_zombie_partial_finalize` y
    `test_p1_a_orphan_reservation_cleanup` daban `TypeError: object() takes no arguments`."""
    sin_uso = re.compile(r"^\s*(?:from|import)\s+(?:" + "|".join(_TERCEROS_SIN_USO) + r")\b", re.M)
    vuelven = [p.name for p in _BACKEND.glob("*.py") if sin_uso.search(p.read_text(encoding="utf-8"))]
    assert not vuelven, f"el backend vuelve a importar un paquete que este test trata como sin uso: {vuelven}"
    backend, falsos = _modulos_del_backend(), {}
    for f in sorted(_TESTS.glob("test_*.py")):
        src = f.read_text(encoding="utf-8", errors="replace")
        if "sys.modules" not in src:
            continue
        try:
            nombres = _suplantados_al_importar(ast.parse(src))
        except SyntaxError:
            continue
        for n in nombres:
            raiz = n.split(".")[0]
            if raiz in backend or raiz in _TERCEROS_SIN_USO or raiz == "routers":
                continue
            mod = sys.modules.get(n)
            if mod is not None and getattr(mod, "__spec__", None) is None:
                falsos.setdefault(n, f.name)
    assert not falsos, f"módulos de terceros que siguen siendo un stub en la sesión (pre-impórtalos en el conftest): {falsos}"


def test_el_preimport_nuevo_va_detras_de_los_de_siempre_y_es_fail_open():
    src = (_TESTS / "conftest.py").read_text(encoding="utf-8")
    i = src.index("_EAGER_BACKEND_MODULES = (")
    assert src.index("import langgraph") < i and src.index("import langchain_openai") < i
    assert src.index("P1-CONFTEST-EAGER-GO") < i
    bloque = src[i:src.index("import ast", i)]
    assert "try:" in bloque and "except Exception" in bloque and "raise" not in bloque
    assert {"agent", "memory_manager", "services", "ai_helpers", "auth"} <= set(_eager_del_conftest())
    assert '_EAGER_THIRD_PARTY_MODULES = ("apscheduler.triggers.cron",)' in bloque


# ─────────────── marcador, documentos y anclas ───────────────
def test_marcador_documentos_y_anclas():
    assert "P1-PLAN-LOTE-52" in (_BACKEND / "app.py").read_text(encoding="utf-8")
    for doc in ("culinary_coherence.md", "plan_pendientes_2026_09_11.md"):
        assert "P1-PLAN-LOTE-52" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f in ("culinary_coherence.py", "dish_structure.py", "tests/conftest.py"):
        assert "P1-PLAN-LOTE-52" in (_BACKEND / f).read_text(encoding="utf-8"), f
