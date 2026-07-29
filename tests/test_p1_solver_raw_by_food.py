"""[P1-SOLVER-RAW-BY-FOOD · 2026-07-25] La CAUSA RAÍZ de la divergencia display↔raw.

`P1-DISPLAY-RAW-QTY-RECONCILE` repara el síntoma en la frontera; `P1-MISALIGN-DEEP-TRACE`
instrumentó el pipeline para encontrar al culpable. Dos generaciones bastaron:

    plan ea79db0e   qty: {"post_macro_engine": 7}          ← 7 de 7 comidas, un solo actor
    plan 0fbd36e8   qty: {"post_macro_engine": 5, "post_humanize": 2, "pre_engine": 1}

Entre las sondas `pre_engine` y `post_macro_engine` hay **una sola llamada**:
`_apply_macro_engine`. Y dentro, en `_apply_macro_solver_to_meal`, estaba esto:

    raw = meal.get("ingredients_raw")
    if isinstance(raw, list) and len(raw) == len(factors):     # ← el guard
        meal["ingredients_raw"] = [rescale(...) for r, f in zip(raw, factors)]

El display se escala SIEMPRE (`meal["ingredients"] = res["ingredients"]`); raw solo si las dos
listas tienen el MISMO número de líneas. Y el tracer midió que el desajuste de largos **nace en
`pre_engine`** — antes del solver — en 7-8 de cada 8 comidas. O sea: en casi todas, el guard
fallaba, el display quedaba escalado y raw se quedaba con las cantidades pre-solver. Como la
lista de compras lee raw (`meal.get("ingredients_raw") or meal.get("ingredients")`), se compraba
una cosa y se leía otra — hasta 4.9× de diferencia en planes entregados.

Es la MISMA clase de error que `P1-PHANTOM-RAW-PARITY`, que cometí yo el día anterior en mis
propios pases: **un guard de alineación por índice que se salta exactamente los casos que
necesitan el trabajo.** Las dos listas no son paralelas por diseño —
`_restore_display_from_raw_orphans` las reconcilia por contenido y trata `len(raw) > len(ings)`
como estado esperado.
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


CATALOGO = [
    {"name": "Mero", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Tomate", "density_g_per_unit": 120, "density_g_per_cup": 180},
    {"name": "Aguacate", "density_g_per_unit": 250, "density_g_per_cup": 150},
    {"name": "Arroz blanco", "density_g_per_unit": None, "density_g_per_cup": 185},
]


@pytest.fixture(autouse=True)
def _catalogos(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: CATALOGO)
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE"):
        monkeypatch.setattr(go, attr, None, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._PHANTOM_CATALOG_INDEX_CACHE = None


# ───────────── 1. el caso que el guard por índice se saltaba ─────────────

def test_escala_raw_aunque_los_largos_difieran():
    """El escenario real: raw tiene una línea extra ('Sal al gusto') y por eso NADA se escalaba."""
    display = ["100 g de mero", "2 tomates", "½ aguacate"]
    factors = [1.5, 1.5, 1.0]
    raw = ["100 g de mero", "2 tomates", "½ aguacate", "Sal al gusto"]
    out, n = go._rescale_raw_by_food(raw, display, factors)
    assert n == 2, out
    assert out[0] == "150 g de mero"
    assert out[1].startswith("3 tomate")
    assert out[2] == "½ aguacate", "factor 1.0 → no se toca"
    assert out[3] == "Sal al gusto", "sin cantidad → intacta"


def test_conserva_el_largo_de_raw():
    """Escalar no puede añadir ni quitar líneas — solo cambiar cantidades."""
    raw = ["100 g de mero", "Sal al gusto", "1 taza de arroz blanco"]
    out, _ = go._rescale_raw_by_food(raw, ["100 g de mero"], [2.0])
    assert len(out) == len(raw)


def test_alimento_con_factores_distintos_se_deja_intacto():
    """Preferimos no escalar que escalar con el factor equivocado."""
    display = ["100 g de mero", "50 g de mero"]      # mismo alimento, dos factores
    factors = [2.0, 0.5]
    out, n = go._rescale_raw_by_food(["100 g de mero"], display, factors)
    assert n == 0 and out == ["100 g de mero"]


def test_alimento_ausente_del_display_no_se_toca():
    out, n = go._rescale_raw_by_food(["1 taza de arroz blanco"], ["100 g de mero"], [2.0])
    assert n == 0 and out == ["1 taza de arroz blanco"]


def test_fail_safe_devuelve_la_lista_original(monkeypatch):
    monkeypatch.setattr(go, "_resolve_line_food_grams",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    raw = ["100 g de mero"]
    out, n = go._rescale_raw_by_food(raw, ["100 g de mero"], [2.0])
    assert out == raw and n == 0


# ───────────── 2. el contrato en el solver ─────────────

def test_el_camino_por_indice_sigue_siendo_el_preferido():
    """Cuando las listas son paralelas DE VERDAD, el índice es exacto y barato: se conserva.

    [P2-RAW-PAIR-BY-FOOD · 2026-07-29] re-anclado: el largo igual ya no basta para tomar ese camino
    (medido: 93.5% de las comidas tiene largo igual y solo el 48.1% de ESAS son paralelas). Lo que
    este test protege es que el camino rápido SIGA EXISTIENDO, no el criterio con que se elige."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_macro_solver_to_meal")
    body = src[i:i + 9000]
    assert "len(raw) == len(factors)" in body, "el camino rápido por índice se conserva"
    assert "_raw_display_parallel_by_food(_ing_strs, raw)" in body, \
        "y ahora exige paralelismo verificado por alimento antes de fiarse del índice"
    assert "elif SOLVER_RAW_BY_FOOD:" in body, "el fallback por alimento cuelga de él"


def test_el_guard_ya_no_es_la_unica_condicion():
    """La regresión que hay que impedir: volver a `if ... and len(raw) == len(factors):` como
    ÚNICA vía, que es lo que dejaba raw sin escalar en 7 de 8 comidas."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_macro_solver_to_meal")
    body = src[i:i + 9000]
    assert "isinstance(raw, list) and len(raw) == len(factors)" not in body, (
        "ese guard era la causa raíz: sin rama alternativa, raw se queda pre-solver"
    )


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'SOLVER_RAW_BY_FOOD = _env_bool("MEALFIT_SOLVER_RAW_BY_FOOD", True)' in src


def test_deja_rastro_para_medir_el_efecto():
    """`_solver_raw_by_food` en el meal permite comprobar en SQL que la causa raíz dejó de
    disparar el reconciliador de la frontera (P1-DISPLAY-RAW-QTY-RECONCILE)."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'meal["_solver_raw_by_food"] = _n' in src
