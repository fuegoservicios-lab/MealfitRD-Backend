"""[P1-CAPS-LAST-WORD + P1-CAP-RAW-BY-FOOD · 2026-07-25] 360 g de queso cottage en un desayuno.

Plan vivo `943c604b`, D2 "Tostadas de Yuca…": entregó **`360g de queso cottage`**, el doble del
techo por comida (`MEALFIT_MEAL_CHEESE_CAP_G=180`). La comida traía a la vez
`_portion_realism_capped=True` y `_protein_closed=True`.

Probado aislado, el cap convierte esa misma línea a 180 g sin problema: **no falla el cap, falla
el ORDEN**. Los caps corren al principio del finalize (L~21451) y en assemble, pero después
siguen actuando pases ADITIVOS —refill de gain-muscle, cerrador de proteína, closer de micros—
que vuelven a inflar líneas ya recortadas.

Y al arreglar el orden apareció el defecto de debajo, que era peor:

    display : 360g de queso cottage   → el cap lo bajó a 180g   ✅
    raw     : 496.8g de queso cottage → **intacto**             ❌  ← la lista compra 4×

`_cap_unrealistic_portions` escribía raw con `raw[idx]` bajo un guard `_lockstep`
(`len(raw)==len(ings)`). Las dos listas están **reordenadas**: el cottage estaba en la posición 2
del display y en la 3 de raw, así que el recorte cayó sobre `"Sal al gusto"` — un no-op
silencioso. Con los largos iguales el guard no protege: **corrompe la línea equivocada**.

Cuarta aparición del mismo patrón en un día (solver, mis dos pases, y ahora los caps): *un guard
de índice sobre dos listas que no son paralelas*.
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


@pytest.fixture(autouse=True)
def _catalogo(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: [
        {"name": "Queso cottage", "density_g_per_unit": None, "density_g_per_cup": 226},
        {"name": "Yuca", "density_g_per_unit": None, "density_g_per_cup": 150},
    ])
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE"):
        monkeypatch.setattr(go, attr, None, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._PHANTOM_CATALOG_INDEX_CACHE = None


def _comida_viva():
    """Copia literal del plan 943c604b — nótese que raw está REORDENADO."""
    return {
        "name": "Tostadas de Yuca con Mantequilla de Maní y Queso Cottage",
        "meal": "Desayuno",
        "ingredients": ["½ taza de yuca rallada (75g)", "2 cdas de mantequilla de maní (32g)",
                        "360g de queso cottage", "Sal al gusto"],
        "ingredients_raw": ["0.5 taza de yuca rallada (75.38g)", "2 cdas de mantequilla de maní (32g)",
                            "Sal al gusto", "496.8g de queso cottage"],
    }


# ───────────── 1. el recorte llega a las DOS listas ─────────────

def test_capea_el_display():
    meal = _comida_viva()
    assert go._cap_unrealistic_portions([{"day": 2, "meals": [meal]}]) >= 1
    linea = next(i for i in meal["ingredients"] if "cottage" in str(i).lower())
    assert linea.startswith("180g"), linea


def test_el_recorte_ALCANZA_la_linea_correcta_de_raw():
    """Lo que hacía que arreglar el display empeorara el dinero: la lista compraba 4×."""
    meal = _comida_viva()
    go._cap_unrealistic_portions([{"day": 2, "meals": [meal]}])
    raw_cottage = next(i for i in meal["ingredients_raw"] if "cottage" in str(i).lower())
    assert "496" not in raw_cottage, f"raw quedó sin recortar: {raw_cottage}"
    assert raw_cottage.startswith("248"), raw_cottage      # 496.8 × 0.5


def test_no_toca_la_linea_equivocada_de_raw():
    """Con `raw[idx]` y las listas reordenadas, el recorte caía sobre 'Sal al gusto'."""
    meal = _comida_viva()
    go._cap_unrealistic_portions([{"day": 2, "meals": [meal]}])
    assert "Sal al gusto" in meal["ingredients_raw"]
    assert len(meal["ingredients_raw"]) == 4, "no se pierden ni duplican líneas"


def test_lineas_sin_cap_intactas():
    meal = _comida_viva()
    go._cap_unrealistic_portions([{"day": 2, "meals": [meal]}])
    assert "2 cdas de mantequilla de maní (32g)" in meal["ingredients"]
    assert "2 cdas de mantequilla de maní (32g)" in meal["ingredients_raw"]


# ───────────── 2. los caps como ÚLTIMA palabra ─────────────

def test_los_caps_se_reejecutan_al_final_del_finalize():
    """Sin esto, cualquier pase aditivo posterior deja una línea por encima del techo."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_fin = src.index("def finalize_plan_data_coherence")
    i_ret = src.index('return (total, ", ".join(parts))', i_fin)
    cola = src[i_ret - 2600:i_ret]
    assert "P1-CAPS-LAST-WORD" in cola
    assert "_cap_unrealistic_portions(days, db=db)" in cola


def test_corre_antes_del_refresh_de_banda():
    """El score persistido debe medir el plato REAL, no el de antes del recorte."""
    from pathlib import Path
    dbp = (Path(go.__file__).resolve().parent / "db_plans.py").read_text(encoding="utf-8")
    i_fin = dbp.index("finalize_plan_data_coherence as _fpc")
    i_band = dbp.index("refresh_clinical_band_score_post_finalize", i_fin)
    assert i_fin < i_band, "el finalize (con los caps) precede al refresh de banda"


def test_knobs():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'CAPS_LAST_WORD = _env_bool("MEALFIT_CAPS_LAST_WORD", True)' in src
    assert "if CAPS_LAST_WORD and PORTION_REALISM_CAP_ENABLED:" in src


def test_el_guard_de_indice_ya_no_es_la_via_principal():
    """La regresión a impedir: volver a `raw[idx]` como camino único."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _cap_unrealistic_portions")
    # La función ronda las 220 líneas y el sitio de escritura está al final; una ventana corta
    # dejaba el assert siempre en verde por no llegar (ya me pasó dos veces hoy con otros tests).
    cuerpo = src[i:i + 22000]
    assert "_rescale_raw_by_food(raw, [s], [factor])" in cuerpo, (
        "el recorte se mapea por ALIMENTO; `raw[idx]` sólo queda como fallback"
    )
