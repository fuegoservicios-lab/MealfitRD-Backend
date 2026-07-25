"""[P1-MISALIGN-DEEP-TRACE · 2026-07-24] Instrumentación para cazar al pase que desalinea.

`P1-DISPLAY-RAW-QTY-RECONCILE` cerró el síntoma (la lista compraba cantidades distintas de las
que el usuario lee) pero no la causa: la divergencia va en ambas direcciones, así que son varios
pases actualizando una lista sin la otra. Esto instrumenta el pipeline para nombrar al culpable
con datos.

El tracer que existía (`P1-RAW-MISALIGN-TRACE`) se quedaba corto en tres cosas:

  1. Sólo miraba `len(display) != len(raw)`. La divergencia de CANTIDAD (24% de los alimentos,
     hasta 4.9×) no cambia el largo → era invisible. Tampoco veía el alimento presente sólo en
     el display, que es el que no se compra nunca.
  2. Marcaba UNA etapa por comida. Una comida que nacía con distinto largo enmascaraba para
     siempre la etapa donde después aparecía la divergencia de cantidad.
  3. Sólo emitía un WARN. Encontrar al culpable exigía mirar logs en vivo con suerte; ahora el
     resumen viaja en `plan_data._raw_misalign_stages` y se consulta con SQL sobre la flota.

⚠️ Y la lección cara de esta implementación: **`_parse_quantity` NO es una función pura.** Cae a
`normalize_name` → `get_semantic_cache` → `_batched_embed_documents`: llamadas de embeddings a
Cohere con `time.sleep` de reintento. El profiler lo puso en negro sobre blanco — 30 de 33
segundos eran `sleep`. Un pase de telemetría que corre 7 veces por generación no puede hacer red.
"""
import time

import pytest

import graph_orchestrator as go
import shopping_calculator as sc


CATALOGO = [
    {"name": "Mero", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Aguacate", "density_g_per_unit": 250, "density_g_per_cup": 150},
    {"name": "Queso blanco", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Espinacas", "density_g_per_unit": None, "density_g_per_cup": 30},
]


@pytest.fixture(autouse=True)
def _catalogos(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: CATALOGO)
    monkeypatch.setattr(go, "_CATALOG_DENSITY_INDEX_CACHE", None, raising=False)
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", None, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._PHANTOM_CATALOG_INDEX_CACHE = None


def _meal():
    return {"name": "Bollos de Harina",
            "ingredients": ["270 g de mero", "15 g de queso blanco", "½ aguacate"],
            "ingredients_raw": ["270 g de mero", "80.19 g de queso blanco"]}


# ───────────── 1. la telemetría no puede hacer red ─────────────

def test_el_tracer_nunca_hace_red(monkeypatch):
    """El contrato que justifica todo el modo `cheap`. Si alguien vuelve a enchufar el tracer al
    resolvedor completo, esto revienta en vez de aparecer como una factura de Cohere."""
    def _boom(*a, **k):
        raise AssertionError("el tracer intentó resolver por red (embeddings/semantic cache)")
    monkeypatch.setattr(sc, "get_semantic_cache", _boom, raising=False)
    monkeypatch.setattr(sc, "_batched_embed_documents", _boom, raising=False)
    monkeypatch.setattr(sc, "_parse_quantity", _boom, raising=False)
    days = [{"day": 1, "meals": [_meal()]}]
    go._trace_misalign(days, "probe")          # no debe lanzar
    assert days[0]["meals"][0].get("_misalign_trace")


def test_el_modo_cheap_es_offline_y_resuelve():
    assert go._resolve_line_food_grams("½ aguacate", cheap=True) == ("aguacate", 125.0)
    assert go._resolve_line_food_grams("270 g de mero", cheap=True) == ("mero", 270.0)
    # Frase con adjetivo: se afirma el CONTRATO (resuelve a la familia del alimento y calcula
    # gramos), no la clave exacta — a qué candidato engancha depende de si el catálogo cargado
    # tiene una fila "espinacas frescas", y eso cambia entre entornos.
    f, g = go._resolve_line_food_grams("½ taza de espinacas frescas", cheap=True)
    assert f and f.startswith("espinacas") and isinstance(g, float) and g > 0


def test_costo_acotado(monkeypatch):
    """Medido: 16 s por generación con el resolvedor completo, 2 ms con el barato. El umbral es
    holgado a propósito (máquinas lentas); lo que caza es una regresión de ORDEN de magnitud."""
    monkeypatch.setattr(sc, "_parse_quantity",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("red")), raising=False)
    days = [{"day": d, "meals": [_meal() for _ in range(4)]} for d in range(1, 8)]
    t = time.perf_counter()
    for _ in range(7):
        go._trace_misalign(days, "probe")
    assert (time.perf_counter() - t) < 2.0


# ───────────── 2. los tres tipos de divergencia ─────────────

def test_detecta_los_tres_tipos():
    fp = go._misalign_fingerprint(_meal())
    assert fp["len"] is True
    assert fp["missing_in_raw"] == ["aguacate"], "el que no se compra nunca"
    assert fp["qty"] == [("queso blanco", 0.19)], "invisible para el tracer viejo (largo igual)"


def test_cada_tipo_recuerda_su_PRIMERA_etapa():
    """El tracer viejo marcaba una sola etapa por comida: la comida que nacía con distinto largo
    enmascaraba para siempre dónde aparecía después la divergencia de cantidad."""
    meal = {"name": "X", "ingredients": ["270 g de mero", "½ aguacate"],
            "ingredients_raw": ["270 g de mero"]}
    days = [{"day": 1, "meals": [meal]}]
    go._trace_misalign(days, "pre_engine")
    assert meal["_misalign_trace"] == {"len": "pre_engine", "missing_in_raw": "pre_engine"}

    # más tarde aparece una divergencia de CANTIDAD, en otra etapa
    meal["ingredients"].append("15 g de queso blanco")
    meal["ingredients_raw"].append("80.19 g de queso blanco")
    go._trace_misalign(days, "post_humanize")
    assert meal["_misalign_trace"]["qty"] == "post_humanize", "la etapa nueva se registra…"
    assert meal["_misalign_trace"]["len"] == "pre_engine", "…sin pisar la primera de las otras"


def test_pareo_por_alimento_no_por_indice():
    """Las dos listas están REORDENADAS entre sí en planes reales. Comparar por posición
    enfrenta '½ tomate' contra '0.5 ají cubanela' y no mide nada — ese error me dio una primera
    medición inflada del 24%."""
    meal = {"name": "X",
            "ingredients": ["270 g de mero", "½ aguacate"],
            "ingredients_raw": ["½ aguacate", "270 g de mero"]}   # mismo contenido, otro orden
    fp = go._misalign_fingerprint(meal)
    assert fp["qty"] == [] and fp["missing_in_raw"] == [] and fp["len"] is False


def test_resumen_agregado_para_sql():
    """Sin esto, encontrar al culpable exige mirar logs en vivo con suerte."""
    days = [{"day": 1, "meals": [_meal(), _meal()]}]
    go._trace_misalign(days, "post_macro_engine")
    resumen = go._summarize_misalign_stages(days)
    assert resumen["qty"] == {"post_macro_engine": 2}
    assert resumen["missing_in_raw"] == {"post_macro_engine": 2}


def test_comida_sana_no_deja_marca():
    meal = {"name": "X", "ingredients": ["270 g de mero"], "ingredients_raw": ["270 g de mero"]}
    days = [{"day": 1, "meals": [meal]}]
    go._trace_misalign(days, "probe")
    assert "_misalign_trace" not in meal
    assert go._summarize_misalign_stages(days) == {}


# ───────────── 3. cableado ─────────────

def test_la_foto_se_toma_ANTES_de_reparar():
    """Si el resumen se tomara después del reconciliador, la telemetría diría siempre 'todo
    bien' y el culpable quedaría tapado por su propio parche."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_trace = src.index('_trace_misalign(result.get("days"), "pre_reconcile")')
    i_sum = src.index("_summarize_misalign_stages(result.get(\"days\"))")
    i_rec = src.index('_reconcile_display_raw_lines(result.get("days")')
    assert i_trace < i_sum < i_rec


def test_los_pases_que_mutan_si_usan_la_resolucion_completa():
    """Dedupe y reconciliador necesitan los alias del catálogo: corren una sola vez y la lista
    de compras hace ese mismo trabajo justo después, así que el caché queda caliente para ella."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _reconcile_display_raw_lines")
    assert "_resolve_line_food_grams(line)" in src[i:i + 3000], (
        "el reconciliador muta datos: resolución completa, no la barata"
    )
