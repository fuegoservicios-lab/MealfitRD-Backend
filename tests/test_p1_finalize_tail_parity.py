"""[P1-FINALIZE-TAIL-PARITY · 2026-07-25] El espejo que declaraba estar completo y no lo estaba.

El refill de gain-muscle dentro de `finalize_plan_data_coherence` (P1-CHUNK-GAINMUSCLE-PARITY)
dice en su propio comentario que corre "espejo del orden de assemble". Pero assemble ganó una
COLA de pases después de su refill y este camino se quedó sin ella.

Medido en vivo (plan `58eb6ed4`, corr=`00471349`) — el log lo fecha sin ambigüedad:

    14:00:29  P1-SOLVER-RAW-BY-FOOD          (cola de assemble)
    14:02:18  P1-DISPLAY-RAW-QTY-RECONCILE   (cola de assemble)
    14:02:53  P1-GAINMUSCLE-KCAL-FLOOR  +200 kcal (arroz cocido)   ← 35 s DESPUÉS

Y en UNA comida produjo las tres cosas que esos pases arreglan:

    ingredients     : ¼ taza de arroz blanco  +  55g de arroz blanco cocido
    ingredients_raw : ¼ taza de arroz blanco  + 100g de arroz blanco cocido

el mismo alimento duplicado, display↔raw a 1.82×, y gramos COCIDOS que resuelven contra la fila
SECA del catálogo (2.76× de sobreconteo en macros).

**Importa más por los CHUNKS que por assemble**: este es el cierre de las semanas 2+ (~27 de los
30 días), y hasta ahora no tenían NINGUNA de estas protecciones. Ahí además la lista de compras
se construye después, así que el arreglo es completo.

⚠️ NO se llama al reconciliador display↔raw: copia las líneas de display verbatim y en este
punto el display ya está humanizado ("1 pechuga (porción)") — borraría las unidades métricas de
`ingredients_raw`, que es exactamente lo que P1-4 existe para proteger.
"""
from pathlib import Path

import pytest

import graph_orchestrator as go
import shopping_calculator as sc

SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _catalogos(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda *a, **k: [{"name": "Arroz blanco", "density_g_per_unit": None,
                                          "density_g_per_cup": 185}])
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE"):
        monkeypatch.setattr(go, attr, None, raising=False)
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE",
                        {"arroz blanco": 358.6, "arroz": 358.6}, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE",
                 "_COOKED_CATALOG_KCAL_CACHE"):
        setattr(go, attr, None)


def _comida_viva():
    """Copia literal de lo que el refill dejó en el plan 58eb6ed4."""
    return [{"day": 2, "meals": [{
        "name": "Soya Texturizada Guisada al Estilo Dominicano",
        "_gainmuscle_kcal_floor": True,
        "ingredients": ["¼ taza de arroz blanco", "55g de arroz blanco cocido"],
        "ingredients_raw": ["¼ taza de arroz blanco", "100g de arroz blanco cocido"],
    }]}]


# ───────────── 1. los dos pases limpian lo que el refill ensucia ─────────────

def test_convierte_los_gramos_cocidos_a_secos():
    """El 2.76×: el catálogo tiene el arroz en SECO y la línea venía en cocido."""
    days = _comida_viva()
    assert go._normalize_cooked_grain_lines(days), "debe reescribir en las dos listas"
    blob = " | ".join(days[0]["meals"][0]["ingredients"] + days[0]["meals"][0]["ingredients_raw"])
    assert "cocido" not in blob, blob


def test_fusiona_el_alimento_duplicado():
    days = _comida_viva()
    go._normalize_cooked_grain_lines(days)
    go._merge_duplicate_food_lines(days)
    m = days[0]["meals"][0]
    assert len(m["ingredients"]) == 1, m["ingredients"]
    assert len(m["ingredients_raw"]) == 1, m["ingredients_raw"]


def test_conserva_la_comida_no_la_recorta():
    """Limpiar no puede quitar arroz: ¼ taza (46 g) + 55 g cocidos (~20 g secos) ≈ 66 g."""
    days = _comida_viva()
    go._normalize_cooked_grain_lines(days)
    go._merge_duplicate_food_lines(days)
    linea = days[0]["meals"][0]["ingredients"][0]
    import re
    g = float(re.search(r"(\d+(?:[.,]\d+)?)", linea).group(1))
    assert 60 <= g <= 75, linea


# ───────────── 2. el cableado y su límite ─────────────

def test_la_cola_cuelga_del_refill_del_finalize():
    i = SRC.index("P1-CHUNK-GAINMUSCLE-PARITY")
    bloque = SRC[i:i + 5200]
    assert "P1-FINALIZE-TAIL-PARITY" in bloque, "la cola va DENTRO del bloque del refill"
    assert "_normalize_cooked_grain_lines(days)" in bloque
    assert "_merge_duplicate_food_lines(days)" in bloque


def test_NO_llama_al_reconciliador_display_raw():
    """La línea roja: en este punto el display ya está humanizado, así que copiar sus líneas a
    raw borraría las unidades métricas que P1-4 protege."""
    i = SRC.index("P1-FINALIZE-TAIL-PARITY")
    bloque = SRC[i:i + 3000]
    assert "_reconcile_display_raw_lines" not in bloque, (
        "reconciliar aquí es destructivo — el display ya pasó por el humanizador"
    )


def test_documenta_por_que_importa_por_los_chunks():
    i = SRC.index("P1-FINALIZE-TAIL-PARITY")
    bloque = SRC[i:i + 3000]
    assert "CHUNK" in bloque.upper(), (
        "sin esa razón escrita, el próximo refactor lo lee como redundante con assemble"
    )


def test_knob_de_rollback():
    assert 'FINALIZE_TAIL_PARITY = _env_bool("MEALFIT_FINALIZE_TAIL_PARITY", True)' in SRC
    assert "if FINALIZE_TAIL_PARITY:" in SRC
