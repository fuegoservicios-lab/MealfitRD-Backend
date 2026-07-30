"""[P1-UPDATE-RAW-BY-FOOD · 2026-07-30] (audit solver+seeder v5 · P1-4)

Dos capas del MISMO defecto, encontradas por 4 lentes independientes del audit.

**(a) Por escritor.** `P2-RAW-PAIR-BY-FOOD` (v4) estableció el contrato "índice solo con
paralelismo VERIFICADO; si no, por alimento" y lo cableó en 3 escritores (solver per-meal,
refinador global, micro-closer). Los 5 escritores del band-closer quedaron fuera y siguen con
el guard viejo:

    raw = m.get("ingredients_raw")
    if isinstance(raw, list) and len(raw) == len(ings):
        raw[idx] = _resc(str(raw[idx]), factor * _f)     # ← índice CIEGO

"Mismo largo" nunca fue "mismo orden": el reconciliador reconstruye raw como
`[conservadas] + [añadidas]` — preserva el largo y cambia el orden. El repo lo MIDIÓ:
93.5% de comidas con largos iguales, y solo el 48.1% de ESAS son paralelas por índice.

**(b) Por superficie.** `_reconcile_display_raw_lines` tenía exactamente 3 callsites (assemble,
finalize, shield pre-INSERT) — CERO en los mutadores de update. Así que en swap-persist,
chat-modify, recipe-expand y regen-day la divergencia introducida se PERSISTE... y peor: el
mismo mutador re-agrega la lista de compras leyendo `ingredients_raw` primero.

Y es silencioso por construcción: `expected_sum_from_recipes` lee `ingredients_raw` en los DOS
lados del coherence guard, así que el guard es ciego a esta divergencia (lección
P1-GRAM-HINT-TRUMPS-QTY). Cero log, cero banner.
"""
from __future__ import annotations

import os

import pytest

import graph_orchestrator as go
import shopping_calculator as sc

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_PLANS = _read(os.path.join("routers", "plans.py"))
_TOOLS = _read("tools.py")

CATALOGO = [
    {"name": "Mero", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Arroz blanco", "density_g_per_unit": None, "density_g_per_cup": 185},
    {"name": "Aceite de oliva", "density_g_per_unit": None, "density_g_per_cup": 218},
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


class _DB:
    """Doble mínimo: macros por gramo de string, sin red ni catálogo real."""

    _PER_G = {
        "mero": {"protein": 0.20, "carbs": 0.0, "fats": 0.01, "kcal": 0.92},
        "arroz": {"protein": 0.027, "carbs": 0.28, "fats": 0.003, "kcal": 1.30},
        "aceite": {"protein": 0.0, "carbs": 0.0, "fats": 1.0, "kcal": 8.84},
    }

    def _key(self, s: str):
        low = str(s).lower()
        return next((k for k in self._PER_G if k in low), None)

    def grams_from_ingredient_string(self, s: str) -> float:
        import re
        m = re.match(r"\s*([\d.]+)\s*g\b", str(s))
        return float(m.group(1)) if m else 0.0

    def macros_from_ingredient_string(self, s: str):
        k = self._key(s)
        if not k:
            return {}
        g = self.grams_from_ingredient_string(s)
        return {m: v * g for m, v in self._PER_G[k].items()}


# ═════════════════ 1 · E2E: el escritor escala el ALIMENTO, no el índice ═════════════════

def test_rebalance_scales_the_right_food_with_rotated_raw(monkeypatch):
    """El caso medido: display y raw del MISMO largo pero en ORDEN distinto (48.1% de
    paralelismo real). El rebalance decide un factor para el arroz (display idx 1); con el guard
    viejo escribía raw[1], que es el MERO."""
    monkeypatch.setattr(go, "_ingredient_macro_group",
                        lambda s, db=None: "carbs" if "arroz" in str(s).lower() else "protein")
    meal = {
        "ingredients": ["100 g de mero", "200 g de arroz blanco"],
        # rotado a propósito: mismo largo, orden distinto (lo que produce el reconciliador)
        "ingredients_raw": ["200 g de arroz blanco", "100 g de mero"],
        "protein": 25, "carbs": 56, "fats": 2, "cals": 350,
    }
    go._rebalance_day_macros_to_target([meal], 28.0, 2.0, _DB(), target_protein=0)
    raw = meal["ingredients_raw"]
    assert "mero" in raw[1].lower(), f"la línea del mero no debe moverse: {raw}"
    assert raw[1] == "100 g de mero", f"el mero quedó escalado — índice ciego: {raw}"
    assert "arroz" in raw[0].lower()
    assert raw[0] != "200 g de arroz blanco", f"el arroz SÍ debía escalarse: {raw}"


def test_rebalance_still_uses_index_when_truly_parallel(monkeypatch):
    """Regresión inversa: con listas realmente paralelas el camino por índice sigue vivo
    (es el barato) y el resultado es el mismo de siempre."""
    monkeypatch.setattr(go, "_ingredient_macro_group",
                        lambda s, db=None: "carbs" if "arroz" in str(s).lower() else "protein")
    meal = {
        "ingredients": ["100 g de mero", "200 g de arroz blanco"],
        "ingredients_raw": ["100 g de mero", "200 g de arroz blanco"],
        "protein": 25, "carbs": 56, "fats": 2, "cals": 350,
    }
    go._rebalance_day_macros_to_target([meal], 28.0, 2.0, _DB(), target_protein=0)
    raw = meal["ingredients_raw"]
    assert raw[0] == "100 g de mero"
    assert raw[1] != "200 g de arroz blanco"


def test_quantize_scales_by_food_with_rotated_raw(monkeypatch):
    """`_apply_portion_quantization` escribe raw en bloque (zip de factores) — misma ceguera."""
    from nutrition_db import quantize_ingredient_string as _q
    if _q("0.37 taza de arroz blanco")[0] == "0.37 taza de arroz blanco":
        pytest.skip("el quantizador no re-snapea este string en este entorno")
    meal = {
        "ingredients": ["0.37 taza de arroz blanco", "100 g de mero"],
        "ingredients_raw": ["100 g de mero", "0.37 taza de arroz blanco"],
        "protein": 25, "carbs": 56, "fats": 2, "cals": 350,
    }
    go._apply_portion_quantization({"days": [{"meals": [meal]}]}, _DB())
    raw = meal["ingredients_raw"]
    assert raw[0] == "100 g de mero", f"el mero no llevaba factor: {raw}"


# ═════════════════ 2 · estructural: los 5 escritores usan el helper SSOT ═════════════════

_WRITERS = [
    "_trim_day_carbs_to_target",
    "_trim_day_fats_to_target",
    "_close_carb_gap_for_day",
    "_rebalance_day_macros_to_target",
    "_apply_portion_quantization",
]


def _fn_body(name: str) -> str:
    i = _GO.index(f"def {name}(")
    return _GO[i:_GO.index("\ndef ", i + 10)]


@pytest.mark.parametrize("fn", _WRITERS)
def test_writer_delegates_raw_to_the_by_food_helper(fn):
    """El contrato v4 en los 5 escritores que quedaron fuera. Anclado al CUERPO de cada función
    (fin-de-bloque relativo), no a una ventana de bytes."""
    body = _fn_body(fn)
    assert ("_sync_one_raw_line" in body) or ("_rescale_raw_by_food" in body), (
        f"{fn} no delega la escritura de ingredients_raw al helper by-food"
    )


@pytest.mark.parametrize("fn", _WRITERS)
def test_writer_has_no_blind_index_write_left(fn):
    """El patrón exacto del bug: `len(raw) == len(...)` como ÚNICO guard antes de `raw[idx] =`.
    Si vuelve a aparecer en cualquiera de los 5, este test lo caza."""
    body = _fn_body(fn)
    ofensas = [ln.strip() for ln in body.splitlines()
               if ("raw[idx] = " in ln or "raw[_ri] = " in ln) and "_sync_one_raw_line" not in ln]
    assert not ofensas, f"{fn} conserva escritura por índice ciego: {ofensas}"


# ═══════════ 3 · por superficie: los mutadores de update reconcilian antes de persistir ═══════════

def _reconcile_callsites(src: str) -> list[int]:
    """Posiciones de las INVOCACIONES del reconciliador, resolviendo el alias.

    El repo importa lazy y con alias — y a menudo en un `from ... import (A as _x, B as _y)`
    multi-línea, así que el ancla NO puede incluir la palabra `import` pegada al nombre."""
    out = []
    i = 0
    key = "_reconcile_display_raw_lines as "
    while True:
        j = src.find(key, i)
        if j < 0:
            break
        alias = src[j + len(key):src.index("\n", j)].strip().rstrip("),")
        if alias:
            k = 0
            while True:
                p = src.find(alias + "(", k)
                if p < 0:
                    break
                out.append(p)
                k = p + 1
        i = j + 1
    return sorted(out)


def test_swap_persist_reconciles_before_rebuilding_shopping_lists():
    """El mutador re-agrega la lista de compras leyendo raw PRIMERO. Si la reconciliación no
    corre antes, la lista se construye sobre el raw divergente que el motor acaba de escribir."""
    hits = _reconcile_callsites(_PLANS)
    assert hits, "swap-persist no reconcilia display↔raw en ninguna parte"
    i_rebuild = _PLANS.index("_rebuild_plan_shopping_lists_inline(")
    # la reconciliación más cercana ANTES del rebuild
    assert any(h < i_rebuild for h in hits), (
        "la reconciliación debe correr ANTES del rebuild de listas")


def test_chat_modify_reconciles_after_the_engine():
    assert _reconcile_callsites(_TOOLS), "chat-modify no reconcilia display↔raw en ninguna parte"


@pytest.mark.parametrize("rel", ["routers/plans.py", "tools.py"])
def test_update_surfaces_reconcile_under_the_same_knob(rel):
    """Mismo knob que el shield pre-INSERT (`RECONCILE_AFTER_BAND_CLOSER`): un solo interruptor
    de rollback para la clase entera, no uno por superficie.

    (Parametrizado por RUTA, no por contenido: pasar el source como parámetro convierte el id
    del test en el archivo entero — 4 MB de ruido en el reporte.)"""
    assert "RECONCILE_AFTER_BAND_CLOSER" in _read(os.path.join(*rel.split("/"))), rel
