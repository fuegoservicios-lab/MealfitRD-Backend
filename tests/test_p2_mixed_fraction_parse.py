"""[P2-MIXED-FRACTION-PARSE · 2026-07-06] "1½ cebollas pequeñas" parseaba a qty=0 + Cohere.

Hallazgo del monitor prod (corr=2eeca23b/c0e07fa0, 2 errores en 1 min): la búsqueda
semántica recibía '1½ cebollas pequeñas' CON la cantidad y Cohere daba timeout (15s).
Causa raíz: `_preprocess_nlp_quantities` cubría la fracción unicode SOLA ("½ taza")
pero no el número MIXTO PEGADO ("1½") → el regex principal de `_parse_quantity` no
matcheaba → (a) qty=0.0 nominal — la cebolla se SUB-CONTABA en la lista; (b) el
string entero entraba como nombre a normalize_name, los tiers léxicos fallaban por
el prefijo numérico y cada recalc quemaba una llamada Cohere.

Fix: "1½" → "1 1/2" en el preprocesador (el parser ya soporta mixtos con espacio)
+ defensa en el ladder: forma sin-cantidad en fuzzy e INTENTO 6 semántico.
"""
import pytest

import shopping_calculator as sc

_MASTER = [
    {"name": "Cebolla", "category": "Vegetales", "aliases": ["cebolla roja", "cebollas"],
     "default_unit": "lb", "price_per_lb": 47.0, "density_g_per_unit": 150.0,
     "market_container": None, "container_weight_g": None, "shelf_life_days": 14},
    {"name": "Avena", "category": "Despensa", "aliases": ["avena en hojuelas"],
     "default_unit": "paquete", "price_per_lb": 40.0, "market_container": "paquete",
     "container_weight_g": 600.0, "shelf_life_days": 365},
]


@pytest.fixture(autouse=True)
def master_stub(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_MASTER))
    # La semántica JAMÁS debe ser necesaria para estos strings (tiers léxicos
    # deben resolver) — si alguien la invoca, el test truena.
    def _no_semantic():
        raise AssertionError("INTENTO 6 (Cohere) invocado — el tier léxico debía resolver")
    monkeypatch.setattr(sc, "get_semantic_cache", _no_semantic)
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


# ───────────── preprocesador ─────────────

@pytest.mark.parametrize("raw,expected", [
    ("1½ cebollas pequeñas", "1 1/2 cebollas pequeñas"),
    ("2¼ tazas de avena", "2 1/4 tazas de avena"),
    ("1 ½ cebollas", "1 1/2 cebollas"),   # con espacio también
    ("3⅔ tazas", "3 2/3 tazas"),
])
def test_mixed_unicode_fraction_expanded(raw, expected):
    assert sc._preprocess_nlp_quantities(raw) == expected


def test_lone_fraction_finally_expands():
    # La expansión de la fracción SOLA estaba MUERTA (el return final devolvía
    # el string original si ningún replacement posterior matcheaba) → "½ taza"
    # caía a qty=0.0 nominal. Ahora expande de verdad.
    assert sc._preprocess_nlp_quantities("½ taza de arroz") == "1/2 taza de arroz"


def test_lone_fraction_parses_qty():
    qty, unit, name = sc._parse_quantity("½ taza de avena", apply_yield_multiplier=False)
    assert qty == 0.5 and unit in ("taza", "tazas") and "avena" in name.lower()


# ───────────── _parse_quantity end-to-end ─────────────

def test_parse_quantity_recovers_qty_and_name():
    qty, unit, name = sc._parse_quantity("1½ cebollas pequeñas", apply_yield_multiplier=False)
    assert qty == 1.5, f"qty debe ser 1.5 (era 0.0 nominal pre-fix): {qty}"
    assert "cebolla" in name.lower(), f"el nombre debe resolver a Cebolla: {name}"


def test_parse_quantity_mixed_with_unit():
    qty, unit, name = sc._parse_quantity("2¼ tazas de avena", apply_yield_multiplier=False)
    assert qty == 2.25
    assert unit in ("taza", "tazas")
    assert "avena" in name.lower()


# ───────────── ladder: sin Cohere para strings con cantidad ─────────────

def test_normalize_name_resolves_without_semantic():
    # Defensa: aunque un caller pase el display crudo CON cantidad, el fuzzy con
    # la forma sin-cantidad resuelve — cero embeddings (el fixture truena si no).
    assert sc.normalize_name("1½ cebollas pequeñas") == "Cebolla"


def test_noqty_form_in_semantic_query_anchor():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("Intento 6: Búsqueda de Similitud Semántica")
    win = src[i:i + 900]
    assert "_sem_q" in win and "_noqty" in win, (
        "el query semántico debe ir SIN prefijo de cantidad (P2-MIXED-FRACTION-PARSE)"
    )
