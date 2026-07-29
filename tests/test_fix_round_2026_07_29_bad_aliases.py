"""[FIX-ROUND-2026-07-29] Findings 3/4/5/6 (lens: lists) — corrección de datos en
`master_ingredients.aliases` (Neon). DB-gated (skip sin conexión — offline/CI, mismo
patrón que `test_food_db_population_coverage.py::test_db_coverage_and_atwater_consistency`).

  - Finding 4 (critical): 'pasta de tomate' / 'puré de tomate' NO deben resolver, vía
    alias, a 'Salsa de tomate' — son alimentos distintos (pasta de tomate ~82 kcal/100g
    USDA vs 28.7 kcal/100g de Salsa de tomate en catálogo, ~3x subestimado).
  - Finding 5 (critical): 'cereza'/'cerezas' (fruta fresca) NO deben resolver, vía alias
    bare, a 'Cereza maraschino' (confitada, 165 kcal / 41.97g carbs — >2.5x el real
    ~63 kcal / ~16g carbs). Los alias LEGÍTIMOS del mismo alimento confitado
    ('cereza maraschino', 'cereza de coctel', 'cereza confitada', 'maraschino cherry')
    se conservan.
  - Finding 6 (important): 'margarina' NO debe resolver, vía alias, a 'Mantequilla'
    (lácteo) — riesgo de clasificación de alérgenos para usuarios que evitan lácteos.
  - Finding 3 (critical): 'Yogurt' SÍ debe traer 'yogurt regular' como alias — el grupo
    MÁS GRANDE (115 productos) de la auditoría `unlinked_products.csv` estaba marcado
    MISSING_FROM_CATALOGUE cuando el catálogo ya tiene una fila 'Yogurt' genérica
    perfectamente aplicable.
"""
import os

import pytest


def _neon_conn():
    url = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg
        return psycopg.connect(url, connect_timeout=8)
    except Exception:
        return None


@pytest.fixture(scope="module")
def _aliases_by_name():
    conn = _neon_conn()
    if conn is None:
        pytest.skip("sin conexión Neon (offline/CI)")
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, aliases FROM master_ingredients "
            "WHERE name IN ('Salsa de tomate', 'Cereza maraschino', 'Mantequilla', 'Yogurt')"
        )
        rows = {name: (aliases or []) for name, aliases in cur.fetchall()}
    finally:
        conn.close()
    return rows


def _lower(aliases):
    return {a.lower() for a in aliases}


def test_salsa_de_tomate_no_longer_aliases_pasta_or_pure(_aliases_by_name):
    aliases = _lower(_aliases_by_name.get("Salsa de tomate", []))
    assert "pasta de tomate" not in aliases, (
        "pasta de tomate NO debe resolver a Salsa de tomate (macros ~3x distintos, finding 4)"
    )
    assert "puré de tomate" not in aliases and "pure de tomate" not in aliases


def test_cereza_maraschino_no_longer_aliases_bare_cereza(_aliases_by_name):
    aliases = _lower(_aliases_by_name.get("Cereza maraschino", []))
    assert "cereza" not in aliases, (
        "cereza (fresca) NO debe resolver a Cereza maraschino (confitada, finding 5)"
    )
    assert "cerezas" not in aliases
    # los alias legítimos del MISMO alimento confitado se conservan (no sobre-corregir).
    assert "cereza maraschino" in aliases
    assert "maraschino cherry" in aliases


def test_mantequilla_no_longer_aliases_margarina(_aliases_by_name):
    aliases = _lower(_aliases_by_name.get("Mantequilla", []))
    assert "margarina" not in aliases, (
        "margarina NO debe resolver a Mantequilla (alimentos distintos, riesgo alérgeno, finding 6)"
    )


def test_yogurt_now_aliases_yogurt_regular(_aliases_by_name):
    aliases = _lower(_aliases_by_name.get("Yogurt", []))
    assert "yogurt regular" in aliases, (
        "Yogurt debe traer 'yogurt regular' como alias — cierra la falsa "
        "MISSING_FROM_CATALOGUE del grupo más grande de unlinked_products.csv (finding 3)"
    )
