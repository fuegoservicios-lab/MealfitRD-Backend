"""[P2-WHITE-FISH-ALIAS-SPLIT · 2026-09-02] Un alias del catálogo resuelve a UNA fila.

Medido: «mero» y «tilapia» eran alias de la fila genérica «Filete de pescado blanco» Y de sus
filas por especie; el índice desempata por longitud y luego alfabético/orden de fila, y la
genérica ganaba ⇒ «filete de mero» compraba el paquete genérico (RD$255) en vez de Mero por
libra (RD$290) o Tilapia por libra (RD$130), ambas con presentación en el súper. Migración
SSOT `p2_white_fish_alias_split_2026_09_02.sql` (ambos dirs, aplicada por el libro).

Tooltip-anchor: P2-WHITE-FISH-ALIAS-SPLIT | array_remove(array_remove(aliases, 'mero'), 'tilapia')
"""
from pathlib import Path

import pytest

import shopping_calculator
from shopping_calculator import normalize_name

BACKEND = Path(__file__).resolve().parents[1]
MIG = "p2_white_fish_alias_split_2026_09_02.sql"

CATALOGO = [
    {"name": "Filete de pescado blanco", "category": "Proteínas",
     "aliases": ["pescado blanco", "filete de pescado", "chillo", "pescado fresco", "pescado"]},
    {"name": "Mero", "category": "Proteínas", "aliases": ["mero", "grouper", "filete de mero", "mero fresco"]},
    {"name": "Tilapia", "category": "Proteínas", "aliases": ["tilapia", "filete de tilapia", "mojarra", "tilapia fresca"]},
    {"name": "Salmón", "category": "Proteínas", "aliases": ["salmon", "filete de salmon"]},
]


@pytest.fixture
def catalogo(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: CATALOGO)


@pytest.mark.parametrize("raw, esperado", [
    ("filete de mero", "Mero"),
    ("mero fresco", "Mero"),
    ("tilapia", "Tilapia"),
    ("filete de tilapia", "Tilapia"),
    ("pescado", "Filete de pescado blanco"),
    ("filete de pescado", "Filete de pescado blanco"),
    ("chillo", "Filete de pescado blanco"),
    ("filete de salmon", "Salmón"),
])
def test_species_resolve_to_their_own_row(catalogo, raw, esperado):
    assert normalize_name(raw) == esperado


def test_migration_present_in_both_ssot_dirs_and_idempotent():
    b = BACKEND / "migrations" / MIG
    assert b.exists(), "falta en backend/migrations"
    sql = b.read_text(encoding="utf-8")
    assert "array_remove(array_remove(aliases, 'mero'), 'tilapia')" in sql
    assert "'mero' = ANY(aliases) OR 'tilapia' = ANY(aliases)" in sql, "idempotente: solo si siguen ahí"
    assert "RAISE EXCEPTION" in sql, "sanity DO $$"
    root = BACKEND.parent / "migrations" / MIG
    if root.exists():
        assert root.read_text(encoding="utf-8").replace("\r\n", "\n") == sql.replace("\r\n", "\n")
