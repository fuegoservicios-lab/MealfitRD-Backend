"""[P1-FLOURS-POOLS · 2026-07-01] (audit creatividad G2)

Las HARINAS y el maíz verificados no rotaban en `DOMINICAN_CARBS` → el ejemplo flagship del owner
("con la harina haces panqueques, bollos, arepas") casi no podía ocurrir como plato principal
fuera del desayuno, pese a que la regla 2.5 de creatividad lo promueve.

Fix: harinas/maíz/tortilla en el pool + backfill del synonym system (los items del pool que no
resolvían ni como key ni como variant eran invisibles para variedad/fatiga/coherencia). El test
de contrato pool↔synonyms vive en test_synonyms (criterio key-O-variant).
"""
from __future__ import annotations

from constants import (DOMINICAN_CARBS, CARB_SYNONYMS, DOMINICAN_PROTEINS, PROTEIN_SYNONYMS,
                       DOMINICAN_VEGGIES_FATS, VEGGIE_FAT_SYNONYMS, DOMINICAN_FRUITS, FRUIT_SYNONYMS)

_NEW_POOL_ITEMS = ("Harina de trigo", "Harina de maíz precocida", "Maíz dulce en granos", "Tortilla de trigo")


def test_flours_in_carb_pool():
    for item in _NEW_POOL_ITEMS:
        assert item in DOMINICAN_CARBS, f"'{item}' debe rotar en DOMINICAN_CARBS (P1-FLOURS-POOLS)"


def test_flours_resolve_in_synonym_system():
    known = {k.lower() for k in CARB_SYNONYMS}
    for variants in CARB_SYNONYMS.values():
        known.update(str(v).lower() for v in variants)
    for item in _NEW_POOL_ITEMS:
        assert item.lower() in known, f"'{item}' no resuelve en CARB_SYNONYMS"


def test_no_bare_harina_or_maiz_variant():
    """Lección P1-NUT-BUTTER-DISTINCT / P1-PREP-COLLAPSE-GUARD: 'harina' y 'maíz' a secas como variant
    colapsarían preparaciones ('harina de avena') a la base equivocada en el tracking."""
    for base, variants in CARB_SYNONYMS.items():
        for v in variants:
            vl = str(v).lower().strip()
            assert vl != "harina", f"variant 'harina' a secas en base '{base}' — colapso de preparaciones"
            assert vl not in ("maíz", "maiz"), f"variant 'maíz' a secas en base '{base}'"


def test_all_pools_resolve_key_or_variant():
    """Cero items de pool desconocidos para el synonym system (invariante P1-FLOURS-POOLS,
    espejo del criterio actualizado de test_synonyms)."""
    # [2026-07-26] El `assert` estaba DENTRO del bucle, así que solo reportaba el primer pool con
    # huecos y escondía los demás: al cerrar 'Granola' y 'Galletas de soda' (CARBS) apareció
    # 'Granada' (FRUITS), que llevaba ahí todo el tiempo. Se acumula y se reporta de una vez —
    # si no, cada arreglo destapa el siguiente y parecen fallos nuevos.
    faltantes: dict[str, list[str]] = {}
    for etiqueta, lst, syn in (
        ("CARBS", DOMINICAN_CARBS, CARB_SYNONYMS),
        ("PROTEINS", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS),
        ("VEGGIES_FATS", DOMINICAN_VEGGIES_FATS, VEGGIE_FAT_SYNONYMS),
        ("FRUITS", DOMINICAN_FRUITS, FRUIT_SYNONYMS),
    ):
        known = {k.lower() for k in syn}
        for variants in syn.values():
            known.update(str(v).lower() for v in variants)
        missing = [i for i in lst if i.lower() not in known]
        if missing:
            faltantes[etiqueta] = missing
    assert not faltantes, (
        f"items de pool sin entrada en su synonym map: {faltantes}. Sin entrada, "
        "`normalize_ingredient_for_tracking` NO colapsa sus variantes y el seguimiento de "
        "frecuencia cuenta 'granola', 'granola sin azúcar' y 'granola con miel' como tres "
        "alimentos distintos — diluye la señal que alimenta la personalización."
    )
