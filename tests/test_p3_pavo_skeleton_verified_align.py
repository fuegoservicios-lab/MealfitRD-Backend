"""[P3-PAVO-SKELETON-VERIFIED-ALIGN · 2026-06-22] El pool de proteínas del esqueleto
(`DOMINICAN_PROTEINS`) debe ofrecer SOLO pavo verificado/comprable.

Bug observado en vivo (corr=4a8d46e1, 2026-06-22): el pool ofrecía "Pavo" (fresco genérico) pero el único
pavo con precio en el catálogo es "Jamón de pavo" → el LLM escribía "Pechuga de pavo" en la receta y la
lista de compras lo dropeaba (no verificado) → receta dice pavo, lista no lo trae (incoherencia).

Fix: "Pavo" → "Jamón de pavo" en el pool. `normalize_name` sigue distinguiendo fresh/procesado (capa de
shopping intacta); este test solo ancla el contenido del pool del esqueleto.
"""
from __future__ import annotations

import constants as C


def test_pool_offers_verified_turkey_only():
    proteins = C.DOMINICAN_PROTEINS
    assert "Jamón de pavo" in proteins, "el pool debe ofrecer el pavo verificado/comprable (Jamón de pavo)"
    # El "Pavo" fresco genérico (no verificado) NO debe estar como entry del pool.
    assert "Pavo" not in proteins, (
        "‘Pavo’ (fresco genérico) no está en el catálogo verificado → no debe ofrecerse en el pool; "
        "usar ‘Jamón de pavo’ (el único pavo con precio)."
    )


def test_normalize_name_resolves_jamon_de_pavo():
    # El nombre del pool resuelve al canónico verificado (no se vuelve a dropear).
    from shopping_calculator import normalize_name
    assert normalize_name("Jamón de pavo") == "Jamón de pavo"
    assert normalize_name("jamón de pavo en lonjas") == "Jamón de pavo"
    # Y la distinción fresh/procesado de normalize_name sigue intacta (no regresión de P3-PROTEIN-CAP-2).
    assert normalize_name("pechuga de pavo") == "Pechuga de pavo"
