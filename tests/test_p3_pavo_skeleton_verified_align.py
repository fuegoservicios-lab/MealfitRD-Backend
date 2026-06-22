"""[P3-PAVO-SKELETON-VERIFIED-ALIGN · 2026-06-22] El pool de proteínas del esqueleto
(`DOMINICAN_PROTEINS`) debe ofrecer pavo VERIFICADO/comprable.

Bug observado en vivo (corr=4a8d46e1, 2026-06-22): el pool ofrecía "Pavo" (fresco genérico, NO verificado)
→ el LLM escribía "Pechuga de pavo" en la receta y la lista de compras lo dropeaba (no priced) → receta
dice pavo, lista no lo trae (incoherencia).

Fix (rev2): el owner agregó "Pechuga de pavo" al catálogo (RD$415/lb, owner_verified) — el pavo que la
regla 69 del day-gen YA prefería sobre el jamón en lonjas. El pool ahora ofrece ese nombre verificado
exacto. `normalize_name` sigue distinguiendo fresh/procesado (capa de shopping intacta).
"""
from __future__ import annotations

import constants as C


def test_pool_offers_verified_turkey_only():
    proteins = C.DOMINICAN_PROTEINS
    assert "Pechuga de pavo" in proteins, "el pool debe ofrecer el pavo verificado (Pechuga de pavo, RD$415)"
    # El "Pavo" fresco genérico (no verificado) NO debe estar como entry del pool.
    assert "Pavo" not in proteins, (
        "‘Pavo’ (fresco genérico) no resuelve a un priced → no debe ofrecerse; usar ‘Pechuga de pavo’."
    )


def test_normalize_name_resolves_turkey_breast_and_keeps_distinction():
    from shopping_calculator import normalize_name
    # El nombre del pool resuelve a su canónico priced (no se vuelve a dropear).
    assert normalize_name("pechuga de pavo") == "Pechuga de pavo"
    assert normalize_name("Pechuga de pavo") == "Pechuga de pavo"
    # La distinción fresh/procesado de normalize_name sigue intacta (no regresión P3-PROTEIN-CAP-2):
    # el jamón en lonjas NO se confunde con la pechuga.
    assert normalize_name("jamón de pavo en lonjas") == "Jamón de pavo"
