"""[P1-PAVO-CATALOG · 2026-07-12] 'Pechuga de pavo' entra al catálogo verificado.

Forense del plan vivo df263d1b (06:36Z): el plato "Pavo Desmenuzado Asado" llevaba
`Pechuga de pavo` en ingredients_raw pero el catálogo solo tenía 'Jamón de pavo' y
'Pavo molido' → `VERIFIED-ONLY-DROP` excluía la proteína del plato de la lista de
compras (fail-safe correcto, lista incompleta). Cierre: seed one-food con el pipeline
de los add_foods_batch (USDA SR Legacy 174515, crudo, Atwater 0 flags, precio estimado
declarado). Este test ancla la INTEGRIDAD del seed — no toca la DB.
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_SEED = os.path.join(_BACKEND, "scripts", "seed_pechuga_pavo_2026_07_12.py")

with open(_SEED, encoding="utf-8") as f:
    _SRC = f.read()


def _record():
    ns: dict = {}
    m = re.search(r"RECORD = \{.*?\n\}", _SRC, re.DOTALL)
    assert m, "bloque RECORD no encontrado en el seed"
    exec(m.group(0), {"__builtins__": {}}, ns)  # literal puro, sin imports
    return ns["RECORD"]


def test_seed_exists_and_gated():
    assert os.path.exists(_SEED)
    assert '"--commit" in sys.argv' in _SRC, "dry-run por default (gate --commit)"
    assert "WHERE name = %s" in _SRC, "idempotente (salta si ya existe)"


def test_atwater_consistency():
    r = _record()
    atwater = r["protein_g_per_100g"] * 4 + r["carbs_g_per_100g"] * 4 + r["fats_g_per_100g"] * 9
    assert abs(atwater - r["kcal_per_100g"]) / r["kcal_per_100g"] < 0.05, \
        f"Atwater {atwater:.1f} vs label {r['kcal_per_100g']} — dato USDA corrupto"


def test_usda_provenance_and_conventions():
    r = _record()
    assert r["fdc_id"] == 174515, "USDA SR Legacy 'Turkey, retail parts, breast, meat only, raw'"
    assert r["category"] == "Proteínas" and r["default_unit"] == "lb", \
        "convención de carnes del catálogo (como Pechuga de pollo)"
    assert r["price_per_lb"] and r["price_per_lb"] > 0, \
        "gate anti-precio-0: sin precio no pasa el filtro de verificados"
    assert r["sodium_mg_per_100g"] < 200, "es carne FRESCA (no curada — eso es Jamón de pavo)"


def test_no_bare_pavo_alias():
    r = _record()
    assert "pavo" not in [a.strip().lower() for a in r["aliases"]], \
        ("'pavo' a secas ya es alias de 'Jamón de pavo' — duplicarlo rompería ese mapping "
         "y el canonicalize_pavo del coherence guard")


def test_folate_omega3_usda_values():
    """El schema exige folate NOT NULL — valores reales del SR 174515 (no 0 inventado)."""
    r = _record()
    assert r["folate_mcg_dfe_per_100g"] == 7, "SR 'Folate, total' = 7 µg (sin fortificar: total≡DFE)"
    assert r["omega3_ala_g_per_100g"] == 0.015, "SR 'PUFA 18:3 n-3 ALA' = 0.015 g"
