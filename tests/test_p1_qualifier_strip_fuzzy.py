"""[P1-QUALIFIER-STRIP-FUZZY + P1-CASABE-NO-BOIL + P1-OFFCATALOG-TOP-OFFENDERS · 2026-07-30]

Tres cierres de la revisión de calidad del owner (plan de 30 días, 2026-07-30):

1. 'Nísperos sin semilla' quedaba FUERA de la lista de compras (18 drops en 40 min de journal)
   mientras 'Nísperos' → 'Níspero' resolvía de sobra: el calificativo tiraba el ratio fuzzy bajo
   0.87. Un calificativo que no se compra por separado no debería poder volver incomprable el
   alimento. Fix: forma adicional sin calificativo NEGATIVO ("sin X", "bajo en X", "libre de X")
   en el pool fuzzy — el umbral 0.87 sigue mandando. "con X" queda fuera a propósito ("yogurt con
   fresas" nombra OTRO producto).

2. Una receta real instruyó "Cocina Casabe en 1½ tazas de agua con sal, tapa y hierve 15 minutos":
   la plantilla de cocción de granos aplicada a una torta seca de yuca ya cocida. Regla de prompt.

3. Los 4 off-catálogo medidos (72 warnings en 40 min): Romero (21), Vino jerez (21), Tortillas de
   maíz (9), Menta (3) — ahora nombrados explícitamente en la regla 5 con su sustituto verificado.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_DG = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")


# ─────────────── 1. el calificativo no vuelve incomprable el alimento ───────────────

def test_nisperos_sin_semilla_resuelve():
    from shopping_calculator import normalize_name
    assert normalize_name("Nísperos sin semilla") == "Níspero"
    assert normalize_name("Nísperos  sin semilla") == "Níspero"      # doble espacio del caso vivo


def test_la_forma_sin_calificativo_entra_al_pool_fuzzy():
    i = _SC.index("P1-QUALIFIER-STRIP-FUZZY")
    seg = _SC[i:i + 1400]
    assert "_noqual" in seg and "_noqual)" in _SC[i:i + 2000], (
        "la forma sin calificativo debe entrar a _fuzz_forms")
    assert r"\bsin\s+" in seg
    assert "con" not in seg.split("re.sub(")[1][:80] or "sin|con" not in seg, (
        "'con X' nombra OTRO producto — no puede recortarse")


def test_solo_toca_el_tier_fuzzy():
    """El recorte NO puede aplicarse antes (tiers exactos/regex) — ahí un match parcial agresivo
    cambiaría resoluciones que hoy son correctas. Solo amplía el pool del fuzzy con umbral 0.87."""
    i_fuzz = _SC.index("INTENTO 5")
    i_qual = _SC.index("P1-QUALIFIER-STRIP-FUZZY")
    assert i_qual > i_fuzz, "el strip de calificativos vive dentro del tier fuzzy, no antes"


# ─────────────── 2. el casabe no se hierve ───────────────

def test_regla_casabe_en_el_prompt():
    i = _DG.index("P1-CASABE-NO-BOIL")
    seg = _DG[i:i + 700]
    for frag in ("JAMÁS se hierve", "torta seca de yuca", "plantilla de cocción de granos"):
        assert frag in seg, f"la regla del casabe perdió {frag!r}"
    assert "pan, tostadas, galletas y tortillas" in seg, (
        "la regla debe cubrir la CLASE (alimentos ya cocidos), no solo el casabe")


# ─────────────── 3. los 4 ofensores medidos, nombrados con sustituto ───────────────

def test_ofensores_medidos_en_la_regla_5():
    i = _DG.index("P1-OFFCATALOG-TOP-OFFENDERS")
    seg = _DG[i:i + 700]
    for ofensor in ("ROMERO", "MENTA", "VINO DE JEREZ", "TORTILLAS DE MAÍZ"):
        assert ofensor in seg, f"falta el ofensor medido {ofensor!r}"
    # cada uno con alternativa del catálogo, no solo la prohibición
    for alt in ("orégano", "cilantro", "vinagre blanco", "tortilla de trigo"):
        assert alt in seg, f"falta el sustituto verificado {alt!r}"
