"""[P2-BLANCH-INGREDIENT-TRUTH + P2-SEGUNDOS-ES-TIEMPO + P2-AUTOFIX-NOTE-EXEMPT
+ tokens blend/girasol · 2026-07-06] Review del plan renovado cd4ae3c3.

Los 4 hallazgos del review visual del owner sobre el plan recién generado:
1. Ceviche: "Blanquea el camaron..." con CERO camarón en el plato (el autofix
   renombró camarón→pescado en pasos, pero el inyector de blanqueo usaba el
   token del TEXTO) + description vendiendo "Camarones cocidos".
2. El mismo ceviche: paso con "60 a 90 segundos" recibía la cola "(~12-15 min
   en agua hirviendo)" — "segundos" no contaba como tiempo para el backstop.
3. Batido: la nota de seguridad quedó "se reemplazó el proteína crudo" — el
   replacer de menciones huérfanas reescribía DENTRO de notas deterministas.
4. Lista: "Aceite de oliva → Blend Girasol 750 Ml" — una mezcla con girasol no
   es aceite de oliva (MUFA distinto).
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


# ───────────── 1. blanch usa el ingrediente real ─────────────

def test_blanch_names_actual_ingredient_not_text_token(monkeypatch):
    monkeypatch.setattr(go, "SEAFOOD_MARINADE_BLANCH_ENABLED", True)
    meal = {
        "name": "Ceviche Fresco de Filete de pescado blanco",
        "ingredients": ["1 filete de pescado", "2 limones", "½ cebolla roja"],
        "recipe": [
            "Mise en place: pica la cebolla y exprime los limones.",
            "Montaje: mezcla el camaron con el jugo de limón y deja marinar 10 minutos.",
        ],
    }
    changed = go._inject_blanch_for_citrus_marinade(meal)
    assert changed is True
    blob = " ".join(str(s) for s in meal["recipe"])
    assert "Blanquea" in blob
    _blanch_step = next(s for s in meal["recipe"] if "Blanquea" in str(s))
    assert "filete de pescado" in _blanch_step.lower(), (
        f"el blanqueo nombra el INGREDIENTE real, no el token huérfano del texto: {_blanch_step}"
    )
    assert "camaron" not in _blanch_step.lower().split("blanquea")[1].split(" en agua")[0], (
        f"cero camarón fantasma en el nombre del blanqueo: {_blanch_step}"
    )


def test_blanch_still_uses_backed_token_when_ingredient_matches(monkeypatch):
    monkeypatch.setattr(go, "SEAFOOD_MARINADE_BLANCH_ENABLED", True)
    meal = {
        "name": "Ceviche de camarón",
        "ingredients": ["200 g de camarones frescos", "2 limones"],
        "recipe": ["Montaje: marina el camarón en el jugo de limón 10 minutos."],
    }
    assert go._inject_blanch_for_citrus_marinade(meal) is True
    _step = next(s for s in meal["recipe"] if "Blanquea" in str(s))
    assert "camaron" in _step.lower(), f"con camarón REAL en el plato, se nombra camarón: {_step}"


# ───────────── 2. segundos = tiempo ─────────────

def test_segundos_counts_as_time_no_contradictory_tail():
    step = ("El Toque de Fuego: añade el filete al agua hirviendo y cocínalo "
            "por 60 a 90 segundos exactos hasta que esté opaco.")
    assert go._CONTRACT_TIME_RE.search(step), "'60 a 90 segundos' ES un tiempo"
    meal = {"name": "Ceviche", "recipe": ["Mise en place: pica.", step, "Montaje: sirve."]}
    changed = go._inject_recipe_time_temp_defaults(meal)
    assert "12-15 min" not in " ".join(str(s) for s in meal["recipe"]), (
        "jamás apendear '(~12-15 min en agua hirviendo)' a un paso que ya dice 60-90 segundos"
    )
    assert changed is False or "segundos" in " ".join(str(s) for s in meal["recipe"])


# ───────────── 3. replacer exento en notas + description ─────────────

def test_autofix_note_exempt_and_description_anchors():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("P2-AUTOFIX-NOTE-EXEMPT")
    win = src[i:i + 2600]
    assert "seguridad alimentaria" in win and "se reemplazo" in win, (
        "las notas deterministas citan el alimento ORIGINAL a propósito — exentas del replace"
    )
    assert "_is_det_note" in win and "(?:es|s)?" in win, (
        "patrón tolerante a plural ('Camarones') y acento ('camaron' sin tilde)"
    )
    assert "P2-AUTOFIX-DESC" in src and 'meal.get("description")' in win, (
        "la description también se reescribe (vendía 'Camarones cocidos' sin camarón)"
    )


# ───────────── 4. blend/girasol fuera del default de aceite de oliva ─────────────

def test_oil_blend_dropped_from_defaults():
    defaults = {"aceite de oliva": [
        {"grams": 750.0, "price": 199.0, "label": "Blend Girasol 750 Ml · Wala", "unit": "botella"},
        {"grams": 500.0, "price": 425.0, "label": "Virgen Extra 500 Ml · Borges", "unit": "botella"},
    ]}
    got = sc._resolve_brand_default("Aceite de oliva", defaults)
    assert got is not None and len(got) == 1 and "Girasol" not in got[0]["label"], (
        "una MEZCLA con girasol no es aceite de oliva — fuera del default"
    )


def test_girasol_allowed_when_in_item_name():
    defaults = {"semillas de girasol": [
        {"grams": 200.0, "price": 84.95, "label": "200 gr · BioEva", "unit": "funda"},
    ]}
    assert sc._resolve_brand_default("Semillas de girasol", defaults), (
        "'girasol' está en el nombre del ítem → legítimo"
    )
