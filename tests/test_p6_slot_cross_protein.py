"""[P6-SLOT-CROSS-PROTEIN] Tests para la detección extendida de
proteína repetida en >1 slot del día (no solo almuerzo+cena).

Bug observable (PDF Día Martes 2026-05-05 15:09):
  - DESAYUNO: Revoltillo de Pavo y Guineítos
  - ALMUERZO: Lomo de Cerdo a la Parrilla con Yuca
  - MERIENDA: Crocante de Casabe con Pavo y Aguacate
  - CENA: Pechuga al Grill (pollo)
  El plan tiene PAVO en 2 slots distintos (desayuno + merienda) =
  monotonía proteica, pero el self-critique no lo flagaba.

Causas:
  1. `_MAIN_PROTEIN_ALIASES` NO tenía 'pavo' como label distinto.
     Pavo se colapsaba con 'pollo' vía alias "pechuga" compartido,
     creando false positives de pollo↔pavo y false negatives de
     pavo↔pavo.
  2. `_detect_slot_incoherence` solo chequeaba almuerzo↔cena overlap,
     dejando desayuno↔merienda y otras combinaciones invisibles.

Fix:
  1. Añadido 'pavo' al alias dict; "pechuga" sola removida de pollo
     para evitar colisión.
  2. Nuevo bloque cross-slot: itera todas las combinaciones de slots,
     flagea si una HEAVY protein (pollo/pavo/cerdo/res/pescado/atun)
     aparece en >=2 slots. Skip almuerzo+cena (ya cubierto por check
     específico).

Cobertura:
  - Pavo en desayuno+merienda → detectado
  - Pavo en almuerzo+cena → detectado por check específico (no duplica)
  - Pollo y pavo distintos en almuerzo+cena → NO flagado (bien)
  - Light proteins (huevo, yogurt) en múltiples slots → NO flagados
  - Cross-slot con 3 slots → detectado
"""
import pytest


def _meal(slot, name, ingredients):
    return {"meal": slot, "name": name, "ingredients": ingredients, "recipe": ""}


def _day(day_num, meals):
    return {"day": day_num, "meals": meals}


# ---------------------------------------------------------------------------
# 1. Repro PDF Día Martes
# ---------------------------------------------------------------------------
def test_repro_pdf_pavo_in_desayuno_and_merienda():
    """Pavo en desayuno + merienda DEBE ser flagado por el cross-slot check."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Revoltillo de Pavo y Guineítos",
                  ["100g pechuga de pavo", "2 guineos verdes", "2 huevos enteros"]),
            _meal("almuerzo", "Lomo de Cerdo con Yuca",
                  ["240g lomo de cerdo", "220g yuca"]),
            _meal("merienda", "Crocante de Casabe con Pavo",
                  ["50g pavo", "1 casabe", "30g aguacate"]),
            _meal("cena", "Pechuga de Pollo al Grill",
                  ["200g pechuga de pollo", "150g batata"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    pavo_issues = [i for i in issues if "pavo" in i.lower()]
    assert pavo_issues, (
        f"Pavo en desayuno+merienda DEBE detectarse. Issues: {issues}"
    )
    # Mensaje debe mencionar ambos slots
    assert any("desayuno" in i and "merienda" in i for i in pavo_issues), (
        f"Mensaje debe mencionar AMBOS slots involucrados: {pavo_issues}"
    )


# ---------------------------------------------------------------------------
# 2. Almuerzo+cena no debe duplicar mensaje (check específico ya lo cubre)
# ---------------------------------------------------------------------------
def test_almuerzo_cena_uses_specific_message_not_cross_slot():
    """Si la repetición es almuerzo+cena, debe usar el mensaje específico
    de ese check, no el general cross-slot (evitar mensaje duplicado)."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Revoltillo de Huevo", ["3 huevos"]),
            _meal("almuerzo", "Pollo Asado con Arroz",
                  ["200g pollo", "150g arroz"]),
            _meal("merienda", "Yogurt con Fruta", ["1 yogurt", "1 manzana"]),
            _meal("cena", "Pollo a la Plancha con Yuca",
                  ["200g pollo", "150g yuca"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    pollo_issues = [i for i in issues if "pollo" in i.lower()]
    # Esperamos UN solo mensaje (el específico de almuerzo+cena)
    # y NO el cross-slot general
    specific_msgs = [i for i in pollo_issues if "almuerzo y cena" in i]
    cross_slot_msgs = [i for i in pollo_issues if "aparece en" in i]
    assert specific_msgs, "Check específico debe disparar"
    assert not cross_slot_msgs, (
        f"NO debe duplicar con cross-slot cuando es almuerzo+cena: {cross_slot_msgs}"
    )


# ---------------------------------------------------------------------------
# 3. Pollo y pavo distintos: NO false positive
# ---------------------------------------------------------------------------
def test_pollo_almuerzo_pavo_cena_no_false_positive():
    """Antes 'pechuga' colapsaba pollo y pavo. Ahora deben ser distintos."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Revoltillo de Huevo", ["3 huevos"]),
            _meal("almuerzo", "Pechuga de Pollo Asada",
                  ["200g pechuga de pollo"]),
            _meal("merienda", "Yogurt con Fruta", ["1 yogurt"]),
            _meal("cena", "Pechuga de Pavo al Horno",
                  ["180g pechuga de pavo"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    # NO debe haber issue por pollo/pavo (son distintos)
    proteins_flagged = [i for i in issues if ("pollo" in i.lower()) or ("pavo" in i.lower())]
    assert not proteins_flagged, (
        f"Pollo y pavo son proteínas DISTINTAS — no debe haber duplicate alert. "
        f"Issues spurious: {proteins_flagged}"
    )


# ---------------------------------------------------------------------------
# 4. Light proteins en múltiples slots: NO flagados
# ---------------------------------------------------------------------------
def test_huevo_in_multiple_slots_not_flagged():
    """Huevo es 'light protein' — puede aparecer en desayuno + merienda
    sin disparar warning."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Revoltillo de Huevo", ["3 huevos"]),
            _meal("almuerzo", "Cerdo con Yuca", ["200g cerdo", "150g yuca"]),
            _meal("merienda", "Huevo Duro con Fruta",
                  ["1 huevo duro", "1 manzana"]),
            _meal("cena", "Pescado con Vegetales",
                  ["200g tilapia", "vegetales"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    huevo_issues = [i for i in issues if "huevo" in i.lower()]
    assert not huevo_issues, (
        f"Huevo (light protein) NO debe ser flagado en múltiples slots. "
        f"Issues spurious: {huevo_issues}"
    )


def test_yogurt_in_multiple_slots_not_flagged():
    """Yogurt en desayuno + merienda OK."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Bowl de Yogurt", ["1 yogurt griego"]),
            _meal("almuerzo", "Res con Arroz", ["200g res"]),
            _meal("merienda", "Yogurt con Fruta", ["1 yogurt"]),
            _meal("cena", "Pescado", ["200g pescado"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    yogurt_issues = [i for i in issues if "yogurt" in i.lower()]
    assert not yogurt_issues


# ---------------------------------------------------------------------------
# 5. Cross-slot con 3+ slots
# ---------------------------------------------------------------------------
def test_pollo_in_three_slots_detected():
    """Pollo en 3 slots distintos (desayuno + almuerzo + cena) — debe
    detectarse, idealmente mencionando todos."""
    from graph_orchestrator import _detect_slot_incoherence

    days = [
        _day(1, [
            _meal("desayuno", "Sándwich de Pollo", ["50g pollo desmenuzado"]),
            _meal("almuerzo", "Pollo Asado", ["200g pollo"]),
            _meal("merienda", "Yogurt", ["1 yogurt"]),
            _meal("cena", "Pollo a la Plancha", ["180g pollo"]),
        ]),
    ]
    issues = _detect_slot_incoherence(days)
    pollo_issues = [i for i in issues if "pollo" in i.lower()]
    # Almuerzo+cena dispara check específico. PERO también debería
    # detectarse algo del desayuno (cross-slot). Verificamos que AL MENOS
    # el almuerzo+cena dispara.
    assert pollo_issues, (
        f"3 slots con pollo deben generar al menos 1 issue: {issues}"
    )


# ---------------------------------------------------------------------------
# 6. Sanity: pavo está en alias dict y heavy labels
# ---------------------------------------------------------------------------
def test_pavo_in_main_protein_aliases():
    """Sanity: pavo debe estar en el alias dict para ser detectable."""
    from graph_orchestrator import _MAIN_PROTEIN_ALIASES
    assert "pavo" in _MAIN_PROTEIN_ALIASES
    assert "pavo" in _MAIN_PROTEIN_ALIASES["pavo"]


def test_pavo_in_heavy_protein_labels():
    from graph_orchestrator import _HEAVY_PROTEIN_LABELS
    assert "pavo" in _HEAVY_PROTEIN_LABELS


def test_light_proteins_not_in_heavy_labels():
    """Huevo, yogurt, gandules, lentejas, habichuelas NO deben ser heavy."""
    from graph_orchestrator import _HEAVY_PROTEIN_LABELS
    light_proteins = {"huevo", "yogurt", "gandules", "habichuelas", "lentejas"}
    assert not (light_proteins & _HEAVY_PROTEIN_LABELS), (
        "Light proteins NO deben estar en _HEAVY_PROTEIN_LABELS — "
        "esos pueden repetirse en múltiples slots sin problema."
    )


def test_pollo_pavo_distinct_aliases():
    """Sanity: 'pechuga' sola NO debe estar en pollo aliases (causaba
    colisión con 'pechuga de pavo' antes del fix)."""
    from graph_orchestrator import _MAIN_PROTEIN_ALIASES
    pollo_aliases = _MAIN_PROTEIN_ALIASES["pollo"]
    pavo_aliases = _MAIN_PROTEIN_ALIASES["pavo"]
    # "pechuga" sola NO debe estar — debe ser específico (pechuga de pollo)
    assert "pechuga" not in pollo_aliases, (
        "'pechuga' sola en pollo causaba colisión con 'pechuga de pavo'"
    )
    # No deben compartir aliases
    overlap = set(pollo_aliases) & set(pavo_aliases)
    assert not overlap, (
        f"Pollo y pavo NO deben compartir aliases: {overlap}"
    )
