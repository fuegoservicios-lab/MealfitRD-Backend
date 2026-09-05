"""[P1-PANTRY-NAME-RESOLUTION · 2026-08-07] La nevera descontaba fantasmas.

Incidente que ancla este archivo (reproducido contra la nevera real del dueño,
43 items, la fila dice "Huevo"):

    coach registra "2 huevos"
      → _parse_quantity → name="Huevos"
      → SELECT ... WHERE ingredient_name = 'Huevos'  → 0 filas
      → quantity = -2 → no entra al INSERT (exige >= 0.01)
                      → no entra al guard de unidad (exige existing_rows)
      → return True                                  ← MIENTE

Ni descontaba, ni escribía en `failed_inventory_deductions`, ni alertaba, y le
devolvía éxito al caller. Los tres sitios que resolvían filas de nevera tenían
su propia copia del `WHERE ingredient_name = %s` exacto.

Este archivo cubre las dos mitades del fix:
  1. La escalera de resolución acierta el plural SIN colapsar sinónimos.
  2. Lo que de verdad no está en la nevera sale como `not_in_pantry`, NO como
     `succeeded`.

Y ancla los tooltip-anchors para que un refactor que quite el resolver falle
acá antes que en producción.
"""
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from constants import (
    canonical_pantry_key,
    pantry_names_match,
    normalize_ingredient_for_tracking,
)

_BACKEND = Path(__file__).resolve().parent.parent
_DB_INVENTORY = _BACKEND / "db_inventory.py"
_CONSTANTS = _BACKEND / "constants.py"
_TOOLS = _BACKEND / "tools.py"


# ---------------------------------------------------------------------------
# 1. El matcher: plurales SÍ, sinónimos NO
# ---------------------------------------------------------------------------

# El caso literal del incidente + variantes ortográficas del MISMO alimento.
_SHOULD_MATCH = [
    ("Huevo", "2 huevos"),        # el incidente exacto (cantidad + plural)
    ("Huevo", "Huevos"),
    ("Huevo", "huevo"),           # solo case
    ("Plátano verde", "platano verde"),   # solo acentos
    ("Plátano verde", "platanos verdes"), # acentos + plural multipalabra
    ("Limón", "limones"),         # -ones → -ón
    ("Frijol", "frijoles"),       # consonante + es
    ("Arroz", "arroces"),         # -ces → -z
    ("Nuez", "nueces"),
    ("Carne", "carnes"),          # vocal + s (la ambigüedad -nes resuelta)
    ("Res", "Reses"),             # plural legítimo de palabra corta
    ("Habichuela", "habichuelas"),
    ("Cebolla roja", "Cebollas rojas"),
    ("Queso blanco", "  queso   blanco  "),  # espacios colapsados
    # [P1-PANTRY-KEY-VULGAR-FRACTIONS · 2026-09-03] las recetas escriben fracciones UNICODE.
    ("Yogurt", "⅓ taza de yogurt"),
    ("Mantequilla de maní", "¾ cucharada de mantequilla de maní"),
    ("Avena", "1½ tazas de avena"),
    ("Kiwi", "½ kiwi"),
]

# Alimentos DISTINTOS que jamás deben resolver a la misma fila física.
_SHOULD_NOT_MATCH = [
    # Sinónimos nutricionales: `GLOBAL_REVERSE_MAP` los colapsa a propósito
    # para tracking/coherencia. Para identidad de fila serían corrupción.
    ("Pechuga de pollo", "Muslo de pollo"),
    ("Pechuga de pollo", "Pollo"),
    ("Lomo de cerdo", "Cerdo"),
    ("Carne molida", "Res"),
    # Subcadenas: la familia de bugs que cita P1-SWAP-PANTRY-PLURAL.
    ("Pollo", "Repollo"),
    ("Sal", "Salsa"),
    ("Res", "Fresco"),
    ("Res", "Fresa"),
    ("Pan", "Pana"),
    ("Maíz", "Maicena"),
    ("Ajo", "Ajonjolí"),
    ("Coco", "Cocoa"),
    ("Aceite", "Aceituna"),
    # Modificadores: compras distintas, con precio y unidad distintos.
    ("Arroz integral", "Arroz"),
    ("Leche descremada", "Leche"),
    ("Leche de coco", "Leche"),
    # Alimentos que en RD son distintos aunque se confundan (P2-VISION-GUINEO-PLATANO).
    ("Guineo", "Plátano"),
]


@pytest.mark.parametrize("a,b", _SHOULD_MATCH)
def test_matches_same_food(a, b):
    assert pantry_names_match(a, b), (
        f"{a!r} y {b!r} son el MISMO alimento y deben resolver a la misma fila. "
        f"claves: {canonical_pantry_key(a)!r} / {canonical_pantry_key(b)!r}"
    )


@pytest.mark.parametrize("a,b", _SHOULD_NOT_MATCH)
def test_rejects_different_foods(a, b):
    assert not pantry_names_match(a, b), (
        f"{a!r} y {b!r} son alimentos DISTINTOS. Matchearlos descuenta del "
        f"alimento equivocado — un fallo silencioso peor que no descontar."
    )


def test_match_is_symmetric_and_reflexive():
    """Si el orden de los argumentos importara, el resultado dependería de
    quién llama — y los tres call sites llaman en órdenes distintos."""
    for a, b in _SHOULD_MATCH + _SHOULD_NOT_MATCH:
        assert pantry_names_match(a, b) == pantry_names_match(b, a), (
            f"asimetría en ({a!r}, {b!r})"
        )
        assert pantry_names_match(a, a)


def test_empty_names_never_match():
    """Dos filas sin nombre no son "la misma fila"."""
    for empty in ("", "   ", None):
        assert not pantry_names_match(empty, "Huevo")
        assert not pantry_names_match("Huevo", empty)
        assert not pantry_names_match(empty, empty)


def test_quantity_only_string_does_not_collapse_to_empty():
    """"200g" sin nombre no debe producir clave vacía: si lo hiciera, dos
    items distintos sin nombre matchearían entre sí."""
    assert canonical_pantry_key("200g") != ""
    assert not pantry_names_match("200g", "500ml")


def test_does_not_reuse_the_synonym_normalizer():
    """Guard de diseño, no de comportamiento.

    `normalize_ingredient_for_tracking` colapsa "pechuga"→"pollo". Si alguien
    "simplifica" el matcher delegando en él, este test cae: es exactamente el
    cambio que haría que comerte una pechuga descuente del muslo.
    """
    assert normalize_ingredient_for_tracking("Pechuga de pollo") == \
        normalize_ingredient_for_tracking("Muslo de pollo"), (
        "premisa del test rota: el mapa de sinónimos ya no colapsa estos dos. "
        "Reconfirma que el matcher de nevera sigue siendo independiente."
    )
    assert not pantry_names_match("Pechuga de pollo", "Muslo de pollo")


# ---------------------------------------------------------------------------
# 2. El resolver: escalera exact → canonical → none
# ---------------------------------------------------------------------------

def _fridge(*names):
    """Filas mínimas con la forma que devuelve `find_pantry_rows_for_name`."""
    return [
        {"id": i + 1, "ingredient_name": n, "quantity": 3.0, "unit": "unidad",
         "reserved_quantity": 0.0, "reservation_details": None}
        for i, n in enumerate(names)
    ]


def test_resolver_exact_match_short_circuits():
    import db_inventory
    rows = _fridge("Huevo")
    with patch.object(db_inventory, "_db_available", return_value=True):
        got, level = db_inventory.find_pantry_rows_for_name(
            "u1", "Huevo", prefetched_rows=rows)
    assert level == "exact"
    assert [r["ingredient_name"] for r in got] == ["Huevo"]


def test_resolver_canonical_match_closes_the_incident():
    """El caso literal: la nevera dice "Huevo", el coach emite "Huevos"."""
    import db_inventory
    rows = _fridge("Huevo", "Queso blanco", "Cerdo")
    with patch.object(db_inventory, "_db_available", return_value=True):
        got, level = db_inventory.find_pantry_rows_for_name(
            "u1", "Huevos", prefetched_rows=rows)
    assert level == "canonical"
    assert [r["ingredient_name"] for r in got] == ["Huevo"]


def test_resolver_reports_none_for_absent_food():
    import db_inventory
    rows = _fridge("Huevo", "Queso blanco")
    with patch.object(db_inventory, "_db_available", return_value=True):
        got, level = db_inventory.find_pantry_rows_for_name(
            "u1", "Pan integral", prefetched_rows=rows)
    assert level == "none"
    assert got == []


def test_resolver_never_crosses_to_a_different_food():
    import db_inventory
    rows = _fridge("Muslo de pollo", "Repollo", "Salsa de tomate")
    with patch.object(db_inventory, "_db_available", return_value=True):
        for probe in ("Pechuga de pollo", "Pollo", "Sal"):
            got, level = db_inventory.find_pantry_rows_for_name(
                "u1", probe, prefetched_rows=rows)
            assert level == "none", f"{probe!r} resolvió a {got!r}"


def test_resolver_is_deterministic_with_legacy_duplicates():
    """El bug histórico pudo dejar "Huevo" y "Huevos" como filas separadas.
    Elegir siempre la misma es lo que las va consolidando con el tiempo."""
    import db_inventory
    rows = _fridge("Huevos", "Huevo")
    with patch.object(db_inventory, "_db_available", return_value=True):
        first, _ = db_inventory.find_pantry_rows_for_name(
            "u1", "huevo", prefetched_rows=rows)
        second, _ = db_inventory.find_pantry_rows_for_name(
            "u1", "huevos", prefetched_rows=list(reversed(rows)))
    assert first[0]["ingredient_name"] == second[0]["ingredient_name"]


def test_knob_off_restores_exact_only_behaviour(monkeypatch):
    """`MEALFIT_PANTRY_CANONICAL_MATCH=false` es el rollback sin redeploy."""
    import db_inventory
    monkeypatch.setenv("MEALFIT_PANTRY_CANONICAL_MATCH", "false")
    import knobs
    knobs._KNOBS_REGISTRY.pop("MEALFIT_PANTRY_CANONICAL_MATCH", None)
    rows = _fridge("Huevo")
    with patch.object(db_inventory, "_db_available", return_value=True):
        got, level = db_inventory.find_pantry_rows_for_name(
            "u1", "Huevos", prefetched_rows=rows)
    assert level == "none" and got == []
    knobs._KNOBS_REGISTRY.pop("MEALFIT_PANTRY_CANONICAL_MATCH", None)


# ---------------------------------------------------------------------------
# 3. El deduct: "no lo tenías" deja de contarse como "descontado"
# ---------------------------------------------------------------------------

def test_absent_item_is_not_reported_as_succeeded():
    """El corazón del fix: pre-fix este item caía en `succeeded` y el resumen
    afirmaba haber descontado algo que nunca bajó."""
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "find_pantry_rows_for_name",
                      return_value=([], "none")), \
         patch.object(db_inventory, "add_or_update_inventory_item") as add_mock, \
         patch.object(db_inventory, "_persist_failed_inventory_deductions") as persist_mock:
        summary = db_inventory.deduct_consumed_meal_from_inventory("u1", ["1 pan integral"])

    assert summary["succeeded"] == []
    assert summary["not_in_pantry"] == ["1 pan integral"]
    assert summary["failed_to_deduct"] == []
    add_mock.assert_not_called(), "no hay fila que mutar — el UPDATE sobraba"
    # La cola de reintentos es para fallos REINTENTABLES. Un item ausente no
    # mejora reintentándolo: solo gasta ticks del cron y ensucia su alerta.
    assert persist_mock.call_args[0][1] == [], (
        "un item ausente no debe entrar a failed_inventory_deductions"
    )


def test_present_item_still_deducts_and_counts_as_succeeded():
    """Anti-regresión del happy path: el fix no debe frenar descuentos buenos."""
    import db_inventory
    rows = _fridge("Huevo")
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "find_pantry_rows_for_name",
                      return_value=(rows, "canonical")), \
         patch.object(db_inventory, "_consume_reserved_inventory", return_value=True), \
         patch.object(db_inventory, "add_or_update_inventory_item", return_value=True) as add_mock, \
         patch.object(db_inventory, "_persist_failed_inventory_deductions"):
        summary = db_inventory.deduct_consumed_meal_from_inventory("u1", ["2 huevos"])

    assert summary["succeeded"] == ["2 huevos"]
    assert summary["not_in_pantry"] == []
    # Debe restar, no sumar.
    assert add_mock.call_args[0][2] < 0
    assert add_mock.call_args[1]["mutation_type"] == "consumption"


def test_summary_keeps_legacy_keys():
    """`sync_inventory_after_chunk_completion` y `tools.log_consumed_meal` leen
    estas claves. Renombrarlas rompería la telemetría del cron en silencio."""
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "find_pantry_rows_for_name",
                      return_value=([], "none")), \
         patch.object(db_inventory, "_persist_failed_inventory_deductions"):
        summary = db_inventory.deduct_consumed_meal_from_inventory("u1", ["1 pan"])
    for key in ("succeeded", "inferred", "failed_to_deduct", "not_in_pantry"):
        assert key in summary and isinstance(summary[key], list)


# ---------------------------------------------------------------------------
# 4. Anclajes parser-based (que un refactor no revierta el fix en silencio)
# ---------------------------------------------------------------------------

def test_tooltip_anchors_alive():
    consts = _CONSTANTS.read_text(encoding="utf-8")
    inv = _DB_INVENTORY.read_text(encoding="utf-8")
    assert "P1-PANTRY-NAME-RESOLUTION-SSOT" in consts
    assert "P1-PANTRY-NAME-RESOLUTION-RESOLVER" in inv


def test_no_call_site_reads_pantry_rows_by_exact_string_again():
    """El bug era literalmente este SELECT, repetido en cuatro sitios.

    Se veta la forma de LECTURA. El `UPDATE user_inventory SET brand = ...
    WHERE ingredient_name = %s` queda fuera a propósito: corre DESPUÉS de que
    el resolver adoptó la ortografía de la nevera, así que ya apunta a la fila
    correcta — ahí la igualdad exacta es la garantía, no el bug.

    El peldaño 1 del propio resolver tampoco matchea: construye el SQL
    concatenando `_COLS`, así que la forma literal no existe en el fuente.
    """
    src = _DB_INVENTORY.read_text(encoding="utf-8")
    hits = re.findall(
        r"FROM user_inventory WHERE user_id = %s AND ingredient_name = %s", src)
    assert not hits, (
        f"{len(hits)} lectura(s) de `user_inventory` por igualdad exacta de "
        f"`ingredient_name` en db_inventory.py. Toda resolución de fila debe "
        f"pasar por `find_pantry_rows_for_name` — si no, el plural vuelve a "
        f"producir el no-op silencioso que cerró P1-PANTRY-NAME-RESOLUTION."
    )


def test_hot_paths_use_the_resolver():
    src = _DB_INVENTORY.read_text(encoding="utf-8")
    for fn in ("def add_or_update_inventory_item", "def _consume_reserved_inventory",
               "def deduct_consumed_meal_from_inventory", "def _apply_reservation_delta"):
        start = src.index(fn)
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        assert "find_pantry_rows_for_name" in body, (
            f"{fn} dejó de usar el resolver — vuelve a estar expuesta al "
            f"no-op silencioso por plural."
        )


def test_tool_message_surfaces_absent_items():
    """Si el coach no lo dice, para el usuario el fallo sigue siendo invisible."""
    src = _TOOLS.read_text(encoding="utf-8")
    assert 'deduct_summary.get("not_in_pantry")' in src, (
        "tools.log_consumed_meal debe leer `not_in_pantry` y avisarle al coach; "
        "si no, sigue afirmando que descontó lo que no descontó."
    )
