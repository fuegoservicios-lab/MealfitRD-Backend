"""[P1-1] Tests para `_should_pause_for_empty_pantry` cubriendo todas las fuentes.

Antes la pausa solo disparaba cuando `fresh_inventory_source == "live"`. Eso dejaba
pasar chunks con snapshots vacíos (TTL aún válido pero pantry sin items): el LLM
generaba el plan sin restricción de nevera, violando la promesa "solo alimentos
en la nevera".

Ahora pausa siempre que items < CHUNK_MIN_FRESH_PANTRY_ITEMS, EXCEPTO en modos
deliberados (flexible_mode, advisory_only, guest).
"""
import pytest


def _is_paused(source, items, snapshot=None, form_data=None):
    from cron_tasks import _should_pause_for_empty_pantry
    return _should_pause_for_empty_pantry(source, items, snapshot, form_data)


# ----------------------------------------------------------------------------
# Comportamiento previo preservado: live + items < min → pausa
# ----------------------------------------------------------------------------

def test_live_source_with_empty_pantry_pauses():
    assert _is_paused("live", []) is True


def test_live_source_with_one_item_pauses():
    assert _is_paused("live", ["pollo"]) is True


def test_live_source_with_min_items_does_not_pause():
    assert _is_paused("live", ["pollo", "arroz", "habichuelas"]) is False


def test_live_source_with_many_items_does_not_pause():
    assert _is_paused("live", ["pollo", "arroz", "habichuelas", "res", "pescado"]) is False


# ----------------------------------------------------------------------------
# [P1-1] Nuevo: snapshot vacío también pausa
# ----------------------------------------------------------------------------

def test_snapshot_source_with_empty_pantry_now_pauses():
    """Antes pasaba; ahora debe pausar. Es la fix central de P1-1."""
    assert _is_paused("snapshot", []) is True


def test_snapshot_source_with_one_item_pauses():
    assert _is_paused("snapshot", ["sal"]) is True


def test_snapshot_source_with_min_items_does_not_pause():
    assert _is_paused("snapshot", ["pollo", "arroz", "habichuelas"]) is False


def test_stale_snapshot_with_empty_pantry_pauses():
    """stale_snapshot que vuelve sin live también debe pausar si está vacío."""
    assert _is_paused("stale_snapshot", []) is True


def test_unknown_source_with_empty_pantry_pauses():
    """Source None / desconocido y pantry vacía: pausa por defecto seguro."""
    assert _is_paused(None, []) is True


# ----------------------------------------------------------------------------
# Excepciones legítimas — degradaciones deliberadas no se re-pausan
# ----------------------------------------------------------------------------

def test_flexible_mode_in_form_data_skips_pause():
    assert _is_paused("snapshot", [], form_data={"_pantry_flexible_mode": True}) is False


def test_flexible_mode_in_snapshot_skips_pause():
    assert _is_paused("snapshot", [], snapshot={"_pantry_flexible_mode": True}) is False


def test_advisory_only_in_form_data_skips_pause():
    """advisory_only es seteado por flujos como live_degraded_snapshot — son
    degradaciones deliberadas que ya manejan TTL/escalación."""
    assert _is_paused("live_degraded_snapshot", [], form_data={"_pantry_advisory_only": True}) is False


def test_advisory_only_in_snapshot_skips_pause():
    assert _is_paused("snapshot", [], snapshot={"_pantry_advisory_only": True}) is False


def test_guest_source_does_not_pause():
    """Guests no tienen perfil para refrescar despensa; pausar sería un dead-end."""
    assert _is_paused("guest", []) is False


def test_guest_with_items_does_not_pause():
    assert _is_paused("guest", ["pollo", "arroz"]) is False


# ----------------------------------------------------------------------------
# Items irrelevantes (sal, aceite, etc.) no cuentan como suficiencia
# ----------------------------------------------------------------------------

def test_only_seasonings_pauses():
    """Si los únicos items son condimentos ignorados, items_meaningful=0 → pausa."""
    assert _is_paused("snapshot", ["sal", "pimienta", "aceite", "vinagre"]) is True


def test_seasonings_plus_real_food_does_not_pause_when_enough():
    """Mix de condimentos y comida real: cuenta solo la comida real."""
    assert _is_paused(
        "snapshot",
        ["sal", "pimienta", "pollo", "arroz", "habichuelas"],
    ) is False
