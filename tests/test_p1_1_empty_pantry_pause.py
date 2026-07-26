"""[P1-1] Tests para `_should_pause_for_empty_pantry` cubriendo todas las fuentes.

Antes la pausa solo disparaba cuando `fresh_inventory_source == "live"`. Eso dejaba
pasar chunks con snapshots vacíos (TTL aún válido pero pantry sin items): el LLM
generaba el plan sin restricción de nevera, violando la promesa "solo alimentos
en la nevera".

Ahora pausa siempre que items < CHUNK_MIN_FRESH_PANTRY_ITEMS, EXCEPTO en modos
deliberados (flexible_mode, advisory_only, guest).

[P1-SUITE-UNBLIND · 2026-07-26] Las listas de alimentos se DERIVAN del knob, no se
escriben a mano. Tres tests de este archivo llevaban rojos desde el 2026-07-11, cuando
P1-PLAN-FREEZE subió `CHUNK_MIN_FRESH_PANTRY_ITEMS` de 3 a 5: seguían afirmando que 3
alimentos bastaban. Producción avanzó, la suite se quedó, y como ya estaba roja nadie
volvió a leer su salida — que es cómo el lazo de P1-PANTRY-GATE-SSOT vivió 14 días en
el subsistema vecino sin que saltara nada.

Un test que fija el número no verifica "hay suficiente comida"; verifica "el umbral vale
3". Derivándolo, el test sigue al knob y solo falla cuando la SEMÁNTICA cambia.
"""
import pytest


def _min_items() -> int:
    from cron_tasks import CHUNK_MIN_FRESH_PANTRY_ITEMS
    return CHUNK_MIN_FRESH_PANTRY_ITEMS


# Alimentos reales distintos (ninguno en el `ignored_terms` de
# `_count_meaningful_pantry_items`), suficientes para cubrir umbrales holgados.
_DESPENSA = [
    "pollo", "arroz", "habichuelas", "res", "pescado",
    "yuca", "platano", "huevos", "queso", "batata",
    "auyama", "lentejas", "cerdo", "tilapia", "avena",
]


def _comida(n: int) -> list:
    """`n` alimentos reales distintos."""
    assert n <= len(_DESPENSA), f"amplía _DESPENSA: el umbral llegó a {n}"
    return _DESPENSA[:n]


def _justo_suficiente() -> list:
    return _comida(_min_items())


def _uno_menos_del_minimo() -> list:
    return _comida(max(1, _min_items() - 1))


def _is_paused(source, items, snapshot=None, form_data=None):
    from cron_tasks import _should_pause_for_empty_pantry
    return _should_pause_for_empty_pantry(source, items, snapshot, form_data)


def test_el_umbral_es_un_knob_vivo():
    """Si esto falla, el knob desapareció y el resto del archivo miente."""
    assert isinstance(_min_items(), int) and _min_items() >= 1


# ----------------------------------------------------------------------------
# Comportamiento previo preservado: live + items < min → pausa
# ----------------------------------------------------------------------------

def test_live_source_with_empty_pantry_pauses():
    assert _is_paused("live", []) is True


def test_live_source_with_one_item_pauses():
    assert _is_paused("live", ["pollo"]) is True


def test_live_source_just_below_min_pauses():
    """Frontera exacta: un alimento menos del mínimo todavía pausa."""
    assert _is_paused("live", _uno_menos_del_minimo()) is True


def test_live_source_with_min_items_does_not_pause():
    assert _is_paused("live", _justo_suficiente()) is False


def test_live_source_with_many_items_does_not_pause():
    assert _is_paused("live", _comida(_min_items() + 2)) is False


# ----------------------------------------------------------------------------
# [P1-1] Nuevo: snapshot vacío también pausa
# ----------------------------------------------------------------------------

def test_snapshot_source_with_empty_pantry_now_pauses():
    """Antes pasaba; ahora debe pausar. Es la fix central de P1-1."""
    assert _is_paused("snapshot", []) is True


def test_snapshot_source_with_one_item_pauses():
    assert _is_paused("snapshot", ["sal"]) is True


def test_snapshot_source_with_min_items_does_not_pause():
    assert _is_paused("snapshot", _justo_suficiente()) is False


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
    assert _is_paused("snapshot", ["sal", "pimienta"] + _justo_suficiente()) is False


def test_seasonings_no_inflan_la_cuenta():
    """La prueba de que los condimentos NO cuentan: con el mínimo de comida real MENOS
    uno, añadir condimentos de sobra no debe salvar la pausa."""
    assert _is_paused(
        "snapshot",
        ["sal", "pimienta", "aceite", "vinagre", "oregano", "canela"] + _uno_menos_del_minimo(),
    ) is True
