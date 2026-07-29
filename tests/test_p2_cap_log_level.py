"""[P2-CAP-LOG-LEVEL · 2026-07-29] Un evento de DISEÑO que ocurre cientos de veces no puede ser
WARNING: el nivel deja de significar "mira esto".

Medido en 8 h de producción: **343 de 460 WARNING = 74,6% del journal** eran topes de perecederos
(P5-VEG-CAP 110, P6-LACTEOS-PERISHABLE-CAP 70, P6-CITRUS-CAP 62, P3-HERB-CAP 48…), o sea la
narración de una decisión de producto ya tomada ("no se compran 30 días de tomate fresco de golpe"),
consumida aguas abajo y comunicada al usuario en el display del ítem. Un operador que abre el journal
tras un incidente ve cientos de líneas de una decisión SANA y las 6 señales reales del día quedan
enterradas.

El fix no borra información: baja el per-ítem a INFO (mismo mensaje, mismo marker, greppable igual) y
emite UN warning agregado por corrida con los topes que sí son señal — los que recortan a la mitad o
más, que ya no hablan del tope sino del MENÚ.
"""
from __future__ import annotations

import logging
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.recs = []

    def emit(self, record):
        self.recs.append((record.levelname, record.getMessage()))


@pytest.fixture()
def cap_logs():
    h = _Capture()
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(h)
    try:
        yield h
    finally:
        root.removeHandler(h)
        root.setLevel(prev)


def test_no_cap_emitter_uses_logging_warning_directly():
    """Anti-regresión estructural: si alguien añade un cap nuevo con `logging.warning`, vuelve el
    74,6%. El canal es `_cap_log`, que decide el nivel."""
    lines = _SC.split("\n")
    offenders = []
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith("logging.warning("):
            blk = "\n".join(lines[i:i + 6])
            if "-CAP]" in blk:
                offenders.append(i + 1)
    assert not offenders, (
        f"emisores de cap todavía en logging.warning (L{offenders}) — usa `_cap_log`, "
        f"que es el que aplica la política de nivel")


def test_per_item_cap_logs_at_info(cap_logs):
    import shopping_calculator as sc
    sc._cap_log("[P5-VEG-CAP] 'Papa' peso cap: 2105g → 750g")
    levels = [lv for lv, m in cap_logs.recs if "P5-VEG-CAP" in m]
    assert levels == ["INFO"], f"el per-ítem debe ser INFO, fue {levels}"


def test_severe_caps_summary_is_one_warning(cap_logs):
    """Lo que SÍ es señal: un tope que recorta >50% habla del MENÚ. Una línea por corrida, no una
    por ítem."""
    import shopping_calculator as sc
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Papa", 2105, 750, "P5-VEG-CAP")      # 36% → severo
    sc._record_cap_applied("Rábano", 68, 50, "P5-VEG-CAP")       # 74% → normal
    sc._record_cap_applied("Tomate", 1575, 750, "P5-VEG-CAP")    # 48% → severo
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 1, f"exactamente UN warning agregado, hubo {len(warns)}"
    assert "2 tope(s)" in warns[0], f"solo los severos: {warns[0]}"
    assert "Papa" in warns[0] and "Tomate" in warns[0]
    assert "Rábano" not in warns[0], "un recorte del 74% no es señal"


def test_no_summary_when_nothing_is_severe(cap_logs):
    import shopping_calculator as sc
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Rábano", 68, 50, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    assert not [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]


def test_rollback_knob_restores_warning(cap_logs, monkeypatch):
    """`=1.0` devuelve TODO a WARNING (comportamiento previo), sin redeploy."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_CAP_LOG_SEVERE_RATIO", 1.0)
    sc._cap_log("[P6-CITRUS-CAP] 'Limón' peso cap: 900g → 300g")
    assert [lv for lv, m in cap_logs.recs if "P6-CITRUS-CAP" in m] == ["WARNING"]


def test_summary_is_wired_at_the_end_of_the_aggregator():
    """Debe correr cuando `_CAPS_APPLIED_LAST_RUN` ya está completo para ESTE run: después del
    reset de entrada y al final del agregador."""
    i_reset = _SC.index("reset_caps_applied_last_run()\n", _SC.index("def aggregate_and_deduct_shopping_list"))
    i_call = _SC.index("_log_severe_caps_summary()", i_reset)
    i_final = _SC.index("[AGGREGATE FINAL]")
    assert i_reset < i_final < i_call, "el resumen va al final del agregador, tras el log FINAL"


def test_knob_is_registered():
    import shopping_calculator  # noqa: F401
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_CAP_LOG_SEVERE_RATIO" in get_knobs_registry_snapshot()
