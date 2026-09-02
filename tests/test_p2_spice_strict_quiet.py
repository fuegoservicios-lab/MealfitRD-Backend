"""[P2-SPICE-STRICT-QUIET · 2026-09-02] En el guard (estricto), «unidad» de un condimento sin
densidad por unidad es lo esperado — el fallback de cucharadita es opt-in de la Nevera — así que
el aviso baja a DEBUG solo para condimentos; para el resto sigue WARNING (dato que falta).
Medido: 1 WARNING de Orégano por cada recálculo de la lista (cada carga del dashboard).

Tooltip-anchor: P2-SPICE-STRICT-QUIET | _spice_log_level
"""
import logging

import db_inventory


def test_level_selector():
    # métodos ligados: comparar por nombre (cada acceso a logger.debug crea un bound method nuevo)
    assert db_inventory._spice_log_level("Orégano dominicano").__name__ == "debug"
    assert db_inventory._spice_log_level("Canela en polvo").__name__ == "debug"
    assert db_inventory._spice_log_level("Fideos").__name__ == "warning"
    assert db_inventory._spice_log_level(None).__name__ == "warning"


def test_both_strict_branches_use_the_selector():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "db_inventory.py").read_text(encoding="utf-8")
    assert src.count("(_spice_log_level(_name))(") == 2
