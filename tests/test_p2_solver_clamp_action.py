"""[P2-SOLVER-CLAMP-ACTION · 2026-07-05] (audit solver+seeder P2-6)

La telemetría per-meal P2-SOLVER-CLAMP-TELEMETRY (flag `_solver_clamp_saturated` cuando un
factor del solver satura el clamp [0.3, 3.5] — el slot queda fuera de banda ANTES de los
closers) existía desde 2026-06-19 pero NADIE la consumía: ni métrica agregada ni serie de
flota. La decisión documentada ("opción b: subir max_scale para proteína-dominantes") no tenía
datos con qué tomarse.

Fix: agregación per-run en el seam de scoring (junto a clinical_band) → métrica
pipeline_metrics node='solver_clamp' con confidence = fracción de meals saturados, metadata
{saturated_meals, total_meals, delivered_was_fallback}. Se emite SIEMPRE que el plan tenga
meals (los ceros son el denominador de la tasa de flota — sin ellos la serie miente).
Solo medición: cero cambio de comportamiento del solver.
"""
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO = (_REPO_ROOT / "backend" / "graph_orchestrator.py").read_text(encoding="utf-8")


def _block():
    i = _GO.index("[P2-SOLVER-CLAMP-ACTION · 2026-07-05]")
    return _GO[i:i + 2200]


def test_marker_anchored():
    assert "P2-SOLVER-CLAMP-ACTION" in _GO


def test_emits_solver_clamp_metric_with_denominator():
    blk = _block()
    assert '"node": "solver_clamp"' in blk
    assert "if _clamp_meals:" in blk, "emite SIEMPRE que haya meals (los ceros son el denominador)"
    assert '"saturated_meals"' in blk and '"total_meals"' in blk
    assert '"delivered_was_fallback"' in blk, "el fallback se excluye en la agregación (macros ~target)"


def test_reads_the_existing_per_meal_flag():
    """Consume el flag que P2-SOLVER-CLAMP-TELEMETRY ya persiste per-meal (cero doble-instrumentación)."""
    assert "P2-SOLVER-CLAMP-TELEMETRY" in _GO, "la telemetría fuente sigue viva en el solver"
    assert '_m4.get("_solver_clamp_saturated")' in _block()


def test_lives_in_scoring_seam_next_to_clinical_band():
    i_band = _GO.index('"node": "clinical_band"')
    i_clamp = _GO.index('"node": "solver_clamp"')
    assert i_band < i_clamp, "la agregación vive en el seam de scoring (post-entrega elegida)"
    # entre ambos emits viven los gates de degradación del scoring (~8-9k chars) — ventana holgada.
    assert (i_clamp - i_band) < 15000, "mismo seam de métricas de scoring (no otro surface)"
