"""[P2-BAND-RETRY-GATE-LOG · 2026-07-05] (audit solver+seeder P2-5)

El log del retry-gate de banda imprimía SIEMPRE "band_score=X < umbral" aunque el agregado
hubiera PASADO y la rama real del rechazo fuera per-macro o el backstop de kcal — el mismo bug
de honestidad que P3-BAND-GATE-LOG-HONESTY ya corrigió en el banner-gate post-scoring. En prod
esto producía diagnósticos falsos ("0.67 < 0.6" con 0.67 ≥ 0.6). Ahora el encabezado es
BRANCH-AWARE: agregado → "band_score=X < thr"; per-macro → "agregado X ≥ umbral thr PERO
macro(s) [...] bajo umbral de celdas"; kcal-backstop → "... PERO kcal Y < thr (backstop)".
Cosmético (cero cambio de comportamiento del gate) — solo el mensaje.
"""
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO = (_REPO_ROOT / "backend" / "graph_orchestrator.py").read_text(encoding="utf-8")


def _gate_block():
    i = _GO.index("[P2-BAND-RETRY-GATE-LOG · 2026-07-05]")
    return _GO[i:i + 1800]


def test_marker_anchored():
    assert "P2-BAND-RETRY-GATE-LOG" in _GO


def test_branch_aware_heads():
    blk = _gate_block()
    assert "if _agg_trigger:" in blk, "rama agregado explícita"
    assert 'f"band_score={_bsr_val} < {_bsr_thr}"' in blk
    assert "PERO macro(s)" in blk, "rama per-macro dice la verdad (agregado ≥ umbral)"
    assert "(backstop)" in blk, "rama kcal-backstop identificada"


def test_mirror_of_banner_gate_honesty():
    """El retry-gate y el banner-gate usan el MISMO patrón de honestidad (agregado ≥ ... PERO)."""
    assert "P3-BAND-GATE-LOG-HONESTY" in _GO, "el espejo del banner-gate sigue vivo"
    assert _GO.count("PERO macro(s)") >= 2, "ambos gates distinguen la rama per-macro"
