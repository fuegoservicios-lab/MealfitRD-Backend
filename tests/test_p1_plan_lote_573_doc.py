# backend/tests/test_p1_plan_lote_573_doc.py
"""[P1-PLAN-LOTE-573 · 2026-09-27] La línea base del banco queda escrita (con sus números) en la doc."""
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "banco_analizador.md"


def test_la_doc_trae_la_linea_base_y_como_repetirla():
    t = DOC.read_text(encoding="utf-8")
    for ancla in ("Línea base", "scripts/banco_analizador_correr.py", "--solo-cache", "Regla de aceptación", "kcal_med"):
        assert ancla in t, ancla
