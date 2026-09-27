# backend/tests/test_p1_plan_lote_579_doc.py
"""[P1-PLAN-LOTE-579 · 2026-09-27] El panel queda documentado: quién entra, qué mide y cómo se enciende."""
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "panel_admin.md"


def test_la_doc_del_panel():
    t = DOC.read_text(encoding="utf-8")
    for ancla in ("MEALFIT_ADMIN_USER_IDS", "MEALFIT_ADMIN_PANEL", "admin_access_log", "vision_scan_resultado",
                  "scan_outcome", "capa 2", "404"):
        assert ancla in t, ancla
