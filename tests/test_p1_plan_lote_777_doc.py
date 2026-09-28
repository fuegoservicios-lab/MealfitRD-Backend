"""[P1-PLAN-LOTE-777 · 2026-09-28] El doc de los regalos de la cuenta existe, dice lo que hay que saber y cada lote
tiene su test."""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_el_doc_cuenta_lo_esencial():
    doc = (_BACKEND / "docs" / "regalos_cuenta.md").read_text(encoding="utf-8")
    for trozo in ("account_grants", "MEALFIT_ACCOUNT_GRANTS", "regalos_cuenta.superponer", "plan_tier_pagado",
                  "X-Admin-Accion", "admin_access_log", "/api/admin/cuentas/buscar", "ON DELETE CASCADE"):
        assert trozo in doc, trozo


def test_cada_lote_tiene_su_test():
    for n in (771, 772, 773, 774):
        assert (_BACKEND / "tests" / f"test_p1_plan_lote_{n}.py").exists(), n


def test_claude_md_enlaza_el_doc():
    claude = _BACKEND.parent / "CLAUDE.md"
    if claude.exists():                      # el repo del backend puede ir solo (CI); el workspace sí lo trae
        assert "backend/docs/regalos_cuenta.md" in claude.read_text(encoding="utf-8")
