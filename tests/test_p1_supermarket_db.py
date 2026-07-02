"""[P1-SUPERMARKET-DB · 2026-07-02] Test ancla MÍNIMO del marker (contrato marker↔test de
test_p2_hist_audit_14_marker_test_link). Contexto: el feature es WIP del owner (Supermercado RD
artificial — catálogo de presentaciones comprables navegable en /supermercado, edición con gate
admin). Este anchor nació durante la reparación de un race de OneDrive: el app.py del owner (marker
bump + include_router) entró en el commit del batch P1-AUDIT-V3-BATCH mientras `routers/supermarket.py`
seguía untracked → origin/main quedó no-importable. El anchor asegura lo estructural (import, prefix,
gate admin, migración SSOT en ambos dirs); el owner puede extenderlo con tests funcionales del feature.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g  # asegura sys.path del backend

_BACKEND = Path(g.__file__).resolve().parent


def test_router_imports_and_prefix():
    # el import roto de este módulo fue exactamente el modo de fallo que rompió origin/main
    from routers.supermarket import router
    assert router.prefix == "/api/supermarket"


def test_marker_wired_in_app_and_router():
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    r_src = (_BACKEND / "routers" / "supermarket.py").read_text(encoding="utf-8")
    assert "P1-SUPERMARKET-DB" in app_src
    assert "from routers.supermarket import router" in app_src
    assert "router = APIRouter" in r_src


def test_admin_gate_present():
    r_src = (_BACKEND / "routers" / "supermarket.py").read_text(encoding="utf-8")
    assert "_verify_admin_token" in r_src, "las mutaciones del catálogo deben ir tras el gate admin"


def test_migration_ssot_both_dirs():
    # [P3-MIGRATIONS-SSOT] la migración debe existir en backend/migrations Y en migrations del workspace root
    name = "p1_supermarket_db_2026_07_02.sql"
    assert (_BACKEND / "migrations" / name).exists()
    assert (_BACKEND.parent / "migrations" / name).exists()
