"""G41: el repo puede verificar las dos migraciones contra el schema vivo."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "verify_country_schema.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_country_schema", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_evaluador_verde_exige_check_indice_y_trece_densidades() -> None:
    verify = _load_script()
    definition = "CHECK ((country IS NULL) OR (country = ANY (ARRAY[" + ",".join(
        f"'{code}'::text" for code in verify.COUNTRY_CODES
    ) + "])))"
    failures, report = verify.evaluate_country_schema(
        [(verify.COUNTRY_CONSTRAINT, definition)],
        [(verify.COUNTRY_INDEX,)],
        [(name, 200.0) for name in verify.DENSITY_NAMES],
    )
    assert failures == []
    assert report["density_rows_found"] == 13
    assert report["density_null"] == []


def test_evaluador_rojo_enumera_todas_las_piezas_ausentes() -> None:
    verify = _load_script()
    failures, report = verify.evaluate_country_schema([], [], [("Nata", None)])
    joined = " | ".join(failures)
    assert verify.COUNTRY_CONSTRAINT in joined
    assert verify.COUNTRY_INDEX in joined
    assert "densidades NULL" in joined
    assert "filas del lote" in joined
    assert report["density_rows_expected"] == 13


def test_conexion_del_detector_es_read_only_y_sondea_catalogos() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "default_transaction_read_only=on" in source
    assert "pg_constraint" in source
    assert "pg_indexes" in source
    assert "density_g_per_cup" in source
    assert "UPDATE " not in source
    assert "INSERT " not in source
    assert "DELETE " not in source


def test_sop_lo_ejecuta_junto_a_health_version() -> None:
    doc = (BACKEND_ROOT / "docs" / "country_system_f1.md").read_text(encoding="utf-8")
    health = doc.index("/health/version")
    verify = doc.index("python scripts/verify_country_schema.py", health)
    assert verify - health < 1000


def test_migraciones_gemelas_siguen_byte_identicas() -> None:
    workspace = BACKEND_ROOT.parent
    for name in (
        "p1_country_keep_density_beta_2026_08_21.sql",
        "p3_country_db_check_2026_08_22.sql",
    ):
        root_copy = workspace / "migrations" / name
        backend_copy = BACKEND_ROOT / "migrations" / name
        if root_copy.is_file():
            assert root_copy.read_bytes() == backend_copy.read_bytes()


def test_pfix_marker_cierra_g41() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "P2-COUNTRY-MIGRACIONES-SIN-APLICAR · 2026-08-23" in script
