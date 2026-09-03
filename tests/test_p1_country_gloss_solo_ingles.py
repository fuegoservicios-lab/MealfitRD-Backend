"""[P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23]

El gloss español es display-only, depende del país (no del locale como proxy)
y no modifica la identidad canónica del ingrediente.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
_MIGRATION = "p1_country_gloss_es_2026_08_23.sql"


def _top_level_blocks(source: str) -> dict[str, str]:
    positions = [(m.start(), m.group(1)) for m in re.finditer(r"^def (\w+)\(", source, re.MULTILINE)]
    return {
        name: source[start:(positions[i + 1][0] if i + 1 < len(positions) else len(source))]
        for i, (start, name) in enumerate(positions)
    }


def test_migracion_gloss_es_existe_en_los_dos_ssot_y_es_identica():
    backend = _BACKEND / "migrations" / _MIGRATION
    root = _ROOT / "migrations" / _MIGRATION
    assert backend.exists() and root.exists()
    assert backend.read_bytes() == root.read_bytes()


def test_migracion_puebla_solo_regionalismos_sin_tocar_identidad():
    sql = (_BACKEND / "migrations" / _MIGRATION).read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS gloss_es TEXT" in sql
    expected = {
        "Auyama": "calabaza",
        "Tayota": "chayote",
        "Lechosa": "papaya",
        "Chinola": "maracuyá",
        "Molondrones": "okra",
        "Guineo": "banana",
        "Yautía": "malanga",
        "Ají cubanela": "pimiento italiano",
    }
    for canonical, gloss in expected.items():
        assert f"('{canonical}', '{gloss}')" in sql
    assert "SET name =" not in sql
    assert "DROP COLUMN" not in sql.upper()
    assert "DO $$" in sql and "RAISE EXCEPTION" in sql


def test_backend_extrae_gloss_es_display_only_sin_mutar_name():
    import shopping_calculator as sc

    item = {"name": "Lechosa", "gloss_es": "  papaya  ", "name_en": "Papaya"}
    before = dict(item)
    assert sc._display_gloss_es_for_item(item) == "papaya"
    assert item == before
    assert sc._display_gloss_es_for_item({"name": "Lechosa"}) is None


def test_backend_adjunta_display_gloss_es_en_los_dos_paths():
    source = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert source.count('market_obj["display_gloss_es"] = _gloss_es') == 2


def test_guard_escopeta_cubre_ambos_glosses_por_propiedad():
    source = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    blocks = _top_level_blocks(source)
    allowed = {
        "_display_name_en_for_item",
        "_display_gloss_es_for_item",
        "aggregate_and_deduct_shopping_list",
    }
    for field in ("name_en", "gloss_es"):
        offenders = sorted(name for name, body in blocks.items() if field in body and name not in allowed)
        assert not offenders, f"{field} salió de la zona display-only: {offenders}"


def test_catalog_api_proyecta_gloss_sin_romper_si_migracion_aun_no_se_aplico():
    source = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    assert "to_jsonb(mi)->>'gloss_es' AS gloss_es" in source
    assert "FROM master_ingredients AS mi ORDER BY name ASC" in source


def test_marker_movil_y_huella_durable_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    shopping = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-COUNTRY-GLOSS-SOLO-INGLES" in app
    assert "P1-COUNTRY-GLOSS-SOLO-INGLES" in shopping
