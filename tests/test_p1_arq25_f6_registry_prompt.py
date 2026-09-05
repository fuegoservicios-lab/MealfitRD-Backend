"""[P1-ARQ25-F6-REGISTRY-PROMPT · 2026-09-05] Fase 6, rebanada 2 (cierre): los candidatos del Dish Registry llegan
al prompt del generador (knob `MEALFIT_DISH_REGISTRY_PROMPT`), la métrica de fidelidad marca `registry_in_prompt`
para medir antes/después, y cada plantilla lleva `logistics` derivada y bloque `editorial`. Las franjas se
normalizan: el motor canoniza a inglés («dinner») y el registry habla español («cena»).
"""
from pathlib import Path

import dish_registry as dr
import horizon as hz

_BACKEND = Path(__file__).resolve().parents[1]


def test_a_franjas_en_ambos_idiomas_resuelven_a_la_del_registry():
    assert dr.canonical_slot_es("dinner") == "cena" and dr.canonical_slot_es("Cena") == "cena"
    assert dr.canonical_slot_es("breakfast") == "desayuno" and dr.canonical_slot_es("snack") == "merienda"
    assert dr.canonical_slot_es("lunch") == dr.canonical_slot_es("almuerzo") == "almuerzo"
    assert dr.canonical_slot_es("rarísima") == "rarisima", "desconocida: tal cual, sin acentos"


def test_b_el_prompt_usa_la_franja_normalizada_y_el_knob_lo_apaga():
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    i = src.index("def registry_prompt_lines(")
    body = src[i:src.index("def _registry_hash_for_effective(", i)]
    assert "key = dr.canonical_slot_es(s_)" in body
    assert 'lines.extend(registry_prompt_lines(effective, sl, day_index=day_index, slot=slot))' in src
    assert '_env_bool("MEALFIT_DISH_REGISTRY_PROMPT", True)' in src
    assert hz.registry_prompt_lines(None, None) == [] and hz.registry_prompt_lines({}, {"days": []}) == []


def test_c_logistica_y_editorial_en_los_snapshots_reales():
    reg = _BACKEND / "data" / "registry"
    files = sorted(reg.glob("dish_registry_*_v*.json")) if reg.exists() else []
    if not files:
        import pytest
        pytest.skip("snapshots no compilados")
    import json
    for f in files:
        snap = json.loads(f.read_text(encoding="utf-8"))
        assert snap["schema_version"] == 3 and snap["compiler_version"] == 3, f.name  # [F7-G]
        for t in snap["templates"]:
            assert t["logistics"]["estimated"] is True and t["editorial"]["status"] == "curated", t["name"]
            assert set(t["logistics"]) >= {"batch_friendly", "freezer_friendly", "min_shelf_life_days", "prep_minutes_est", "difficulty_est"}
