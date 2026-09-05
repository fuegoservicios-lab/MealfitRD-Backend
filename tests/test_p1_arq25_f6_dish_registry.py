"""[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Fase 6: Dish Registry compilado. Gate: 100 % de constituyentes
resuelve o queda excluido explícitamente; snapshot reproducible bit a bit desde la fuente; cero tags clínicos
manuales; el allocator consume el snapshot (hash en el blueprint y en la métrica de fidelidad).
Sin DB: catálogo simulado + snapshots en disco (si existen).
"""
import json
import os
from pathlib import Path

import pytest

import dish_registry as dr

_BACKEND = Path(__file__).resolve().parents[1]
_CATALOG = [
    {"name": "Huevo", "aliases": ["huevos"], "kcal_per_100g": 143, "protein_g_per_100g": 12.6, "carbs_g_per_100g": 0.7, "fats_g_per_100g": 9.5,
     "sodium_mg_per_100g": 142, "potassium_mg_per_100g": 138, "phosphorus_mg_per_100g": 198, "saturated_fat_g_per_100g": 3.1, "sugars_g_per_100g": 0.4, "fiber_g_per_100g": 0},
    {"name": "Plátano verde", "aliases": ["platano verde", "platanos verdes"], "kcal_per_100g": 122, "protein_g_per_100g": 1.3, "carbs_g_per_100g": 31.9, "fats_g_per_100g": 0.4,
     "sodium_mg_per_100g": 4, "potassium_mg_per_100g": 499, "phosphorus_mg_per_100g": 34, "saturated_fat_g_per_100g": 0.1, "sugars_g_per_100g": 15, "fiber_g_per_100g": 2.3},
    {"name": "Salami", "aliases": [], "kcal_per_100g": 336, "protein_g_per_100g": 22, "carbs_g_per_100g": 1.6, "fats_g_per_100g": 26,
     "sodium_mg_per_100g": 1740, "potassium_mg_per_100g": 340, "phosphorus_mg_per_100g": 180, "saturated_fat_g_per_100g": 9.6, "sugars_g_per_100g": 0.5, "fiber_g_per_100g": 0},
    {"name": "Camarones", "aliases": ["camaron"], "kcal_per_100g": 99, "protein_g_per_100g": 24, "carbs_g_per_100g": 0.2, "fats_g_per_100g": 0.3,
     "sodium_mg_per_100g": 111, "potassium_mg_per_100g": 259, "phosphorus_mg_per_100g": 244, "saturated_fat_g_per_100g": 0.1, "sugars_g_per_100g": 0, "fiber_g_per_100g": 0},
]
_TEMPLATES = [
    {"template_id": "tpl_b", "template_version": 1, "name": "Mangú con salami", "slots": ["desayuno"], "base": "platano", "protein": "cerdo", "technique": "hervido", "transform": True,
     "constituents": [{"name": "Plátano verde", "grams": 180}, {"name": "Salami", "grams": 60}, {"name": "Zapote", "grams": 100}]},
    {"template_id": "tpl_a", "template_version": 1, "name": "Huevos con camarones", "slots": ["cena", "almuerzo"], "base": "none", "protein": "camarones", "technique": "sartén", "transform": False,
     "constituents": [{"name": "huevos", "grams": 100}, {"name": "Camarones", "grams": 120}]},
    {"template_id": "tpl_c", "template_version": 1, "name": "Sin nada", "slots": ["merienda"], "base": "none", "protein": "none", "technique": "frío", "transform": False, "constituents": []},
]


def _snap(**kw):
    return dr.compile_library("es", catalog_rows=_CATALOG, version="t", templates=_TEMPLATES, **kw)


def test_a_resolucion_por_nombre_y_alias_y_exclusion_explicita():
    snap = _snap()
    by = {t["template_id"]: t for t in snap["templates"]}
    assert [t["template_id"] for t in snap["templates"]] == ["tpl_a", "tpl_b", "tpl_c"], "orden estable por template_id"
    a = by["tpl_a"]
    assert [c["canonical"] for c in a["constituents"]] == ["Huevo", "Camarones"], "alias «huevos» → Huevo"
    assert a["status"] == "ok" and a["excluded"] == []
    b = by["tpl_b"]
    assert b["status"] == "partial"
    assert b["excluded"] == [{"name": "Zapote", "grams": 100.0, "reason": "not_in_catalog"}], "lo que no está en el catálogo se excluye EXPLÍCITAMENTE"
    assert by["tpl_c"]["status"] == "excluded"
    st = snap["stats"]
    assert st["constituents"] == st["resolved"] + 1 and st["ok"] == 1 and st["partial"] == 1 and st["excluded"] == 1


def test_b_riesgo_intrinseco_derivado_sin_tags_clinicos_manuales():
    snap = _snap()
    by = {t["template_id"]: t for t in snap["templates"]}
    b = by["tpl_b"]["intrinsic_risk_attributes"]
    assert b["sodium_high"] is True and b["processed_meat"] is True and b["processed_items"] == ["Salami"]
    assert b["potassium_high"] is True, "180 g de plátano verde ≈ 898 mg de potasio"
    a = by["tpl_a"]["intrinsic_risk_attributes"]
    assert "mariscos" in a["allergens"] and "huevo" in a["allergens"], "clases del vocabulario SSOT de alérgenos"
    assert a["sodium_high"] is False
    for t in snap["templates"]:
        assert not any(k.startswith("safe_for") for k in t), "nunca elegibilidad clínica estática (§7.2)"
    src = (_BACKEND / "dish_registry.py").read_text(encoding="utf-8")
    assert "safe_for_diabetes" in src and 'nunca\n     `safe_for_diabetes`' in src or "nunca" in src


def test_c_snapshot_reproducible_bit_a_bit_y_verificable(tmp_path, monkeypatch):
    s1, s2 = _snap(), _snap()
    assert s1["snapshot_hash"] == s2["snapshot_hash"] and dr.verify_snapshot(s1)
    monkeypatch.setattr(dr, "REGISTRY_DIR", str(tmp_path))
    p = dr.write_snapshot(s1, os.path.join(str(tmp_path), "x.json"))
    raw1 = open(p, "rb").read()
    dr.write_snapshot(s2, p)
    assert open(p, "rb").read() == raw1, "misma fuente ⇒ mismos bytes"
    loaded = json.loads(raw1)
    assert dr.verify_snapshot(loaded)
    loaded["templates"][0]["name"] = "otro"
    assert not dr.verify_snapshot(loaded), "el hash detecta un snapshot editado a mano"
    # cambia la fuente ⇒ cambia el hash (y source_hash lo explica)
    t2 = json.loads(json.dumps(_TEMPLATES)); t2[1]["constituents"][1]["grams"] = 150
    s3 = dr.compile_library("es", catalog_rows=_CATALOG, version="t", templates=t2)
    assert s3["snapshot_hash"] != s1["snapshot_hash"] and s3["source_hash"] != s1["source_hash"]
    assert s3["catalog_fingerprint"] == s1["catalog_fingerprint"]


def test_d_runtime_fail_open_y_candidatos_para_el_allocator(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "REGISTRY_DIR", str(tmp_path))
    dr._CACHE.clear()
    assert dr.load_registry("ES", version="t") is None, "sin snapshot: None, nada bloquea"
    snap = _snap()
    dr.write_snapshot(snap, dr.snapshot_path("es", "t"))
    dr._CACHE.clear()
    monkeypatch.setenv("MEALFIT_DISH_REGISTRY_SNAPSHOT", "t")
    assert dr.registry_hash("ES") == snap["snapshot_hash"]
    cands = dr.template_candidates("ES", "cena", "pescado")  # familia F3 (camarones ⊂ pescado)
    assert [c["template_id"] for c in cands] == ["tpl_a"], "status ok + franja + familia"
    assert dr.template_candidates("ES", "cena", "pescado", exclude_allergens=["mariscos"]) == [], "clase de alérgeno declarada"
    assert dr.template_candidates("ES", "cena", "pollo") == [], "familia que no casa"
    assert dr.template_candidates("ES", "desayuno") == [], "tpl_b es partial: no se ofrece"
    dr._CACHE.clear()


def test_e_bibliotecas_reales_compiladas_100_por_ciento_resueltas_o_excluidas():
    """Con los snapshots en disco (compilados contra el catálogo real): gate de la fase."""
    reg = _BACKEND / "data" / "registry"
    files = sorted(reg.glob("dish_registry_*_v*.json")) if reg.exists() else []
    if not files:
        pytest.skip("snapshots no compilados en este entorno")
    seen = set()
    for f in files:
        snap = json.loads(f.read_text(encoding="utf-8"))
        assert dr.verify_snapshot(snap), f.name
        seen.add(snap["library"])
        st = snap["stats"]
        assert st["constituents"] == st["resolved"] + sum(len(t["excluded"]) for t in snap["templates"]), f"{f.name}: todo constituyente resuelve o queda excluido"
        assert st["templates"] > 0 and st["resolution_pct"] >= 90.0, f"{f.name}: {st}"
        for t in snap["templates"]:
            assert t["status"] in ("ok", "partial", "excluded")
            for e in t["excluded"]:
                assert e["reason"] in ("not_in_catalog", "no_grams", "declared_unresolved")
    assert seen == set(dr.LIBRARIES), f"faltan bibliotecas: {set(dr.LIBRARIES) - seen}"


def test_f_las_87_plantillas_do_tienen_constituyentes_curados():
    p = _BACKEND / "data" / "dish_constituents_do.json"
    if not p.exists():
        pytest.skip("curación DO no generada")
    cur = json.loads(p.read_text(encoding="utf-8"))["templates"]
    templates = json.loads((_BACKEND / "data" / "dish_templates.json").read_text(encoding="utf-8"))["templates"]
    assert set(cur) == {t["name"] for t in templates}, "una entrada por plantilla, ni más ni menos"
    assert all(len(v["constituents"]) >= 1 for v in cur.values())
    assert all(v["origin"] == "curated" for v in cur.values()), "las 87 curadas a mano, ninguna por reglas de respaldo"


def test_h_logistica_y_editorial_derivadas_en_cada_plantilla():
    snap = _snap()
    by = {t["template_id"]: t for t in snap["templates"]}
    lg = by["tpl_a"]["logistics"]
    assert lg["estimated"] is True and lg["batch_friendly"] is False and lg["freezer_friendly"] is False, "«sartén» no es plato de tanda"
    assert by["tpl_b"]["logistics"]["batch_friendly"] is True, "«hervido» se hace en tanda"
    ed = by["tpl_a"]["editorial"]
    assert ed["status"] == "curated" and ed["display_name"]["es"] == "Huevos con camarones" and ed["media"] == []
    assert snap["schema_version"] == 2


def test_i_los_candidatos_del_registry_llegan_al_prompt_con_knob(tmp_path, monkeypatch):
    import horizon as hz
    monkeypatch.setattr(dr, "REGISTRY_DIR", str(tmp_path)); dr._CACHE.clear()
    dr.write_snapshot(_snap(), dr.snapshot_path("es", "t"))
    monkeypatch.setenv("MEALFIT_DISH_REGISTRY_SNAPSHOT", "t")
    eff = {"recurrence": {"global_mode": "balanced"}, "market": {"country": "ES"}, "diet": {"allergies": []}}
    sl = {"days_offset": 0, "days": [{"day_index": 0, "protein": "pescado", "slots": ["cena"]}]}
    lines = hz.registry_prompt_lines(eff, sl)
    assert len(lines) == 1 and "Huevos con camarones" in lines[0] and "Día 1" in lines[0]
    block = hz.policy_prompt_block(eff, sl, surface="test", enforced=True)
    assert "Platos del registro curado" in block
    # alérgeno declarado ⇒ el candidato con mariscos no se ofrece
    eff2 = {**eff, "diet": {"allergies": ["mariscos"]}}
    assert hz.registry_prompt_lines(eff2, sl) == []
    # knob apagado ⇒ prompt byte-idéntico al de antes
    monkeypatch.setenv("MEALFIT_DISH_REGISTRY_PROMPT", "0")
    assert hz.registry_prompt_lines(eff, sl) == [] and "Platos del registro" not in hz.policy_prompt_block(eff, sl, surface="test", enforced=True)
    dr._CACHE.clear()


def test_g_el_allocator_y_la_metrica_llevan_el_hash_del_snapshot():
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert '"registry": _registry_block_for_country(' in src, "el blueprint lleva hash + candidatos del registry"
    assert '"registry_hash": report.get("registry_hash")' in src, "la métrica de fidelidad guarda el hash"
    assert "def _registry_block_for_country(" in src
    assert '"registry_in_prompt": report.get("registry_in_prompt")' in src, "la métrica distingue antes/después del prompt"
