"""[P1-ARQ25-F7-CULTURE · 2026-09-05 · subfase G] Compra única y congelador sincronizados con el motor.

`master_ingredients.shelf_life_days` es un relleno (lechuga, fresas y carne valen 14 por igual), así que la
durabilidad es una tabla de reglas (`pantry_durability`) y de ella cuelgan: el registry (días que aguanta cada
plato sin/con congelador), el filtro de candidatos por día del ciclo, el validador de fidelidad, el prompt por modo
de congelador, la política (que ya NO acorta el ciclo sin congelador) y el copy del PDF. Además, las seis
bibliotecas ganan familias de «misma despensa, distinta preparación» que aguantan las semanas 2-4.
"""
import json
import re
from pathlib import Path

import pytest

import pantry_durability as pd

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"


def test_a_clases_de_durabilidad_por_nombre_no_por_la_columna_de_relleno():
    assert pd.classify("Atún en agua")["cls"] == "pantry" and pd.classify("Sardinas en lata")["days_fresh"] >= 90
    assert pd.classify("Huevo") == {"cls": "cold", "days_fresh": 35, "days_frozen": 35}
    assert pd.classify("Pechuga de pollo") == {"cls": "freezable", "days_fresh": 3, "days_frozen": 90}
    assert pd.classify("Lechuga")["cls"] == "fresh" and pd.classify("Tomate")["cls"] == "fresh"
    assert pd.classify("Repollo")["days_fresh"] >= 21 and pd.classify("Cebolla")["days_fresh"] >= 21
    # tokens cortos son palabra exacta: «sal» no es «salmón»; los largos casan por prefijo: «yogur» → «yogurt»
    assert pd.classify("Sal")["cls"] == "pantry" and pd.classify("Salmón")["cls"] == "freezable"
    assert pd.classify("Yogurt griego sin azúcar")["cls"] == "cold", "«sin azúcar» no lo vuelve despensa"
    assert pd.classify("Huevos rellenos")["cls"] == "fresh" and pd.classify("Panecillos de mantequilla")["cls"] == "fresh"
    assert pd.classify("Chorizo mexicano")["cls"] == "freezable" and pd.classify("Chorizo español")["cls"] == "cold"
    assert pd.classify("marte", "Despensa")["cls"] == "pantry" and pd.classify("marte")["cls"] == "fresh", "default por categoría; sin nada, fresco"


def test_b_ventana_de_congelacion_y_exigencia_por_dia():
    assert pd.freeze_window_days("none", 30) == 0 and pd.freeze_window_days("limited", 30) == 14 and pd.freeze_window_days("full", 30) == 30
    eff = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "limited"}}
    assert pd.single_trip_requirements(eff, 3) is None, "la primera semana es de frescos: sin exigencia"
    assert pd.single_trip_requirements(eff, 7) == {"need_days": 8, "allow_frozen": True, "freezer_mode": "limited", "freeze_window_days": 14}
    assert pd.single_trip_requirements(eff, 14)["allow_frozen"] is False
    assert pd.single_trip_requirements({"shopping": {"main_cycle_days": 30, "fresh_topup_days": 7}}, 20) is None, "con reposición no aplica"
    assert pd.single_trip_requirements({"shopping": {"main_cycle_days": 7}}, 20) is None
    # un plato de pollo fresco cabe el día 10 solo congelando; uno de atún y arroz cabe siempre
    pollo = pd.durability_of(["Arroz blanco", "Pechuga de pollo", "Repollo"])
    assert pd.template_fits(pollo["days_fresh_min"], pollo["days_with_freezer_min"], 10, allow_frozen=True)
    assert not pd.template_fits(pollo["days_fresh_min"], pollo["days_with_freezer_min"], 10, allow_frozen=False)
    atun = pd.durability_of(["Arroz blanco", "Atún en agua", "Repollo", "Cebolla"])
    assert atun["pantry_only"] and pd.template_fits(atun["days_fresh_min"], atun["days_with_freezer_min"], 30, allow_frozen=False)
    assert pd.ingredient_issue_beyond_horizon("Pechuga de pollo", 20, allow_frozen=False) == "protein_beyond_freeze_window"
    assert pd.ingredient_issue_beyond_horizon("Lechuga", 9, allow_frozen=True) == "fresh_beyond_horizon"
    assert pd.ingredient_issue_beyond_horizon("Huevo", 29, allow_frozen=False) is None


def test_c_el_registry_filtra_candidatos_por_durabilidad_y_las_bibliotecas_tienen_despensa():
    import dish_registry as dr
    if not Path(dr.snapshot_path("do")).exists():
        pytest.skip("snapshots no compilados")
    for lib, (_, cc, _c) in dr.LIBRARIES.items():
        snap = json.loads(Path(dr.snapshot_path(lib)).read_text(encoding="utf-8"))
        assert snap["schema_version"] == 3
        pantry = [t for t in snap["templates"] if (t.get("logistics") or {}).get("pantry_only")]
        assert len(pantry) >= 12, f"[{lib}] solo {len(pantry)} platos de despensa (≥21 días sin congelador)"
        for t in snap["templates"][:3]:
            assert all("durability" in c and "days_fresh" in c for c in t["constituents"])
        # día 29 sin congelador: cada franja principal sigue teniendo candidatos, y todos aguantan
        for slot in ("almuerzo", "cena", "desayuno"):
            c = dr.template_candidates(cc, slot, None, k=50, need_days=30, allow_frozen=False)
            assert c, f"[{lib}] sin candidatos de despensa en {slot} para el día 30"
            assert all(x["pantry_only"] or (x["logistics"].get("days_fresh_min") or 0) >= 30 for x in c)
        # con congelador, los de proteína fresca vuelven a entrar
        with_fz = dr.template_candidates(cc, "almuerzo", "pollo", k=50, need_days=10, allow_frozen=True)
        without = dr.template_candidates(cc, "almuerzo", "pollo", k=50, need_days=10, allow_frozen=False)
        assert len(with_fz) >= len(without)


def test_d_blueprint_y_prompt_exigen_durabilidad_solo_bajo_compra_unica():
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert "def _dur_kwargs(" in src
    assert src.count("**_dur_kwargs(") == 2, "el bloque del blueprint y las líneas del prompt filtran por día"
    assert "from pantry_durability import freeze_window_days" in src, "_freeze_horizon_days delega en el SSOT"
    import horizon as hz
    assert hz._freeze_horizon_days("limited", 30) == 14 and hz._freeze_horizon_days("none", 30) == 0
    assert hz._dur_kwargs({"shopping": {"main_cycle_days": 30, "freezer_mode": "none"}}, 12) == {"need_days": 13, "allow_frozen": False}
    assert hz._dur_kwargs({"shopping": {"main_cycle_days": 7}}, 12) == {}


def test_e_validador_y_prompt_por_modo_de_congelador():
    import horizon as hz
    eff = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"}}
    days = [{"meals": [{"ingredients": ["Pechuga de pollo 150 g", "Lechuga", "Arroz blanco", "Huevo"]}]}]
    issues = hz.fresh_beyond_horizon_issues(days, {"days_offset": 20}, eff)
    codes = {i["code"] for i in issues}
    assert codes == {"protein_beyond_freeze_window", "fresh_beyond_horizon"}, codes
    assert not [i for i in issues if "Huevo" in i["ingredient"] or "Arroz" in i["ingredient"]]
    eff_full = {**eff, "shopping": {**eff["shopping"], "freezer_mode": "full"}}
    assert {i["code"] for i in hz.fresh_beyond_horizon_issues(days, {"days_offset": 20}, eff_full)} == {"fresh_beyond_horizon"}
    lines = hz.single_trip_prompt_lines({**eff, "shopping": {**eff["shopping"], "freezer_mode": "limited"}}, {"days": [{"day_index": 9}]})
    joined = " ".join(lines)
    assert "congelada del día 8 al 14" in joined and "Misma despensa, distinta PREPARACIÓN" in joined
    assert "Sin congelador" in " ".join(hz.single_trip_prompt_lines(eff, {"days": []}))


def test_f_la_politica_respeta_el_ciclo_sin_congelador_y_lo_declara():
    import plan_policy as pp
    form = {"groceryDuration": "monthly", "freezerMode": "none", "freshTopup": "no", "mealOrganization": "balanced"}
    out = pp.compile_from_form(form)
    eff, rels = out["effective"], out.get("relaxations") or []
    assert eff["shopping"]["main_cycle_days"] == 30, "una compra para el mes se respeta"
    assert eff["shopping"]["fresh_topup_days"] is None
    r = [x for x in rels if x.get("reason_code") == "pantry_proteins_after_first_week"]
    assert r and r[0].get("action") == "applied"
    assert "pantry_proteins_after_first_week" in pp._REASON_COPY
    js = (_FRONT / "config" / "planPolicy.js").read_text(encoding="utf-8")
    assert "'pantry_proteins_after_first_week'" in js and "case 'pantry_proteins_after_first_week':" in js


def test_g_el_pdf_explica_la_compra_unica_segun_el_congelador():
    dash = (_FRONT / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    assert "const _freezerMode = String(_policyShopping?.freezer_mode || 'limited');" in dash
    assert "t('PERECEDEROS — UNA SOLA COMPRA (SIN CONGELADOR: CONSUME PRIMERO)')" in dash
    assert "t('PERECEDEROS — UNA SOLA COMPRA (CONGELA LAS PROTEÍNAS EL DÍA DE LA COMPRA)')" in dash
    assert "t('PERECEDEROS — UNA SOLA COMPRA (CONGELA LO DE LA SEGUNDA SEMANA)')" in dash
    assert "no congelas: la proteína fresca es para la primera semana" in dash
    assert "CONGELA O CONSUME PRIMERO" not in dash, "el copy genérico que mentía sin congelador desapareció"


def test_h_las_familias_de_despensa_repiten_ingredientes_con_distinta_preparacion():
    """La petición del dueño: arroz + atún, arroz + huevo, arroz + sardinas… distinta técnica, misma despensa."""
    import dish_registry as dr
    if not Path(dr.snapshot_path("do")).exists():
        pytest.skip("snapshots no compilados")
    for lib in dr.LIBRARIES:
        snap = json.loads(Path(dr.snapshot_path(lib)).read_text(encoding="utf-8"))
        pantry = [t for t in snap["templates"] if (t.get("logistics") or {}).get("pantry_only")]
        techniques = {str(t.get("technique")).lower() for t in pantry}
        assert len(techniques) >= 4, f"[{lib}] la despensa se prepara de {len(techniques)} formas; hacen falta ≥ 4"
        proteins = {str(t.get("protein")).lower() for t in pantry}
        assert {"atun", "huevo", "legumbre"} <= proteins, f"[{lib}] faltan familias de despensa: {proteins}"
