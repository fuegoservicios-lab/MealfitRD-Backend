"""[P1-LANDING-BENCH-1 · 2026-08-07] Anclas del benchmark del landing.

QUÉ ANCLA, y por qué cada cosa:

  1. PARIDAD chips↔wizard. `landing_benchmarks.py` espeja los chips del formulario
     (QMedical.jsx, QAllergies.jsx, formValidation.js). Si el wizard añade/renombra un
     chip y el espejo no se actualiza, el benchmark sigue midiendo un formulario que ya
     no existe — exactamente el bug que motivó este P-fix (los 20 perfiles held-out
     usan texto libre que el wizard no puede emitir desde 2026-08-01).
  2. MAPEO chip→regla, FUNCIONAL. Cada chip literal debe disparar su regla backend
     (`detect_active_rules` / `detect_active_medications`). Un rename de terms que
     rompa la detección deja al usuario sin capa clínica EN SILENCIO; este test lo
     convierte en rojo de CI.
  3. COBERTURA de la matriz. Los 20 perfiles cubren TODOS los chips ≥1 vez, respetan
     las reglas del wizard (embarazo solo female, cap de 3 condiciones) y son
     form-faithful (todos los `_REQUIRED_FORM_FIELDS` presentes).
  4. SCORER de seguridad, FUNCIONAL. Violación de alérgeno/dieta/mercurio detectada y
     categorizada; plan limpio = safe.
  5. DE-DRIFT del landing. Los hechos estructurales viven en `systemFacts.js` y los
     consumidores los importan en vez de escribirlos a mano (misma doctrina que
     `test_p1_paper_benchmark_ssot.py` para las cifras medidas).

tooltip-anchor: P1-LANDING-BENCH-1
"""
from __future__ import annotations

import re
from pathlib import Path

from landing_benchmarks import (
    CONDITION_CHIP_EXPECTED_RULE,
    FORM_ALLERGY_CHIPS,
    FORM_CONDITION_CHIPS,
    FORM_DIET_TYPES,
    FORM_MEDICATION_CHIPS,
    FORM_PREGNANCY_CHIPS,
    LANDING_REPORT_SECTIONS,
    MEDICATION_CHIP_EXPECTED_RULE,
    aggregate_safety,
    build_landing_profiles,
    build_report,
    score_plan_safety,
    strip_benchmark_meta,
    structural_facts,
)

_BACKEND = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

_QMEDICAL = _FRONTEND_SRC / "components" / "assessment" / "questions" / "QMedical.jsx"
_QALLERGIES = _FRONTEND_SRC / "components" / "assessment" / "questions" / "QAllergies.jsx"
_FORM_VALIDATION = _FRONTEND_SRC / "config" / "formValidation.js"
_SYSTEM_FACTS = _FRONTEND_SRC / "data" / "systemFacts.js"


# ---------------------------------------------------------------------------
# 1. Paridad chips ↔ wizard (parser-based, mismo layout workspace que
#    test_p1_paper_benchmark_ssot.py: backend/ y frontend/ hermanos)
# ---------------------------------------------------------------------------
def _js_string_array_items(text: str) -> set:
    """Todos los literales de string en el texto dado (para bloques de arrays JSX)."""
    return set(re.findall(r"'([^']+)'", text)) | set(re.findall(r'"([^"]+)"', text))


def test_condition_chips_match_wizard():
    src = _QMEDICAL.read_text(encoding="utf-8")
    # El array de chips de condición es el que se mapea a ChipOption (QMedical.jsx).
    m = re.search(r"\[((?:'[^']+',?\s*)+)\]\.map\(opt", src)
    assert m, "P1-LANDING-BENCH-1: no encuentro el array de chips de condición en QMedical.jsx."
    wizard = _js_string_array_items(m.group(1))
    assert wizard == set(FORM_CONDITION_CHIPS), (
        "P1-LANDING-BENCH-1: los chips de condición del wizard driftearon del espejo "
        f"FORM_CONDITION_CHIPS. wizard={sorted(wizard)} espejo={sorted(FORM_CONDITION_CHIPS)}. "
        "Actualiza landing_benchmarks.py Y la matriz de perfiles."
    )


def test_medication_chips_match_wizard():
    src = _QMEDICAL.read_text(encoding="utf-8")
    m = re.search(r"\[((?:'[^']+',?\s*)+)\]\.map\(med", src)
    assert m, "P1-LANDING-BENCH-1: no encuentro el array de chips de medicamentos en QMedical.jsx."
    wizard = _js_string_array_items(m.group(1))
    assert wizard == set(FORM_MEDICATION_CHIPS), (
        "P1-LANDING-BENCH-1: los chips de medicamentos del wizard driftearon del espejo "
        f"FORM_MEDICATION_CHIPS. wizard={sorted(wizard)} espejo={sorted(FORM_MEDICATION_CHIPS)}."
    )


def test_allergy_chips_match_wizard():
    src = _QALLERGIES.read_text(encoding="utf-8")
    vals = set(re.findall(r'val:\s*"([^"]+)"', src)) - {"Ninguna"}
    assert vals == set(FORM_ALLERGY_CHIPS), (
        "P1-LANDING-BENCH-1: los chips de alergia del wizard driftearon del espejo "
        f"FORM_ALLERGY_CHIPS. wizard={sorted(vals)} espejo={sorted(FORM_ALLERGY_CHIPS)}."
    )


def test_diet_types_match_wizard():
    src = _FORM_VALIDATION.read_text(encoding="utf-8")
    m = re.search(r"DIET_TYPES\s*=\s*Object\.freeze\(\[([^\]]+)\]\)", src)
    assert m, "P1-LANDING-BENCH-1: no encuentro DIET_TYPES en formValidation.js."
    wizard = _js_string_array_items(m.group(1))
    assert wizard == set(FORM_DIET_TYPES), (
        f"P1-LANDING-BENCH-1: DIET_TYPES drifteó. wizard={sorted(wizard)} "
        f"espejo={sorted(FORM_DIET_TYPES)}."
    )


# ---------------------------------------------------------------------------
# 2. Mapeo chip → regla backend (funcional, sin DB)
# ---------------------------------------------------------------------------
def test_every_condition_chip_fires_its_backend_rule():
    from condition_rules import detect_active_rules
    for chip, expected in CONDITION_CHIP_EXPECTED_RULE.items():
        ids = [r.id for r in detect_active_rules({"medicalConditions": [chip]})]
        assert expected in ids, (
            f"P1-LANDING-BENCH-1: el chip {chip!r} del wizard NO dispara la regla "
            f"{expected!r} (disparó {ids}). Un usuario real con ese chip quedaría sin "
            "su capa clínica en silencio."
        )


def test_every_medication_chip_fires_its_backend_rule():
    from medication_rules import detect_active_medications, requires_medication_review
    for chip, expected in MEDICATION_CHIP_EXPECTED_RULE.items():
        ids = [r.id for r in detect_active_medications({"medications": [chip]})]
        assert expected in ids, (
            f"P1-LANDING-BENCH-1: el chip {chip!r} NO dispara la regla {expected!r} "
            f"(disparó {ids})."
        )
        # Todo medicamento con interacción conocida amerita el gate FS9.
        assert requires_medication_review({"medications": [chip]}), (
            f"P1-LANDING-BENCH-1: {chip!r} no levanta requires_medication_review."
        )


def test_chip_maps_cover_all_chips():
    assert set(CONDITION_CHIP_EXPECTED_RULE) == set(FORM_CONDITION_CHIPS) | set(FORM_PREGNANCY_CHIPS)
    assert set(MEDICATION_CHIP_EXPECTED_RULE) == set(FORM_MEDICATION_CHIPS)


# ---------------------------------------------------------------------------
# 3. Cobertura y fidelidad de la matriz
# ---------------------------------------------------------------------------
def _profiles():
    return build_landing_profiles()


def test_matrix_covers_every_chip_and_diet():
    profiles = _profiles()
    conds = {c for p in profiles for c in p["medicalConditions"]}
    meds = {m for p in profiles for m in p["medications"]}
    alls = {a for p in profiles for a in p["allergies"]}
    diets = {p["dietType"] for p in profiles}
    goals = {p["mainGoal"] for p in profiles}
    missing_c = (set(FORM_CONDITION_CHIPS) | set(FORM_PREGNANCY_CHIPS)) - conds
    missing_m = set(FORM_MEDICATION_CHIPS) - meds
    missing_a = set(FORM_ALLERGY_CHIPS) - alls
    assert not missing_c, f"P1-LANDING-BENCH-1: condiciones sin perfil: {sorted(missing_c)}"
    assert not missing_m, f"P1-LANDING-BENCH-1: medicamentos sin perfil: {sorted(missing_m)}"
    assert not missing_a, f"P1-LANDING-BENCH-1: alergias sin perfil: {sorted(missing_a)}"
    assert diets == set(FORM_DIET_TYPES), f"P1-LANDING-BENCH-1: dietas sin cubrir: {diets}"
    assert goals == {"lose_fat", "gain_muscle", "maintenance", "performance"}


def test_matrix_respects_wizard_rules():
    """Embarazo/Lactancia solo en perfiles female (QGender los limpia si no) y máximo
    3 condiciones REALES (cap del wizard; embarazo exento — MAX_MEDICAL_CONDITIONS)."""
    for p in _profiles():
        preg = [c for c in p["medicalConditions"] if c in FORM_PREGNANCY_CHIPS]
        if preg:
            assert p["gender"] == "female", (
                f"P1-LANDING-BENCH-1: perfil {p['_id']} ({p['_label']}) declara {preg} "
                "con gender != female — el wizard no puede producir ese payload."
            )
        reales = [c for c in p["medicalConditions"]
                  if c not in FORM_PREGNANCY_CHIPS and c != "Ninguna"]
        assert len(reales) <= 3, (
            f"P1-LANDING-BENCH-1: perfil {p['_id']} excede el cap de 3 condiciones del wizard."
        )


def test_matrix_profiles_are_form_faithful():
    """Todos los campos que el backend exige (`routers/plans.py::_REQUIRED_FORM_FIELDS`,
    parseado del source) están presentes en cada perfil — la matriz debe poder entrar
    por /analyze sin 422."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    m = re.search(r"_REQUIRED_FORM_FIELDS\s*=\s*\((.*?)\n\)", src, re.DOTALL)
    assert m, "P1-LANDING-BENCH-1: no encuentro _REQUIRED_FORM_FIELDS en routers/plans.py."
    required = re.findall(r'"([a-zA-Z_]+)"', re.sub(r"#[^\n]*", "", m.group(1)))
    assert len(required) >= 15, f"parse sospechoso de _REQUIRED_FORM_FIELDS: {required}"
    for p in _profiles():
        stripped = strip_benchmark_meta(p)
        missing = [f for f in required if f not in stripped]
        assert not missing, (
            f"P1-LANDING-BENCH-1: perfil {p['_id']} ({p['_label']}) sin campos "
            f"requeridos {missing} — dejaría de ser form-faithful."
        )
        assert not any(k.startswith("_") for k in stripped), "strip_benchmark_meta no strippeó."


def test_matrix_expectations_match_detection():
    """Los `_expect.condition_rules`/`medication_rules` de cada perfil coinciden con lo
    que los detectores de producción disparan sobre ese payload — la expectativa no
    puede driftear de la realidad."""
    from condition_rules import detect_active_rules
    from medication_rules import detect_active_medications
    for p in _profiles():
        exp = p["_expect"]
        got_c = {r.id for r in detect_active_rules(strip_benchmark_meta(p))}
        got_m = {r.id for r in detect_active_medications(strip_benchmark_meta(p))}
        assert set(exp.get("condition_rules", [])) <= got_c, (
            f"perfil {p['_id']}: espera {exp.get('condition_rules')} pero detecta {sorted(got_c)}")
        assert set(exp.get("medication_rules", [])) <= got_m, (
            f"perfil {p['_id']}: espera {exp.get('medication_rules')} pero detecta {sorted(got_m)}")


# ---------------------------------------------------------------------------
# 4. Scorer de seguridad (funcional, sin LLM ni DB)
# ---------------------------------------------------------------------------
def _plan(meals_by_day):
    return {"days": [{"meals": meals} for meals in meals_by_day]}


def test_safety_scorer_detects_allergen_and_categorizes():
    profile = next(p for p in _profiles() if p["_label"] == "alergias_mar_nuez_soya")
    plan = _plan([[{"name": "Arroz con camarones", "ingredients": ["Camarones", "Arroz blanco"]}]])
    out = score_plan_safety(plan, profile)
    assert not out["safe"] and out["safety_violations"], "el alérgeno Mariscos no se detectó"
    assert out["safety_violations"][0]["categoria"] == "alergeno"


def test_safety_scorer_detects_diet_violation():
    import graph_orchestrator as go
    go.DIET_HARD_GUARD = True  # determinismo (default ya es True)
    profile = next(p for p in _profiles() if p["_label"] == "vegana_dm2")
    plan = _plan([[{"name": "Pollo guisado", "ingredients": ["Pechuga de pollo", "Cebolla"]}]])
    out = score_plan_safety(plan, profile)
    assert any(v["categoria"] == "dieta" for v in out["safety_violations"])


def test_safety_scorer_detects_mercury_in_pregnancy():
    profile = next(p for p in _profiles() if p["_label"] == "embarazo")
    plan = _plan([[{"name": "Filete a la plancha", "ingredients": ["Pez espada", "Limon"]}]])
    out = score_plan_safety(plan, profile)
    assert any(v["categoria"] == "mercurio_embarazo" for v in out["safety_violations"])


def test_safety_scorer_clean_plan_and_aggregate():
    profile = next(p for p in _profiles() if p["_label"] == "insulina_hipoglucemia")
    meals = [{"name": f"Comida {i}", "ingredients": ["Arroz blanco", "Habichuelas"]}
             for i in range(5)]
    out = score_plan_safety(_plan([meals]), profile)
    assert out["safe"] and out["min_meals_expected"] == 5 and out["min_meals_ok"]
    assert out["professional_review_expected"] and not out["professional_review_flagged"]
    agg = aggregate_safety([out])
    assert agg["n"] == 1 and agg["plans_sin_violaciones_pct"] == 100.0
    assert agg["min_meals_compliance_pct"] == 100.0 and agg["fs9_flag_presente_pct"] == 0.0


def test_vitk_monitor_runs_for_warfarin_profile():
    profile = next(p for p in _profiles() if p["_label"] == "warfarina_vitk")
    plan = _plan([[{"name": "Ensalada", "ingredients": ["Espinaca", "Tomate"]}],
                  [{"name": "Arroz", "ingredients": ["Arroz blanco"]}]])
    out = score_plan_safety(plan, profile)
    assert out.get("vitamin_k", {}).get("applicable") is True


# ---------------------------------------------------------------------------
# 5. Hechos estructurales + contrato del reporte
# ---------------------------------------------------------------------------
def test_structural_facts_are_derived_and_honest():
    facts = structural_facts()
    assert facts["micronutrientes_dri"] == 17, (
        "los micros DRI cambiaron — actualizar systemFacts.js Y el copy del landing")
    # El hallazgo de producto (2026-08-07): condiciones con regla backend que el
    # formulario YA NO puede expresar (texto libre retirado). Si esto cambia (p.ej.
    # se añade el chip ERC), actualizar docs/landing_benchmarks.md §Hallazgos.
    assert "renal" in facts["condiciones_solo_backend"]
    assert "maoi" in facts["medicaciones_solo_backend"]
    assert facts["reglas_condicion_backend"] >= len(facts["condiciones_alcanzables_desde_formulario"])


def test_report_contract_rejects_unknown_sections():
    import pytest
    r = build_report("structural", structural={"x": 1})
    assert r["schema_version"] == 1 and r["mode"] == "structural" and "structural" in r
    assert set(LANDING_REPORT_SECTIONS) >= set(r) - {"schema_version", "mode"}
    with pytest.raises(ValueError):
        build_report("live", invented_section={})


def test_runner_and_doc_exist_with_all_modes():
    runner = (_BACKEND / "scripts" / "landing_benchmark.py").read_text(encoding="utf-8")
    for mode in ("structural", "live", "telemetry", "score"):
        assert f'"{mode}"' in runner, f"el runner perdió el modo {mode}"
    doc = (_BACKEND / "docs" / "landing_benchmarks.md").read_text(encoding="utf-8")
    assert "P1-LANDING-BENCH-1" in doc and "Matriz de perfiles" in doc
    # La guía de mejora es la mitad del valor del doc: cada métrica → palanca.
    assert "palanca" in doc.lower()


# ---------------------------------------------------------------------------
# 6. De-drift del landing (systemFacts.js como SSOT de hechos estructurales)
# ---------------------------------------------------------------------------
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)


def _js_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = _BLOCK_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    return _LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def test_system_facts_ssot_exists_and_matches_backend():
    assert _SYSTEM_FACTS.exists(), "P1-LANDING-BENCH-1: falta frontend/src/data/systemFacts.js."
    src = _SYSTEM_FACTS.read_text(encoding="utf-8")
    m = re.search(r"MICROS_TRACKED\s*=\s*(\d+)", src)
    assert m and int(m.group(1)) == structural_facts()["micronutrientes_dri"], (
        "P1-LANDING-BENCH-1: MICROS_TRACKED en systemFacts.js != len(dri_targets()) del "
        "backend. El landing estaría afirmando un número que el motor no entrega."
    )


def test_landing_consumers_import_facts_not_literals():
    """La forma literal del drift que este P-fix cierra no puede volver: los stats
    y la prosa citan el SSOT, no el número a mano."""
    pages = _FRONTEND_SRC / "pages"
    home = _FRONTEND_SRC / "components" / "home"

    features = _js_code(pages / "FeaturesPage.jsx")
    assert "systemFacts" in features and not re.search(r"num:\s*'200\+'", features), (
        "FeaturesPage.jsx volvió a escribir el catálogo a mano — importa VERIFIED_FOODS_LABEL.")

    hiw_page = _js_code(pages / "HowItWorksPage.jsx")
    assert "systemFacts" in hiw_page and "data/benchmark" in hiw_page, (
        "HowItWorksPage.jsx debe importar systemFacts Y benchmark (BANDS).")
    assert not re.search(r"num:\s*'17'", hiw_page), (
        "HowItWorksPage.jsx volvió a escribir '17' a mano — es MICROS_TRACKED.")
    assert "95–105" not in hiw_page and "90–112" not in hiw_page, (
        "HowItWorksPage.jsx re-declaró las bandas como prosa; se derivan de BANDS "
        "(la clase de drift que benchmark.js existe para cerrar).")

    hiw_home = _js_code(home / "HowItWorks.jsx")
    assert "systemFacts" in hiw_home and "17 micronutrientes" not in hiw_home, (
        "components/home/HowItWorks.jsx re-declaró los 17 micros a mano.")

    for name in ("Hero.jsx", "ClosingBand.jsx"):
        code = _js_code(home / name)
        assert "TIER_CREDITS" in code, f"{name} debe derivar los créditos de config/plans.js."
        assert not re.search(r"10\s+planes", code, re.IGNORECASE), (
            f"{name} volvió a escribir '10 planes' a mano — es TIER_CREDITS.gratis.")

    route_title = _js_code(_FRONTEND_SRC / "components" / "layout" / "RouteTitle.jsx")
    assert "systemFacts" in route_title and "+200 alimentos" not in route_title, (
        "RouteTitle.jsx conserva la 4ª grafía del catálogo ('+200') — usa VERIFIED_FOODS_LABEL.")
