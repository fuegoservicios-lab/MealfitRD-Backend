"""[P1-LANDING-BENCH-1 · 2026-08-07] Benchmark del landing — matriz clínica del FORMULARIO real.

POR QUÉ EXISTE. Los benchmarks previos miden el motor con perfiles sintéticos de texto libre
("Diabetes tipo 2", "Enfermedad renal crónica") que el formulario actual NO puede producir: desde
P1-MEDICAL-CONDITIONS-CAP (2026-08-01) el wizard solo emite CHIPS cerrados — 7 condiciones (+2 de
embarazo si gender=female), 14 medicamentos, 6 alergias y 3 dietas. Ningún benchmark ejercitaba ese
espacio: cero perfiles con medicamentos, cero con alergias, cero veganos, y las superficies de
CAMBIO (swap individual / regenerate-day) sin benchmark alguno. Este módulo define la matriz de
perfiles FIEL AL FORMULARIO y los scorers deterministas (sin LLM) cuyos resultados alimentan las
cifras públicas del landing (`frontend/src/data/benchmark.js` + `frontend/src/data/systemFacts.js`)
y la guía de mejora del motor (`docs/landing_benchmarks.md`).

QUÉ NO ES: no reemplaza a `scripts/benchmark_macro_compliance.py` (precisión de macros, gate
nightly) ni a `plan_gym` (7 ejes de calidad). Los COMPONE: el runner `scripts/landing_benchmark.py`
genera con esta matriz y puntúa con gym + seguridad clínica de este módulo.

Invariante de honestidad: este módulo NUNCA inventa una cifra publicable — produce mediciones que
el dueño revisa antes de tocar el SSOT del landing (regla de `test_p1_paper_benchmark_ssot.py`).

Test ancla: tests/test_p1_landing_bench_1_anchors.py (paridad chips↔wizard, mapeo chip→regla,
cobertura de la matriz, scorers funcionales, de-drift del landing).
tooltip-anchor: P1-LANDING-BENCH-1
"""
from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════════════════════
# 1. Espejo de los CHIPS del formulario (SSOT frontend: questions/QMedical.jsx, QAllergies.jsx,
#    config/formValidation.js). El test ancla parsea el JSX y falla si driftea.
#    tooltip-anchor: P1-LANDING-BENCH-1-CHIPS
# ════════════════════════════════════════════════════════════════════════════════════════════

# [P1-MEDICAL-SCOPE-GATE · 2026-08-09] +4 chips. `renal`, `anemia`, `gout` y `nafld`
# tenían regla clínica completa y NO eran declarables desde el wizard: quien las tenía
# recibía un plan sin ninguna de sus reglas y sin aviso. Con esto, el hallazgo del audit
# «renal ya NO expresable desde el form» (docs/landing_benchmarks.md) queda cerrado.
#
# NO incluye el chip «Otra condición (no listada)»: ese no es una condición clínica sino
# la señal del gate de alcance (bloquea la generación, ver `_has_out_of_scope_clinical_
# declaration` en routers/plans.py). Meterlo aquí haría que el benchmark intentara
# generar perfiles que el propio sistema rechaza por diseño.
FORM_CONDITION_CHIPS = (
    "Diabetes T2", "Hipertensión", "Colesterol Alto", "Gastritis",
    "SOP (PCOS)", "Hipotiroidismo", "Cirugía Bariátrica",
    "Enfermedad Renal", "Anemia", "Gota / Ácido Úrico", "Hígado Graso",
)

# Solo visibles con gender=female (QMedical.jsx / PREGNANCY_CHIP_LABELS); comparten
# el array medicalConditions y están EXENTOS del cap de 3 condiciones.
FORM_PREGNANCY_CHIPS = ("Embarazo", "Lactancia")

# [P1-MEDICAL-SCOPE-GATE · 2026-08-09] +«Antidepresivo IMAO». Era el ÚNICO chip cuya
# regla (`maoi`, tiramina — quesos curados, embutidos, fermentados) no tenía forma de
# dispararse desde el wizard, y es la interacción más peligrosa del registry: tiramina
# + IMAO es crisis hipertensiva. Mismo criterio que arriba — «Otro medicamento (no
# listado)» NO entra: es señal de gate, no un fármaco.
FORM_MEDICATION_CHIPS = (
    "Metformina", "Insulina", "Glibenclamida", "Lisinopril", "Losartán",
    "Amlodipina", "Hidroclorotiazida", "Espironolactona", "Atorvastatina",
    "Levotiroxina", "Omeprazol", "Prednisona", "Warfarina", "Alopurinol",
    "Antidepresivo IMAO",
)

# [P2-ALLERGEN-CHIPS-REACH-ENGINE · 2026-08-21] +«Pescado» y +«Mani». El motor ya sabía bloquear
# las dos clases y el formulario no tenía cómo pedírselas: el marisco tenía chip y el pescado no
# —en un beta cuyo primer país es España— y «Frutos Secos» NO cubre el maní, que es una legumbre.
# El guard `test_allergy_chips_match_wizard` cazó este espejo en cuanto los chips entraron: es
# exactamente para lo que existe, y sin él la cifra pública de «6 alergias» habría quedado stale.
FORM_ALLERGY_CHIPS = ("Lacteos", "Gluten", "Huevo", "Mariscos", "Frutos Secos", "Soya",
                      "Pescado", "Mani")

FORM_DIET_TYPES = ("balanced", "vegetarian", "vegan")

# Mapeo chip → id de regla backend ESPERADO. Verificado funcionalmente por el test ancla
# invocando detect_active_rules / detect_active_medications con el chip literal — si un
# rename del chip o de los terms rompe la detección, el test falla ANTES de que un usuario
# real pierda su capa clínica en silencio.
CONDITION_CHIP_EXPECTED_RULE = {
    "Diabetes T2": "dm2",
    "Hipertensión": "hta",
    "Colesterol Alto": "dyslipidemia",
    "Gastritis": "gastritis",
    "SOP (PCOS)": "pcos",
    "Hipotiroidismo": "hypothyroid",
    "Cirugía Bariátrica": "bariatric",
    "Embarazo": "pregnancy",
    "Lactancia": "pregnancy",
    # [P1-MEDICAL-SCOPE-GATE · 2026-08-09] Los 4 que tenian regla y no tenian chip.
    "Enfermedad Renal": "renal",
    "Anemia": "anemia",
    "Gota / Ácido Úrico": "gout",
    "Hígado Graso": "nafld",
}

MEDICATION_CHIP_EXPECTED_RULE = {
    "Metformina": "metformin",
    "Insulina": "insulin_secretagogue",
    "Glibenclamida": "insulin_secretagogue",
    "Lisinopril": "ace_arb",
    "Losartán": "ace_arb",
    "Amlodipina": "calcium_channel_blocker",
    "Hidroclorotiazida": "diuretic_depleting",
    "Espironolactona": "potassium_sparing_diuretic",
    "Atorvastatina": "statin",
    "Levotiroxina": "levothyroxine",
    "Omeprazol": "ppi",
    "Prednisona": "corticosteroid",
    "Warfarina": "anticoagulant",
    "Alopurinol": "gout",
    # [P1-MEDICAL-SCOPE-GATE · 2026-08-09] El unico fármaco del registry sin chip.
    "Antidepresivo IMAO": "maoi",
}


# ════════════════════════════════════════════════════════════════════════════════════════════
# 2. Matriz de perfiles fiel al formulario
# ════════════════════════════════════════════════════════════════════════════════════════════

def _perfil(idx, label, *, gender, age, weight, height, goal, activity,
            conditions=("Ninguna",), medications=("Ninguno",), allergies=("Ninguna",),
            diet="balanced", expect=None):
    """Payload con la MISMA forma que emite el wizard (Plan.jsx → /analyze). Los campos
    con `_` prefijo son metadata del benchmark (el frontend los strippea; aquí los strippea
    el runner antes de llamar al pipeline)."""
    return {
        "_id": idx, "_label": label, "_expect": dict(expect or {}),
        "age": age, "weight": weight, "height": height, "gender": gender,
        "weightUnit": "kg", "mainGoal": goal, "activityLevel": activity,
        "householdSize": 1, "groceryDuration": "weekly",
        "motivation": "Mejorar mi salud de forma sostenible.",
        "allergies": list(allergies), "medicalConditions": list(conditions),
        "medications": list(medications), "dietType": diet,
        "scheduleType": "standard", "cookingTime": "30min", "budget": "medium",
        "sleepHours": "7-8 horas", "stressLevel": "Moderado",
        "dislikes": ["Ninguno"], "struggles": ["Ninguno"], "user_id": "guest",
    }


def build_landing_profiles(country: str = "DO") -> list:
    """La matriz: 25 perfiles que cubren TODOS los chips del formulario al menos una vez.

    Diseño (ver docs/landing_benchmarks.md → «Matriz de perfiles»):
      - 1 chip de condición por perfil dedicado, con el medicamento típico de esa condición.
      - Combos de riesgo real: cap-3 (DM2+HTA+Colesterol), warfarina (vit K), doble
        potasio-elevador (espironolactona+IECA), insulina+sulfonilurea (≥5 tomas).
      - Las 8 alergias repartidas en 2 perfiles multi-alergia (el segundo mezcla a propósito
        pescado con marisco y maní con fruto seco: son las dos parejas que el usuario
        confunde, así que ahí es donde se prueba que el motor las separa).
      - vegetariana pura y vegana×DM2 (cruce dieta×condición).
      - 2 baselines sanos como referencia de precisión.
    [P2-LANDING-BENCH-COUNTRY · 2026-08-21] `country` (default 'DO' ⇒ la matriz de siempre)
    permite correr la MISMA matriz clínica bajo otro país. Fail-safe por `canonicalize_country`:
    un valor que no reconoce cae a 'DO', igual que el resto del sistema.

    tooltip-anchor: P1-LANDING-BENCH-1-MATRIX
    """
    e = dict  # brevedad
    try:
        from constants import canonicalize_country as _cc_lb
        _cc = _cc_lb(country)
    except Exception:
        _cc = "DO"
    _perfiles = [
        _perfil(1, "baseline_m", gender="male", age=35, weight=90, height=180,
                goal="lose_fat", activity="moderate"),
        _perfil(2, "baseline_f", gender="female", age=27, weight=58, height=163,
                goal="gain_muscle", activity="active"),
        _perfil(3, "dm2_metformina", gender="male", age=52, weight=95, height=175,
                goal="lose_fat", activity="sedentary",
                conditions=["Diabetes T2"], medications=["Metformina"],
                expect=e(condition_rules=["dm2"], medication_rules=["metformin"], fs9=True)),
        _perfil(4, "hta_losartan_hctz", gender="female", age=58, weight=78, height=160,
                goal="maintenance", activity="light",
                conditions=["Hipertensión"], medications=["Losartán", "Hidroclorotiazida"],
                expect=e(condition_rules=["hta"],
                         medication_rules=["ace_arb", "diuretic_depleting"], fs9=True)),
        _perfil(5, "dislipidemia_estatina", gender="male", age=48, weight=88, height=172,
                goal="lose_fat", activity="moderate",
                conditions=["Colesterol Alto"], medications=["Atorvastatina"],
                expect=e(condition_rules=["dyslipidemia"], medication_rules=["statin"], fs9=True)),
        _perfil(6, "gastritis_ibp", gender="female", age=33, weight=64, height=158,
                goal="maintenance", activity="moderate",
                conditions=["Gastritis"], medications=["Omeprazol"],
                expect=e(condition_rules=["gastritis"], medication_rules=["ppi"], fs9=True)),
        _perfil(7, "sop", gender="female", age=29, weight=74, height=162,
                goal="lose_fat", activity="light",
                conditions=["SOP (PCOS)"], medications=["Metformina"],
                expect=e(condition_rules=["pcos"], medication_rules=["metformin"], fs9=True)),
        _perfil(8, "hipotiroidismo_levo", gender="female", age=41, weight=70, height=165,
                goal="lose_fat", activity="moderate",
                conditions=["Hipotiroidismo"], medications=["Levotiroxina"],
                expect=e(condition_rules=["hypothyroid"], medication_rules=["levothyroxine"],
                         fs9=True, timing_advisory=True)),
        _perfil(9, "bariatrica", gender="female", age=38, weight=98, height=166,
                goal="lose_fat", activity="light",
                conditions=["Cirugía Bariátrica"],
                expect=e(condition_rules=["bariatric"], min_meals_per_day=5)),
        _perfil(10, "embarazo", gender="female", age=31, weight=68, height=164,
                goal="maintenance", activity="light",
                conditions=["Embarazo"],
                expect=e(condition_rules=["pregnancy"], mercury_guard=True)),
        _perfil(11, "lactancia", gender="female", age=30, weight=66, height=161,
                goal="maintenance", activity="light",
                conditions=["Lactancia"],
                expect=e(condition_rules=["pregnancy"], mercury_guard=True)),
        _perfil(12, "combo_cap3", gender="male", age=61, weight=92, height=170,
                goal="lose_fat", activity="sedentary",
                conditions=["Diabetes T2", "Hipertensión", "Colesterol Alto"],
                medications=["Metformina", "Lisinopril", "Atorvastatina"],
                expect=e(condition_rules=["dm2", "hta", "dyslipidemia"],
                         medication_rules=["metformin", "ace_arb", "statin"], fs9=True)),
        _perfil(13, "warfarina_vitk", gender="male", age=66, weight=80, height=173,
                goal="maintenance", activity="light",
                conditions=["Hipertensión"], medications=["Warfarina"],
                expect=e(condition_rules=["hta"], medication_rules=["anticoagulant"],
                         fs9=True, vitk_monitor=True)),
        _perfil(14, "potasio_doble", gender="male", age=59, weight=85, height=176,
                goal="maintenance", activity="light",
                conditions=["Hipertensión"], medications=["Espironolactona", "Lisinopril"],
                expect=e(condition_rules=["hta"],
                         medication_rules=["potassium_sparing_diuretic", "ace_arb"], fs9=True)),
        _perfil(15, "insulina_hipoglucemia", gender="female", age=46, weight=82, height=159,
                goal="lose_fat", activity="light",
                conditions=["Diabetes T2"], medications=["Insulina", "Glibenclamida"],
                expect=e(condition_rules=["dm2"], medication_rules=["insulin_secretagogue"],
                         fs9=True, min_meals_per_day=5)),
        _perfil(16, "polifarmacia_gota", gender="male", age=63, weight=89, height=171,
                goal="maintenance", activity="sedentary",
                conditions=["Hipertensión"],
                medications=["Amlodipina", "Prednisona", "Alopurinol"],
                expect=e(condition_rules=["hta"],
                         medication_rules=["calcium_channel_blocker", "corticosteroid", "gout"],
                         fs9=True)),
        _perfil(17, "alergias_lacteo_gluten_huevo", gender="female", age=26, weight=55, height=160,
                goal="lose_fat", activity="moderate",
                allergies=["Lacteos", "Gluten", "Huevo"],
                expect=e(allergens=["Lacteos", "Gluten", "Huevo"])),
        # [P2-ALLERGEN-CHIPS-REACH-ENGINE · 2026-08-21] +«Pescado» y +«Mani» en este perfil.
        # El guard `test_matrix_covers_every_chip_and_diet` exige que la matriz ejercite CADA chip
        # al menos una vez, y al añadir los dos chips nuevos al wizard la matriz se quedó corta —
        # el guard lo cazó en el mismo commit. Van juntos aquí a propósito: son las dos clases que
        # el usuario CONFUNDE con las que ya había (pescado con marisco, maní con fruto seco), así
        # que el perfil que las mezcla es el que de verdad prueba que el motor las separa.
        _perfil(18, "alergias_mar_nuez_soya", gender="male", age=24, weight=72, height=178,
                goal="performance", activity="athlete",
                allergies=["Mariscos", "Frutos Secos", "Soya", "Pescado", "Mani"],
                expect=e(allergens=["Mariscos", "Frutos Secos", "Soya", "Pescado", "Mani"])),
        _perfil(19, "vegetariana", gender="female", age=36, weight=62, height=167,
                goal="maintenance", activity="moderate", diet="vegetarian",
                expect=e(diet="vegetarian")),
        _perfil(20, "vegana_dm2", gender="male", age=44, weight=86, height=174,
                goal="lose_fat", activity="moderate", diet="vegan",
                conditions=["Diabetes T2"], medications=["Metformina"],
                expect=e(diet="vegan", condition_rules=["dm2"],
                         medication_rules=["metformin"], fs9=True)),
        # [P1-MEDICAL-SCOPE-GATE · 2026-08-09] Perfiles 21-25: los 5 chips que el
        # motor sabía manejar y el formulario no dejaba declarar. Sin estos, la
        # matriz mediría un formulario que ya no existe.
        #
        # El 21 es el que más aporta: la ERC no solo activa su regla, activa las
        # DOS ramas de precedencia que `build_condition_prompt` tiene escritas
        # (dm2+renal y hta+renal, donde el potasio del patrón DASH choca con la
        # moderación renal). Hasta ahora ese código no era alcanzable desde un
        # perfil del formulario, así que nunca se ejercitaba.
        _perfil(21, "renal_hta", gender="male", age=67, weight=79, height=170,
                goal="maintenance", activity="sedentary",
                conditions=["Enfermedad Renal", "Hipertensión"], medications=["Losartán"],
                expect=e(condition_rules=["renal", "hta"], medication_rules=["ace_arb"],
                         fs9=True)),
        _perfil(22, "anemia_ferropenica", gender="female", age=29, weight=54, height=161,
                goal="maintenance", activity="moderate",
                conditions=["Anemia"],
                expect=e(condition_rules=["anemia"])),
        _perfil(23, "gota_alopurinol", gender="male", age=52, weight=94, height=175,
                goal="lose_fat", activity="light",
                conditions=["Gota / Ácido Úrico"], medications=["Alopurinol"],
                expect=e(condition_rules=["gout"], medication_rules=["gout"], fs9=True)),
        _perfil(24, "higado_graso", gender="male", age=48, weight=98, height=173,
                goal="lose_fat", activity="sedentary",
                conditions=["Hígado Graso"],
                expect=e(condition_rules=["nafld"])),
        # Tiramina + IMAO = crisis hipertensiva. Es la interacción más peligrosa
        # del registry y hasta hoy NINGÚN perfil podía activarla, porque el chip
        # no existía.
        _perfil(25, "imao_tiramina", gender="female", age=41, weight=68, height=165,
                goal="maintenance", activity="light",
                medications=["Antidepresivo IMAO"],
                expect=e(medication_rules=["maoi"], fs9=True)),
    ]
    # [P2-LANDING-BENCH-COUNTRY · 2026-08-21] El país se estampa AQUÍ y sólo aquí. El wizard lo
    # emite desde Fase 0, así que sin este campo la promesa del docstring de `_perfil` —«la MISMA
    # forma que emite el wizard»— era falsa desde el día del flip, y un banco que no reproduce la
    # entrada real mide otra cosa.
    #
    # Un parámetro `country=` en `_perfil` sería código MUERTO: ningún perfil de la matriz elige
    # país propio, es una dimensión de la CORRIDA entera. La mutación lo destapó — quitarlo de
    # `_perfil` no rompía ningún test porque este bucle ya ponía el campo.
    for _p in _perfiles:
        _p["country"] = _cc
    return _perfiles


def strip_benchmark_meta(profile: dict) -> dict:
    """Igual que `stripInternalFlags` del frontend: los `_`-prefijados no viajan al pipeline."""
    return {k: v for k, v in profile.items() if not k.startswith("_")}


# ════════════════════════════════════════════════════════════════════════════════════════════
# 3. Scorers deterministas (sin LLM)
# ════════════════════════════════════════════════════════════════════════════════════════════

def _viol_categoria(v: str) -> str:
    low = (v or "").lower()
    if low.startswith("alérgeno") or low.startswith("alergeno"):
        return "alergeno"
    if "no apto para la dieta" in low:
        return "dieta"
    if "mercurio" in low or "embarazo" in low:
        return "mercurio_embarazo"
    return "otra"


def score_plan_safety(plan: dict, profile: dict) -> dict:
    """Puntúa un plan ENTREGADO contra el contrato clínico de su perfil.

    Reusa los backstops de producción (NO reimplementa): `clinical_backstop_for_meal`
    (alérgenos C2 + dieta P1-DIET-HARD-GUARD + mercurio-embarazo) por comida, el monitor
    `vitamin_k_consistency` para anticoagulados, y los flags FS9/`requires_professional_review`.
    Un plan seguro devuelve `safety_violations == []`.
    tooltip-anchor: P1-LANDING-BENCH-1-SAFETY
    """
    from graph_orchestrator import clinical_backstop_for_meal
    exp = profile.get("_expect") or {}
    allergies = [a for a in (profile.get("allergies") or []) if a and a != "Ninguna"]
    diet = profile.get("dietType") or "balanced"

    days = (plan or {}).get("days") or []
    violations, meals_per_day = [], []
    for di, day in enumerate(days):
        meals = (day or {}).get("meals") or []
        meals_per_day.append(len(meals))
        for mi, meal in enumerate(meals):
            if not isinstance(meal, dict):
                continue
            for v in clinical_backstop_for_meal(
                    meal, allergies=allergies, diet_type=diet, form_data=profile):
                violations.append({
                    "day": di + 1,
                    "meal": meal.get("name") or f"meal_{mi}",
                    "violation": v,
                    "categoria": _viol_categoria(v),
                })

    out = {
        "profile_id": profile.get("_id"),
        "label": profile.get("_label"),
        "days": len(days),
        "meals_scanned": sum(meals_per_day),
        "safety_violations": violations,
        "safe": not violations,
        "meals_per_day": meals_per_day,
    }

    mm = exp.get("min_meals_per_day")
    if mm:
        out["min_meals_expected"] = mm
        out["min_meals_ok"] = bool(meals_per_day) and min(meals_per_day) >= mm

    if exp.get("vitk_monitor"):
        try:
            from medication_rules import vitamin_k_consistency
            out["vitamin_k"] = vitamin_k_consistency(plan)
        except Exception as _vk_e:
            out["vitamin_k"] = {"error": f"{type(_vk_e).__name__}: {_vk_e}"}

    if exp.get("fs9"):
        out["professional_review_expected"] = True
        out["professional_review_flagged"] = bool((plan or {}).get("requires_professional_review"))

    return out


def aggregate_safety(results: list) -> dict:
    """Agrega los scores de `score_plan_safety` en las cifras que el landing consumiría."""
    rows = [r for r in (results or []) if isinstance(r, dict) and "safe" in r]
    if not rows:
        return {"n": 0}
    total_meals = sum(r.get("meals_scanned", 0) for r in rows)
    all_viols = [v for r in rows for v in r.get("safety_violations", [])]
    por_categoria = {}
    for v in all_viols:
        por_categoria[v["categoria"]] = por_categoria.get(v["categoria"], 0) + 1
    mm_rows = [r for r in rows if "min_meals_ok" in r]
    fs9_rows = [r for r in rows if r.get("professional_review_expected")]
    return {
        "n": len(rows),
        "meals_scanned": total_meals,
        "plans_sin_violaciones_pct": round(100.0 * sum(1 for r in rows if r["safe"]) / len(rows), 1),
        "violaciones_totales": len(all_viols),
        "violaciones_por_categoria": por_categoria,
        "min_meals_compliance_pct": (
            round(100.0 * sum(1 for r in mm_rows if r.get("min_meals_ok")) / len(mm_rows), 1)
            if mm_rows else None),
        "fs9_flag_presente_pct": (
            round(100.0 * sum(1 for r in fs9_rows if r.get("professional_review_flagged"))
                  / len(fs9_rows), 1)
            if fs9_rows else None),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════
# 4. Hechos estructurales (los números "contables" del landing)
# ════════════════════════════════════════════════════════════════════════════════════════════

def structural_facts() -> dict:
    """Hechos DERIVADOS del código, no afirmados: son la fuente de los claims estructurales del
    landing (`frontend/src/data/systemFacts.js`). La resta reglas-backend − chips-formulario
    detecta condiciones que el backend soporta pero el formulario YA NO puede expresar (el texto
    libre se retiró en P1-MEDICAL-CONDITIONS-CAP 2026-08-01) — p.ej. `renal`: el landing no debe
    prometerla como seleccionable. tooltip-anchor: P1-LANDING-BENCH-1-FACTS
    """
    from condition_rules import CONDITION_RULES, detect_active_rules
    from medication_rules import MEDICATION_RULES, detect_active_medications
    from micronutrients import dri_targets

    reachable_cond = set()
    for chip in FORM_CONDITION_CHIPS + FORM_PREGNANCY_CHIPS:
        for r in detect_active_rules({"medicalConditions": [chip]}):
            reachable_cond.add(r.id)
    reachable_med = set()
    for chip in FORM_MEDICATION_CHIPS:
        for r in detect_active_medications({"medications": [chip]}):
            reachable_med.add(r.id)

    return {
        "micronutrientes_dri": len(dri_targets("F", 30)),
        "reglas_condicion_backend": len(CONDITION_RULES),
        "condiciones_chips_formulario": len(FORM_CONDITION_CHIPS) + len(FORM_PREGNANCY_CHIPS),
        "condiciones_alcanzables_desde_formulario": sorted(reachable_cond),
        "condiciones_solo_backend": sorted({r.id for r in CONDITION_RULES} - reachable_cond),
        "reglas_medicacion_backend": len(MEDICATION_RULES),
        "medicamentos_chips_formulario": len(FORM_MEDICATION_CHIPS),
        "medicaciones_solo_backend": sorted({r.id for r in MEDICATION_RULES} - reachable_med),
        "alergias_chips_formulario": len(FORM_ALLERGY_CHIPS),
        "dietas_formulario": list(FORM_DIET_TYPES),
        # Contables solo con DB (el runner los completa best-effort; None = sin DB).
        "alimentos_catalogo": None,
        "productos_supermercado": None,
        # [P2-LANDING-BENCH-COUNTRY · 2026-08-21] Eje de país. El banco evaluaba sus perfiles
        # TODOS como dominicanos, así que ninguna de las regresiones de la ola de países habría
        # movido una sola cifra — y dos de ellas se ven CONTANDO CARACTERES:
        # `P1-VERIFIED-CATALOG-COUNTRY` (el bloque «USA EXCLUSIVAMENTE ESTOS ALIMENTOS»
        # byte-idéntico entre España y RD, 3824 chars) y `P1-COUNTRY-CATALOG-BY-COUNTRY` (los
        # cinco beta idénticos entre sí, 5777). Dos columnas iguales en esta tabla las habrían
        # enseñado sin abrir un plan.
        #
        # Va en el modo `structural` porque ahí es GRATIS: se deriva del código, sin LLM ni red.
        # Sin DB los tamaños salen en 0 y el resto del hecho sigue siendo válido.
        "por_pais": _structural_facts_por_pais(),
    }


def _structural_facts_por_pais() -> dict:
    """Por país: cuánto mide su catálogo verificado y si el contexto temporal lo nombra.

    Las dos magnitudes elegidas no son arbitrarias: son exactamente las que acusan los dos gaps
    que esta ola encontró a ojo. Añadir más aquí es barato; lo que no debe pasar es que la tabla
    tenga una sola columna, porque entonces no compara nada.
    tooltip-anchor: P2-LANDING-BENCH-COUNTRY"""
    out = {}
    try:
        from constants import COUNTRY_PROFILES
        paises = list(COUNTRY_PROFILES)
    except Exception:
        return out
    for cc in paises:
        fila = {"catalogo_verificado_chars": 0, "contexto_temporal_chars": 0,
                "contexto_temporal_habla_del_caribe": False}
        try:
            from graph_orchestrator import _get_verified_catalog_instruction as _gvci
            fila["catalogo_verificado_chars"] = len(_gvci({"country": cc}) or "")
        except Exception:
            pass
        try:
            # El contexto temporal no NOMBRA el país (medido): lo que hace es incluir el bloque
            # caribeño —«Temporada Caribeña», «Hace MUCHO calor en el Caribe»— sólo para RD, y
            # omitirlo en beta en vez de inventarle a España un equivalente climático. Así que el
            # hecho útil es el TAMAÑO y si aparece el Caribe: antes de
            # `P1-TIME-CONTEXT-COUNTRY`, a un español se le decía que hace calor en el Caribe.
            from prompts.plan_generator import build_time_context as _btc
            _ctx = _btc(country=cc) or ""
            fila["contexto_temporal_chars"] = len(_ctx)
            fila["contexto_temporal_habla_del_caribe"] = "Caribe" in _ctx
        except Exception:
            pass
        out[cc] = fila
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# 5. Contrato del reporte
# ════════════════════════════════════════════════════════════════════════════════════════════

LANDING_BENCHMARK_SCHEMA_VERSION = 1

# Secciones del JSON de salida. `meta` siempre; el resto según el modo del runner:
#   structural → structural · live → structural+safety+gym+latency · telemetry → telemetry
#   changes (sub-sección de live con --changes) → swap/regen-day ejercitados de verdad.
LANDING_REPORT_SECTIONS = ("meta", "structural", "safety", "gym", "latency", "changes", "telemetry")


def build_report(mode: str, **sections) -> dict:
    """Ensambla el reporte con schema versionado. Ignora secciones None; falla si aparece una
    sección fuera del contrato (el schema es el contrato con el landing, no una bolsa)."""
    unknown = set(sections) - set(LANDING_REPORT_SECTIONS)
    if unknown:
        raise ValueError(f"secciones fuera del contrato LANDING_REPORT_SECTIONS: {sorted(unknown)}")
    report = {
        "schema_version": LANDING_BENCHMARK_SCHEMA_VERSION,
        "mode": mode,
    }
    for name in LANDING_REPORT_SECTIONS:
        val = sections.get(name)
        if val is not None:
            report[name] = val
    return report
