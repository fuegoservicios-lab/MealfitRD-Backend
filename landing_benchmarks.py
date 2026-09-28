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

import re

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
# [P0-ALLERGEN-EU14-CLASES-I18N · 2026-08-23] El wizard ganó dos chips (Lactosa y
# Sesamo, de las 4 clases EU-14 que faltaban); el espejo avanza con él.
FORM_ALLERGY_CHIPS = ("Lacteos", "Gluten", "Huevo", "Mariscos", "Frutos Secos", "Soya",
                      "Pescado", "Mani", "Lactosa", "Sesamo")

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
                # [P0-ALLERGEN-EU14-CLASES-I18N · 2026-08-23] Lactosa entra JUNTO a
                # Lacteos a propósito: son la pareja confundible (intolerancia vs alergia)
                # y este perfil es el que prueba que el motor las trata por separado.
                allergies=["Lacteos", "Gluten", "Huevo", "Lactosa"],
                expect=e(allergens=["Lacteos", "Gluten", "Huevo", "Lactosa"])),
        # [P2-ALLERGEN-CHIPS-REACH-ENGINE · 2026-08-21] +«Pescado» y +«Mani» en este perfil.
        # El guard `test_matrix_covers_every_chip_and_diet` exige que la matriz ejercite CADA chip
        # al menos una vez, y al añadir los dos chips nuevos al wizard la matriz se quedó corta —
        # el guard lo cazó en el mismo commit. Van juntos aquí a propósito: son las dos clases que
        # el usuario CONFUNDE con las que ya había (pescado con marisco, maní con fruto seco), así
        # que el perfil que las mezcla es el que de verdad prueba que el motor las separa.
        _perfil(18, "alergias_mar_nuez_soya", gender="male", age=24, weight=72, height=178,
                goal="performance", activity="athlete",
                # Sesamo entra aquí por la misma razón: semilla que el usuario mezcla
                # con los frutos secos, y este perfil es el que separa esas clases.
                allergies=["Mariscos", "Frutos Secos", "Soya", "Pescado", "Mani", "Sesamo"],
                expect=e(allergens=["Mariscos", "Frutos Secos", "Soya", "Pescado", "Mani", "Sesamo"])),
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
# 3b. Nutrición del plan ENTREGADO — MAPE por macro, peor macro y días 4-en-banda
#     [P1-PLAN-LOTE-749 · 2026-09-28] El contrato del landing (pilar B-01) pide tres cifras que
#     el arnés NO calculaba: sólo existía el eje `banda` del gym, que usa otra banda (sin el techo
#     kcal de ganancia muscular) y divide los días 4-en-banda por días VACÍOS incluidos. Estas
#     funciones son puras (ni LLM ni DB) y reproducen la banda de `compute_clinical_band_score`
#     del motor — la paridad la ancla test_p1_plan_lote_749.py contra el propio motor.
#     tooltip-anchor: P1-PLAN-LOTE-749-NUTRITION
# ════════════════════════════════════════════════════════════════════════════════════════════

NUTRITION_MACROS = ("kcal", "protein", "carbs", "fats")

# Espejo de los defaults del motor: la banda REAL se lee con `engine_band_definition()` (que
# respeta los knobs MEALFIT_BAND_SCORE_* del entorno). Esto sólo se usa si el motor no importa.
_DEFAULT_ENGINE_BAND = {"macro": [0.90, 1.12], "kcal": [0.95, 1.05], "kcal_upper_gain_muscle": 1.10}

# Espejo de `graph_orchestrator._GAINMUSCLE_GOAL_TOKENS`. Paridad anclada por
# test_p1_plan_lote_749.py::test_copias_del_motor_iguales_al_motor (igualdad de la tupla y de
# `_macro_num`/`_goal_is_gain_muscle` contra las funciones del motor). Es copia y no import para
# que el scorer no cargue el grafo entero por cada comida.
_GAINMUSCLE_GOAL_TOKENS = ("gain_muscle", "ganar_musculo", "ganancia", "bulk", "superavit")


def engine_band_definition() -> dict:
    """La banda clínica tal como la aplica el MOTOR (`compute_clinical_band_score`): proteína,
    carbos y grasas en [BAND_SCORE_LOWER, BAND_SCORE_UPPER] × objetivo; kcal en [0.95, 1.05]
    (literal dentro del motor) con techo GAINMUSCLE_KCAL_BAND_UPPER en ganancia muscular.
    Si el motor no se puede importar devuelve los defaults marcados `source="default"` — el
    reporte lo dice, no lo esconde. tooltip-anchor: P1-PLAN-LOTE-749-BAND"""
    try:
        import graph_orchestrator as _go
        return {"macro": [float(_go.BAND_SCORE_LOWER), float(_go.BAND_SCORE_UPPER)],
                "kcal": [0.95, 1.05],
                "kcal_upper_gain_muscle": float(_go.GAINMUSCLE_KCAL_BAND_UPPER),
                "source": "graph_orchestrator"}
    except Exception as _e:
        return dict(_DEFAULT_ENGINE_BAND, source=f"default ({type(_e).__name__})")


def _macro_num(x) -> float:
    """Espejo de `graph_orchestrator._meal_macro_num`: '154g'/'464 kcal'/None → float; NaN/Inf → 0."""
    try:
        s = str(x).strip().lower().replace("g", "").replace("kcal", "").strip()
        v = float(s) if s else 0.0
        return v if (v == v and v not in (float("inf"), float("-inf"))) else 0.0
    except Exception:
        return 0.0


def _goal_is_gain_muscle(plan: dict, goal=None) -> bool:
    """Espejo de `graph_orchestrator._plan_goal_is_gainmuscle`: goal explícito → `main_goal` del
    plan (la etiqueta persistida, p.ej. «Ganancia Muscular (Superávit 8%…)») → form_data."""
    import unicodedata
    raw = goal
    if not raw and isinstance(plan, dict):
        fd = plan.get("form_data") if isinstance(plan.get("form_data"), dict) else {}
        raw = plan.get("main_goal") or fd.get("mainGoal") or fd.get("goal")
    if not raw:
        return False
    g = unicodedata.normalize("NFD", str(raw).lower()).encode("ascii", "ignore").decode("ascii")
    return any(tok in g for tok in _GAINMUSCLE_GOAL_TOKENS)


def score_plan_nutrition(plan: dict, *, goal=None, band: dict = None) -> dict:
    """Puntúa la nutrición de UN plan entregado contra sus propios objetivos (cabecera del plan:
    `calories` + `macros`, que el motor fija al OBJETIVO en assemble — `delivered_*` es lo real).

    Por día y por macro (kcal, proteína, carbos, grasas): ratio entregado/objetivo con el total
    RECALCULADO desde las comidas (suma de `cals`/`protein`/`carbs`/`fats`), igual que el motor.
      - error porcentual absoluto |entregado − objetivo| / objetivo (base del MAPE);
      - en banda si cae en la banda del motor; un día es 4-en-banda si tiene las 4 celdas y
        las 4 están dentro (un día sin comidas cuenta y queda fuera, como en el motor).
    Un macro con objetivo ≤ 0 no genera celda. Sin objetivos o sin días → `scored=False`.
    tooltip-anchor: P1-PLAN-LOTE-749-NUTRITION"""
    b = band or _DEFAULT_ENGINE_BAND
    if not isinstance(plan, dict):
        return {"scored": False, "reason": "plan no es dict"}
    pm = plan.get("macros") if isinstance(plan.get("macros"), dict) else {}
    targets = {
        "kcal": _macro_num(plan.get("calories")),
        "protein": _macro_num(pm.get("protein")),
        "carbs": _macro_num(pm.get("carbs")),
        "fats": _macro_num(pm.get("fats")),
    }
    days = [d for d in (plan.get("days") or []) if isinstance(d, dict)]
    if not days:
        return {"scored": False, "reason": "sin días", "targets": targets}
    if not any(t > 0 for t in targets.values()):
        return {"scored": False, "reason": "sin objetivos (calories/macros)", "targets": targets}

    gain = _goal_is_gain_muscle(plan, goal)
    mlo, mhi = b["macro"]
    klo, khi = b["kcal"]
    if gain:
        khi = b.get("kcal_upper_gain_muscle", khi)
    bands = {"kcal": (klo, khi), "protein": (mlo, mhi), "carbs": (mlo, mhi), "fats": (mlo, mhi)}

    per = {m: {"n": 0, "abs_pct_err_sum": 0.0, "in_band": 0} for m in NUTRITION_MACROS}
    per_day, days_eval, all4 = [], 0, 0
    for i, day in enumerate(days):
        meals = [m for m in (day.get("meals") or []) if isinstance(m, dict)]
        # [P1-PLAN-LOTE-749 · ronda 1] kcal SOLO de `cals`, como `compute_clinical_band_score`: la
        # primera versión sumaba `calories` cuando faltaba `cals` y daba otro número que el motor.
        delivered = {
            "kcal": sum(_macro_num(m.get("cals")) for m in meals),
            "protein": sum(_macro_num(m.get("protein")) for m in meals),
            "carbs": sum(_macro_num(m.get("carbs")) for m in meals),
            "fats": sum(_macro_num(m.get("fats")) for m in meals),
        }
        cells = inside = 0
        ratios = {}
        for mac in NUTRITION_MACROS:
            t = targets[mac]
            if t <= 0:
                continue
            cells += 1
            ratio = delivered[mac] / t
            ratios[mac] = round(ratio, 3)
            per[mac]["n"] += 1
            per[mac]["abs_pct_err_sum"] += abs(ratio - 1.0)
            lo, hi = bands[mac]
            if lo * t <= delivered[mac] <= hi * t:
                per[mac]["in_band"] += 1
                inside += 1
        if cells:
            days_eval += 1
            ok = cells >= 4 and inside == cells
            all4 += 1 if ok else 0
            per_day.append({"day": day.get("day") or (i + 1), "ratios": ratios, "all4_in_band": ok})

    mape = {m: round(100.0 * v["abs_pct_err_sum"] / v["n"], 2) for m, v in per.items() if v["n"]}
    worst = max(mape, key=mape.get) if mape else None
    return {
        "scored": True, "days": len(days), "days_evaluated": days_eval, "gain_muscle": gain,
        "targets": targets, "per_macro": per, "per_macro_mape_pct": mape,
        "worst_macro": worst, "worst_macro_mape_pct": mape.get(worst) if worst else None,
        "four_macros_in_band_days": all4,
        "four_macros_in_band_pct": round(100.0 * all4 / days_eval, 1) if days_eval else None,
        "per_day": per_day,
    }


def aggregate_nutrition(results: list, *, n_attempted: int = None, band: dict = None) -> dict:
    """Agrega por DÍA evaluado (todas las celdas de todos los planes; como el nightly y como define
    B-01 «días evaluados»). Claves que consume el importador del landing (schema v2):
    `n_scored`, `n_attempted`, `macro_mape_pct` (media de los 4 MAPE por macro),
    `worst_macro_mape_pct` (el mayor de los 4) y `four_macros_in_band_pct`. Sin planes puntuados
    NO emite cifras (el importador las salta; no se inventa un 0). tooltip-anchor: P1-PLAN-LOTE-749-NUTRITION"""
    rows = [r for r in (results or []) if isinstance(r, dict)]
    scored = [r for r in rows if r.get("scored")]
    out = {
        "n_scored": len(scored),
        "n_attempted": int(n_attempted) if n_attempted is not None else len(rows),
        "band": band or dict(_DEFAULT_ENGINE_BAND, source="default"),
        "definition": ("error = |total recalculado del día − objetivo| / objetivo, por macro "
                       "(kcal, proteína, carbos, grasas); MAPE agregado por día evaluado; "
                       "macro_mape = media de los 4; 4-en-banda = día con las 4 celdas dentro de "
                       "la banda del motor"),
    }
    if not scored:
        return out
    per = {m: {"n": 0, "err": 0.0, "in": 0} for m in NUTRITION_MACROS}
    days_eval = all4 = 0
    for r in scored:
        days_eval += r.get("days_evaluated") or 0
        all4 += r.get("four_macros_in_band_days") or 0
        for m, v in (r.get("per_macro") or {}).items():
            if m in per:
                per[m]["n"] += v.get("n") or 0
                per[m]["err"] += v.get("abs_pct_err_sum") or 0.0
                per[m]["in"] += v.get("in_band") or 0
    mape = {m: round(100.0 * v["err"] / v["n"], 2) for m, v in per.items() if v["n"]}
    if mape:
        worst = max(mape, key=mape.get)
        out.update({
            "per_macro_mape_pct": mape,
            "macro_mape_pct": round(sum(mape.values()) / len(mape), 2),
            "worst_macro": worst,
            "worst_macro_mape_pct": mape[worst],
            "per_macro_in_band_pct": {m: round(100.0 * v["in"] / v["n"], 1)
                                      for m, v in per.items() if v["n"]},
        })
    out["days_evaluated"] = days_eval
    if days_eval:
        out["four_macros_in_band_pct"] = round(100.0 * all4 / days_eval, 1)
    out["plans_all_days_in_band_pct"] = round(
        100.0 * sum(1 for r in scored if r.get("days_evaluated")
                    and r.get("four_macros_in_band_days") == r.get("days_evaluated")) / len(scored), 1)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# 3c. Entrega y confiabilidad — qué llegó al usuario y qué no
#     [P1-PLAN-LOTE-749 · 2026-09-28] Un plan con `_is_fallback` y SIN `_partial_repair` lo
#     DESCARTA el router (422 rechazo crítico / 503 «IA saturada», routers/plans.py
#     FALLBACK-GUARD): nunca llega al usuario. El modo `live` lo contaba como entregado y lo
#     puntuaba; el `remote` lo veía como error genérico. Ahora los dos hablan el mismo idioma.
#     tooltip-anchor: P1-PLAN-LOTE-749-DELIVERY
# ════════════════════════════════════════════════════════════════════════════════════════════

DELIVERED_STATES = ("delivered", "delivered_fallback")


def plan_delivery_state(plan) -> str:
    """`delivered` · `delivered_fallback` (reparación parcial: revisada y persistida) ·
    `discarded_fallback` (el router no la entrega) · `error` (no hay plan)."""
    if not isinstance(plan, dict):
        return "error"
    if plan.get("_is_fallback"):
        return "delivered_fallback" if plan.get("_partial_repair") else "discarded_fallback"
    return "delivered"


# Los dos únicos `detail` de TEXTO del 503 del FALLBACK-GUARD síncrono (routers/plans.py:
# spending cap / saturación). Otros 503 del mismo endpoint NO son un fallback: «no pudimos
# guardarlo» (P2-PLAN-PERSIST-FAILED), `server_busy_generating` (dict) o el HTML de nginx.
_FALLBACK_GUARD_503_DETAILS = ("La IA está temporalmente saturada",
                               "El servicio de IA no está disponible")


def classify_remote_error(message: str) -> str:
    """Estado de entrega de un error del modo remote. Solo el FALLBACK-GUARD es un fallback
    descartado; el resto se distingue por el `detail` que el runner guarda en el mensaje:
      · SSE `code=critical_restriction` / `code=llm_unavailable` → `discarded_fallback`.
      · Síncrono 422 con `detail` de TEXTO → rechazo crítico (`discarded_fallback`): es el contrato
        P2-CRITICAL-REJECTION-CODE (el frontend lo reconoce por `typeof detail === 'string'`).
      · Síncrono 422 con `detail` OBJETO con `code` (`missing_required_fields`,
        `invalid_biometric_range`, `budget_insufficient`, `too_many_medical_conditions`,
        `clinical_scope_exceeded`, `invalid_total_days`…) → `rejected_request`: el servidor rechazó
        la PETICIÓN antes de generar; no es un fallback y no debe inflar esa tasa.
      · Síncrono 503 con uno de `_FALLBACK_GUARD_503_DETAILS` → `discarded_fallback`.
      · Todo lo demás (503 «no pudimos guardarlo», `server_busy_generating`, nginx, SSE
        `plan_persist_failed`, timeouts) → `error`.
    [P1-PLAN-LOTE-749 · ronda 1] La primera versión contaba CUALQUIER 422/503 como fallback.
    tooltip-anchor: P1-PLAN-LOTE-749-DELIVERY"""
    msg = str(message or "")
    if re.search(r"code=(critical_restriction|llm_unavailable)\b", msg):
        return "discarded_fallback"
    # [ronda 2] El stream también se clasifica por su `detail`: un rechazo suyo ya no se reenvía
    # al síncrono (scripts/landing_benchmark.py::_remote_generate_stream).
    m = re.search(r"HTTP (\d{3}) en /api/plans/analyze(?:/stream)?\b:?\s*(.*)", msg, re.DOTALL)
    if not m:
        return "error"
    status, body = m.group(1), m.group(2)
    detail_str = re.match(r'\s*\{\s*"detail"\s*:\s*"((?:[^"\\]|\\.)*)', body)
    detail_code = re.match(r'\s*\{\s*"detail"\s*:\s*\{.*?"code"\s*:\s*"(\w+)"', body, re.DOTALL)
    if status == "422":
        if detail_str:
            return "discarded_fallback"
        if detail_code:
            return "rejected_request"
        return "error"
    if status == "503" and detail_str and detail_str.group(1).startswith(_FALLBACK_GUARD_503_DETAILS):
        return "discarded_fallback"
    return "error"


def latency_percentiles(values) -> dict:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return {"n": 0}
    out = {"n": len(vals)}
    for p in (0.5, 0.95):
        idx = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
        out[f"p{int(p * 100)}"] = vals[idx]
    return out


def aggregate_reliability(rows: list, *, n_attempted: int = None) -> dict:
    """Pilar B-04 con los fallos EN el denominador: entrega = entregados / iniciados; fallback =
    fallbacks entregados + descartados / iniciados («sin excluir fallbacks exitosos ni fallidos»);
    latencia de TODO intento hasta entrega o fallo terminal, y aparte la de los entregados."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    n = int(n_attempted) if n_attempted is not None else len(rows)
    states = [r.get("delivery") or "error" for r in rows]
    n_del = sum(1 for s in states if s in DELIVERED_STATES)
    n_fb_del = states.count("delivered_fallback")
    n_fb_desc = states.count("discarded_fallback")
    pct = (lambda k: round(100.0 * k / n, 1) if n else None)
    return {
        "n_attempted": n,
        "n_delivered": n_del,
        "n_delivered_fallback": n_fb_del,
        "n_discarded_fallback": n_fb_desc,
        # Petición rechazada ANTES de generar (422 de validación): sigue en el denominador (no se
        # entregó) pero no es un fallback ni un error de infraestructura.
        "n_rejected_request": states.count("rejected_request"),
        "n_errors": states.count("error"),
        "delivery_rate_pct": pct(n_del),
        "fallback_rate_pct": pct(n_fb_del + n_fb_desc),
        "latency_all_s": latency_percentiles(r.get("duration_s") for r in rows),
        "latency_delivered_s": latency_percentiles(r.get("duration_s") for r, s in zip(rows, states)
                                     if s in DELIVERED_STATES),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════
# 3d. Expectativas clínicas derivadas del FORMULARIO
#     [P1-PLAN-LOTE-749 · 2026-09-28] Para re-puntuar planes que NO son de la matriz (corpus de
#     baterías reales) hace falta el `_expect` que la matriz escribe a mano. Mismas reglas que la
#     matriz — la paridad perfil a perfil la ancla el test. tooltip-anchor: P1-PLAN-LOTE-749-EXPECT
# ════════════════════════════════════════════════════════════════════════════════════════════

_NONE_CHIPS = {"", "ninguna", "ninguno", "none"}


def _chips(values) -> list:
    if isinstance(values, str):
        values = [values]
    return [str(v) for v in (values or []) if str(v).strip().lower() not in _NONE_CHIPS]


def derive_expectations(form: dict) -> dict:
    """`_expect` de un formulario: reglas clínicas por chip, FS9 si declara medicación, ≥5 tomas
    con bariátrica o insulina/sulfonilurea, monitor de vitamina K con warfarina, mercurio en
    embarazo/lactancia, alérgenos y dieta no-balanced. Solo claves con valor (como la matriz)."""
    form = form if isinstance(form, dict) else {}
    conds = _chips(form.get("medicalConditions"))
    meds = _chips(form.get("medications"))
    alls = _chips(form.get("allergies"))
    out = {}

    def _uniq(xs):
        seen, res = set(), []
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                res.append(x)
        return res

    cr = _uniq(CONDITION_CHIP_EXPECTED_RULE.get(c) for c in conds)
    mr = _uniq(MEDICATION_CHIP_EXPECTED_RULE.get(m) for m in meds)
    if alls:
        out["allergens"] = alls
    diet = form.get("dietType") or "balanced"
    if diet != "balanced":
        out["diet"] = diet
    if cr:
        out["condition_rules"] = cr
    if mr:
        out["medication_rules"] = mr
    if meds:
        out["fs9"] = True
    if "Levotiroxina" in meds:
        out["timing_advisory"] = True
    if "Cirugía Bariátrica" in conds or {"Insulina", "Glibenclamida"} & set(meds):
        out["min_meals_per_day"] = 5
    if {"Embarazo", "Lactancia"} & set(conds):
        out["mercury_guard"] = True
    if "Warfarina" in meds:
        out["vitk_monitor"] = True
    return out


# Campos de texto libre que producción une a sus listas (`_OTHER_TEXT_FIELD_MAP` del motor).
_FREE_TEXT_FIELDS = ("otherAllergies", "otherConditions", "otherDislikes", "otherStruggles")


def corpus_profile(form: dict, pid: str) -> dict:
    """Perfil de un plan de CORPUS con la MISMA unión de texto libre que hace producción.

    [P1-PLAN-LOTE-749 · ronda 1] El generador une `otherAllergies`/`otherConditions`/… a sus listas
    al empezar (`graph_orchestrator._merge_other_text_fields`, vía `profile_with_free_text` sobre una
    copia), y `score_plan_safety` solo lee `allergies`. Armar el perfil con el formulario CRUDO
    dejaba ciego al scorer: camarones a quien escribió «frutos del mar» salía «seguro».

    Se usa la función del motor (no una copia) y las expectativas se derivan del perfil YA unido.
    Con el centinela «Ninguna» producción DESCARTA el texto (P0-FORM-1); el scorer sigue esa regla,
    pero el perfil lo deja a la vista en `_free_text_discarded` (campos cuyo texto no llegó al
    motor) para que la corrida no lo calle. tooltip-anchor: P1-PLAN-LOTE-749-EXPECT"""
    from graph_orchestrator import profile_with_free_text
    raw = dict(form or {})
    merged = profile_with_free_text(raw)
    discarded = [f for f in _FREE_TEXT_FIELDS
                 if str(raw.get(f) or "").strip() and not str(merged.get(f) or "").strip()]
    return dict(merged, _id=pid, _label=pid, _expect=derive_expectations(merged),
                _free_text_discarded=discarded)


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

# [P1-PLAN-LOTE-749 · 2026-09-28] v1 → v2: el importador del landing (bioboros-cinematic,
# `benchmark_import.py`) exige `schema_version == 2`, un bloque `run` trazable (commit de origen,
# protocolo, cohorte) y `nutrition.aggregate`. Sin eso rechaza el reporte entero.
LANDING_BENCHMARK_SCHEMA_VERSION = 2

# Versión del protocolo congelado del landing (`contract/benchmark-v22.json → protocol.version`).
# El importador rechaza un reporte cuyo `run.protocol_version` no coincida: si el landing congela
# un protocolo nuevo, se sube AQUÍ (o se pasa `--protocol-version`), a sabiendas.
LANDING_BENCHMARK_PROTOCOL_VERSION = "2.2-prep.1"

# Secciones del JSON de salida. `run` siempre (lo pone el runner); el resto según el modo:
#   structural → structural · live/remote → safety+nutrition+gym+latency+reliability(+changes)
#   score → safety+nutrition+gym · telemetry → telemetry
LANDING_REPORT_SECTIONS = ("run", "meta", "structural", "safety", "nutrition", "gym", "latency",
                           "reliability", "changes", "telemetry")

# Modos cuyo `profile_count` es la cohorte de la matriz (los demás no tienen perfiles).
_PROFILE_MODES = ("live", "remote", "score")


_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)

# De quién es `run.source_commit` según el modo. En `live`/`structural` el motor corre EN el mismo
# proceso: el commit es el del motor medido. En `remote` el motor es el del SERVIDOR y en `score`
# los planes se generaron en otra parte: el commit es el de los scorers. En `telemetry`, el de las
# consultas (los datos son de lo que corrió en prod en la ventana).
_SOURCE_COMMIT_ROLE = {"live": "engine_and_scorers", "structural": "engine_and_scorers",
                       "remote": "scorers", "score": "scorers", "telemetry": "queries"}


def engine_identity(start, end) -> dict:
    """Commit del MOTOR medido en una corrida remote, a partir de `/health/version` al empezar y al
    terminar (`git_sha`, que el deploy inyecta por env `GIT_SHA`; hoy puede valer `"unknown"`).

    `verified` solo si los dos extremos dan el MISMO sha válido; `changed_during_run` si hubo un
    deploy a mitad (la corrida mezcla dos binarios); `not_exposed` si el servidor no lo publica;
    `unreachable` si no se pudo leer. Solo `verified` rellena `engine_commit`.
    [P1-PLAN-LOTE-749 · ronda 1] tooltip-anchor: P1-PLAN-LOTE-749-RUN"""
    def _sha(v):
        s = str((v or {}).get("git_sha") or "").strip() if isinstance(v, dict) else ""
        return s if _SHA_RE.fullmatch(s) else None
    if not isinstance(start, dict) or not isinstance(end, dict) or "error" in start or "error" in end:
        return {"engine_commit": None, "engine_commit_status": "unreachable"}
    a, b = _sha(start), _sha(end)
    if a is None or b is None:
        return {"engine_commit": None, "engine_commit_status": "not_exposed"}
    if a.lower() != b.lower():
        return {"engine_commit": None, "engine_commit_status": "changed_during_run"}
    return {"engine_commit": a, "engine_commit_status": "verified"}


# [P1-PLAN-LOTE-749 · ronda 2] Claves de `/health/version` que cambian con un deploy o un reinicio.
# `cambio_durante_la_corrida` se decidía solo con `git_sha`, y prod publica `git_sha:"unknown"`
# (el deploy aún no inyecta `GIT_SHA`): un redeploy a mitad de corrida salía `false`.
_SERVER_CHANGE_KEYS = ("git_sha", "deploy_timestamp", "last_known_pfix", "process_started_at")
_NOT_EXPOSED = ("", "unknown", "none", "null")


def server_change_during_run(start, end) -> dict:
    """¿Cambió el servidor entre el `/health/version` del inicio y el del final? Compara cada clave
    de `_SERVER_CHANGE_KEYS` que los DOS extremos publican con un valor real (no «unknown»).
    `cambio`: True si alguna difiere (`claves` dice cuáles), False si hubo al menos una comparable y
    ninguna difiere, None si no se puede verificar (extremo ilegible o nada comparable) — nunca un
    `false` que nadie comprobó. tooltip-anchor: P1-PLAN-LOTE-749-RUN"""
    if not isinstance(start, dict) or not isinstance(end, dict) or "error" in start or "error" in end:
        return {"cambio": None, "claves": []}

    def _val(v, k):
        s = str(v.get(k) if v.get(k) is not None else "").strip()
        return None if s.lower() in _NOT_EXPOSED else s
    comparables, distintas = 0, []
    for k in sorted(_SERVER_CHANGE_KEYS):
        a, b = _val(start, k), _val(end, k)
        if a is None or b is None:
            continue
        comparables += 1
        if a != b:
            distintas.append(k)
    if distintas:
        return {"cambio": True, "claves": distintas}
    return {"cambio": False if comparables else None, "claves": []}


def build_run_meta(*, mode: str, started_at: str, finished_at: str, source_commit, source_dirty,
                   architecture: str, protocol_version: str, country_scope: list,
                   profile_ids: list, full_profile_ids: list, parameters: dict,
                   cohort: str = "matrix", source_commit_role: str = None,
                   engine: dict = None) -> dict:
    """Bloque `run` del reporte v2 (contrato del importador del landing). Puro.

    `cohort_status`: `complete` si la corrida intentó EXACTAMENTE la matriz entera, `partial` si
    un subconjunto, `not_applicable` en structural/telemetry o en un corpus que no es la matriz.
    `publication_status` nace SIEMPRE `candidate` (G-06: el importador crea candidatos; publica
    una persona). `source_dirty=None` = no verificable, y el importador lo rechaza igual que True.

    [P1-PLAN-LOTE-749 · ronda 1] G-04 (trazabilidad) pide el commit de ORIGEN del resultado. En
    remote `source_commit` es el del runner/scorers, no el del binario medido: `source_commit_role`
    lo dice y `engine_commit`/`engine_commit_status` (de `engine_identity`) dan el del motor cuando
    el servidor lo publica. En live el motor corre en proceso: `engine_commit = source_commit`.
    tooltip-anchor: P1-PLAN-LOTE-749-RUN"""
    pids = list(profile_ids or [])
    full = list(full_profile_ids or [])
    if mode not in _PROFILE_MODES or cohort != "matrix":
        cohort_status = "not_applicable"
    elif pids and sorted(map(str, pids)) == sorted(map(str, full)):
        cohort_status = "complete"
    else:
        cohort_status = "partial"
    stamp = re.sub(r"[^0-9t]", "", str(started_at).lower().replace("z", ""))[:15] or "sinfecha"
    commit7 = str(source_commit)[:7].lower() if source_commit else "nocommit"
    role = source_commit_role or _SOURCE_COMMIT_ROLE.get(mode, "scorers")
    if engine is None:
        engine = ({"engine_commit": source_commit, "engine_commit_status": "in_process"}
                  if role == "engine_and_scorers"
                  else {"engine_commit": None, "engine_commit_status": "not_applicable"})
    return {
        "id": f"{mode}-{stamp}z-{commit7}",
        "mode": mode,
        "architecture": architecture or "unspecified",
        "protocol_version": protocol_version,
        "started_at": started_at,
        "finished_at": finished_at,
        "source_commit": source_commit,
        "source_commit_role": role,
        "engine_commit": engine.get("engine_commit"),
        "engine_commit_status": engine.get("engine_commit_status"),
        "source_dirty": source_dirty,
        "country_scope": list(country_scope or []),
        "full_profile_count": len(full),
        "profile_count": len(pids) if mode in _PROFILE_MODES else 0,
        "cohort_status": cohort_status,
        "publication_status": "candidate",
        "parameters": dict(parameters or {}),
    }


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


# ════════════════════════════════════════════════════════════════════════════════════════════
# 6. Telemetría de producción — SOLO lo entregado
#    [P1-PLAN-LOTE-749 · 2026-09-28] Medido en prod (30 días, 2026-09-28): de 408 filas
#    `clinical_band_final`, 269 eran `assemble-tail` — la lectura INTERMEDIA pre-review, que puede
#    reintentarse o descartarse — y sólo 81 `pre-INSERT` (lo que se guarda y ve el usuario). La
#    media las mezclaba. Y `clinical_band` se emite por CORRIDA del pipeline, incluidas las que
#    acaban en un fallback que el FALLBACK-GUARD del router descarta (422/503): contar su
#    `delivered_was_fallback` como «tasa de fallback» y su duración como «latencia de generación»
#    mide intentos, no entregas.
#    tooltip-anchor: P1-PLAN-LOTE-749-TELEMETRY
# ════════════════════════════════════════════════════════════════════════════════════════════

# Superficies de `clinical_band_final` que son la ENTREGA: el INSERT del plan inicial
# (`db_plans._finalize_plan_data_for_insert`, surface por defecto) y el merge T1 de cada bloque
# posterior (`cron_tasks`, «chunk-T1 semana N»: tras él no corre ningún pase más sobre esos días).
# Todas las demás (assemble-tail, review-band-gate, *-budget-convergence, post-review-patch) son
# estados intermedios del mismo plan.
DELIVERED_BAND_SURFACE_EXACT = ("pre-INSERT",)
DELIVERED_BAND_SURFACE_PREFIXES = ("chunk-T1",)


def _delivered_surface_sql() -> str:
    exact = " OR ".join(f"metadata->>'surface' = '{s}'" for s in DELIVERED_BAND_SURFACE_EXACT)
    pref = " OR ".join(f"metadata->>'surface' LIKE '{p}%%'" for p in DELIVERED_BAND_SURFACE_PREFIXES)
    return f"({exact} OR {pref})"


# ── [P1-PLAN-LOTE-749 · ronda 2] El TIPO de una entrega sale de su SUPERFICIE ──
# La ronda 1 lo decidía con `session_id = 'unknown'` de la corrida. Eso solo aparta los
# `rolling_refill` (su form no lleva sesión): los bloques `chunk_kind='initial_plan'` —los días 8-30
# del horizonte, generados días después y CON la sesión del formulario— salían «plan inicial».
# Re-verificado en prod (solo lectura, 28-sep): los 5 «planes iniciales por chunk-T1» eran bloques de
# semana 2-3 generados 3-8 días después de crear el plan; ninguna fila «chunk-T1 semana 1».
# Plan inicial = `pre-INSERT` (el INSERT del plan, y también el relleno del placeholder del Bloque 1
# por la cola: `fill_placeholder_meal_plan_atomic` finaliza con la superficie por defecto) o
# `chunk-T1 semana 1` (defensivo: hoy no se emite). Cualquier otro `chunk-T1 semana N` es un bloque
# posterior. tooltip-anchor: P1-PLAN-LOTE-749-DELIVERY-KIND
INITIAL_DELIVERY_SURFACES = ("pre-INSERT", "chunk-T1 semana 1")


def delivery_kind(surface):
    """`plan_inicial` · `bloque_posterior` · None (no es una superficie de entrega). Espejo en
    Python de `delivery_kind_sql` (el test evalúa las dos con las mismas superficies)."""
    s = str(surface or "")
    if s in INITIAL_DELIVERY_SURFACES:
        return "plan_inicial"
    if any(s.startswith(p) for p in DELIVERED_BAND_SURFACE_PREFIXES):
        return "bloque_posterior"
    return None


def delivery_kind_sql(col: str) -> str:
    """Expresión CASE (SQL estándar: `IN` + `LIKE`) con el tipo de entrega de la columna `col`,
    que guarda la superficie. tooltip-anchor: P1-PLAN-LOTE-749-DELIVERY-KIND"""
    ini = ", ".join(f"'{s}'" for s in INITIAL_DELIVERY_SURFACES)
    pref = " OR ".join(f"{col} LIKE '{p}%%'" for p in DELIVERED_BAND_SURFACE_PREFIXES)
    return (f"CASE WHEN {col} IN ({ini}) THEN 'plan_inicial' "
            f"WHEN {pref} THEN 'bloque_posterior' ELSE 'otra' END")


# ── [P1-PLAN-LOTE-749 · ronda 1] Emparejar cada fila de banda con la CORRIDA que la produjo ──
# La primera versión filtraba por superficie y daba por hecho que toda fila `pre-INSERT` era una
# entrega. Medido en prod (solo lectura, 28-sep, ventana de 30 días): de 81 filas `pre-INSERT`, 67
# (todas del 2-7 sep) no tenían ninguna corrida `clinical_band` detrás ni `meal_plans` cerca — con
# `user_id` NULL y sesión `post-finalize`, que es exactamente lo que deja CUALQUIER llamada a
# `db_plans._finalize_plan_data_for_insert` fuera de una generación (tests, scripts). Su media
# (0,95) tapaba la de las entregas reales. Y la latencia promediaba 49 corridas de pipeline cuando
# hubo 14 planes iniciales: entraban los bloques del chunk worker y los reintentos que nunca se
# fusionaron.
#
# Regla: una fila de banda es una entrega si hay una corrida `clinical_band` del MISMO usuario
# ≤ `_PAIR_WINDOW` antes. `pre-INSERT` guarda `user_id` NULL (el dict que recibe el finalize no lo
# lleva), así que ahí el usuario se prueba por su fila de `meal_plans`, creada entre el arranque de
# esa corrida y el INSERT. Medido: fila↔corrida a 1-21 s en pre-INSERT y a 3-48 s en chunk-T1;
# `meal_plans.created_at` ≈ arranque de la corrida (el plan nace al empezar a generar). Un invitado
# no persiste su plan: su corrida no tiene fila de entrega y se cuenta aparte, sin inventarla.
# El tipo (plan inicial / bloque posterior) sale de la SUPERFICIE de la entrega (`delivery_kind_sql`),
# no de la sesión de la corrida [ronda 2]: `chunk_kind='initial_plan'` son los días 8-30 del horizonte
# (bloques posteriores, con sesión), no el Bloque 1.
_PAIR_WINDOW = "5 minutes"


def _delivery_pairing_ctes() -> str:
    """CTEs `win`, `corridas`, `entregas` y `pares` (entrega ↔ corrida). UN solo parámetro: los días
    de la ventana. tooltip-anchor: P1-PLAN-LOTE-749-TELEMETRY"""
    return f"""WITH win AS (SELECT NOW() - make_interval(days => %s) AS desde),
            corridas AS (
                SELECT pm.id, pm.user_id, pm.session_id, pm.created_at, pm.duration_ms,
                       COALESCE(pm.metadata->>'delivered_was_fallback', 'false') = 'true' AS con_fallback
                FROM pipeline_metrics pm, win
                WHERE pm.node = 'clinical_band'
                  AND pm.created_at >= win.desde - interval '{_PAIR_WINDOW}'),
            entregas AS (
                SELECT f.id, f.user_id, f.created_at, f.confidence, f.metadata->>'surface' AS surface
                FROM pipeline_metrics f, win
                WHERE f.node = 'clinical_band_final' AND {_delivered_surface_sql()}
                  AND f.created_at >= win.desde),
            pares AS (
                SELECT e.id AS entrega_id, e.confidence, e.surface, c.id AS corrida_id,
                       c.session_id, c.duration_ms, c.con_fallback
                FROM entregas e
                JOIN LATERAL (
                    SELECT c.id, c.session_id, c.duration_ms, c.con_fallback
                    FROM corridas c
                    WHERE c.user_id IS NOT NULL
                      AND c.created_at BETWEEN e.created_at - interval '{_PAIR_WINDOW}' AND e.created_at
                      AND (c.user_id = e.user_id
                           OR (e.user_id IS NULL AND EXISTS (
                                 SELECT 1 FROM meal_plans mp
                                 WHERE mp.user_id::text = c.user_id
                                   AND mp.created_at BETWEEN
                                       c.created_at - make_interval(secs => c.duration_ms / 1000.0)
                                           - interval '10 minutes'
                                       AND e.created_at + interval '{_PAIR_WINDOW}')))
                    ORDER BY c.created_at DESC
                    LIMIT 1) c ON TRUE)
            """


def telemetry_queries(days: int) -> dict:
    """{nombre: (sql, params)} del modo telemetry. Todas filtran por ventana `days` (un parámetro).
    Banda y latencia cuentan SOLO pares entrega↔corrida (`_delivery_pairing_ctes`); lo que no se
    empareja se reporta APARTE (`banda_excluida_sin_corrida`, `corridas_por_entrega`) — nunca
    mezclado con lo entregado. tooltip-anchor: P1-PLAN-LOTE-749-TELEMETRY"""
    d = (int(days),)
    win = "created_at >= NOW() - make_interval(days => %s)"
    ctes = _delivery_pairing_ctes()
    return {
        # [P1-CHANGE-OUTCOME-TELEMETRY] ¿los cambios de plato salen a la primera?
        "changes": (
            f"""SELECT node, metadata->>'outcome' AS outcome, COUNT(*) AS n,
                      ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)) AS p50_ms,
                      ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)) AS p95_ms
               FROM pipeline_metrics
               WHERE node IN ('change_swap','change_regen_day') AND {win}
               GROUP BY 1, 2 ORDER BY 1, 2""", d),
        # Banda de lo ENTREGADO: solo filas emparejadas con su corrida; tipo según la SUPERFICIE.
        "banda_entregada": (
            f"""{ctes}
               SELECT {delivery_kind_sql('p.surface')} AS entrega,
                      COUNT(*) AS n, ROUND(AVG(p.confidence)::numeric, 3) AS media,
                      ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY p.confidence)::numeric, 3) AS p50
               FROM pares p
               GROUP BY 1 ORDER BY 1""", d),
        # Filas de superficie de entrega SIN corrida detrás (tests/scripts contra prod): aparte.
        "banda_excluida_sin_corrida": (
            f"""{ctes}
               SELECT CASE WHEN e.surface = 'pre-INSERT' THEN 'pre-INSERT' ELSE 'chunk-T1' END AS superficie,
                      COUNT(*) AS n, ROUND(AVG(e.confidence)::numeric, 3) AS media,
                      MIN(e.created_at)::date AS desde, MAX(e.created_at)::date AS hasta
               FROM entregas e
               WHERE NOT EXISTS (SELECT 1 FROM pares p WHERE p.entrega_id = e.id)
               GROUP BY 1 ORDER BY 1""", d),
        "fallback_rate": (
            f"""SELECT COUNT(*) AS planes_entregados,
                      COUNT(*) FILTER (WHERE plan_data->>'_is_fallback' = 'true') AS con_fallback,
                      ROUND((COUNT(*) FILTER (WHERE plan_data->>'_is_fallback' = 'true'))::numeric
                            / NULLIF(COUNT(*), 0), 3) AS rate
               FROM meal_plans
               WHERE {win}""", d),
        # Latencia del pipeline de las corridas que SÍ se entregaron (emparejadas), tipadas por la
        # SUPERFICIE de su entrega (ronda 2). Una reparación parcial entregada cuenta (llegó al
        # usuario); un fallback descartado no tiene fila de entrega y no entra. DISTINCT ON: una
        # corrida cuenta una vez aunque casara con dos filas de entrega.
        "generacion_latencia": (
            f"""{ctes}
               SELECT {delivery_kind_sql('pe.surface')} AS tipo, COUNT(*) AS n,
                      ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY c.duration_ms) / 1000.0)::numeric) AS p50_s,
                      ROUND((percentile_cont(0.95) WITHIN GROUP (ORDER BY c.duration_ms) / 1000.0)::numeric) AS p95_s
               FROM corridas c
               JOIN (SELECT DISTINCT ON (p.corrida_id) p.corrida_id, p.surface
                     FROM pares p ORDER BY p.corrida_id, p.surface DESC) pe
                 ON pe.corrida_id = c.id, win
               WHERE c.created_at >= win.desde
               GROUP BY 1 ORDER BY 1""", d),
        # Denominador honesto: TODAS las corridas del pipeline, con y sin entrega. Antes
        # `no_entregados` contaba solo las marcadas fallback (0/49) cuando ≥23 no se entregaron.
        # [ronda 2] Una corrida SIN entrega no tiene superficie, así que su tipo no se puede probar:
        # las filas se agrupan por lo que SÍ se mide (`origen`) y el tipo de entrega va en columnas.
        #   · `sin_usuario`: `user_id` NULL — invitado o script (no se distinguen desde aquí).
        #   · `sesion_unknown`: el form de la corrida no llevaba sesión (p. ej. el relleno rolling
        #     del chunk worker).
        #   · `con_sesion`: el resto — planes iniciales Y bloques `initial_plan` (días 8-30).
        "corridas_por_entrega": (
            f"""{ctes}
               SELECT CASE WHEN c.user_id IS NULL THEN 'sin_usuario'
                           WHEN c.session_id = 'unknown' THEN 'sesion_unknown'
                           ELSE 'con_sesion' END AS origen,
                      COUNT(*) AS corridas,
                      COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM pares p WHERE p.corrida_id = c.id
                          AND {delivery_kind_sql('p.surface')} = 'plan_inicial')) AS entregadas_plan_inicial,
                      COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM pares p WHERE p.corrida_id = c.id
                          AND {delivery_kind_sql('p.surface')} = 'bloque_posterior')) AS entregadas_bloque_posterior,
                      COUNT(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM pares p WHERE p.corrida_id = c.id)) AS sin_entrega,
                      COUNT(*) FILTER (WHERE c.con_fallback) AS con_fallback,
                      MIN(c.created_at)::date AS desde, MAX(c.created_at)::date AS hasta
               FROM corridas c, win
               WHERE c.created_at >= win.desde
               GROUP BY 1 ORDER BY 1""", d),
        "quality_index": (
            f"""SELECT COUNT(*) AS n,
                      ROUND(AVG((plan_data->'_quality_index'->>'score')::float)::numeric, 1) AS media
               FROM meal_plans
               WHERE plan_data ? '_quality_index' AND {win}""", d),
        "costo_por_nodo": (
            f"""SELECT node, model, COUNT(*) AS calls,
                      ROUND((SUM(cost_usd_micros) / 1e6)::numeric, 4) AS usd
               FROM llm_usage_events
               WHERE {win}
               GROUP BY 1, 2 ORDER BY usd DESC NULLS LAST LIMIT 12""", d),
    }
