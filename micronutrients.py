"""[P3-MICRONUTRIENTS · 2026-06-13] Panel clínico de micronutrientes del plan vs DRI/WHO.

Cierra el hallazgo de la auditoría clínica (FS4): el solver optimizaba solo kcal+macros →
sodio/fibra/azúcar/vit D/calcio/hierro/B12/potasio eran gaps estructurales invisibles. Este
módulo computa el panel desde los micros poblados en `master_ingredients` (P3-MICRONUTRIENTS),
lo compara vs los pisos/techos DRI/WHO (sex-aware), y construye un REPORTE ADVISORY (no un
gate duro: la vit D y otros rara vez se alcanzan con alimentos enteros → se marca como gap
conocido con sugerencia de suplemento, NO se rechaza el plan → cero loops de regen).

LIMITACIÓN honesta: los totales se computan solo de los ingredientes resueltos en el catálogo
`master_ingredients` (cobertura parcial) y NO incluyen la sal AÑADIDA ("sal al gusto", sin
gramaje). Por eso el reporte expone `coverage` y trata los pisos como ESTIMADOS: un piso que
se cumple con la suma parcial es seguro; uno que no, es incierto (puede subir con lo no
resuelto). El techo de sodio que se dispara es señal fuerte; el que no, es incierto (sal añadida).
"""
from __future__ import annotations

# [P2-ANEMIA-TARGET · 2026-06-15] Surface advisory de un condition_target de anemia en el panel (paralelo
# a DM2/HTA/dislipidemia). NO eleva el piso de hierro (usa el RDA 18F/8M ya vigente) → NO crea objetivo
# inalcanzable nuevo → imposible de loopear (el panel es advisory por contrato, no alimenta should_retry).
# Anemia es un DÉFICIT (no un exceso): no hay ingrediente-ofensor que sustituir + el catálogo es-DO no
# tiene hígado y el swap magro→res perdería proteína → NO se añade `substitutions` a la fila anemia (ver
# condition_rules.py). Default OFF (user-facing en panel/PDF). Knob canónico auto-registrado en _KNOBS_REGISTRY.
try:
    from knobs import _env_bool as _mn_env_bool, _env_float as _mn_env_float
    _ANEMIA_CONDITION_TARGET_ENABLED = _mn_env_bool("MEALFIT_ANEMIA_CONDITION_TARGET", False)
    # [P1-RENAL-K-CEILING · 2026-06-26] (auditoría gap #8) Techo OBSERVABLE de potasio en ERC (hiperkalemia
    # = riesgo AGUDO de arritmia, lo que el audit flageó como "no capado"). NO es un veto/retiro de alimentos
    # (respeta la decisión P2-RENAL-POTASSIUM-DETERMINISTIC de NO hacer un veto ciego): es un techo de PANEL
    # que DEGRADA (banner) cuando el plan excede, usando la columna potassium_mg ya poblada desde USDA. El
    # umbral fino por estadio (G3a→G5/diálisis) lo define el nefrólogo; el default es un piso de seguridad.
    _RENAL_K_CEILING_ENABLED = _mn_env_bool("MEALFIT_RENAL_K_CEILING", True)
    _RENAL_K_CEILING_MG = _mn_env_float("MEALFIT_RENAL_K_CEILING_MG", 3000.0)
except Exception:  # pragma: no cover - knobs siempre disponible en prod
    _ANEMIA_CONDITION_TARGET_ENABLED = False
    _RENAL_K_CEILING_ENABLED = True
    _RENAL_K_CEILING_MG = 3000.0

# Términos de azúcar AÑADIDA (free sugars) — el techo WHO aplica a estos, NO al azúcar
# intrínseco de fruta/leche (que no es preocupación de salud).
_ADDED_SUGAR_TERMS = ("miel", "azucar", "azúcar", "sirope", "jarabe", "panela",
                      "melaza", "glasead", "honey", "sugar", "dulce de leche")


# Términos de sexo MASCULINO (mismo set que nutrition_calculator.calculate_bmr; ojo: "mujer"
# empieza con M pero NO es masculino → la lista evita ese falso positivo).
_MALE_TERMS = ("male", "masculino", "m", "hombre")

# [P3-CONDITION-RULES · 2026-06-14] Términos de diabetes (sin acentos; el caller pasa las
# condiciones ya normalizadas con strip_accents). ADA 2025/2026 abandonó el %carbos/IG fijo →
# calidad del carbohidrato: fibra ≥14 g/1000 kcal. Aquí elevamos el piso de fibra para diabéticos.
_DIABETES_TERMS = ("diabet", "dm2", "dm-2", "dm 2", "t2dm", "prediabet", "pre-diabet",
                   "hiperglucem", "resistencia a la insulina", "resistencia insulinica",
                   "glucemia alta", "azucar alta", "intolerancia a la glucosa", "intolerancia a glucosa")
# Fibra mínima por 1000 kcal en DM2 (ADA 2026 "calidad del carbohidrato").
_DM2_FIBER_PER_1000KCAL = 14.0


def _has_diabetes(conditions) -> bool:
    """True si alguna condición (lista de strings ya lowercased/sin-acentos) es diabetes."""
    if not conditions:
        return False
    try:
        return any(any(t in str(c) for t in _DIABETES_TERMS) for c in conditions)
    except Exception:
        return False


def dri_targets(sex: str | None = "F", age: int | None = None, pregnant: bool = False) -> dict:
    """Pisos/techos DRI (IOM) + WHO por nutriente para un adulto. Sex-aware donde importa
    (hierro 18 vs 8 mg; fibra 25 vs 38 g; potasio 2600 vs 3400 mg). Default conservador
    femenino (hierro alto) cuando el sexo es desconocido.

    [P2-DRI-AGE-AWARE · 2026-06-15] (gap-audit G15) Age-aware en los dos micros donde el DRI real cambia
    con la edad y eso afecta la suplementación advisory: el HIERRO de la mujer baja de 18 mg (19-50) a 8 mg
    post-menopausia (51+) — antes 18 mg para todas sobre-flageaba déficit en mayores; el CALCIO sube de
    1000 a 1200 mg para mujeres 51+ y hombres 71+ — antes 1000 fijo sub-reforzaba en mayores. `age=None`
    (desconocida) → valores de adulto joven (conservadores: hierro alto, calcio base).

    [P2-DRI-PREGNANCY-AWARE · 2026-06-19] (audit fresco P2-4) `pregnant=True` sube los micros COMPUTADOS
    cuyo RDA cambia en embarazo/lactancia y donde el déficit es más caro: HIERRO 27 mg (RDA gestación, vs
    18) y B12 2.6 mcg (vs 2.4). El FOLATO (RDA 600 mcg DFE) NO se añade a `targets` a propósito — el catálogo
    no tiene columna de folato, así que un floor sin dato mostraría 'bajo' siempre (falso). El folato se
    surfacea como condition_target advisory en `build_micronutrient_report`. El embarazo tiene precedencia
    sobre el ajuste post-menopáusico del hierro (una embarazada >51 es excepcional pero el RDA gestación manda)."""
    male = str(sex or "").strip().lower() in _MALE_TERMS
    try:
        _age = int(age) if age is not None else None
    except (TypeError, ValueError):
        _age = None
    # Hierro (IOM/DRI): hombre 8 mg; mujer 18 mg (19-50) → 8 mg (51+, post-menopausia).
    # [P2-DRI-PREGNANCY-AWARE] embarazo → 27 mg (manda sobre sexo/edad).
    _iron = 27.0 if pregnant else (8.0 if male else (8.0 if (_age is not None and _age >= 51) else 18.0))
    # Calcio (IOM/DRI): 1200 mg para mujer 51+ / hombre 71+; 1000 mg resto.
    _calcium = 1200.0 if (_age is not None and ((not male and _age >= 51) or (male and _age >= 71))) else 1000.0
    # [P2-DRI-AGE-AWARE-VITD · 2026-06-18] (audit fresco P2) Vit D (IOM/DRI): 15 mcg (600 UI) para 1-70 años;
    # 20 mcg (800 UI) para 71+ — antes 15 fijo sub-flageaba déficit en el grupo de mayor riesgo de
    # insuficiencia. Simétrico al ajuste por edad de hierro/calcio (el set original solo cubría esos dos).
    _vit_d = 20.0 if (_age is not None and _age >= 71) else 15.0
    # [P1-FOOD-DB-EXTENDED-MICROS · 2026-06-25] Panel exhaustivo (DRI/IOM, sex/embarazo-aware donde el RDA
    # cambia). Folato DFE (clave embarazo, ya advisory → ahora computado), zinc (músculo/inmunidad), vit A
    # RAE, vit C, vit E, vit K (AI; ojo warfarina, ver nota), selenio, omega-3 ALA (AI).
    _zinc     = 11.0 if (pregnant or male) else 8.0     # RDA mg (lactancia 12; embarazo 11)
    _folate   = 600.0 if pregnant else 400.0            # mcg DFE (embarazo 600)
    _vit_a    = 770.0 if pregnant else (900.0 if male else 700.0)   # mcg RAE (embarazo 770)
    _vit_c    = 85.0 if pregnant else (90.0 if male else 75.0)      # mg (embarazo 85)
    _vit_e    = 15.0                                    # mg alfa-tocoferol (adulto)
    _vit_k    = 120.0 if male else 90.0                 # mcg (AI)
    _selenium = 60.0 if pregnant else 55.0             # mcg (embarazo 60)
    _omega3   = 1.4 if pregnant else (1.6 if male else 1.1)         # g ALA (AI)
    return {
        "fiber_g":       {"floor": 38.0 if male else 25.0, "unit": "g"},
        "sodium_mg":     {"ceiling": 2000.0, "unit": "mg"},          # WHO <2000
        "free_sugars_g": {"ceiling": 25.0, "unit": "g"},             # WHO condicional <5% E
        "vit_d_mcg":     {"floor": _vit_d, "unit": "mcg"},           # DRI 600 UI (15 mcg) / 800 UI (20 mcg) 71+
        "calcium_mg":    {"floor": _calcium, "unit": "mg"},
        "iron_mg":       {"floor": _iron, "unit": "mg"},
        "b12_mcg":       {"floor": 2.6 if pregnant else 2.4, "unit": "mcg"},   # [P2-DRI-PREGNANCY-AWARE]
        "potassium_mg":  {"floor": 3400.0 if male else 2600.0, "unit": "mg"},
        "magnesium_mg":  {"floor": 420.0 if male else 320.0, "unit": "mg"},   # [P4] DRI IOM
        # [P1-FOOD-DB-EXTENDED-MICROS] panel exhaustivo (todos floors).
        "zinc_mg":       {"floor": _zinc, "unit": "mg"},
        "folate_mcg":    {"floor": _folate, "unit": "mcg"},
        "vit_a_mcg":     {"floor": _vit_a, "unit": "mcg"},
        "vit_c_mg":      {"floor": _vit_c, "unit": "mg"},
        "vit_e_mg":      {"floor": _vit_e, "unit": "mg"},
        "vit_k_mcg":     {"floor": _vit_k, "unit": "mcg"},
        "selenium_mcg":  {"floor": _selenium, "unit": "mcg"},
        "omega3_g":      {"floor": _omega3, "unit": "g"},
    }


_LABELS = {
    "fiber_g": "Fibra", "sodium_mg": "Sodio", "free_sugars_g": "Azúcares añadidos",
    "vit_d_mcg": "Vitamina D", "calcium_mg": "Calcio", "iron_mg": "Hierro",
    "b12_mcg": "Vitamina B12", "potassium_mg": "Potasio",
    "magnesium_mg": "Magnesio", "saturated_fat_g": "Grasa saturada",
    # [P1-FOOD-DB-EXTENDED-MICROS] panel exhaustivo
    "zinc_mg": "Zinc", "folate_mcg": "Folato", "vit_a_mcg": "Vitamina A",
    "vit_c_mg": "Vitamina C", "vit_e_mg": "Vitamina E", "vit_k_mcg": "Vitamina K",
    "selenium_mcg": "Selenio", "omega3_g": "Omega-3",
}

_SUPPLEMENT_NOTE = {
    "fiber_g": "Aumenta vegetales, frutas con cáscara, legumbres (habichuelas) y granos integrales.",
    "vit_d_mcg": "Una dieta de alimentos enteros rara vez alcanza la vit D: añade pescado graso "
                 "(salmón/sardina 1-2x/sem) o lácteo fortificado, o considera un suplemento de 600-800 UI.",
    "calcium_mg": "Refuerza con lácteos (yogur/queso) o vegetales de hoja verde y sésamo.",
    "iron_mg": "Refuerza con legumbres (habichuelas), carnes rojas magras y hígado; acompaña con vit C "
               "(naranja/limón) para mejorar la absorción.",
    "b12_mcg": "Asegura fuentes animales (huevo, lácteos, carne, pescado); si eres vegano, suplemento de B12.",
    "potassium_mg": "Aumenta frutas, vegetales y legumbres (habichuelas, guineo, batata, espinaca).",
    "sodium_mg": "Reduce la sal añadida (≤1 g/día) y usa especias sin sodio (ajo, comino, orégano, limón).",
    "free_sugars_g": "Reduce miel/azúcares añadidos; endulza con fruta o estevia.",
    "magnesium_mg": "Aumenta vegetales de hoja verde, legumbres, nueces/semillas y granos integrales (clave del patrón DASH).",
    "saturated_fat_g": "Reduce frituras, piel de pollo, grasa visible de carnes, embutidos, mantequilla y lácteos enteros; "
                       "usa cocción al horno/plancha/hervido y grasas insaturadas (aguacate, aceite de oliva).",
    # [P1-FOOD-DB-EXTENDED-MICROS] notas accionables (comida primero) del panel exhaustivo.
    "zinc_mg": "Refuerza con carnes (res/cerdo), mariscos, huevo, legumbres, nueces/semillas (calabaza/ajonjolí).",
    "folate_mcg": "Aumenta vegetales de hoja verde, legumbres (habichuelas/lentejas), aguacate, cítricos y granos fortificados.",
    "vit_a_mcg": "Aumenta vegetales naranja/verde oscuro (zanahoria, auyama, batata, espinaca), huevo y lácteos.",
    "vit_c_mg": "Aumenta cítricos (naranja/limón), guayaba, pimiento/ají, brócoli y frutas frescas; ayuda a absorber el hierro.",
    "vit_e_mg": "Refuerza con nueces/semillas (almendra, girasol), aceites vegetales, aguacate y hoja verde.",
    "vit_k_mcg": "Se obtiene de vegetales de hoja verde (espinaca, brócoli). IMPORTANTE: si tomas anticoagulante "
                 "(warfarina), NO la aumentes de golpe — mantén una ingesta CONSISTENTE y coordina con tu médico.",
    "selenium_mcg": "Refuerza con pescado/mariscos, huevo, carnes y nuez de Brasil (1-2 al día bastan).",
    "omega3_g": "Aumenta pescado graso (sardina/salmón), linaza/chía, nueces y aceite de canola.",
}

# [P1-POTASSIUM-PANEL-MED-AWARE · 2026-06-19] (audit fresco P1-1) Nota ALTERNATIVA del potasio cuando el perfil
# toma un fármaco que lo eleva (ahorrador-K/IECA-ARA-II): el piso baja al DRI (no DASH 4700), pero si el plan
# entrega potasio bajo el DRI, la nota estándar `_SUPPLEMENT_NOTE["potassium_mg"]` ("come más guineo/aguacate")
# CONTRADIRÍA al `medication_review` ("NO maximices potasio") en el mismo PDF. Esta nota lo modera en su lugar,
# cerrando el contradictorio también en la capa del gap del piso (la fila DASH ya se moderó arriba).
_POTASSIUM_RESTRICTED_NOTE = ("Mantén el potasio en porciones MODERADAS y parejas (no lo maximices): tu "
                              "medicación puede elevar el potasio en sangre (riesgo de hiperkalemia). El "
                              "balance fino lo define tu médico con análisis.")

# [P2-RENAL-FIBER-NOTE · 2026-06-19] (audit fresco P2, cluster S1) Nota de fibra para perfiles con ERC: la fibra
# NO se restringe en enfermedad renal, pero su FUENTE sí — la nota estándar empuja "legumbres (habichuelas)",
# altas en potasio/fósforo que KDIGO pide moderar. Esta variante orienta la fibra a vegetales/frutas bajos en
# potasio, coherente con la moderación renal (cierra la asimetría con el guard de potasio del panel).
_FIBER_RENAL_NOTE = ("Aumenta la fibra con vegetales y frutas bajos en potasio (ej. manzana, pera, repollo, "
                     "zanahoria, pepino) y avena; MODERA las leguminosas y granos muy altos en potasio/fósforo "
                     "si tienes enfermedad renal — el balance fino lo define tu nefrólogo.")

# [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (gap-audit G5) Umbral de cobertura POR-NUTRIENTE bajo el cual un
# TECHO en apariencia 'ok' se reporta 'estimado_alto' (incierto). Mismo 0.6 que el 'estimado_bajo' de los
# pisos, para simetría. Caveat honesto para el panel/PDF.
_CEILING_COVERAGE_FLOOR = 0.6
_CEILING_ESTIMADO_NOTE = ("Estimado: algunos ingredientes no tienen dato de este nutriente en el catálogo, "
                          "por lo que el total mostrado puede estar SUBESTIMADO y el valor real superar el "
                          "techo. Verifícalo con tu nutricionista, especialmente si tienes una condición "
                          "cardiometabólica.")


# [P4-UNIFIED-RESOLVER · 2026-06-14] Detección de HTA y dislipidemia para los targets condicionales
# (DASH Mg/K, techo de grasa saturada) — SSOT de términos en constants (mismo registro del motor).
def _has_condition(conditions, terms) -> bool:
    if not conditions:
        return False
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    try:
        # Normaliza la condición (lower + sin acentos) — los `terms` ya son ascii lowercase. Robusto
        # tanto si el caller pasó el string normalizado como crudo ("Hipertensión").
        return any(any(t in _sa(str(c).lower()) for t in terms) for c in conditions)
    except Exception:
        return False


def _has_hta(conditions) -> bool:
    try:
        from constants import HTA_CONDITION_TERMS
        return _has_condition(conditions, HTA_CONDITION_TERMS)
    except Exception:
        return _has_condition(conditions, ("hipertens", "presion alta", "presión alta", "hta"))


def _has_dyslipidemia(conditions) -> bool:
    try:
        from constants import DYSLIPIDEMIA_CONDITION_TERMS
        return _has_condition(conditions, DYSLIPIDEMIA_CONDITION_TERMS)
    except Exception:
        return _has_condition(conditions, ("colesterol", "dislipid", "trigliceri", "ldl alto"))


def _has_anemia(conditions) -> bool:
    # [P2-ANEMIA-TARGET · 2026-06-15] Paralelo a _has_hta/_has_dyslipidemia.
    try:
        from constants import ANEMIA_CONDITION_TERMS
        return _has_condition(conditions, ANEMIA_CONDITION_TERMS)
    except Exception:
        return _has_condition(conditions, ("anemia", "ferropen", "hierro bajo", "ferritina baja"))


def _has_renal(conditions) -> bool:
    # [P1-RENAL-SODIUM-SUBS · 2026-06-19] (audit fresco P1-4) Paralelo a _has_hta. El panel imponía
    # condition_target de sodio solo para HTA → un perfil ERC-puro (sin HTA) no surfaceaba la restricción
    # de sodio renal (estándar-de-cuidado KDIGO). Cierra la asimetría de OBSERVABILIDAD del panel/PDF.
    try:
        from constants import RENAL_CONDITION_TERMS
        return _has_condition(conditions, RENAL_CONDITION_TERMS)
    except Exception:
        return _has_condition(conditions, ("renal", "rinon", "erc", "ckd", "nefro", "enfermedad renal",
                                           "insuficiencia renal", "nefropat"))


def compute_plan_micronutrient_totals(plan: dict, db) -> dict:
    """Suma los micros de todos los ingredientes resueltos del plan y devuelve el PROMEDIO
    diario + metadata de cobertura. `free_sugars_g` solo cuenta el azúcar de ingredientes
    de azúcar AÑADIDA (miel/sirope/glaseado), no el intrínseco de fruta/leche."""
    days = plan.get("days") or []
    num_days = max(1, len(days))
    acc = {k: 0.0 for k in ("fiber_g", "sodium_mg", "free_sugars_g", "vit_d_mcg",
                            "calcium_mg", "iron_mg", "b12_mcg", "potassium_mg",
                            "magnesium_mg", "saturated_fat_g",
                            # [P1-FOOD-DB-EXTENDED-MICROS] panel exhaustivo
                            "zinc_mg", "folate_mcg", "vit_a_mcg", "vit_c_mg",
                            "vit_e_mg", "vit_k_mcg", "selenium_mcg", "omega3_g")}
    # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (gap-audit G5) Cobertura POR-NUTRIENTE: de los ingredientes
    # RESUELTOS, cuántos traían dato NO-NULL de cada micro. `coverage` (global) NO basta para los TECHOS:
    # un ingrediente puede resolver (tiene macros) pero traer la columna del micro NULL (p.ej. embutidos DD
    # con saturated_fat_g sin poblar) → suma 0.0 → el techo reportaría 'ok' falso para un dislipidémico. La
    # cobertura por-nutriente distingue "0 real" de "0 por NULL" → permite reportar 'estimado_alto' (incierto).
    _SRC_KEY = {"fiber_g": "fiber", "sodium_mg": "sodium_mg", "vit_d_mcg": "vit_d_mcg",
                "calcium_mg": "calcium_mg", "iron_mg": "iron_mg", "b12_mcg": "b12_mcg",
                "potassium_mg": "potassium_mg", "magnesium_mg": "magnesium_mg",
                "saturated_fat_g": "saturated_fat_g",
                # [P1-FOOD-DB-EXTENDED-MICROS] panel exhaustivo (report-key → micros-dict key, idénticos)
                "zinc_mg": "zinc_mg", "folate_mcg": "folate_mcg", "vit_a_mcg": "vit_a_mcg",
                "vit_c_mg": "vit_c_mg", "vit_e_mg": "vit_e_mg", "vit_k_mcg": "vit_k_mcg",
                "selenium_mcg": "selenium_mcg", "omega3_g": "omega3_g"}
    present = {k: 0 for k in _SRC_KEY}
    total_ings = resolved_ings = 0
    # [P1-MICRO-PERDAY-FLOOR · 2026-07-02] Acumulación POR DÍA además del promedio del plan.
    # El panel se evaluaba SOLO sobre el promedio (v/num_days) — un plan con varianza alta
    # podía "cumplir" en promedio con días individuales deficitarios (asimetría vs macros,
    # que se evalúan per-día×celda). `per_day` alimenta el resumen worst-day del reporte.
    per_day: list = []
    for day in days:
        day_acc = {k: 0.0 for k in acc}
        for meal in day.get("meals", []) or []:
            # [P1-MICRONUTRIENT-RAW-INGREDIENTS · 2026-06-25] Prefiere `ingredients_raw` (forma CANÓNICA
            # con gramaje explícito + nombre real del protein-closer: "20g de queso mozzarella cocido")
            # sobre `ingredients` (forma DISPLAY: "¾ lonja/pedazo de queso", que NO resuelve nombre/unidad).
            # El display pierde resolubilidad → sub-cuenta los micros del closer (queso/yogurt: calcio/B12/
            # magnesio) y, peor, un RECÓMPUTO post-generación (chunk worker P1-MICRONUTRIENT-CHUNK-RECOMPUTE
            # / swap / regenerate-day) sobre el display daría un total MENOR que el de assemble (que se
            # computó sobre la forma raw) → panel inconsistente. Usar raw recupera ~7% de magnesio
            # (436→467 medido en vivo) y hace recompute == assemble. Fallback a `ingredients` si no hay raw
            # (planes viejos / meals sin la lista). tooltip-anchor: P1-MICRONUTRIENT-RAW-INGREDIENTS
            for ing in (meal.get("ingredients_raw") or meal.get("ingredients") or []):
                total_ings += 1
                m = db.micros_from_ingredient_string(str(ing))
                if not m:
                    continue
                resolved_ings += 1
                day_acc["fiber_g"] += m.get("fiber") or 0.0
                day_acc["sodium_mg"] += m.get("sodium_mg") or 0.0
                day_acc["vit_d_mcg"] += m.get("vit_d_mcg") or 0.0
                day_acc["calcium_mg"] += m.get("calcium_mg") or 0.0
                day_acc["iron_mg"] += m.get("iron_mg") or 0.0
                day_acc["b12_mcg"] += m.get("b12_mcg") or 0.0
                day_acc["potassium_mg"] += m.get("potassium_mg") or 0.0
                day_acc["magnesium_mg"] += m.get("magnesium_mg") or 0.0           # [P4-UNIFIED-RESOLVER]
                day_acc["saturated_fat_g"] += m.get("saturated_fat_g") or 0.0     # [P4-UNIFIED-RESOLVER]
                # [P1-FOOD-DB-EXTENDED-MICROS] panel exhaustivo
                day_acc["zinc_mg"] += m.get("zinc_mg") or 0.0
                day_acc["folate_mcg"] += m.get("folate_mcg") or 0.0
                day_acc["vit_a_mcg"] += m.get("vit_a_mcg") or 0.0
                day_acc["vit_c_mg"] += m.get("vit_c_mg") or 0.0
                day_acc["vit_e_mg"] += m.get("vit_e_mg") or 0.0
                day_acc["vit_k_mcg"] += m.get("vit_k_mcg") or 0.0
                day_acc["selenium_mcg"] += m.get("selenium_mcg") or 0.0
                day_acc["omega3_g"] += m.get("omega3_g") or 0.0
                # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] cuenta presencia NO-NULL por micro (G5).
                for _ak, _sk in _SRC_KEY.items():
                    if m.get(_sk) is not None:
                        present[_ak] += 1
                ing_low = str(ing).lower()
                if any(t in ing_low for t in _ADDED_SUGAR_TERMS):
                    day_acc["free_sugars_g"] += m.get("sugars_g") or 0.0
        # [P1-MICRO-PERDAY-FLOOR] merge del día al acumulador del plan + snapshot per-día.
        for _dk, _dv in day_acc.items():
            acc[_dk] += _dv
        per_day.append({_dk: round(_dv, 1) for _dk, _dv in day_acc.items()})
    daily = {k: round(v / num_days, 1) for k, v in acc.items()}
    coverage = round(resolved_ings / total_ings, 2) if total_ings else 0.0
    # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (G5) fracción de resueltos con dato NO-NULL por micro.
    nutrient_coverage = {k: (round(present[k] / resolved_ings, 2) if resolved_ings else 0.0)
                         for k in present}
    return {"daily": daily, "coverage": coverage, "nutrient_coverage": nutrient_coverage,
            "resolved_ings": resolved_ings, "total_ings": total_ings, "num_days": num_days,
            "per_day": per_day}  # [P1-MICRO-PERDAY-FLOOR]


def build_micronutrient_report(plan: dict, db, sex: str | None = "F",
                               conditions=None, daily_kcal: float | None = None,
                               fiber_per_1000kcal: float = _DM2_FIBER_PER_1000KCAL,
                               age: int | None = None, pregnant: bool = False,
                               k_elevating_med: bool = False) -> dict:
    """Reporte advisory: panel de micros diarios vs DRI/WHO con status + nota accionable.
    status ∈ {ok, bajo, alto, estimado_bajo, estimado_alto}. Floors incumplidos con cobertura
    parcial → 'estimado_bajo' (incierto, puede subir con lo no resuelto). Techos en apariencia
    'ok' pero con cobertura POR-NUTRIENTE parcial → 'estimado_alto' (incierto, el real puede
    superar el techo — dirección peligrosa, G5). NO rechaza el plan.

    [P3-CONDITION-RULES] Si `conditions` incluye diabetes, eleva el PISO de fibra a la regla
    ADA 2026 de calidad del carbohidrato (≥14 g/1000 kcal, usando `daily_kcal`) en vez del DRI
    general — convierte la fibra de un advisory genérico en un objetivo clínico citable por DM2."""
    totals = compute_plan_micronutrient_totals(plan, db)
    daily = totals["daily"]
    coverage = totals["coverage"]
    targets = dri_targets(sex, age, pregnant=pregnant)   # [P2-DRI-AGE-AWARE/PREGNANCY-AWARE] hierro/calcio/B12
    condition_targets = []
    # [P2-DRI-PREGNANCY-AWARE · 2026-06-19] (audit fresco P2-4) Embarazo/lactancia → condition_target citable
    # con folato (RDA 600 mcg DFE — NO computable, sin columna en el catálogo, por eso advisory y no floor) +
    # hierro 27 mg (sí computado, ya elevado en `targets`). Cierra la falsa tranquilidad del panel en el estado
    # de mayor consecuencia por déficit (anemia materna / defectos del tubo neural).
    if pregnant:
        condition_targets.append({
            "condicion": "Embarazo / lactancia",
            "regla": "Folato ≥600 mcg/día (hoja verde, leguminosas, cítricos) + Hierro ≥27 mg/día (con vit C)",
            "guia": "RDA gestación (IOM) — prioriza folato (tubo neural) y hierro (anemia materna); evita listeria",
            "actual": {"hierro": daily.get("iron_mg", 0.0)},
        })
    # [P3-CONDITION-RULES] DM2: piso de fibra = max(DRI, 14 g/1000 kcal). ADA 2025/2026 reemplazó
    # el target de %carbos/IG por "calidad del carbohidrato" — la fibra es el proxy citable.
    if _has_diabetes(conditions) and daily_kcal and daily_kcal > 0:
        dm2_fiber_floor = round(fiber_per_1000kcal * (daily_kcal / 1000.0), 1)
        if dm2_fiber_floor > targets["fiber_g"]["floor"]:
            targets["fiber_g"]["floor"] = dm2_fiber_floor
        condition_targets.append({
            "condicion": "Diabetes T2 / prediabetes",
            "regla": f"Fibra ≥{targets['fiber_g']['floor']}g/día (≥{int(fiber_per_1000kcal)} g/1000 kcal)",
            "guia": "ADA 2025/2026 — calidad del carbohidrato (reemplaza %carbos/índice glucémico)",
            "actual": daily.get("fiber_g", 0.0),
        })
    # [P4-UNIFIED-RESOLVER] HTA → patrón DASH: eleva el PISO de potasio (4700 mg) y magnesio (500 mg)
    # sobre el DRI general. Antes el balance DASH era PROMPT-confiado; ahora se evalúa con dato real
    # (columnas potassium/magnesium pobladas desde USDA). El techo de sodio ya vive en el DRI general.
    # [P2-RENAL-HTA-POTASSIUM-GUARD · 2026-06-19] (audit fresco P2-2) El piso DASH de potasio (4700) y
    # magnesio (500) se aplica SOLO si NO hay ERC comórbida: en enfermedad renal el potasio se RESTRINGE
    # (riesgo de hiperkalemia → arritmia), así que el piso DASH está contraindicado y manda el techo renal.
    # Para renal+HTA, el sodio bajo de DASH sí aplica (cubierto por la rama renal de abajo + el cap/subs);
    # solo se omite la maximización de potasio/magnesio. La HTA es la causa #1 de ERC → comorbilidad común.
    if _has_hta(conditions) and not _has_renal(conditions):
        # Magnesio (DASH) siempre aplica: un ahorrador de potasio / IECA-ARA-II no contraindica el magnesio.
        if 500.0 > targets["magnesium_mg"]["floor"]:
            targets["magnesium_mg"]["floor"] = 500.0
        # [P1-POTASSIUM-PANEL-MED-AWARE · 2026-06-19] (audit fresco P1-1) El piso DASH de potasio (4700) se
        # eleva SOLO si el perfil NO toma un fármaco que ELEVE el potasio sérico (`k_elevating_med`: ahorrador
        # de potasio —espironolactona— o IECA/ARA-II). Antes el panel era CIEGO a `medications`: subía el piso a
        # 4700 y emitía la nota "come más guineo/aguacate/leguminosas" mientras `medication_review` decía lo
        # contrario ("NO maximices potasio") → señales OPUESTAS en el mismo PDF, y el panel determinista empujaba
        # la dirección PELIGROSA (hiperkalemia → arritmia). Es la asimetría exacta del guard renal `not _has_renal`:
        # la ERC ya suprime el piso DASH-K; un fármaco que sube el potasio debe suprimirlo igual. El sodio bajo de
        # DASH (techo 2000 del DRI) y el magnesio sí aplican; solo se omite la MAXIMIZACIÓN del potasio.
        if not k_elevating_med:
            if 4700.0 > targets["potassium_mg"]["floor"]:
                targets["potassium_mg"]["floor"] = 4700.0
            condition_targets.append({
                "condicion": "Hipertensión (patrón DASH)",
                "regla": "Potasio ≥4700 mg/día + Magnesio ≥500 mg/día + Sodio <2000 mg/día",
                "guia": "DASH (NHLBI/AHA-ACC) — el balance Na/K/Mg/Ca baja la presión arterial",
                "actual": {"potasio": daily.get("potassium_mg", 0.0), "magnesio": daily.get("magnesium_mg", 0.0),
                           "sodio": daily.get("sodium_mg", 0.0)},
            })
        else:
            condition_targets.append({
                "condicion": "Hipertensión (patrón DASH) — potasio moderado por medicación",
                "regla": "Magnesio ≥500 mg/día + Sodio <2000 mg/día — NO maximizar el potasio (medicación que lo eleva)",
                "guia": ("DASH adaptado: tu medicamento (ahorrador de potasio o IECA/ARA-II) ELEVA el potasio "
                         "sérico → mantén porciones moderadas de guineo/aguacate/leguminosas; el balance fino lo "
                         "define tu médico con análisis (riesgo de hiperkalemia)."),
                "actual": {"magnesio": daily.get("magnesium_mg", 0.0), "sodio": daily.get("sodium_mg", 0.0)},
            })
    # [P1-RENAL-SODIUM-SUBS · 2026-06-19] (audit fresco P1-4) ERC → surface la restricción de SODIO como
    # condition_target citable (paralelo a HTA), cerrando la asimetría de observabilidad: antes solo HTA
    # generaba un target de sodio en el panel, así que un perfil ERC-puro (sin HTA) no lo veía. NO modifica
    # `targets` (el techo `sodium_mg` 2000 del DRI ya aplica a todos → sin enforcement nuevo, sin objetivo
    # inalcanzable). El cap de proteína KDIGO y los swaps de sodio (condition_rules) son el enforcement; esto
    # es solo el espejo de display. NO eleva potasio (lo contrario al riesgo renal — comorbilidad renal+HTA
    # es el P2-2 pendiente).
    if _has_renal(conditions):
        condition_targets.append({
            "condicion": "Enfermedad renal crónica",
            "regla": "Sodio <2000 mg/día (sal medida mínima) — modera potasio/fósforo si aparecen en exceso",
            "guia": "KDIGO 2024 — la restricción de sodio reduce sobrecarga de volumen, edema y proteinuria",
            "actual": daily.get("sodium_mg", 0.0),
        })
        # [P1-RENAL-K-CEILING · 2026-06-26] (auditoría gap #8) Techo OBSERVABLE de potasio (hiperkalemia
        # AGUDA = arritmia). Convierte el floor DRI de potasio en un TECHO para ERC (el piso DASH-K ya se
        # suprime arriba con `not _has_renal`). El panel marca 'alto' (→ banner/degrade) cuando el plan
        # excede → cierra el gap "hiperkalemia ERC no capada". NO veta alimentos ni fuerza retry (la
        # variedad/costo se preservan); es observabilidad + steering. Knob MEALFIT_RENAL_K_CEILING.
        if _RENAL_K_CEILING_ENABLED:
            targets["potassium_mg"] = {"ceiling": _RENAL_K_CEILING_MG, "unit": "mg"}
            condition_targets.append({
                "condicion": "Enfermedad renal crónica — potasio moderado",
                "regla": f"Potasio ≤{int(_RENAL_K_CEILING_MG)} mg/día (moderar — riesgo de hiperkalemia/arritmia)",
                "guia": "KDIGO — en ERC el potasio se MODERA; el umbral exacto por estadio lo define tu nefrólogo",
                "actual": daily.get("potassium_mg", 0.0),
            })
    # [P4-UNIFIED-RESOLVER] Dislipidemia → techo de GRASA SATURADA <7% de las kcal (AHA/ACC). Antes era
    # PROMPT-only (faltaba la columna satfat); ahora se evalúa con dato real. Solo se añade el target
    # cuando hay dislipidemia (todos consumen algo de satfat; la regla <7% es condicional).
    if _has_dyslipidemia(conditions) and daily_kcal and daily_kcal > 0:
        _satfat_ceiling = round(0.07 * daily_kcal / 9.0, 1)   # 7% de kcal → gramos (grasa = 9 kcal/g)
        targets["saturated_fat_g"] = {"ceiling": _satfat_ceiling, "unit": "g"}
        condition_targets.append({
            "condicion": "Dislipidemia / colesterol alto",
            "regla": f"Grasa saturada ≤{_satfat_ceiling}g/día (<7% de las kcal)",
            "guia": "AHA 2021 / ACC 2025 — la grasa saturada eleva el LDL",
            "actual": daily.get("saturated_fat_g", 0.0),
        })
    # [P2-ANEMIA-TARGET · 2026-06-15] Anemia ferropénica → surface el target de hierro como condition_target
    # citable (paralelo a DM2/HTA/dislipidemia). NO modifica `targets` (el piso `iron_mg` 18F/8M YA es el RDA
    # por sexo vigente → sin enforcement nuevo, sin objetivo inalcanzable nuevo → imposible de loopear). Solo
    # añade una fila descriptiva al panel/PDF. Gateado por knob default OFF (user-facing).
    if _ANEMIA_CONDITION_TARGET_ENABLED and _has_anemia(conditions):
        _iron_floor = (targets.get("iron_mg") or {}).get("floor")
        condition_targets.append({
            "condicion": "Anemia ferropénica",
            "regla": (f"Hierro ≥{_iron_floor}mg/día (RDA por sexo) + vitamina C en la misma comida"
                      if _iron_floor else "Hierro alto (RDA por sexo) + vitamina C en la misma comida"),
            "guia": "OMS/RDA — prioriza hierro hemo (carnes magras) + leguminosas; separa de café/té/lácteos",
            "actual": daily.get("iron_mg", 0.0),
        })
    panel, gaps = [], []
    nutrient_coverage = totals.get("nutrient_coverage", {})
    for key, tgt in targets.items():
        val = daily.get(key, 0.0)
        unit = tgt["unit"]
        if "ceiling" in tgt:
            ceil = tgt["ceiling"]
            # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (G5) La dirección PELIGROSA de un techo es la
            # SUB-estimación (real > techo pero reportamos 'ok'). Si la cobertura POR-NUTRIENTE es parcial
            # (ingredientes resueltos con la columna NULL — p.ej. embutidos DD sin satfat poblado), el total
            # está subestimado → 'estimado_alto' (incierto + caveat), NUNCA 'ok' liso. Simétrico a 'estimado_bajo'.
            _ncov = nutrient_coverage.get(key, coverage)
            if val > ceil:
                status = "alto"
            elif _ncov < _CEILING_COVERAGE_FLOOR:
                status = "estimado_alto"
            else:
                status = "ok"
            entry = {"nutriente": _LABELS[key], "key": key, "valor": val, "unidad": unit,
                     "techo": ceil, "status": status, "cobertura_nutriente": _ncov}
            if status == "alto":
                entry["nota"] = _SUPPLEMENT_NOTE.get(key, "")
                gaps.append(entry)
            elif status == "estimado_alto":
                entry["nota"] = _CEILING_ESTIMADO_NOTE
                gaps.append(entry)
        else:
            floor = tgt["floor"]
            # [P1-FLOOR-COVERAGE-AWARE · 2026-06-24] Simétrico a P1-CEILING-COVERAGE-AWARE (rama
            # ceiling arriba): un PISO incumplido es INCIERTO no solo cuando la cobertura GLOBAL es
            # parcial, sino también cuando la cobertura POR-NUTRIENTE de ESTE micro es parcial
            # (ingredientes RESUELTOS pero con la columna del micro en NULL → suman 0.0 → SUBESTIMAN
            # el total de este micro específico). Antes la rama floor solo miraba `coverage` global
            # → un piso como el magnesio reportaba 'bajo' CONFIADO aunque sus contribuyentes de alta
            # frecuencia (yogurt/casabe/ajo con magnesium NULL) lo subestimaran sistemáticamente. El
            # backfill del catálogo (P1-MICRONUTRIENT-CATALOG-BACKFILL) corrige el VALOR; esta rama
            # corrige la HONESTIDAD del status para los NULL residuales (filas nuevas / no resueltos).
            _ncov_floor = nutrient_coverage.get(key, coverage)
            if val >= floor:
                status = "ok"
            elif coverage < 0.6 or _ncov_floor < _CEILING_COVERAGE_FLOOR:
                status = "estimado_bajo"  # cobertura global o por-nutriente parcial → incierto
            else:
                status = "bajo"
            entry = {"nutriente": _LABELS[key], "key": key, "valor": val, "unidad": unit,
                     "piso": floor, "status": status}
            if status in ("bajo", "estimado_bajo"):
                # [P1-POTASSIUM-PANEL-MED-AWARE · 2026-06-19] (audit fresco P1-1) Con un fármaco que eleva el
                # potasio, NO emitir la nota DRI "come más guineo/aguacate" (contradiría la moderación de
                # potasio del medication_review → hiperkalemia). El potasio basal sigue siendo necesario, pero
                # el panel/PDF no debe NUDGEAR a subirlo. Gateado por el mismo flag de P1-1 (rollback vía knob).
                if key == "potassium_mg" and k_elevating_med:
                    entry["nota"] = _POTASSIUM_RESTRICTED_NOTE
                elif key == "fiber_g" and _has_renal(conditions):
                    entry["nota"] = _FIBER_RENAL_NOTE   # [P2-RENAL-FIBER-NOTE] no empujar leguminosas en ERC
                else:
                    entry["nota"] = _SUPPLEMENT_NOTE.get(key, "")
                # [P3-FLOOR-ESTIMADO-CAVEAT · 2026-07-01] (audit v2 micros GAP-6, batch P3-AUDIT-V2-
                # RESIDUALS) Asimetría de honestidad: la rama ceiling tiene nota dedicada para lo
                # INCIERTO (_CEILING_ESTIMADO_NOTE) pero el piso mostraba la MISMA nota de "come más X"
                # para 'bajo' confiado y 'estimado_bajo' (que puede ser solo dato faltante del catálogo,
                # no un déficit real). Caveat simétrico anexado, sin reemplazar el consejo (sigue siendo
                # accionable si el déficit es real).
                if status == "estimado_bajo" and entry.get("nota"):
                    entry["nota"] = (str(entry["nota"]).rstrip() +
                                     " (Dato estimado: algunos ingredientes de tu plan no tienen este "
                                     "nutriente medido en el catálogo — el valor real puede ser mayor.)")
                gaps.append(entry)
        panel.append(entry)
    # [P1-MICRO-PERDAY-FLOOR · 2026-07-02] Resumen worst-day: el panel de arriba evalúa el
    # PROMEDIO del plan; aquí evaluamos cada día contra los pisos y resumimos el peor día
    # (compacto — NO persistimos la matriz 17×N). Solo pisos con cobertura CIERTA (misma
    # semántica que 'bajo' vs 'estimado_bajo') y solo micros accionables con comida:
    # vit D / B12 (inalcanzables, regla de oro del panel) y vit K (consistencia INR — no
    # nudgear hojas verdes) quedan FUERA; potasio queda fuera bajo fármaco K-elevador.
    per_day_floors = None
    try:
        import os as _os_pd
        _pd_ratio = min(1.0, max(0.1, float(_os_pd.environ.get("MEALFIT_MICRO_PERDAY_RATIO", "0.6"))))
        _pd_min = max(1, int(float(_os_pd.environ.get("MEALFIT_MICRO_PERDAY_MIN_MICROS", "2"))))
        _pd_days = totals.get("per_day") or []
        _pd_excluded = {"vit_d_mcg", "b12_mcg", "vit_k_mcg"}
        if k_elevating_med:
            _pd_excluded.add("potassium_mg")
        if len(_pd_days) >= 2 and coverage >= 0.6:
            _pd_low_by_day = []
            for _pd_vals in _pd_days:
                _lows = []
                for _pk, _ptgt in targets.items():
                    if "ceiling" in _ptgt or _pk in _pd_excluded:
                        continue
                    if nutrient_coverage.get(_pk, coverage) < _CEILING_COVERAGE_FLOOR:
                        continue  # cobertura por-nutriente incierta → no acusar el día
                    try:
                        if float(_pd_vals.get(_pk, 0.0)) < float(_ptgt["floor"]) * _pd_ratio:
                            _lows.append(_pk)
                    except (TypeError, ValueError):
                        continue
                _pd_low_by_day.append(_lows)
            _worst_idx = max(range(len(_pd_low_by_day)), key=lambda i: len(_pd_low_by_day[i]))
            _days_below = sum(1 for _l in _pd_low_by_day if len(_l) >= _pd_min)
            per_day_floors = {
                "days_evaluated": len(_pd_days),
                "low_ratio_threshold": _pd_ratio,
                "min_micros": _pd_min,
                "worst_day": {
                    "day_index": _worst_idx,
                    "low": _pd_low_by_day[_worst_idx],
                },
                "days_below": _days_below,
                "flagged": len(_pd_low_by_day[_worst_idx]) >= _pd_min,
            }
    except Exception:
        per_day_floors = None
    # [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-M1) Resumen worst-day de TECHOS — espejo compacto de
    # per_day_floors, que saltaba explícitamente todo target con 'ceiling': un día-pico de sodio
    # (6 días a 1500mg + 1 a 4400mg → promedio 1914 "ok") o de POTASIO en ERC (donde el riesgo de
    # hiperkalemia es AGUDO per-día y la conversión floor→ceiling lo sacaba del único chequeo
    # per-día existente) era invisible para todos los consumidores (todos leen el promedio).
    # SOLO observabilidad/banner — jamás retry ni trim (respeta la decisión G9 techo-no-duro y
    # P2-RENAL-POTASSIUM-DETERMINISTIC). El per-día de sodio hereda la subestimación por
    # sal-al-gusto (misma semántica 'estimado' del panel). Un solo micro excedido ya flaggea
    # (los excesos agudos no requieren co-ocurrencia, a diferencia de los pisos).
    per_day_ceilings = None
    try:
        import os as _os_pc
        _pc_ratio = min(3.0, max(1.0, float(_os_pc.environ.get("MEALFIT_MICRO_PERDAY_CEILING_RATIO", "1.5"))))
        _pc_days = totals.get("per_day") or []
        if len(_pc_days) >= 2 and coverage >= 0.6:
            _pc_high_by_day = []
            for _pc_vals in _pc_days:
                _highs = []
                for _pk, _ptgt in targets.items():
                    if "ceiling" not in _ptgt:
                        continue
                    if nutrient_coverage.get(_pk, coverage) < _CEILING_COVERAGE_FLOOR:
                        continue  # cobertura por-nutriente incierta → no acusar el día
                    try:
                        _ceil_v = float(_ptgt["ceiling"])
                        if _ceil_v > 0 and float(_pc_vals.get(_pk, 0.0)) > _ceil_v * _pc_ratio:
                            _highs.append(_pk)
                    except (TypeError, ValueError):
                        continue
                _pc_high_by_day.append(_highs)
            _worst_c_idx = max(range(len(_pc_high_by_day)), key=lambda i: len(_pc_high_by_day[i]))
            _days_above = sum(1 for _h in _pc_high_by_day if _h)
            per_day_ceilings = {
                "days_evaluated": len(_pc_days),
                "high_ratio_threshold": _pc_ratio,
                "worst_day": {
                    "day_index": _worst_c_idx,
                    "high": _pc_high_by_day[_worst_c_idx],
                },
                "days_above": _days_above,
                "flagged": bool(_pc_high_by_day[_worst_c_idx]),
            }
    except Exception:
        per_day_ceilings = None
    return {
        "panel": panel,
        "gaps": gaps,
        "coverage": coverage,
        "resolved_ings": totals["resolved_ings"],
        "total_ings": totals["total_ings"],
        "sex": "M" if str(sex or "").strip().lower() in _MALE_TERMS else "F",
        "condition_targets": condition_targets,
        "per_day_ceilings": per_day_ceilings,  # [P2-AUDIT-V5-BATCH GAP-M1]
        "per_day_floors": per_day_floors,  # [P1-MICRO-PERDAY-FLOOR]
        "disclaimer": ("Estimado desde el catálogo nutricional (cobertura "
                       f"{int(coverage*100)}%); NO incluye la sal añadida 'al gusto'. "
                       "Orientativo, no sustituye evaluación de un nutricionista."),
    }


# [P3-SUPPLEMENT-ADVICE · 2026-06-13] Plantillas de suplementación por micronutriente floor.
# Dosis de referencia para adulto sano (RDA/UL conservador); el `dose_fn` ajusta por sexo.
# Cierra honestamente el gap que una dieta de alimentos enteros rara vez alcanza (vit D, hierro
# en mujeres menstruantes, B12 en veganos): en vez de solo marcar "BAJO", da un plan accionable.
_SUPPLEMENT_TEMPLATES = {
    "vit_d_mcg": {
        "nombre": "Vitamina D3",
        "dosis": "600–800 UI/día (15–20 mcg)",
        "alimentos": "pescado graso (salmón/sardina enlatada 1–2x/sem), yema de huevo, lácteo fortificado, exposición solar 10–15 min",
        "precaucion": "no exceder 4000 UI/día sin control médico (UL).",
    },
    "calcium_mg": {
        "nombre": "Calcio (citrato o carbonato)",
        "dosis": "500 mg/día solo si no alcanzas con la dieta",
        "alimentos": "yogur/queso, sardina con espina, vegetales de hoja verde, sésamo/ajonjolí, tofu",
        "precaucion": "tómalo separado del hierro (compiten); no exceder 2500 mg/día totales.",
    },
    "iron_mg": {
        "nombre": "Hierro (bisglicinato, mejor tolerado)",
        "dosis_m": "8 mg/día solo si hay déficit confirmado",
        "dosis_f": "18 mg/día (especialmente si menstrúas)",
        # [P2-IRON-DOSE-AGE-AWARE · 2026-06-19] (audit fresco P2-7) post-menopausia (≥51): el RDA baja a 8 mg
        # y el exceso de hierro en mayores se asocia a riesgo CV → dosis menor + sin el copy de "si menstrúas".
        "dosis_f_post": "8 mg/día solo si hay déficit confirmado (post-menopausia)",
        # [P2-SUPPLEMENT-PREGNANCY-AWARE · 2026-06-19] (review P2) embarazo: RDA 27 mg — coherente con el piso
        # del panel (dri_targets pregnant→27); manda sobre el ajuste por sexo/edad. Bajo control prenatal.
        "dosis_f_preg": "27 mg/día (embarazo/lactancia) — bajo control prenatal/médico",
        "alimentos": "habichuelas/lentejas, carnes rojas magras, hígado, espinaca; acompaña con vit C (naranja/limón)",
        "precaucion": "separado de lácteos/café/té; confirma déficit con análisis (ferritina) antes de suplementar dosis altas.",
    },
    "b12_mcg": {
        "nombre": "Vitamina B12 (cianocobalamina)",
        "dosis": "2.4 mcg/día (o 250–500 mcg/sem si suplementas)",
        "alimentos": "huevo, lácteos, carne, pescado",
        "precaucion": "ESENCIAL si tu dieta es vegana/vegetariana estricta — no es opcional en ese caso.",
    },
    # [P1-FOOD-DB-EXTENDED-MICROS] suplementos del panel exhaustivo (solo los que aplica suplementar).
    "zinc_mg": {
        "nombre": "Zinc (citrato o gluconato)",
        "dosis": "8–11 mg/día solo si no alcanzas con la dieta",
        "alimentos": "carnes, mariscos, huevo, legumbres, semillas de calabaza/ajonjolí",
        "precaucion": "no exceder 40 mg/día (UL); el exceso compite con el cobre.",
    },
    "folate_mcg": {
        "nombre": "Ácido fólico / Folato",
        "dosis": "400 mcg/día (600 mcg en embarazo, bajo control prenatal)",
        "alimentos": "hoja verde, legumbres, aguacate, cítricos",
        "precaucion": "en embarazo es ESENCIAL (tubo neural); confírmalo con tu médico antes/durante la gestación.",
    },
    "vit_c_mg": {
        "nombre": "Vitamina C (ácido ascórbico)",
        "dosis": "75–90 mg/día solo si no alcanzas con fruta/vegetales",
        "alimentos": "naranja/limón, guayaba, pimiento/ají, brócoli",
        "precaucion": "no exceder 2000 mg/día (UL); dosis altas pueden causar molestias digestivas.",
    },
    "omega3_g": {
        "nombre": "Omega-3 (aceite de pescado EPA/DHA o de algas)",
        "dosis": "250–500 mg/día de EPA+DHA (o linaza/chía para ALA)",
        "alimentos": "pescado graso (sardina/salmón), linaza, chía, nueces",
        "precaucion": "si tomas anticoagulante, consulta a tu médico (puede afectar la coagulación).",
    },
    # [P1-SUPPLEMENT-COMPLETE · 2026-06-25] Magnesio/vit E/fibra: el panel exhaustivo los marca
    # "bajo" pero no tenían plantilla → la tarjeta de suplemento caía en silencio. Magnesio es un
    # suplemento seguro y común (glicinato); vit E y fibra van con encuadre "comida primero"
    # (forma suplementaria existe pero el alimento es preferible). Decisión del owner: cerrar los
    # gaps con SUPLEMENTACIÓN, no forzando alimentos en el menú.
    "magnesium_mg": {
        "nombre": "Magnesio (glicinato o citrato)",
        # [P1-SUPPLEMENT-UL-SAFE · 2026-06-26] (audit gap #4) El tope del rango era 400 mg, que EXCEDE su
        # propio UL suplementario de 350 mg declarado abajo → la tarjeta se auto-contradecía. Alineado a 350.
        "dosis": "200–350 mg/día solo si no alcanzas con la dieta (glicinato = mejor tolerado)",
        "ul": 350,  # [P1-SUPPLEMENT-UL-SAFE] UL suplementario (mg). Test de regresión ancla dosis_max ≤ ul.
        "alimentos": "hoja verde, legumbres, nueces/semillas (calabaza/ajonjolí), avena, cacao, aguacate",
        "precaucion": "no exceder 350 mg/día en forma suplementaria (UL); el exceso es laxante. Precaución en enfermedad renal.",
    },
    "vit_e_mg": {
        "nombre": "Vitamina E (d-alfa-tocoferol)",
        "dosis": "comida primero; ~15 mg/día solo si no alcanzas (evita dosis altas de rutina)",
        "alimentos": "almendras/avellanas, semillas de girasol, aceites vegetales, aguacate, hoja verde",
        "precaucion": "no exceder 1000 mg/día (UL); dosis altas + anticoagulante aumentan riesgo de sangrado.",
    },
    "fiber_g": {
        "nombre": "Fibra (psyllium / cáscara de plántago)",
        "dosis": "comida primero; 5–10 g/día de psyllium si no alcanzas (sube gradual + agua)",
        "alimentos": "legumbres, avena, frutas con cáscara, vegetales, granos integrales, linaza/chía",
        "precaucion": "aumenta gradual y con suficiente agua; sepáralo de medicamentos (puede reducir su absorción).",
    },
}


# [P1-SUPPLEMENT-CONDITION-AWARE · 2026-06-26] (audit gap #4) Suplementos a SUPRIMIR ante una condición
# médica donde la forma suplementaria es un riesgo activo (no solo "precaución"). Magnesio en ERC →
# hipermagnesemia (el riñón insuficiente no excreta el exceso → riesgo cardíaco/neuromuscular): NO se
# sugiere dosis. El paciente igual ve el micro "bajo" en el panel (informativo), pero sin tarjeta de dosis.
# Mismo principio fail-secure que el potasio (que nunca tiene plantilla). El balance fino lo define su
# nefrólogo. Si añades un suplemento riesgoso en otra condición, agrégalo aquí con su razón.
_SUPPLEMENT_RENAL_SUPPRESS = frozenset({"magnesium_mg"})


def build_supplement_recommendations(report: dict, sex: str | None = "F", age: int | None = None,
                                     pregnant: bool = False, conditions=None) -> dict:
    """[P3-SUPPLEMENT-ADVICE · 2026-06-13] A partir de los gaps FLOOR del reporte de
    micronutrientes (vit D/calcio/hierro/B12 bajo), construye recomendaciones de
    suplementación ACCIONABLES (suplemento + dosis sex-aware + alternativa alimentaria +
    precaución). Cierra de forma honesta lo que los alimentos enteros rara vez alcanzan.
    NO prescribe: incluye disclaimer profesional. Retorna {items, disclaimer, count}.

    [P1-SUPPLEMENT-CONDITION-AWARE · 2026-06-26] `conditions` (lista de strings del perfil) suprime
    suplementos riesgosos para una condición declarada — hoy: magnesio en ERC (hipermagnesemia)."""
    male = str(sex or "").strip().lower() in _MALE_TERMS
    renal = _has_renal(conditions) if conditions else False
    items = []
    for g in (report.get("gaps") or []):
        key = g.get("key")
        tpl = _SUPPLEMENT_TEMPLATES.get(key)
        if not tpl or g.get("status") not in ("bajo", "estimado_bajo"):
            continue  # solo floors realmente bajos; ceilings (sodio/azúcar) no son suplemento
        if renal and key in _SUPPLEMENT_RENAL_SUPPRESS:
            # [P1-SUPPLEMENT-CONDITION-AWARE] fail-secure: en ERC NO sugerimos dosis de magnesio.
            continue
        dose = tpl.get("dosis")
        if key == "iron_mg":
            # [P2-IRON-DOSE-AGE-AWARE · 2026-06-19] (P2-7) age-aware como el piso DRI: mujer ≥51 → dosis
            # post-menopáusica (8 mg), no la de menstruante (18 mg). Cierra la incoherencia floor-age-aware
            # vs dose-sex-only. `age` puede faltar → cae a la dosis de menstruante (conservador).
            if pregnant:
                # [P2-SUPPLEMENT-PREGNANCY-AWARE · 2026-06-19] embarazo manda sobre sexo/edad (RDA 27 mg),
                # coherente con el piso 27 del panel — evita la tarjeta contradictoria (panel 27 / dosis 18).
                dose = tpl["dosis_f_preg"]
            elif male:
                dose = tpl["dosis_m"]
            else:
                try:
                    _a = int(age) if age is not None else None
                except (TypeError, ValueError):
                    _a = None
                dose = tpl["dosis_f_post"] if (_a is not None and _a >= 51) else tpl["dosis_f"]
        items.append({
            "nutriente": g.get("nutriente"),
            "key": key,
            "actual": g.get("valor"),
            "objetivo": g.get("piso"),
            "unidad": g.get("unidad"),
            "suplemento": tpl["nombre"],
            "dosis_sugerida": dose,
            "primero_alimentos": tpl["alimentos"],
            "precaucion": tpl["precaucion"],
        })
    return {
        "items": items,
        "count": len(items),
        # [P3-MICRO-SUGGEST-CLARITY · 2026-06-19] Disclaimer más corto y claro
        # (antes era un párrafo denso). Conserva los 2 puntos clave: comida primero
        # + consultar al médico, e indica que la dosis depende del caso/análisis.
        "disclaimer": ("Orientativo, no es una prescripción. Cubre los gaps con comida "
                       "primero y consulta a tu médico antes de tomar suplementos "
                       "(la dosis depende de tu análisis y tu caso)."),
    }


def build_micronutrient_targets_directive(sex: str | None = "female", age: int | None = None,
                                          conditions=None, daily_kcal: float | None = None,
                                          pregnant: bool = False,
                                          k_elevating_med: bool = False,
                                          goal: str | None = None) -> str:
    """[P1-MICRONUTRIENT-STEER · 2026-06-24] Directiva CUANTITATIVA de micronutrientes para el
    prompt del day-generator. Convierte la guía HEURÍSTICA histórica ("usa legumbres para fibra/
    hierro") en PISOS NUMÉRICOS accionables para los micros ALCANZABLES con alimentos enteros
    (magnesio, calcio, hierro, fibra, potasio). Es el lado "steer" del par steer/gate: GUÍA al
    generador hacia densidad nutricional, NO es un gate (el `build_micronutrient_report` advisory
    sigue siendo la fuente de verdad post-hoc — este módulo NUNCA rechaza un plan, ver docstring).

    Diseño deliberado:
    - Usa los MISMOS pisos `dri_targets(sex, age, pregnant)` + las MISMAS elevaciones por condición
      (DM2 fibra ADA, DASH Mg/K) que el reporte → la guía y el panel quedan COHERENTES.
    - EXCLUYE la vitamina D: rara vez se alcanza con comida entera → se cubre con suplemento
      (`build_supplement_recommendations`); forzarla distorsionaría el plan (anti-patrón del módulo).
    - EXCLUYE los TECHOS (sodio/azúcar): "come menos sal" es otra instrucción, ya cubierta aparte.
    - Respeta `k_elevating_med`: con un fármaco que eleva el potasio sérico NO empuja a maximizarlo
      (mismo guard de seguridad que el panel P1-POTASSIUM-PANEL-MED-AWARE → evita nudge a hiperkalemia).
    - [P1-MICRONUTRIENT-STEER-PROTEIN-AWARE · 2026-06-25] Respeta `goal`: en `gain_muscle` el piso de
      proteína es alto y MANDA. La directiva antepone una línea de PRIORIDAD (proteína animal de alta
      densidad en cada comida principal) y reordena Mg/Fe para que la leguminosa sea GUARNICIÓN, no
      plato-base — evita el nudge a déficit de proteína que disparaba retries (observado: 3 intentos en vivo).
    - Fail-safe: cualquier error → "" (prompt sin la sección, jamás rompe la generación)."""
    try:
        is_muscle = str(goal or "").strip().lower() in ("gain_muscle", "muscle_gain", "ganar_musculo", "ganar musculo")
        targets = dri_targets(sex, age, pregnant=pregnant)
        fiber_floor = targets["fiber_g"]["floor"]
        mg_floor = targets["magnesium_mg"]["floor"]
        k_floor = targets["potassium_mg"]["floor"]
        ca_floor = targets["calcium_mg"]["floor"]
        fe_floor = targets["iron_mg"]["floor"]
        # Elevaciones por condición — espejo EXACTO de build_micronutrient_report (coherencia guía↔panel).
        if _has_diabetes(conditions) and daily_kcal and daily_kcal > 0:
            fiber_floor = max(fiber_floor, round(_DM2_FIBER_PER_1000KCAL * (daily_kcal / 1000.0), 1))
        if _has_hta(conditions) and not _has_renal(conditions):
            mg_floor = max(mg_floor, 500.0)
            if not k_elevating_med:
                k_floor = max(k_floor, 4700.0)
        lines = ["--- OBJETIVOS DE MICRONUTRIENTES (densidad nutricional del día) ---"]
        if is_muscle:
            lines.append(
                "PRIORIDAD: el piso de proteína y una fuente animal de ALTA densidad (pollo, pescado, "
                "cerdo, res, huevos, queso) en CADA comida principal (almuerzo y cena) MANDAN. Logra los "
                "micros de abajo con GUARNICIONES, vegetales, semillas y nueces como COMPLEMENTO — NUNCA "
                "reemplaces la proteína principal por leguminosas o almidón para subir un micronutriente."
            )
        lines.append(
            "Además de las calorías y los macros, busca que el día APUNTE de forma NATURAL a estos "
            "pisos diarios. NO fuerces un solo alimento ni distorsiones las porciones: intégralos en "
            "comidas dominicanas variadas y sabrosas."
        )
        # Magnesio y Hierro: en gain_muscle se reordenan para NO empujar la leguminosa como plato-base
        # (compite con el piso de proteína); en el resto de objetivos la leguminosa lidera (fuente barata
        # y eficiente de ambos micros + fibra).
        if is_muscle:
            lines.append(f"• Magnesio ≥{int(round(mg_floor))} mg → nueces/semillas (linaza, maní, ajonjolí), "
                         "vegetales de hoja verde, avena y granos integrales (leguminosas solo como guarnición).")
            lines.append(f"• Hierro ≥{fe_floor:g} mg → carnes rojas magras y huevo (hierro hemo); acompaña con "
                         "vitamina C (naranja/limón) en la misma comida para absorber mejor.")
        else:
            lines.append(f"• Magnesio ≥{int(round(mg_floor))} mg → vegetales de hoja verde, legumbres (habichuelas), "
                         "nueces/semillas (linaza, maní, ajonjolí), avena y granos integrales.")
            lines.append(f"• Hierro ≥{fe_floor:g} mg → legumbres, carnes rojas magras, huevo; acompaña con vitamina C "
                         "(naranja/limón) en la misma comida para mejorar la absorción.")
        lines.append(f"• Calcio ≥{int(round(ca_floor))} mg → lácteos (yogur, queso), ajonjolí/sésamo, hoja verde, "
                     "sardina con espina.")
        lines.append(f"• Fibra ≥{int(round(fiber_floor))} g → vegetales, frutas con cáscara, legumbres y granos integrales.")
        # [P1-FOOD-DB-EXTENDED-MICROS · 2026-06-25] micros nuevos GANABLES con comida y sin conflicto de
        # medicación/UL (zinc, folato, vit C). El resto del panel exhaustivo (vit A/E/K/selenio/omega-3)
        # queda informativo en el medidor, NO se mete al steering (evita sobre-restringir + el choque vit K↔warfarina).
        try:
            lines.append(f"• Zinc ≥{targets['zinc_mg']['floor']:g} mg → carnes, mariscos, huevo, legumbres, semillas de calabaza/ajonjolí.")
            lines.append(f"• Folato ≥{int(round(targets['folate_mcg']['floor']))} mcg → hoja verde, legumbres, aguacate, cítricos.")
            lines.append(f"• Vitamina C ≥{int(round(targets['vit_c_mg']['floor']))} mg → cítricos, guayaba, pimiento/ají, brócoli (mejora la absorción del hierro).")
            # [P1-MICRO-STEER-OMEGA3-VITE · 2026-06-25] Vit E y Omega-3 también son alcanzables con comida
            # (nueces/semillas/pescado) y eran 2 de los bajos típicos → al steering para empujarlos a meta.
            lines.append(f"• Vitamina E ≥{targets['vit_e_mg']['floor']:g} mg → nueces/almendras, semillas de girasol, aguacate y aceites vegetales.")
            lines.append(f"• Omega-3 ≥{targets['omega3_g']['floor']:g} g → linaza/chía, pescado graso (sardina/salmón) 1-2x/sem, nueces.")
        except Exception:
            pass
        if k_elevating_med:
            lines.append("• Potasio: mantén porciones MODERADAS y parejas (NO lo maximices) — el perfil "
                         "toma un fármaco que eleva el potasio sérico (riesgo de hiperkalemia).")
        else:
            lines.append(f"• Potasio ≥{int(round(k_floor))} mg → guineo, plátano, batata, aguacate, "
                         "espinaca, legumbres y naranja.")
        lines.append("La vitamina D casi nunca se alcanza solo con alimentos: NO la fuerces (se cubre "
                     "con un consejo de suplemento aparte).")
        return "\n".join(lines)
    except Exception:
        return ""
