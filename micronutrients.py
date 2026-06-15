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
    from knobs import _env_bool as _mn_env_bool
    _ANEMIA_CONDITION_TARGET_ENABLED = _mn_env_bool("MEALFIT_ANEMIA_CONDITION_TARGET", False)
except Exception:  # pragma: no cover - knobs siempre disponible en prod
    _ANEMIA_CONDITION_TARGET_ENABLED = False

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


def dri_targets(sex: str | None = "F", age: int | None = None) -> dict:
    """Pisos/techos DRI (IOM) + WHO por nutriente para un adulto. Sex-aware donde importa
    (hierro 18 vs 8 mg; fibra 25 vs 38 g; potasio 2600 vs 3400 mg). Default conservador
    femenino (hierro alto) cuando el sexo es desconocido.

    [P2-DRI-AGE-AWARE · 2026-06-15] (gap-audit G15) Age-aware en los dos micros donde el DRI real cambia
    con la edad y eso afecta la suplementación advisory: el HIERRO de la mujer baja de 18 mg (19-50) a 8 mg
    post-menopausia (51+) — antes 18 mg para todas sobre-flageaba déficit en mayores; el CALCIO sube de
    1000 a 1200 mg para mujeres 51+ y hombres 71+ — antes 1000 fijo sub-reforzaba en mayores. `age=None`
    (desconocida) → valores de adulto joven (conservadores: hierro alto, calcio base)."""
    male = str(sex or "").strip().lower() in _MALE_TERMS
    try:
        _age = int(age) if age is not None else None
    except (TypeError, ValueError):
        _age = None
    # Hierro (IOM/DRI): hombre 8 mg; mujer 18 mg (19-50) → 8 mg (51+, post-menopausia).
    _iron = 8.0 if male else (8.0 if (_age is not None and _age >= 51) else 18.0)
    # Calcio (IOM/DRI): 1200 mg para mujer 51+ / hombre 71+; 1000 mg resto.
    _calcium = 1200.0 if (_age is not None and ((not male and _age >= 51) or (male and _age >= 71))) else 1000.0
    return {
        "fiber_g":       {"floor": 38.0 if male else 25.0, "unit": "g"},
        "sodium_mg":     {"ceiling": 2000.0, "unit": "mg"},          # WHO <2000
        "free_sugars_g": {"ceiling": 25.0, "unit": "g"},             # WHO condicional <5% E
        "vit_d_mcg":     {"floor": 15.0, "unit": "mcg"},             # DRI 600 UI
        "calcium_mg":    {"floor": _calcium, "unit": "mg"},
        "iron_mg":       {"floor": _iron, "unit": "mg"},
        "b12_mcg":       {"floor": 2.4, "unit": "mcg"},
        "potassium_mg":  {"floor": 3400.0 if male else 2600.0, "unit": "mg"},
        "magnesium_mg":  {"floor": 420.0 if male else 320.0, "unit": "mg"},   # [P4] DRI IOM
    }


_LABELS = {
    "fiber_g": "Fibra", "sodium_mg": "Sodio", "free_sugars_g": "Azúcares añadidos",
    "vit_d_mcg": "Vitamina D", "calcium_mg": "Calcio", "iron_mg": "Hierro",
    "b12_mcg": "Vitamina B12", "potassium_mg": "Potasio",
    "magnesium_mg": "Magnesio", "saturated_fat_g": "Grasa saturada",
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
}

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


def compute_plan_micronutrient_totals(plan: dict, db) -> dict:
    """Suma los micros de todos los ingredientes resueltos del plan y devuelve el PROMEDIO
    diario + metadata de cobertura. `free_sugars_g` solo cuenta el azúcar de ingredientes
    de azúcar AÑADIDA (miel/sirope/glaseado), no el intrínseco de fruta/leche."""
    days = plan.get("days") or []
    num_days = max(1, len(days))
    acc = {k: 0.0 for k in ("fiber_g", "sodium_mg", "free_sugars_g", "vit_d_mcg",
                            "calcium_mg", "iron_mg", "b12_mcg", "potassium_mg",
                            "magnesium_mg", "saturated_fat_g")}  # [P4] DASH Mg + dislipidemia satfat
    # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (gap-audit G5) Cobertura POR-NUTRIENTE: de los ingredientes
    # RESUELTOS, cuántos traían dato NO-NULL de cada micro. `coverage` (global) NO basta para los TECHOS:
    # un ingrediente puede resolver (tiene macros) pero traer la columna del micro NULL (p.ej. embutidos DD
    # con saturated_fat_g sin poblar) → suma 0.0 → el techo reportaría 'ok' falso para un dislipidémico. La
    # cobertura por-nutriente distingue "0 real" de "0 por NULL" → permite reportar 'estimado_alto' (incierto).
    _SRC_KEY = {"fiber_g": "fiber", "sodium_mg": "sodium_mg", "vit_d_mcg": "vit_d_mcg",
                "calcium_mg": "calcium_mg", "iron_mg": "iron_mg", "b12_mcg": "b12_mcg",
                "potassium_mg": "potassium_mg", "magnesium_mg": "magnesium_mg",
                "saturated_fat_g": "saturated_fat_g"}
    present = {k: 0 for k in _SRC_KEY}
    total_ings = resolved_ings = 0
    for day in days:
        for meal in day.get("meals", []) or []:
            for ing in meal.get("ingredients", []) or []:
                total_ings += 1
                m = db.micros_from_ingredient_string(str(ing))
                if not m:
                    continue
                resolved_ings += 1
                acc["fiber_g"] += m.get("fiber") or 0.0
                acc["sodium_mg"] += m.get("sodium_mg") or 0.0
                acc["vit_d_mcg"] += m.get("vit_d_mcg") or 0.0
                acc["calcium_mg"] += m.get("calcium_mg") or 0.0
                acc["iron_mg"] += m.get("iron_mg") or 0.0
                acc["b12_mcg"] += m.get("b12_mcg") or 0.0
                acc["potassium_mg"] += m.get("potassium_mg") or 0.0
                acc["magnesium_mg"] += m.get("magnesium_mg") or 0.0           # [P4-UNIFIED-RESOLVER]
                acc["saturated_fat_g"] += m.get("saturated_fat_g") or 0.0     # [P4-UNIFIED-RESOLVER]
                # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] cuenta presencia NO-NULL por micro (G5).
                for _ak, _sk in _SRC_KEY.items():
                    if m.get(_sk) is not None:
                        present[_ak] += 1
                ing_low = str(ing).lower()
                if any(t in ing_low for t in _ADDED_SUGAR_TERMS):
                    acc["free_sugars_g"] += m.get("sugars_g") or 0.0
    daily = {k: round(v / num_days, 1) for k, v in acc.items()}
    coverage = round(resolved_ings / total_ings, 2) if total_ings else 0.0
    # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (G5) fracción de resueltos con dato NO-NULL por micro.
    nutrient_coverage = {k: (round(present[k] / resolved_ings, 2) if resolved_ings else 0.0)
                         for k in present}
    return {"daily": daily, "coverage": coverage, "nutrient_coverage": nutrient_coverage,
            "resolved_ings": resolved_ings, "total_ings": total_ings, "num_days": num_days}


def build_micronutrient_report(plan: dict, db, sex: str | None = "F",
                               conditions=None, daily_kcal: float | None = None,
                               fiber_per_1000kcal: float = _DM2_FIBER_PER_1000KCAL,
                               age: int | None = None) -> dict:
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
    targets = dri_targets(sex, age)   # [P2-DRI-AGE-AWARE · 2026-06-15] (G15) hierro/calcio age-aware
    condition_targets = []
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
    if _has_hta(conditions):
        if 4700.0 > targets["potassium_mg"]["floor"]:
            targets["potassium_mg"]["floor"] = 4700.0
        if 500.0 > targets["magnesium_mg"]["floor"]:
            targets["magnesium_mg"]["floor"] = 500.0
        condition_targets.append({
            "condicion": "Hipertensión (patrón DASH)",
            "regla": "Potasio ≥4700 mg/día + Magnesio ≥500 mg/día + Sodio <2000 mg/día",
            "guia": "DASH (NHLBI/AHA-ACC) — el balance Na/K/Mg/Ca baja la presión arterial",
            "actual": {"potasio": daily.get("potassium_mg", 0.0), "magnesio": daily.get("magnesium_mg", 0.0),
                       "sodio": daily.get("sodium_mg", 0.0)},
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
            if val >= floor:
                status = "ok"
            elif coverage < 0.6:
                status = "estimado_bajo"  # cobertura parcial → incierto
            else:
                status = "bajo"
            entry = {"nutriente": _LABELS[key], "key": key, "valor": val, "unidad": unit,
                     "piso": floor, "status": status}
            if status in ("bajo", "estimado_bajo"):
                entry["nota"] = _SUPPLEMENT_NOTE.get(key, "")
                gaps.append(entry)
        panel.append(entry)
    return {
        "panel": panel,
        "gaps": gaps,
        "coverage": coverage,
        "resolved_ings": totals["resolved_ings"],
        "total_ings": totals["total_ings"],
        "sex": "M" if str(sex or "").strip().lower() in _MALE_TERMS else "F",
        "condition_targets": condition_targets,
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
        "alimentos": "habichuelas/lentejas, carnes rojas magras, hígado, espinaca; acompaña con vit C (naranja/limón)",
        "precaucion": "separado de lácteos/café/té; confirma déficit con análisis (ferritina) antes de suplementar dosis altas.",
    },
    "b12_mcg": {
        "nombre": "Vitamina B12 (cianocobalamina)",
        "dosis": "2.4 mcg/día (o 250–500 mcg/sem si suplementas)",
        "alimentos": "huevo, lácteos, carne, pescado",
        "precaucion": "ESENCIAL si tu dieta es vegana/vegetariana estricta — no es opcional en ese caso.",
    },
}


def build_supplement_recommendations(report: dict, sex: str | None = "F") -> dict:
    """[P3-SUPPLEMENT-ADVICE · 2026-06-13] A partir de los gaps FLOOR del reporte de
    micronutrientes (vit D/calcio/hierro/B12 bajo), construye recomendaciones de
    suplementación ACCIONABLES (suplemento + dosis sex-aware + alternativa alimentaria +
    precaución). Cierra de forma honesta lo que los alimentos enteros rara vez alcanzan.
    NO prescribe: incluye disclaimer profesional. Retorna {items, disclaimer, count}."""
    male = str(sex or "").strip().lower() in _MALE_TERMS
    items = []
    for g in (report.get("gaps") or []):
        key = g.get("key")
        tpl = _SUPPLEMENT_TEMPLATES.get(key)
        if not tpl or g.get("status") not in ("bajo", "estimado_bajo"):
            continue  # solo floors realmente bajos; ceilings (sodio/azúcar) no son suplemento
        dose = tpl.get("dosis")
        if key == "iron_mg":
            dose = tpl["dosis_m"] if male else tpl["dosis_f"]
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
        "disclaimer": ("Recomendación orientativa, NO una prescripción. Prioriza cerrar los "
                       "gaps con ALIMENTOS primero; consulta a tu médico/nutricionista antes "
                       "de iniciar cualquier suplemento (dosis y necesidad dependen de tu "
                       "análisis de sangre y condiciones individuales)."),
    }
