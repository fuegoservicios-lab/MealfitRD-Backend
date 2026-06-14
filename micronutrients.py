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

# Términos de azúcar AÑADIDA (free sugars) — el techo WHO aplica a estos, NO al azúcar
# intrínseco de fruta/leche (que no es preocupación de salud).
_ADDED_SUGAR_TERMS = ("miel", "azucar", "azúcar", "sirope", "jarabe", "panela",
                      "melaza", "glasead", "honey", "sugar", "dulce de leche")


# Términos de sexo MASCULINO (mismo set que nutrition_calculator.calculate_bmr; ojo: "mujer"
# empieza con M pero NO es masculino → la lista evita ese falso positivo).
_MALE_TERMS = ("male", "masculino", "m", "hombre")


def dri_targets(sex: str | None = "F") -> dict:
    """Pisos/techos DRI (IOM) + WHO por nutriente para un adulto. Sex-aware donde importa
    (hierro 18 vs 8 mg; fibra 25 vs 38 g; potasio 2600 vs 3400 mg). Default conservador
    femenino (hierro alto) cuando el sexo es desconocido."""
    male = str(sex or "").strip().lower() in _MALE_TERMS
    return {
        "fiber_g":       {"floor": 38.0 if male else 25.0, "unit": "g"},
        "sodium_mg":     {"ceiling": 2000.0, "unit": "mg"},          # WHO <2000
        "free_sugars_g": {"ceiling": 25.0, "unit": "g"},             # WHO condicional <5% E
        "vit_d_mcg":     {"floor": 15.0, "unit": "mcg"},             # DRI 600 UI
        "calcium_mg":    {"floor": 1000.0, "unit": "mg"},
        "iron_mg":       {"floor": 8.0 if male else 18.0, "unit": "mg"},
        "b12_mcg":       {"floor": 2.4, "unit": "mcg"},
        "potassium_mg":  {"floor": 3400.0 if male else 2600.0, "unit": "mg"},
    }


_LABELS = {
    "fiber_g": "Fibra", "sodium_mg": "Sodio", "free_sugars_g": "Azúcares añadidos",
    "vit_d_mcg": "Vitamina D", "calcium_mg": "Calcio", "iron_mg": "Hierro",
    "b12_mcg": "Vitamina B12", "potassium_mg": "Potasio",
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
}


def compute_plan_micronutrient_totals(plan: dict, db) -> dict:
    """Suma los micros de todos los ingredientes resueltos del plan y devuelve el PROMEDIO
    diario + metadata de cobertura. `free_sugars_g` solo cuenta el azúcar de ingredientes
    de azúcar AÑADIDA (miel/sirope/glaseado), no el intrínseco de fruta/leche."""
    days = plan.get("days") or []
    num_days = max(1, len(days))
    acc = {k: 0.0 for k in ("fiber_g", "sodium_mg", "free_sugars_g", "vit_d_mcg",
                            "calcium_mg", "iron_mg", "b12_mcg", "potassium_mg")}
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
                ing_low = str(ing).lower()
                if any(t in ing_low for t in _ADDED_SUGAR_TERMS):
                    acc["free_sugars_g"] += m.get("sugars_g") or 0.0
    daily = {k: round(v / num_days, 1) for k, v in acc.items()}
    coverage = round(resolved_ings / total_ings, 2) if total_ings else 0.0
    return {"daily": daily, "coverage": coverage,
            "resolved_ings": resolved_ings, "total_ings": total_ings, "num_days": num_days}


def build_micronutrient_report(plan: dict, db, sex: str | None = "F") -> dict:
    """Reporte advisory: panel de micros diarios vs DRI/WHO con status + nota accionable.
    status ∈ {ok, bajo, alto, estimado_bajo}. Floors incumplidos con cobertura parcial →
    'estimado_bajo' (incierto, puede subir con lo no resuelto). NO rechaza el plan."""
    totals = compute_plan_micronutrient_totals(plan, db)
    daily = totals["daily"]
    coverage = totals["coverage"]
    targets = dri_targets(sex)
    panel, gaps = [], []
    for key, tgt in targets.items():
        val = daily.get(key, 0.0)
        unit = tgt["unit"]
        if "ceiling" in tgt:
            ceil = tgt["ceiling"]
            status = "alto" if val > ceil else "ok"
            entry = {"nutriente": _LABELS[key], "key": key, "valor": val, "unidad": unit,
                     "techo": ceil, "status": status}
            if status == "alto":
                entry["nota"] = _SUPPLEMENT_NOTE.get(key, "")
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
