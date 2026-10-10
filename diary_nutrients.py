"""Versioned food diary references. RDA/AI are not toxicity thresholds.

Adult references: NIH ODS fact sheets, reviewed 2026-10-10. Unknown age/sex
does not silently select a reference; pediatric targets are outside this app.
"""
import math
import unicodedata

CATALOG_VERSION = "diary-micros-2026-10-10-v1"
NUTRIENTS = (
    ("fiber_g", "fiber", "g", "DietaryFiber"),
    ("sodium_mg", "sodium_mg", "mg", "Sodium"),
    ("potassium_mg", "potassium_mg", "mg", "Potassium"),
    ("calcium_mg", "calcium_mg", "mg", "Calcium"),
    ("iron_mg", "iron_mg", "mg", "Iron"),
    ("vit_c_mg", "vit_c_mg", "mg", "VitaminC"),
    ("vit_a_mcg", "vit_a_mcg", "mcg RAE", "VitaminA"),
    ("vit_d_mcg", "vit_d_mcg", "mcg", "VitaminD"),
    ("magnesium_mg", "magnesium_mg", "mg", "Magnesium"),
    ("zinc_mg", "zinc_mg", "mg", "Zinc"),
    ("b12_mcg", "b12_mcg", "mcg", "VitaminB12"),
    ("folate_mcg", "folate_mcg", "mcg DFE", "Folate"),
    ("vit_e_mg", "vit_e_mg", "mg", "VitaminE"),
    ("vit_k_mcg", "vit_k_mcg", "mcg", "VitaminK"),
    ("selenium_mcg", "selenium_mcg", "mcg", "Selenium"),
    ("vit_b6_mg", "vit_b6_mg", "mg", "VitaminB6"),
    ("iodine_mcg", "iodine_mcg", "mcg", "Iodine"),
)


def valid_amount(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def life_stage(profile):
    def norm(s):
        return "".join(c for c in unicodedata.normalize("NFD", str(s).lower()) if not unicodedata.combining(c))
    def yes(v):
        return v is True or norm(v).strip() in ("true", "1", "si", "yes", "on")
    raw = profile.get("medicalConditions") or profile.get("medical_conditions") or []
    terms = " ".join(norm(v) for v in (raw if isinstance(raw, list) else [raw]))
    lactating = any(yes(profile.get(k)) for k in ("isLactating", "lactating", "breastfeeding")) or any(k in terms for k in ("lactan", "lactat", "breastfeed", "nursing", "amamant", "amament", "allatt", "allait"))
    pregnant = any(yes(profile.get(k)) for k in ("isPregnant", "pregnant", "embarazada")) or any(k in terms for k in ("embaraz", "pregnan", "gestac", "gravidez", "gravidanza", "grossesse"))
    return pregnant, lactating


def references(sex, age, pregnant=False, lactating=False):
    from micronutrients import dri_targets
    gender = str(sex or "").strip().lower()
    if gender not in ("male", "masculino", "masculina", "hombre", "m", "female", "femenino", "femenina", "mujer", "f"):
        return {}
    try:
        age = float(age)
        if not math.isfinite(age) or age < 19 or age > 120:
            return {}
    except (ValueError, TypeError):
        return {}
    male = gender in ("male", "masculino", "masculina", "hombre", "m")
    # Pregnancy takes precedence when both flags are present.
    lactating = bool(lactating and not pregnant)
    base = dri_targets("male" if male else "female", age, pregnant)
    overrides = {
        "fiber_g": 30 if male and age >= 51 else 21 if not male and age >= 51 else 38 if male else 25,
        "magnesium_mg": (350 if age < 31 else 360) if pregnant else ((400 if age < 31 else 420) if male else (310 if age < 31 else 320)),
        "vit_b6_mg": 1.9 if pregnant else 2.0 if lactating else 1.3 if age < 51 else 1.7 if male else 1.5,
        "iodine_mcg": 220 if pregnant else 290 if lactating else 150,
    }
    if pregnant:
        overrides.update(fiber_g=28, potassium_mg=2900, vit_k_mcg=90)
    if lactating:
        overrides.update(fiber_g=29, iron_mg=9, b12_mcg=2.8, zinc_mg=12,
                         folate_mcg=500, vit_a_mcg=1300, vit_c_mg=120,
                         vit_e_mg=19, selenium_mcg=70, potassium_mg=2800, vit_k_mcg=90)
    result = {}
    for key, _, unit, sheet in NUTRIENTS:
        d = base.get(key, {})
        target = overrides.get(key, d.get("floor", d.get("ceiling")))
        if target is None:
            continue
        source = ("https://www.who.int/news-room/fact-sheets/detail/salt-reduction" if key == "sodium_mg" else
                  "https://nap.nationalacademies.org/read/10490/chapter/1" if key == "fiber_g" else
                  f"https://ods.od.nih.gov/factsheets/{sheet}-HealthProfessional/")
        result[key] = {"target": float(target), "kind": "ceiling" if key == "sodium_mg" else "floor",
                       "unit": unit, "source": source, "reference_type": "WHO" if key == "sodium_mg" else "AI" if key in ("fiber_g", "potassium_mg", "vit_k_mcg") else "RDA",
                       "reviewed_at": "2026-10-10", "scope": "food", "upper_limit": None}
    return result
