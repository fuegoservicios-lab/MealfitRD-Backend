"""[P1-PANTRY-SUFFICIENCY · 2026-06-23] Evaluador de suficiencia de la Nevera.

Responde la pregunta del requisito del owner para los botones "actualizar platos":
¿la Nevera (`user_inventory`) tiene suficiente comida — y suficientes MACROS/MICROS
para el OBJETIVO del usuario — para generar un plato (o un día) nuevo? Si NO, los
endpoints de actualizar platos NO deben llamar al LLM y deben avisar "agrega más
ítems a tu Nevera".

Diseño:
  - Suma macros/micros DISPONIBLES desde `user_inventory` (qty disponible × master
    macros/100g) y los compara contra el target del usuario escalado al scope
    (1 comida vs día completo).
  - CONSERVADOR: ítems sin master row / sin `kcal_per_100g` / no convertibles a gramos
    NO se cuentan (`uncountable_items`) — nunca inventa disponibilidad (fail-secure
    hacia "insuficiente", nunca hacia "alcanza de más").
  - PALANCA = PROTEÍNA (coherente con el piso de proteína goal-aware del generador) +
    un piso de kcal. MICROS = advisory por default (las columnas de micros son
    nullable → bloquear arriesga falsos negativos; se SURFACEAN como déficits pero no
    bloquean salvo que se active el gate de micros).
  - Fail-OPEN ante error interno: si el evaluador falla, retorna `sufficient=True`
    (no bloquea al usuario por un bug del evaluador; el soft-fail reactivo de
    `swap_meal` sigue cubriendo la nevera vacía dura).

El GATE (bloquear o no) lo decide el ENDPOINT vía knob `MEALFIT_PANTRY_SUFFICIENCY_GATE`;
este módulo solo COMPUTA. Reutiliza los motores existentes (db_inventory, nutrition_db,
nutrition_calculator, micronutrients); sin DDL, sin callsites por defecto.
"""
import os
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Knobs (defaults seguros; el gate per-se lo controla el endpoint)
# --------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Cobertura mínima por macro crítico para considerar la nevera "suficiente".
_PROTEIN_RATIO = _env_float("MEALFIT_PANTRY_SUFFICIENCY_PROTEIN_RATIO", 0.90)
_KCAL_RATIO = _env_float("MEALFIT_PANTRY_SUFFICIENCY_KCAL_RATIO", 0.80)
# Micros: advisory por default (no bloquea). Si se activa, usa _MICROS_RATIO.
_MICROS_GATE = _env_bool("MEALFIT_PANTRY_SUFFICIENCY_MICROS_GATE", False)
_MICROS_RATIO = _env_float("MEALFIT_PANTRY_SUFFICIENCY_MICROS_RATIO", 0.50)
# Fracción del día que representa UNA comida cuando el caller no pasa el target del slot.
_DEFAULT_MEAL_FRACTION = _env_float("MEALFIT_PANTRY_SUFFICIENCY_MEAL_FRACTION", 0.30)

# Micros (DRI floors) que el evaluador surfacea. Sodio/azúcar son TECHOS → no son
# "déficit por faltante", se excluyen del gate de suficiencia.
_MICRO_FLOOR_KEYS = (
    "fiber_g", "iron_mg", "calcium_mg", "potassium_mg", "magnesium_mg", "vit_d_mcg", "b12_mcg",
)
# dri_targets usa otras keys → mapa floor_key(panel) → dri_key.
_DRI_KEY = {
    "fiber_g": "fiber_g", "iron_mg": "iron_mg", "calcium_mg": "calcium_mg",
    "potassium_mg": "potassium_mg", "magnesium_mg": "magnesium_mg",
    "vit_d_mcg": "vit_d_mcg", "b12_mcg": "b12_mcg",
}

_SUGGESTIONS = {
    "protein_g": "Agrega proteína a tu Nevera: pollo, huevos, pescado, carne o habichuelas.",
    "kcal": "Tu Nevera está muy vacía — agrega más alimentos (granos, víveres, frutas, proteínas).",
    "carbs_g": "Agrega carbohidratos: arroz, pan, avena, plátano o víveres.",
    "fats_g": "Agrega grasas saludables: aguacate, aceite de oliva, nueces o maní.",
    "fiber_g": "Agrega fibra: vegetales, frutas, avena o habichuelas.",
    "iron_mg": "Agrega hierro: carne roja, hígado, lentejas o espinaca.",
    "calcium_mg": "Agrega calcio: leche, queso o yogurt.",
    "potassium_mg": "Agrega potasio: guineo, plátano, papa, ñame o aguacate.",
    "magnesium_mg": "Agrega magnesio: nueces, semillas, espinaca o habichuelas.",
    "vit_d_mcg": "Agrega vitamina D: huevos, pescado graso o lácteos fortificados.",
    "b12_mcg": "Agrega vitamina B12: carne, pescado, huevos o lácteos.",
}

# Etiqueta legible por nutriente (para el mensaje al usuario).
_LABEL = {
    "protein_g": "proteína", "kcal": "calorías", "carbs_g": "carbohidratos", "fats_g": "grasas",
    "fiber_g": "fibra", "iron_mg": "hierro", "calcium_mg": "calcio", "potassium_mg": "potasio",
    "magnesium_mg": "magnesio", "vit_d_mcg": "vitamina D", "b12_mcg": "vitamina B12",
}


def _empty_panel() -> dict:
    return {
        "kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0, "fiber_g": 0.0,
        "iron_mg": 0.0, "calcium_mg": 0.0, "potassium_mg": 0.0, "magnesium_mg": 0.0,
        "vit_d_mcg": 0.0, "b12_mcg": 0.0,
    }


def available_macros_from_inventory(rows: list, db) -> tuple[dict, list]:
    """Suma el aporte de macros/micros de las filas de `user_inventory`.

    `rows`: salida de `get_raw_user_inventory` (cada fila con `ingredient_name`,
    `available_quantity`/`quantity`, `unit`). `db`: IngredientNutritionDB.

    Retorna `(panel, uncountable_items)`. Ítems sin master / sin kcal / no
    convertibles a gramos → `uncountable_items`, NO contados (conservador)."""
    panel = _empty_panel()
    uncountable: list = []
    for row in rows or []:
        try:
            name = (row.get("ingredient_name") or "").strip()
            qty = row.get("available_quantity")
            if qty is None:
                qty = row.get("quantity") or 0
            qty = float(qty or 0)
            unit = row.get("unit") or ""
            if not name or qty <= 0:
                continue
            info = db.lookup(name)
            if info is None:  # sin master row o kcal_per_100g NULL
                uncountable.append(name)
                continue
            grams = db.to_grams(qty, unit, info)
            if grams is None or grams <= 0:  # unidad discreta sin densidad / container sin peso
                uncountable.append(name)
                continue
            f = grams / 100.0
            panel["kcal"] += (info.kcal or 0.0) * f
            panel["protein_g"] += (info.protein or 0.0) * f
            panel["carbs_g"] += (info.carbs or 0.0) * f
            panel["fats_g"] += (info.fats or 0.0) * f
            panel["fiber_g"] += (info.fiber or 0.0) * f
            panel["iron_mg"] += (info.iron_mg or 0.0) * f
            panel["calcium_mg"] += (info.calcium_mg or 0.0) * f
            panel["potassium_mg"] += (info.potassium_mg or 0.0) * f
            panel["magnesium_mg"] += (info.magnesium_mg or 0.0) * f
            panel["vit_d_mcg"] += (info.vit_d_mcg or 0.0) * f
            panel["b12_mcg"] += (info.b12_mcg or 0.0) * f
        except Exception as _row_e:  # una fila mala no tumba el panel completo
            logger.debug(f"[PANTRY-SUFFICIENCY] fila inválida ignorada: {_row_e}")
            continue
    return panel, uncountable


def _daily_targets(form_data: dict) -> dict:
    """Macros diarios del objetivo del usuario → panel {kcal, protein_g, carbs_g, fats_g}."""
    from nutrition_calculator import get_nutrition_targets
    t = get_nutrition_targets(form_data) or {}
    macros = t.get("macros") or {}
    return {
        "kcal": float(t.get("target_calories") or 0),
        "protein_g": float(macros.get("protein_g") or 0),
        "carbs_g": float(macros.get("carbs_g") or 0),
        "fats_g": float(macros.get("fats_g") or 0),
    }


def _daily_micro_floors(form_data: dict) -> dict:
    """Floors DRI de micros para el sexo/edad/embarazo del usuario → {panel_key: floor}."""
    try:
        from micronutrients import dri_targets
        sex = form_data.get("gender") or form_data.get("sex")
        age = form_data.get("age")
        conds = form_data.get("conditions") or form_data.get("medicalConditions") or []
        if isinstance(conds, str):
            conds = [conds]
        pregnant = bool(form_data.get("pregnant")) or any(
            ("embaraz" in str(c).lower() or "lactan" in str(c).lower()) for c in conds
        )
        dri = dri_targets(sex=sex, age=age, pregnant=pregnant) or {}
    except Exception as _dri_e:
        logger.debug(f"[PANTRY-SUFFICIENCY] dri_targets falló: {_dri_e}")
        return {}
    out = {}
    for pk in _MICRO_FLOOR_KEYS:
        spec = dri.get(_DRI_KEY[pk]) or {}
        floor = spec.get("floor")
        if floor:
            out[pk] = float(floor)
    return out


def evaluate_pantry_sufficiency(
    user_id: str,
    form_data: dict,
    scope: str = "meal",
    *,
    meal_target: dict | None = None,
    nutrition_db=None,
    inventory_rows: list | None = None,
) -> dict:
    """¿La Nevera del usuario alcanza para generar un plato/día acorde al objetivo?

    Args:
      user_id: dueño de la Nevera.
      form_data: perfil (goal/gender/age/weight/...) — para targets de macro/micro.
      scope: "meal" (un plato) | "day" (día completo).
      meal_target: (scope="meal") macros del slot a cubrir {kcal, protein_g, carbs_g,
        fats_g} (típicamente las del plato original). Si None → daily × fracción default.
      nutrition_db: IngredientNutritionDB reusable (singleton); se crea si None.
      inventory_rows: override de filas de inventario (para tests); si None → DB.

    Returns dict con: sufficient(bool), available(panel), required(panel),
      coverage(ratio por nutriente crítico), deficits[] (accionables), uncountable_items[],
      message(str|None resumen para el usuario)."""
    try:
        if nutrition_db is None:
            from nutrition_db import IngredientNutritionDB
            nutrition_db = IngredientNutritionDB()

        if inventory_rows is None:
            from db import get_raw_user_inventory
            inventory_rows = get_raw_user_inventory(user_id)

        available, uncountable = available_macros_from_inventory(inventory_rows, nutrition_db)

        daily = _daily_targets(form_data)
        if scope == "day":
            frac = 1.0
            if meal_target:
                # El caller (regenerate-day) pasa el target REAL del día = suma de las
                # macros de los platos del plan (ya goal-correcto) → más fiable que
                # recomputar desde form_data potencialmente incompleto.
                required = {
                    "kcal": float(meal_target.get("kcal") or meal_target.get("cals") or 0),
                    "protein_g": float(meal_target.get("protein_g") or meal_target.get("protein") or 0),
                    "carbs_g": float(meal_target.get("carbs_g") or meal_target.get("carbs") or 0),
                    "fats_g": float(meal_target.get("fats_g") or meal_target.get("fats") or 0),
                }
                # Si el caller no pobló macros (todo 0) → caer al target diario calculado.
                if required["protein_g"] <= 0 and required["kcal"] <= 0:
                    required = dict(daily)
                # [P5-DAY-PROTEIN-FALLBACK · 2026-06-23] (audit inteligencia P2-11) Espejo del fallback de
                # scope='meal': si el caller pobló kcal pero NO la proteína (meals con `cals` sin key
                # `protein` — plan legacy/degradado/parcial), required[protein_g]=0 → `_check` la salta →
                # el gate degrada a SOLO-kcal y una Nevera de arroz+aceite (mucha kcal, poca proteína)
                # pasaría el día bajo el piso de proteína (la PALANCA, decisión D1 lever-only). frac=1.0
                # para day → target diario COMPLETO (NO escalar por fracción).
                elif required["protein_g"] <= 0 and required["kcal"] > 0 and daily.get("protein_g", 0) > 0:
                    required["protein_g"] = daily["protein_g"]
            else:
                required = dict(daily)
        else:  # "meal"
            if meal_target:
                required = {
                    "kcal": float(meal_target.get("kcal") or meal_target.get("cals") or 0),
                    "protein_g": float(meal_target.get("protein_g") or meal_target.get("protein") or 0),
                    "carbs_g": float(meal_target.get("carbs_g") or meal_target.get("carbs") or 0),
                    "fats_g": float(meal_target.get("fats_g") or meal_target.get("fats") or 0),
                }
                frac = (required["kcal"] / daily["kcal"]) if daily["kcal"] else _DEFAULT_MEAL_FRACTION
                # [P5-MEAL-PROTEIN-FALLBACK · 2026-06-23] La PALANCA del gate es la PROTEÍNA.
                # Si el caller pobló kcal pero NO la proteína del plato (frontend viejo
                # cacheado pre-P5, o un meal cuyo objeto guardó solo `cals`), required[protein]
                # quedaría en 0 → `_check` lo salta → el gate degrada a SOLO-kcal y una Nevera
                # de arroz+aceite (mucha kcal, poca proteína) pasaría. Evitamos esa pérdida
                # silenciosa de la palanca: caemos al target diario escalado por la fracción
                # calórica de este plato (espejo del fallback all-zero de scope='day').
                if required["protein_g"] <= 0 and required["kcal"] > 0 and daily.get("protein_g", 0) > 0:
                    required["protein_g"] = daily["protein_g"] * frac
            else:
                frac = _DEFAULT_MEAL_FRACTION
                required = {k: v * frac for k, v in daily.items()}

        # Micros requeridos (escalados al scope por la fracción del día).
        micro_floors = _daily_micro_floors(form_data)
        required_micros = {k: v * frac for k, v in micro_floors.items()}

        coverage: dict = {}
        deficits: list = []
        sufficient = True

        def _check(key: str, req: float, *, gate: bool, ratio_min: float, advisory: bool = False):
            nonlocal sufficient
            if not req or req <= 0:
                return
            have = available.get(key, 0.0)
            cov = have / req
            coverage[key] = round(cov, 3)
            if cov < ratio_min:
                deficits.append({
                    "nutrient": key,
                    "label": _LABEL.get(key, key),
                    "have": round(have, 1),
                    "need": round(req, 1),
                    "ratio": round(cov, 3),
                    "advisory": advisory,
                    "suggestion": _SUGGESTIONS.get(key, "Agrega más ítems a tu Nevera."),
                })
                if gate:
                    sufficient = False

        # Macros crítico (palanca): proteína + kcal bloquean.
        _check("protein_g", required["protein_g"], gate=True, ratio_min=_PROTEIN_RATIO)
        _check("kcal", required["kcal"], gate=True, ratio_min=_KCAL_RATIO)
        # Micros: advisory por default (gate solo si _MICROS_GATE).
        for mk, req in required_micros.items():
            _check(mk, req, gate=_MICROS_GATE, ratio_min=_MICROS_RATIO, advisory=not _MICROS_GATE)

        # Mensaje accionable (solo déficits que bloquean, o el peor advisory si no hay gate).
        message = None
        blocking = [d for d in deficits if not d["advisory"]]
        if blocking:
            faltan = ", ".join(d["label"] for d in blocking[:3])
            message = (
                f"Tu Nevera no alcanza para cubrir tu objetivo en este plato: falta {faltan}. "
                f"{blocking[0]['suggestion']}"
            )

        return {
            "sufficient": bool(sufficient),
            "available": {k: round(v, 1) for k, v in available.items()},
            "required": {k: round(v, 1) for k, v in required.items()},
            "coverage": coverage,
            "deficits": deficits,
            "uncountable_items": uncountable,
            "scope": scope,
            "message": message,
        }
    except Exception as _e:
        # Fail-OPEN: nunca bloquear por un bug del evaluador.
        logger.warning(f"[PANTRY-SUFFICIENCY] evaluador falló ({_e}) → fail-open (sufficient=True)")
        return {
            "sufficient": True, "available": {}, "required": {}, "coverage": {},
            "deficits": [], "uncountable_items": [], "scope": scope, "message": None,
            "error": str(_e),
        }
