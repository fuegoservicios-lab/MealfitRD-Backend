"""[P3-CONDITION-ENGINE · 2026-06-14] Motor de constraints clínicos DECLARATIVO.

Generaliza el patrón ad-hoc por-condición (que vivía disperso en ~7 callsites de graph_orchestrator
+ plan_generator) a un REGISTRO de datos: cada condición del set Pareto cardiometabólico DR es una
fila `ConditionRule` con (detección SSOT, bloque de prompt citable, sustituciones deterministas de
ingredientes, precedencia para comorbilidad, clasificación safety/derivación). Los consumidores
(prompt builder, guard de sustituciones, gate FS9) ITERAN el registro → añadir HTA/dislipidemia/
anemia es declarar una fila, no editar 5 sitios. Cierra el drift detector↔prompt (un solo SSOT) y la
explosión combinatoria de comorbilidades.

Estado del enforcement por condición (honesto):
- ERC: cap de proteína 0.8 g/kg + gate nefrólogo → enforced en graph (no migrado aquí aún) + referral.
- DM2: fibra ADA (advisory) + sustitución azúcar→stevia (enforced, via este motor).
- HTA: sustitución de sodio (embutidos/cubitos/bacalao→fresco) ENFORCED via este motor (NUEVO).
- Dislipidemia: prompt (satfat<7% requiere columna satfat que falta en master_ingredients → M3).
- Anemia: prompt + densidad de hierro (el panel ya computa iron_mg).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from constants import (
    RENAL_CONDITION_TERMS, DIABETES_CONDITION_TERMS, HTA_CONDITION_TERMS,
    DYSLIPIDEMIA_CONDITION_TERMS, ANEMIA_CONDITION_TERMS,
)

SAFETY_HARD = "safety_hard"            # reglas que el motor REESCRIBE determinísticamente
CLINICAL_REFERRAL = "clinical_referral"  # se derivan al profesional (no se auto-prescriben)

_MEDICAL_NONE_SENTINELS = frozenset({
    "", "ninguna", "ninguno", "ningunas", "ningunos", "none", "n/a", "na",
    "no", "nada", "sin alergias", "sin condiciones", "ninguna alergia",
})


@dataclass(frozen=True)
class ConditionRule:
    """Una condición del set Pareto como DATO. `substitutions` = ((tokens...), reemplazo, etiqueta,
    preserve_qty?); `preserve_qty` (4º elem, opcional, default False) marca staples que aportan peso
    (embutidos/bacalao) para los que el guard preserva el prefijo de cantidad y recalcula macros.
    `sub_negatives` = frases que vetan la sustitución (ej. 'baja en sodio'). `precedence` menor =
    mayor prioridad clínica de seguridad (ERC manda)."""
    id: str
    label: str
    terms: tuple
    prompt_block: str
    substitutions: tuple = ()
    sub_negatives: tuple = ()
    precedence: int = 50
    classification: str = SAFETY_HARD


# ── Tablas de sustitución determinista (patrón generalizado del sugar-guard) ──
# Cada fila: (tokens, reemplazo, etiqueta, preserve_qty). `preserve_qty=True` SOLO para staples que
# aportan peso/macros (embutidos, bacalao salado) → el guard preserva el prefijo de cantidad
# ("100g de longaniza" → "100g de Pechuga de pollo") y recalcula los macros del plato por delta.
# `False` para condimentos/azúcares donde la UNIDAD misma es lo contraindicado ("1 cubito de…",
# "1 cda de miel") o son "al gusto" — preservar el prefijo dejaría la palabra ofensora en el string.
_DM2_SUGAR_SUBS = (
    (("miel", "honey"), "Stevia al gusto", "miel", False),
    (("sirope", "jarabe", "syrup"), "Stevia al gusto", "sirope/jarabe", False),
    (("panela", "melaza", "molasses"), "Stevia al gusto", "panela/melaza", False),
    (("leche condensada", "dulce de leche", "condensed milk"), "Leche evaporada sin azúcar", "leche condensada", False),
    (("refresco", "gaseosa", "soda", "malta", "jugo de caja", "jugo concentrado",
      "jugo embotellado", "jugo de cajita"), "Agua", "bebida azucarada", False),
    (("azucar", "azúcar", "sugar"), "Stevia al gusto", "azúcar", False),
)
_DM2_SUGAR_NEGATIVES = ("sin azucar", "no azucar", "0 azucar", "cero azucar", "libre de azucar",
                        "0% azucar", "sin azucares", "bajo en azucar")

_HTA_SODIUM_SUBS = (
    (("embutido", "salami", "longaniza", "salchicha", "mortadela", "tocineta",
      "bacon", "chorizo", "jamon", "jamón"), "Pechuga de pollo", "embutidos", True),
    # OJO: tokens estrechos para no colisionar con proteína legítima — 'cubito de' (NO 'cubito', que
    # matchearía 'cubitos de pollo' = pollo en cubos); 'salsa de soya' (NO 'soya' desnudo, que
    # borraría tofu/leche-de-soya/carne-de-soya, proteína vegetal). Bug encontrado por review adversaria.
    (("cubito de", "sazon en polvo", "sazón en polvo", "sazon completa", "caldo en cubo",
      "consome", "consomé", "maggi", "sopita", "sazonador"), "Especias naturales (ajo, cebolla, orégano, comino)", "cubitos/sazón en polvo", False),
    (("bacalao salado", "bacalao seco", "arenque salado"), "Pescado fresco", "pescado salado", True),
    (("salsa de soya", "salsa de soja", "teriyaki"), "Limón con especias", "salsa de soya", False),
    (("tajin", "tajín", "sal de ajo", "sal de cebolla"), "Especias sin sal añadida", "sazonadores salados", False),
)
_HTA_SODIUM_NEGATIVES = ("baja en sodio", "bajo en sodio", "sin sal", "sin sodio", "reducido en sodio")


# ── El REGISTRO declarativo (SSOT del comportamiento por condición) ──
CONDITION_RULES: tuple = (
    ConditionRule(
        id="renal", label="Enfermedad renal crónica", terms=RENAL_CONDITION_TERMS,
        precedence=10, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🫘 REGLA CLÍNICA — ENFERMEDAD RENAL (KDIGO 2024) — PRECAUCIÓN, REQUIERE NEFRÓLOGO:\n"
            "   • Proteína MODERADA, NO alta: porciones modestas de proteína de alta calidad (huevo, "
            "pescado, pollo) — NO maximices la proteína (lo contrario a un plan de hipertrofia).\n"
            "   • SODIO BAJO: sal medida y mínima; evita embutidos, cubitos/sazón, bacalao salado, ultra-procesados.\n"
            "   • Modera alimentos MUY altos en potasio/fósforo si aparecen en exceso (vísceras, lácteos en gran "
            "cantidad, exceso de guineo/aguacate); el balance fino lo define el nefrólogo.\n"
            "   • Este plan es ORIENTATIVO y NO sustituye la indicación de un profesional de salud renal."),
    ),
    ConditionRule(
        id="dm2", label="Diabetes T2 / prediabetes", terms=DIABETES_CONDITION_TERMS,
        precedence=30, substitutions=_DM2_SUGAR_SUBS, sub_negatives=_DM2_SUGAR_NEGATIVES,
        prompt_block=(
            "🩸 REGLA CLÍNICA — DIABETES T2 / PREDIABETES (ADA 2025/2026, CALIDAD DEL CARBOHIDRATO):\n"
            "   • NO se trata de 'bajar los carbohidratos' ni de un % fijo: prioriza la CALIDAD del carbohidrato.\n"
            "   • FIBRA ALTA (objetivo ≥14 g por cada 1000 kcal): incluye leguminosas (habichuelas, lentejas, "
            "gandules), avena, vegetales abundantes y fruta entera con cáscara.\n"
            "   • GRANOS INTEGRALES INTACTOS: arroz integral, avena y víveres con fibra (batata, yuca, plátano "
            "verde) sobre harinas refinadas, pan blanco y arroz blanco pelado.\n"
            "   • PROHIBIDAS las bebidas azucaradas y los azúcares añadidos (miel, sirope, dulces); endulza con "
            "fruta o estevia.\n"
            "   • Combina SIEMPRE el carbohidrato con proteína + grasa saludable + fibra en la misma comida."),
    ),
    ConditionRule(
        id="hta", label="Hipertensión arterial", terms=HTA_CONDITION_TERMS,
        precedence=40, substitutions=_HTA_SODIUM_SUBS, sub_negatives=_HTA_SODIUM_NEGATIVES,
        prompt_block=(
            "🧂 REGLA CLÍNICA — HIPERTENSIÓN (patrón DASH, NHLBI/AHA-ACC 2025):\n"
            "   • SODIO BAJO (meta ≤1500 mg/día): NADA de embutidos (salami, longaniza, jamón), cubitos/sazón "
            "en polvo, bacalao salado, ni 'sal al gusto' genérica. Especifica sal medida mínima (≤1 g/día).\n"
            "   • POTASIO/MAGNESIO/CALCIO ALTOS: prioriza vegetales, frutas (guineo, aguacate con moderación), "
            "leguminosas, lácteos bajos en grasa, vegetales de hoja verde.\n"
            "   • Sabor sin sal: ajo, cebolla, orégano, comino, cilantro, limón, vinagre.\n"
            "   • Evita ultra-procesados y enlatados altos en sodio."),
    ),
    ConditionRule(
        id="dyslipidemia", label="Dislipidemia / colesterol alto", terms=DYSLIPIDEMIA_CONDITION_TERMS,
        precedence=45,
        prompt_block=(
            "🫀 REGLA CLÍNICA — DISLIPIDEMIA (AHA 2021/ACC 2025):\n"
            "   • GRASA SATURADA BAJA (<7% de las calorías): evita frituras, piel de pollo, grasa visible de "
            "carnes, embutidos, mantequilla y lácteos enteros. Usa cocción al horno/plancha/hervido.\n"
            "   • FIBRA SOLUBLE alta (baja el LDL): avena, habichuelas/lentejas, berenjena, manzana, cítricos.\n"
            "   • GRASAS SALUDABLES: aguacate, aceite de oliva, pescado graso (sardina/salmón), nueces — con moderación.\n"
            "   • Sin grasas trans (margarina dura, productos de repostería industrial)."),
    ),
    ConditionRule(
        id="anemia", label="Anemia ferropénica", terms=ANEMIA_CONDITION_TERMS,
        precedence=60,
        prompt_block=(
            "🩸 REGLA CLÍNICA — ANEMIA FERROPÉNICA (densidad de hierro):\n"
            "   • HIERRO HEMO: incluye carnes rojas magras, hígado (1x/sem), pollo, pescado.\n"
            "   • POTENCIA LA ABSORCIÓN: acompaña el hierro con vitamina C (naranja, limón, tomate, pimiento).\n"
            "   • FOLATO + B12: leguminosas, vegetales de hoja verde, huevo.\n"
            "   • SEPARA del café/té/lácteos en la misma comida (inhiben la absorción de hierro)."),
    ),
)

_RULES_BY_ID = {r.id: r for r in CONDITION_RULES}


def _norm_conditions(form_data) -> list:
    """Lista normalizada (lower + strip_accents, sin sentinel) de las condiciones del form."""
    if not isinstance(form_data, dict):
        return []
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    raw = form_data.get("medicalConditions") or form_data.get("medical_conditions") or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for c in raw:
        s = str(c).strip().lower()
        if not s or s in _MEDICAL_NONE_SENTINELS:
            continue
        try:
            s = _sa(s)
        except Exception:
            pass
        out.append(s)
    return out


def detect_active_rules(form_data) -> list:
    """Reglas activas para el perfil, ordenadas por precedencia (seguridad primero)."""
    conds = _norm_conditions(form_data)
    if not conds:
        return []
    active = [r for r in CONDITION_RULES
              if any(any(t in c for t in r.terms) for c in conds)]
    return sorted(active, key=lambda r: r.precedence)


def build_condition_prompt(form_data) -> str:
    """Bloque de reglas nutricionales por condición (registry-driven) + nota de comorbilidad."""
    active = detect_active_rules(form_data)
    if not active:
        return ""
    blocks = [r.prompt_block for r in active]
    ids = {r.id for r in active}
    if "dm2" in ids and "renal" in ids:
        blocks.append(
            "⚖️ PRECEDENCIA CLÍNICA — DIABETES + ENFERMEDAD RENAL JUNTAS (diabético-nefropatía):\n"
            "   • La regla RENAL MANDA sobre el target de fibra/leguminosas de la diabetes: las leguminosas y "
            "granos integrales (altos en potasio/fósforo) se MODERAN — prioriza vegetales BAJOS en potasio.\n"
            "   • Mantén proteína MODERADA (renal) y sodio bajo; NO subas la carga de carbohidrato. El balance "
            "fino lo define el nefrólogo.")
    elif len(active) >= 2:
        labels = ", ".join(r.label for r in active)
        blocks.append(
            f"⚖️ PRECEDENCIA CLÍNICA — CONDICIONES MÚLTIPLES ({labels}): cuando dos reglas chocan, gana la MÁS "
            "RESTRICTIVA en dirección de seguridad (el techo más bajo de sodio, proteína o grasa saturada). "
            "Este plan es orientativo; el balance individual lo define tu profesional de salud.")
    return ("\n--- REGLAS NUTRICIONALES POR CONDICIÓN MÉDICA (DETERMINISTAS, CITABLES) ---\n"
            + "\n\n".join(blocks)
            + "\n----------------------------------------\n")


def collect_substitutions(form_data) -> list:
    """Sustituciones deterministas de ingredientes activas para el perfil, en orden de precedencia.
    Cada item: {tokens, replacement, label, negatives, condition, preserve_qty}. El guard las aplica
    en un solo pase. Tolera filas legacy de 3 elementos (preserve_qty → False por defecto)."""
    out = []
    for r in detect_active_rules(form_data):
        for sub in (r.substitutions or ()):
            tokens, repl, label = sub[0], sub[1], sub[2]
            preserve_qty = bool(sub[3]) if len(sub) > 3 else False
            out.append({"tokens": tokens, "replacement": repl, "label": label,
                        "negatives": r.sub_negatives or (), "condition": r.id,
                        "preserve_qty": preserve_qty})
    return out


def active_condition_labels(form_data) -> list:
    return [r.label for r in detect_active_rules(form_data)]
