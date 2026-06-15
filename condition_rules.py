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
- Dislipidemia: sustitución de grasa saturada (mantequilla→aceite, lácteos enteros→bajos en grasa,
  tocino/chicharrón→lean) ENFORCED via este motor + techo satfat<7% kcal medido en el panel
  (columna saturated_fat_g poblada desde USDA, P4-UNIFIED-RESOLVER). El swap baja el LDL preservando
  proteína; el techo del panel marca el residual. [P4-DYSLIPIDEMIA-ENFORCED · 2026-06-14]
- Anemia: prompt + densidad de hierro (el panel ya computa iron_mg).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from constants import (
    RENAL_CONDITION_TERMS, DIABETES_CONDITION_TERMS, HTA_CONDITION_TERMS,
    DYSLIPIDEMIA_CONDITION_TERMS, ANEMIA_CONDITION_TERMS,
    PREGNANCY_CONDITION_TERMS, HYPOTHYROID_CONDITION_TERMS, GOUT_CONDITION_TERMS,
    NAFLD_CONDITION_TERMS, PCOS_CONDITION_TERMS,
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
    # [P2-SUBS-RESOLVE · 2026-06-15] "Filete de pescado blanco" (nombre del catálogo) — NO "Pescado
    # fresco" (no resolvía → fantasma en la lista + delta de macros perdía el ingrediente). Audit P2.
    (("bacalao salado", "bacalao seco", "arenque salado"), "Filete de pescado blanco", "pescado salado", True),
    (("salsa de soya", "salsa de soja", "teriyaki"), "Limón con especias", "salsa de soya", False),
    (("tajin", "tajín", "sal de ajo", "sal de cebolla"), "Especias sin sal añadida", "sazonadores salados", False),
)
_HTA_SODIUM_NEGATIVES = ("baja en sodio", "bajo en sodio", "sin sal", "sin sodio", "reducido en sodio")

# [P4-DYSLIPIDEMIA-ENFORCED · 2026-06-14] Dislipidemia → grasa saturada: swaps fuente-grasa-saturada →
# versión magra/insaturada de la MISMA categoría (preserva proteína, baja el LDL). preserve_qty=True: el
# reemplazo conserva la cantidad/volumen (misma comida, versión leaner) y el delta de macros lo ajusta el
# guard. Tokens ESTRECHOS (lección del bug 'soya'): nada de 'manteca' desnudo (matchea 'mantecado'),
# 'nata' (matchea 'natural'/'natilla') ni 'crema' (matchea 'crema de maní') — solo frases inequívocas.
_DYSLIPIDEMIA_SATFAT_SUBS = (
    (("mantequilla", "manteca de cerdo", "manteca vegetal", "margarina"), "Aceite de oliva", "mantequilla/manteca/margarina", True),
    # NOTA: 'leche entera' NO se sustituye — el catálogo conflaciona 'leche entera' y 'leche descremada'
    # en la MISMA fila `Leche`, así que el swap sería un no-op clínico (sin reducción de grasa real). Se
    # reactivará cuando exista una fila distinta de leche descremada. (Hallazgo de review adversaria.)
    (("crema de leche", "crema espesa"), "Leche evaporada", "crema de leche", True),
    (("queso amarillo", "queso cheddar", "queso crema"), "Queso cottage", "queso alto en grasa", True),
    (("yogur griego entero", "yogurt griego entero", "yogur entero", "yogurt entero", "yogur natural entero"),
     "Yogurt griego sin azúcar", "yogur entero", True),
    (("tocino", "tocineta", "chicharron", "chicharrón"), "Pechuga de pollo", "tocino/chicharrón", True),
)
_DYSLIPIDEMIA_NEGATIVES = ("descremad", "baja en grasa", "bajo en grasa", "light", "desnatad",
                           "sin grasa", "0% grasa", "0 grasa",
                           # nueces/semillas = grasa INSATURADA (saludable): 'mantequilla de maní/almendra'
                           # NO debe caer en el swap de 'mantequilla'. (Veto análogo al 'baja en sodio' de HTA.)
                           "mani", "maní", "cacahuate", "almendra", "marañon", "maranon", "semilla",
                           # [review adversaria] 'chicharron'/'tocino' desnudos colisionan con
                           # 'chicharrón DE POLLO' (pollo frito criollo = pollo), 'tocino DE PAVO' (ya magro)
                           # y 'tocino DE CIELO' (postre). Ningún swap de dislipidemia debe disparar sobre
                           # un '…de pollo/pavo'. 'coco' (crema de coco) NO es lácteo → no mislabelear.
                           "de pollo", "de pavo", "de cielo", "coco")


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
        precedence=45, substitutions=_DYSLIPIDEMIA_SATFAT_SUBS, sub_negatives=_DYSLIPIDEMIA_NEGATIVES,
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
    # [P1-CONDITION-COVERAGE · 2026-06-14] Condiciones comunes que faltaban. ADVISORY (prompt_block +
    # gate de derivación FS9, classification CLINICAL_REFERRAL): la guía es estándar-de-cuidado general;
    # la regla fina (qué limitar/sustituir por estadio/severidad) la valida el profesional, NO el motor
    # determinista (evita enforcement clínico sin revisión humana — cautela del audit P1). Embarazo es
    # además SEGURIDAD: el déficit calórico ya lo bloquea el gate en nutrition_calculator.
    ConditionRule(
        id="pregnancy", label="Embarazo / lactancia", terms=PREGNANCY_CONDITION_TERMS,
        precedence=15, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🤰 REGLA CLÍNICA — EMBARAZO / LACTANCIA (SEGURIDAD — REQUIERE OBSTETRA/NUTRICIONISTA):\n"
            "   • NUNCA un déficit calórico: usa AL MENOS mantenimiento (el requerimiento sube en 2º/3º "
            "trimestre y lactancia). NO es momento de perder peso.\n"
            "   • FOLATO + HIERRO altos: vegetales de hoja verde, leguminosas, carnes magras, huevo; "
            "acompaña el hierro con vitamina C.\n"
            "   • CALCIO + PROTEÍNA suficientes: lácteos PASTEURIZADOS, pescado bajo en mercurio.\n"
            "   • EVITA por listeria/seguridad: embutidos y quesos/lácteos NO pasteurizados, pescado "
            "crudo, carne/huevo poco cocidos, pescados altos en mercurio (tiburón, pez espada, atún "
            "grande), alcohol y exceso de cafeína.\n"
            "   • Este plan es ORIENTATIVO y NO sustituye el control prenatal."),
    ),
    ConditionRule(
        id="hypothyroid", label="Hipotiroidismo", terms=HYPOTHYROID_CONDITION_TERMS,
        precedence=55, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🦋 REGLA CLÍNICA — HIPOTIROIDISMO (orientativa, requiere endocrinólogo):\n"
            "   • YODO ADECUADO (sin exceso): sal yodada con moderación, pescado/mariscos, huevo, lácteos.\n"
            "   • SELENIO + ZINC: nuez (1-2/día si no hay alergia), huevo, pollo, mariscos.\n"
            "   • Modera los GOITRÓGENOS CRUDOS en gran cantidad (repollo, brócoli, coliflor, yuca cruda); "
            "cocidos pierden el efecto — no es necesario eliminarlos.\n"
            "   • Separa los suplementos de hierro/calcio de la levotiroxina (interfieren su absorción)."),
    ),
    ConditionRule(
        id="gout", label="Gota / ácido úrico alto", terms=GOUT_CONDITION_TERMS,
        precedence=55, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🦶 REGLA CLÍNICA — GOTA / HIPERURICEMIA (orientativa, requiere médico):\n"
            "   • PURINAS BAJAS: limita vísceras (hígado, riñón, mollejas), mariscos, sardina/anchoa y "
            "el exceso de carnes rojas.\n"
            "   • CERO alcohol (especialmente cerveza) y bebidas/alimentos altos en fructosa (refrescos, "
            "jugos azucarados).\n"
            "   • HIDRATACIÓN abundante (agua) + lácteos BAJOS en grasa (protectores).\n"
            "   • Vegetales, fruta entera, granos integrales y proteína magra con moderación."),
    ),
    ConditionRule(
        id="nafld", label="Hígado graso (NAFLD/MAFLD)", terms=NAFLD_CONDITION_TERMS,
        precedence=50, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🫥 REGLA CLÍNICA — HÍGADO GRASO (NAFLD/MAFLD) (orientativa, requiere médico):\n"
            "   • REDUCE azúcar añadida y FRUCTOSA (refrescos, jugos, dulces, sirope) y carbohidratos "
            "refinados (pan blanco, harinas) — son el principal motor de la grasa hepática.\n"
            "   • PATRÓN MEDITERRÁNEO: aceite de oliva, pescado, vegetales, leguminosas, granos integrales.\n"
            "   • CERO alcohol.\n"
            "   • Si hay sobrepeso, una pérdida gradual de peso (7-10%) mejora la esteatosis."),
    ),
    ConditionRule(
        id="pcos", label="SOP (ovario poliquístico)", terms=PCOS_CONDITION_TERMS,
        precedence=50, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🌸 REGLA CLÍNICA — SOP / OVARIO POLIQUÍSTICO (orientativa, requiere ginecólogo/endocrino):\n"
            "   • CALIDAD DEL CARBOHIDRATO (resistencia a la insulina): prioriza fibra alta, granos "
            "integrales intactos y leguminosas; evita azúcares añadidos y harinas refinadas.\n"
            "   • PROTEÍNA + GRASA SALUDABLE en cada comida para estabilizar la glucosa.\n"
            "   • Patrón antiinflamatorio (pescado graso, aceite de oliva, vegetales).\n"
            "   • Combina con actividad física; el manejo del peso mejora la sensibilidad a la insulina."),
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


# ════════════════════════════════════════════════════════════════════════════════════════════════
# [P0-ALLERGEN-SUBS · 2026-06-14] Sustitución determinista de ALÉRGENOS IgE declarados
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Cierra el gap del audit clínico (2026-06-14): las alergias declaradas (`form_data['allergies']`)
# dependían del prompt + el backstop romo `_scan_allergen_violations` (que NUKE el plan entero a
# fallback matemático). Este registro hace QUIRÚRGICA la defensa: para los alérgenos con un reemplazo
# SEGURO que RESUELVE al catálogo `master_ingredients`, sustituye el ingrediente ofensor in-place
# (preserva cantidad + recalcula macros por delta, vía el mismo motor que las sustituciones por
# condición), conservando el plan rico del LLM. `_scan_allergen_violations` queda como red de
# seguridad post-swap: cualquier residual sigue escalando a rechazo crítico → fallback.
#
# DECISIÓN HONESTA (documentada): lácteos, huevo, maní y frutos secos NO se sustituyen aquí — el
# catálogo es-DO NO tiene un target libre del alérgeno que resuelva (no hay leche/queso vegetal, ni
# mantequilla de semillas). Para esos, el path crítico→fallback existente sigue EXCLUYÉNDOLOS (cero
# regresión). Habilitar lácteos/huevo requeriría filas nuevas en el catálogo (palanca de DATOS).
#
# TOKENS accent-free + lowercase (se matchean contra `strip_accents(ingrediente).lower()`) y
# ESTRECHOS (lección del bug 'soya'/'pana'): nada de raíces ambiguas — 'pan de agua' (NO 'pan'
# desnudo, que matchea 'pana'=fruta de pan); 'tofu'/'salsa de soya' (NO 'soya' desnudo). Lo que un
# token estrecho no atrape (p.ej. 'pan' a secas) lo recoge el backstop `_scan_allergen_violations`.
# Cada fila: (tokens, reemplazo, etiqueta, preserve_qty). Los reemplazos son nombres del catálogo
# (con acentos) → resuelven en el delta de macros. preserve_qty=True para proteínas/almidones-staple.

_ALLERGEN_FISH_SUBS = (
    (("pescado", "bacalao", "atun", "salmon", "tilapia", "mero", "chillo", "sardina",
      "merluza", "carite", "dorado", "filete de pescado"),
     "Pechuga de pollo", "pescado", True),
)
_ALLERGEN_SHELLFISH_SUBS = (
    (("camaron", "langosta", "langostino", "cangrejo", "marisco", "pulpo", "calamar",
      "almeja", "ostra", "lambi", "gamba"),
     "Pechuga de pollo", "mariscos", True),
)
_ALLERGEN_SOY_SUBS = (
    # condimento (qty pequeña, 'al gusto') → reemplazo bare, NO preserva prefijo de cantidad.
    (("salsa de soya", "salsa de soja", "teriyaki"), "Limón con especias", "salsa de soya", False),
    # proteína de soya → swap a pollo preservando el peso comprable.
    (("tofu", "edamame", "proteina de soya", "carne de soya", "soja texturizada", "soya texturizada"),
     "Pechuga de pollo", "soya (tofu/edamame)", True),
)
_ALLERGEN_GLUTEN_SUBS = (
    (("harina de trigo",), "Harina de maíz precocida", "harina de trigo", True),
    (("pan de agua", "pan integral", "pan blanco", "pan de trigo", "pan tostado", "pan sandwich",
      "pan pita", "tostada", "tostadas"),
     "Casabe", "pan de trigo", True),
    (("pasta integral", "espagueti", "macarron", "coditos", "fideo", "lasana", "tallarin",
      "pasta de trigo", "penne", "ravioli", "ñoqui", "noqui"),
     "Arroz blanco", "pasta de trigo", True),
    (("galleta de soda", "galletas de soda", "galleta de trigo", "galletas de trigo"),
     "Galletas de arroz", "galletas de trigo", True),
    # [P0-ALLERGEN-SUBS live-fix · 2026-06-14] La avena es naturalmente sin gluten PERO el revisor
    # médico la rechaza por contaminación cruzada (estándar conservador) → sin este swap, un plan con
    # avena para un alérgico a gluten cae a fallback. Swap a Quinoa (GF nativo, en catálogo, alto en
    # proteína → ideal para volumen). 'avena' no es substring de 'avellana' (frutos secos) ni de otro
    # ingrediente → token seguro. Hallazgo de la prueba en vivo P0-ALLERGEN-SUBS.
    (("avena", "hojuelas de avena", "harina de avena", "salvado de avena"),
     "Quinoa", "avena (gluten por contaminación)", True),
    (("cebada", "centeno", "cuscus", "bulgur", "germen de trigo", "salvado de trigo", "tortilla de trigo",
      "tortilla de harina", "tortilla integral"),
     "Arroz blanco", "cereal con gluten", True),
)
# Vetos (accent-free): evitan swappear un alimento que YA es libre del alérgeno (queda SAFE igual,
# pero no lo cambiamos innecesariamente). Solo gluten los necesita (variantes GF en el catálogo es-DO).
_ALLERGEN_GLUTEN_NEGATIVES = ("sin gluten", "libre de gluten", "gluten free", "de maiz", "de arroz",
                              "de yuca", "de almendra", "casabe", "pana")

# Detección: alergia DECLARADA (texto del form, strip_accents+lower) → categoría con tabla de swaps.
# Términos accent-free; matching `term in declared` (substring) — over-detección es la dirección SEGURA.
_ALLERGEN_DETECT = {
    "fish": ("pescado", "fish", "atun", "salmon", "bacalao", "tilapia", "sardina"),
    "shellfish": ("marisco", "camaron", "langosta", "cangrejo", "shellfish", "crustaceo", "molusco"),
    "soy": ("soya", "soja", "soy", "tofu", "edamame"),
    "gluten": ("gluten", "trigo", "wheat", "celiac", "celiaqu", "tacc"),
}
_ALLERGEN_SUBS_BY_CAT = {
    "fish": _ALLERGEN_FISH_SUBS,
    "shellfish": _ALLERGEN_SHELLFISH_SUBS,
    "soy": _ALLERGEN_SOY_SUBS,
    "gluten": _ALLERGEN_GLUTEN_SUBS,
}
_ALLERGEN_NEGATIVES_BY_CAT = {
    "gluten": _ALLERGEN_GLUTEN_NEGATIVES,
}


def collect_allergen_substitutions(form_data) -> list:
    """[P0-ALLERGEN-SUBS · 2026-06-14] Sustituciones deterministas para los alérgenos IgE DECLARADOS
    (`form_data['allergies']`) que tienen un reemplazo seguro que RESUELVE al catálogo es-DO
    (fish/shellfish/soy/gluten). Mismo shape que `collect_substitutions` → reusa el motor compartido
    `_apply_substitutions_core`. Cada item: {tokens, replacement, label, negatives, condition,
    preserve_qty}. Lácteos/huevo/maní/frutos secos NO se incluyen (sin target que resuelva) → siguen
    por el path crítico→fallback. Sentinel: P0-ALLERGEN-SUBS."""
    if not isinstance(form_data, dict):
        return []
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    raw = form_data.get("allergies") or []
    if isinstance(raw, str):
        raw = [raw]
    declared = []
    for a in raw:
        s = str(a).strip().lower()
        if not s or s in _MEDICAL_NONE_SENTINELS:
            continue
        try:
            s = _sa(s)
        except Exception:
            pass
        declared.append(s)
    if not declared:
        return []
    out = []
    for cat, terms in _ALLERGEN_DETECT.items():
        if not any(t in d for t in terms for d in declared):
            continue
        negs = _ALLERGEN_NEGATIVES_BY_CAT.get(cat, ())
        for sub in _ALLERGEN_SUBS_BY_CAT[cat]:
            tokens, repl, label = sub[0], sub[1], sub[2]
            preserve_qty = bool(sub[3]) if len(sub) > 3 else False
            out.append({"tokens": tokens, "replacement": repl, "label": label,
                        "negatives": negs, "condition": f"allergen:{cat}",
                        "preserve_qty": preserve_qty})
    return out


def active_allergen_labels(form_data) -> list:
    """Etiquetas es-DO de las categorías de alérgeno con swap determinista activo para el perfil."""
    seen, out = set(), []
    for s in collect_allergen_substitutions(form_data):
        cat = s["condition"]
        if cat not in seen:
            seen.add(cat)
            out.append(s["label"])
    return out
