"""[P1-MEDICATION-RULES · 2026-06-18] (audit fresco P1-A/P1-B) Motor DECLARATIVO de interacciones
fármaco-alimento. Espejo del patrón de `condition_rules.py`: cada interacción dietética relevante es
una fila `MedicationRule` (detección por términos accent-free, directiva citable de prompt, interacción
+ recomendación es-DO). Los medicamentos se declaran en `form_data['medications']` (chips del wizard,
OPCIONAL) y, como defensa-en-profundidad, se escanean también los campos de texto médico libre
(`otherConditions`/`medicalConditions`) — el usuario suele escribir "tomo warfarina" en "otra condición".

Por qué este motor existe (audit fresco 2026-06-18, P1-A): el formulario capturaba `allergies` y
`medicalConditions` pero NO los medicamentos → punto ciego de interacciones bien establecidas (warfarina
↔ vitamina K, metformina ↔ B12, IECA/ARA-II ↔ potasio, levotiroxina ↔ calcio/hierro). El único guard
previo era REACTIVO (depende de que la LLM detecte el concern). Este motor es PROACTIVO: pregunta por el
medicamento e inyecta la directiva determinista + levanta el gate FS9.

DECISIÓN HONESTA (clínica): los medicamentos NO se auto-prescriben ni reescriben el plan (no hay
`substitutions`). Toda fila es CLINICAL_REFERRAL → directiva de prompt (sesga selección/timing) + gate
FS9 (`requires_professional_review`). El balance fino (dosis, INR, potasio sérico) lo define el médico/
farmacéutico. Es la simétrica de los gates de embarazo/menores aplicada a fármacos.

VITAMINA K (P1-B, [P1-WARFARIN-VITAMIN-K]): el riesgo de la warfarina NO es el valor absoluto de vit K
sino su CONSISTENCIA día a día (cambios bruscos desestabilizan el INR → sangrado/coágulos). El catálogo
`master_ingredients` NO tiene columna de vitamina K (poblarla es trabajo de DATOS, follow-up), así que
NO computamos mcg — `vitamin_k_consistency` usa una heurística por NOMBRE (vegetales de hoja verde por
día) que detecta la VARIABILIDAD, que es justo el riesgo clínico. El advisory primario (mantener
consistente, no eliminar) es además la intervención clínicamente correcta.
"""
from __future__ import annotations

from dataclasses import dataclass


def _strip_accents(s: str) -> str:
    try:
        from constants import strip_accents as _sa
        return _sa(s)
    except Exception:  # pragma: no cover - constants siempre disponible en prod
        return s


CLINICAL_REFERRAL = "clinical_referral"  # las interacciones se derivan al profesional (no se prescriben)

# Sentinels de "sin medicamentos" (el chip-UI no emite sentinel — array vacío basta — pero somos
# defensivos por si un campo legacy/texto-libre trae "Ninguno").
_MED_NONE_SENTINELS = frozenset({
    "", "ninguno", "ninguna", "ningunos", "ningunas", "none", "n/a", "na", "no", "nada",
    "sin medicamentos", "ningun medicamento", "ningún medicamento", "no tomo medicamentos",
})


@dataclass(frozen=True)
class MedicationRule:
    """Una interacción fármaco-alimento como DATO. `terms`: substrings accent-free lowercase (genéricos
    + marcas comunes en RD). `anticoagulant`: marca la fila warfarina (gatea el monitor de consistencia
    de vit K, P1-WARFARIN-VITAMIN-K). Todas las filas son CLINICAL_REFERRAL (no se auto-prescriben).
    `precedence` menor = mayor prioridad clínica de seguridad (anticoagulante manda)."""
    id: str
    label: str
    terms: tuple
    interaction: str       # qué interactúa (es-DO, citable en PDF)
    recommendation: str    # qué hacer (es-DO, accionable)
    prompt_block: str      # directiva determinista al generador LLM
    anticoagulant: bool = False
    # [P2-MEDICATION-TIMING-ADVISORY · 2026-06-18] (audit fresco P2-1) marca fármacos cuya ABSORCIÓN
    # depende del TIMING respecto a las comidas (levotiroxina ↔ calcio/hierro/soya) → surface un banner
    # de timing dedicado (medication_review.timing_advisories), distinto del advisory de interacción general.
    timing_sensitive: bool = False
    precedence: int = 50


# ── El REGISTRO declarativo (SSOT de las interacciones fármaco-alimento) ──
MEDICATION_RULES: tuple = (
    MedicationRule(
        id="anticoagulant", label="Anticoagulante (warfarina/acenocumarol)",
        terms=("warfarina", "warfarin", "coumadin", "acenocumarol", "sintrom", "anticoagulante",
               "anticoagulant", "cumarina", "cumarin"),
        anticoagulant=True, precedence=10,
        interaction=("La warfarina y similares dependen de una ingesta ESTABLE de vitamina K (vegetales "
                     "de hoja verde: espinaca, brócoli, repollo, lechuga). Cambios bruscos día a día "
                     "desestabilizan el INR (riesgo de sangrado o de coágulos)."),
        recommendation=("Mantén una cantidad CONSISTENTE de vegetales de hoja verde cada día (no los "
                        "elimines —son saludables— pero no alternes días de mucha hoja verde con días sin). "
                        "Tu médico controla tu INR; consúltale antes de cambiar tu patrón de verduras."),
        prompt_block=(
            "💊 INTERACCIÓN FÁRMACO-ALIMENTO — ANTICOAGULANTE (WARFARINA/ACENOCUMAROL) — REQUIERE MÉDICO:\n"
            "   • VITAMINA K CONSISTENTE: distribuye los vegetales de hoja verde (espinaca, brócoli, "
            "repollo, lechuga, acelga) de forma PAREJA entre todos los días — NO concentres muchos en un "
            "día y ninguno en otro. La ESTABILIDAD importa más que la cantidad.\n"
            "   • Cantidades MODERADAS y similares cada día (no las elimines: la vitamina K es saludable).\n"
            "   • EVITA jugo de toronja/pomelo y suplementos de vitamina K/E sin indicación médica.\n"
            "   • Este plan es ORIENTATIVO; el ajuste fino lo define tu médico según tu INR."),
    ),
    MedicationRule(
        id="metformin", label="Metformina",
        terms=("metformina", "metformin", "glucophage", "glafornil", "dimefor", "glucamet"),
        precedence=30,
        interaction=("El uso prolongado de metformina puede reducir la absorción de vitamina B12 (riesgo "
                     "de déficit con el tiempo)."),
        recommendation=("Asegura fuentes de vitamina B12 (huevo, lácteos, carne, pescado); si eres "
                        "vegano/vegetariano, considera un suplemento de B12. Tu médico puede medir tu B12."),
        prompt_block=(
            "💊 INTERACCIÓN FÁRMACO-ALIMENTO — METFORMINA:\n"
            "   • Refuerza la VITAMINA B12 (huevo, lácteos, carne, pescado) — la metformina reduce su "
            "absorción con el uso prolongado.\n"
            "   • Mantén las pautas de diabetes (fibra alta, sin bebidas azucaradas) si aplican."),
    ),
    MedicationRule(
        id="ace_arb", label="Antihipertensivo IECA/ARA-II",
        terms=("lisinopril", "enalapril", "ramipril", "captopril", "benazepril", "perindopril", "ieca",
               "losartan", "valsartan", "telmisartan", "candesartan", "irbesartan", "olmesartan",
               "ara ii", "ara-ii", "araii"),
        precedence=40,
        interaction=("Los IECA/ARA-II pueden ELEVAR el potasio en sangre. Los sustitutos de sal (cloruro "
                     "de potasio) y los suplementos de potasio aumentan ese riesgo."),
        recommendation=("EVITA los sustitutos de sal 'lite'/'sin sodio' (son cloruro de potasio) y los "
                        "suplementos de potasio salvo indicación médica. Tu médico controla tu potasio sérico."),
        prompt_block=(
            "💊 INTERACCIÓN FÁRMACO-ALIMENTO — ANTIHIPERTENSIVO IECA/ARA-II:\n"
            "   • NO uses sustitutos de sal 'lite'/'sin sodio' (son CLORURO DE POTASIO) ni suplementos de "
            "potasio — estos fármacos ya elevan el potasio en sangre.\n"
            "   • Los alimentos altos en potasio (guineo, aguacate, leguminosas) son aceptables con "
            "moderación; el balance fino lo controla el médico con análisis."),
    ),
    MedicationRule(
        id="levothyroxine", label="Levotiroxina (tiroides)",
        terms=("levotiroxina", "levothyroxine", "eutirox", "synthroid", "euthyrox", "tiroxina",
               "liotironina", "liothyronine"),
        timing_sensitive=True, precedence=45,
        interaction=("El calcio, el hierro y la soya/fibra reducen la absorción de la levotiroxina si se "
                     "toman junto con ella."),
        recommendation=("Toma la levotiroxina en ayunas con agua y espera 30-60 min antes de desayunar; "
                        "separa los lácteos/calcio, el hierro y la soya al menos 4 horas de la pastilla."),
        prompt_block=(
            "💊 INTERACCIÓN FÁRMACO-ALIMENTO — LEVOTIROXINA (TIROIDES):\n"
            "   • TIMING: la levotiroxina se toma en ayunas; el desayuno (sobre todo lácteos/calcio, "
            "hierro, café y soya) debe ir 30-60 min DESPUÉS de la pastilla.\n"
            "   • Separa los suplementos de calcio/hierro al menos 4 h de la pastilla (reducen su absorción)."),
    ),
)

_RULES_BY_ID = {r.id: r for r in MEDICATION_RULES}


def _norm_medications(form_data) -> list:
    """Lista normalizada (lower + strip_accents, sin sentinel) de los medicamentos del form. Fuente
    primaria: el campo `medications` (chips). Backstop defensa-en-profundidad: el texto médico libre
    (`otherConditions`/`medicalConditions`/`otherMedications`) donde el usuario suele escribir su med."""
    if not isinstance(form_data, dict):
        return []
    sources = []
    for key in ("medications", "medication", "medicamentos"):
        v = form_data.get(key)
        if isinstance(v, str):
            sources.append(v)
        elif isinstance(v, (list, tuple)):
            sources.extend(v)
    # Backstop: texto médico libre (un med escrito en "otra condición" sigue contando).
    for key in ("otherMedications", "otherConditions", "medicalConditions", "medical_conditions"):
        v = form_data.get(key)
        if isinstance(v, str):
            sources.append(v)
        elif isinstance(v, (list, tuple)):
            sources.extend(v)
    out = []
    for raw in sources:
        s = str(raw).strip().lower()
        if not s or s in _MED_NONE_SENTINELS:
            continue
        out.append(_strip_accents(s))
    return out


def detect_active_medications(form_data) -> list:
    """Reglas de medicación activas para el perfil, ordenadas por precedencia (seguridad primero)."""
    meds = _norm_medications(form_data)
    if not meds:
        return []
    active = [r for r in MEDICATION_RULES
              if any(any(t in m for t in r.terms) for m in meds)]
    return sorted(active, key=lambda r: r.precedence)


def build_medication_prompt(form_data) -> str:
    """Bloque de directivas de interacción fármaco-alimento (registry-driven). No-op si no hay
    medicamento cubierto. Emite SOLO los `prompt_block` canned de las reglas DETECTADAS (NUNCA re-emite
    el texto crudo del usuario → no es vector de prompt-injection, igual que `build_condition_prompt`)."""
    active = detect_active_medications(form_data)
    if not active:
        return ""
    blocks = [r.prompt_block for r in active]
    return ("\n--- INTERACCIONES FÁRMACO-ALIMENTO (DETERMINISTAS, REQUIEREN MÉDICO) ---\n"
            + "\n\n".join(blocks)
            + "\n----------------------------------------\n")


def active_medication_labels(form_data) -> list:
    """Etiquetas canónicas de las interacciones activas (para el FS9 note / PDF). Devuelve los LABELS
    de las reglas, NO el texto crudo del usuario."""
    return [r.label for r in detect_active_medications(form_data)]


def build_medication_advisories(active_rules_or_form) -> list:
    """Items advisory para el panel/PDF: {medicamento, interaccion, recomendacion}. Acepta una lista de
    `MedicationRule` o directamente el `form_data` dict."""
    if isinstance(active_rules_or_form, dict):
        active_rules_or_form = detect_active_medications(active_rules_or_form)
    return [{"medicamento": r.label, "interaccion": r.interaction, "recomendacion": r.recommendation}
            for r in (active_rules_or_form or [])]


def detect_anticoagulant(form_data) -> bool:
    """True si el perfil declara un anticoagulante (warfarina/acenocumarol) → gatea el monitor de
    consistencia de vitamina K (P1-WARFARIN-VITAMIN-K)."""
    return any(r.anticoagulant for r in detect_active_medications(form_data))


def build_timing_advisories(active_rules_or_form) -> list:
    """[P2-MEDICATION-TIMING-ADVISORY · 2026-06-18] (audit fresco P2-1) Subconjunto de advisories para
    fármacos TIMING-SENSITIVE (la absorción depende de cuándo se toman respecto a las comidas — p.ej.
    levotiroxina ↔ calcio/hierro/soya). El frontend lo renderiza como un banner de timing dedicado,
    distinto del advisory de interacción general. Acepta lista de `MedicationRule` o `form_data` dict."""
    if isinstance(active_rules_or_form, dict):
        active_rules_or_form = detect_active_medications(active_rules_or_form)
    return [{"medicamento": r.label, "recomendacion": r.recommendation}
            for r in (active_rules_or_form or []) if getattr(r, "timing_sensitive", False)]


def requires_medication_review(form_data) -> bool:
    """True si hay cualquier interacción fármaco-alimento activa → gate FS9 (revisión profesional).
    Conservador: todo medicamento con interacción dietética conocida amerita coordinación médica."""
    return bool(detect_active_medications(form_data))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# [P1-WARFARIN-VITAMIN-K · 2026-06-18] (audit fresco P1-B) Monitor de consistencia de vitamina K
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Alimentos es-DO altos en vitamina K (filoquinona), accent-free lowercase substrings. NO es exhaustivo
# ni numérico: el catálogo `master_ingredients` NO tiene columna `vit_k_mcg` (poblarla desde USDA es
# follow-up de DATOS). Detecta la VARIABILIDAD día a día de hoja verde, que es el riesgo clínico real de
# la warfarina (estabilidad del INR), no el valor absoluto. Honestidad: `method=name_presence_heuristic`.
_HIGH_VIT_K_TERMS = (
    "espinaca", "brocoli", "col rizada", "kale", "berza", "acelga", "berro", "lechuga", "repollo",
    "coles de bruselas", "col de bruselas", "perejil", "cilantro", "esparrago", "grelos",
    "hojas de nabo", "hojas de mostaza", "rucula", "vainita", "habichuela tierna", "quimbombo",
    "molondron", "okra",
)


def _high_vit_k_count_per_day(plan) -> list:
    """Cuenta ingredientes altos en vit K por día (match por nombre, accent-free)."""
    days = (plan or {}).get("days") or []
    per_day = []
    for d in days:
        cnt = 0
        for meal in (d.get("meals") or []):
            for ing in (meal.get("ingredients") or []):
                s = _strip_accents(str(ing)).lower()
                if any(t in s for t in _HIGH_VIT_K_TERMS):
                    cnt += 1
        per_day.append(cnt)
    return per_day


def vitamin_k_consistency(plan) -> dict:
    """[P1-WARFARIN-VITAMIN-K · 2026-06-18] Heurística de consistencia de vitamina K para usuarios de
    anticoagulante. NO mide mcg (sin dato en catálogo) — cuenta alimentos de hoja verde por día y mide
    la VARIABILIDAD día a día (`spread = max - min`). El riesgo de la warfarina es la inconsistencia, no
    el valor absoluto. Retorna {applicable, per_day, spread, variability, note, method}."""
    per_day = _high_vit_k_count_per_day(plan)
    note = ("Mantén una cantidad CONSISTENTE de vegetales de hoja verde (espinaca, brócoli, repollo, "
            "lechuga) cada día. NO los elimines —son saludables— pero evita alternar días de mucha hoja "
            "verde con días sin. Tu médico controla tu INR: consúltale antes de cambiar tu patrón de "
            "verduras. (Estimación por presencia de alimentos, no por mg de vitamina K.)")
    if not per_day:
        return {"applicable": True, "per_day": [], "spread": 0, "variability": "unknown",
                "note": note, "method": "name_presence_heuristic"}
    spread = max(per_day) - min(per_day)
    variability = "low" if spread <= 1 else ("moderate" if spread <= 3 else "high")
    return {"applicable": True, "per_day": per_day, "spread": spread, "variability": variability,
            "note": note, "method": "name_presence_heuristic"}
