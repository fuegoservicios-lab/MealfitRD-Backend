"""[P1-MACRO-GAP-DINNER-CRON · 2026-11-03]
Evaluador de brecha de macros para el cron de la cena (7:00 PM).

Problema resuelto:
A las 7:00 PM, el cron de cena no solo debe evitar repetir ingredientes (BB-003),
sino que debe operar con rigor presupuestario y matemático:
1. Si el usuario ya consumió el 95%+ de sus calorías diarias o le quedan <= 100 kcal,
   el cron NO debe empujar una cena sólida ni platos calóricos: cambia a modo cierre de jornada / hidratación.
2. Si le quedan pocas calorías (< 350 kcal) pero mucha proteína pendiente (>= 25 g),
   fuerza sugerencias de proteína magra pura de alta densidad (atún en agua, claras, whey con agua, pechuga),
   excluyendo los ingredientes ya consumidos hoy.
3. Si tiene amplio margen calórico y proteico, sugiere una cena completa y balanceada.
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Fuentes de proteína magra pura (alta densidad proteica, mínimas calorías)
_FUENTES_PROTEINA_MAGRA = [
    {"nombre": "atún en agua", "claves": ("atun", "tuna")},
    {"nombre": "claras de huevo revueltas con vegetales", "claves": ("clara", "claras", "huevo", "huevos")},
    {"nombre": "pechuga de pollo a la plancha limpia", "claves": ("pechuga", "pollo")},
    {"nombre": "batido de proteína en agua", "claves": ("proteina", "whey", "batida", "batido")},
    {"nombre": "queso cottage bajo en grasa", "claves": ("cottage", "queso")},
    {"nombre": "yogur griego 0% grasa", "claves": ("yogur", "yogurt", "griego")},
    {"nombre": "filete de pescado blanco a la plancha", "claves": ("pescado", "tilapia", "mero")},
]

# Fuentes ultra-aisladas para micro-rescate proteico (BB-029: ~100-120 kcal, 20-25g proteína limpia)
_FUENTES_MICRO_RESCATE = [
    {"nombre": "claras de huevo revueltas o cocidas (4-5 claras, ~80 kcal, 18-20 g proteína)", "claves": ("clara", "claras")},
    {"nombre": "batido de proteína whey en agua fría (1 scoop en agua, ~110-120 kcal, 24-25 g proteína)", "claves": ("proteina", "whey", "batida", "batido")},
    {"nombre": "atún en agua bien escurrido (1 lata, ~100-110 kcal, 24-26 g proteína)", "claves": ("atun", "tuna")},
]


@dataclass
class MacroGapResult:
    mode: str  # 'cierre_hidratacion' | 'micro_rescate_proteico' | 'proteina_magra' | 'cena_completa' | 'estandar' | 'desconocido'
    target_kcal: int
    target_protein: int
    consumed_kcal: int
    consumed_protein: int
    remaining_kcal: int
    remaining_protein: int
    pct_kcal_consumed: float
    instruccion_prompt: str
    alimentos_magros_filtrados: List[str]


def _normalizar(texto: str) -> str:
    if not texto:
        return ""
    n = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _extraer_alimentos_consumidos(consumed: List[Dict[str, Any]]) -> set[str]:
    """Extrae palabras clave de alimentos consumidos hoy a partir de meal_name e ingredients."""
    consumidos = set()
    for m in consumed or []:
        if not isinstance(m, dict):
            continue
        nombre = _normalizar(m.get("meal_name", ""))
        for p in nombre.split():
            if len(p) >= 3:
                consumidos.add(p)
        for ing in m.get("ingredients") or []:
            ing_norm = _normalizar(str(ing))
            for p in ing_norm.split():
                if len(p) >= 3:
                    consumidos.add(p)
    return consumidos


def _filtrar_fuentes_magras(consumidos_hoy: set[str]) -> List[str]:
    """Devuelve las fuentes de proteína magra que NO coinciden con alimentos consumidos hoy."""
    permitidos = []
    for f in _FUENTES_PROTEINA_MAGRA:
        if not any(k in consumidos_hoy for k in f["claves"]):
            permitidos.append(f["nombre"])
    # Si todos coincidieron, incluir al menos las dos opciones universales más puras
    if not permitidos:
        permitidos = ["atún en agua", "claras de huevo", "batido de proteína en agua"]
    return permitidos


def _filtrar_fuentes_micro_rescate(consumidos_hoy: set[str]) -> List[str]:
    """[BB-029] Devuelve fuentes ultra-aisladas (~100-120 kcal, 20-25g proteína limpia) para micro-rescate."""
    permitidos = []
    for f in _FUENTES_MICRO_RESCATE:
        if not any(k in consumidos_hoy for k in f["claves"]):
            permitidos.append(f["nombre"])
    if not permitidos:
        permitidos = [
            "claras de huevo revueltas o cocidas (4-5 claras, ~80 kcal, 18-20 g proteína)",
            "batido de proteína whey en agua fría (~110-120 kcal, 24-25 g proteína)",
        ]
    return permitidos


def _obtener_metas(health: Optional[Dict[str, Any]], plan_vigente: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """Obtiene (target_calories, target_protein) combinando plan, calculator y profile."""
    hp = health if isinstance(health, dict) else {}
    plan = plan_vigente if isinstance(plan_vigente, dict) else {}

    # 1. Del plan vigente
    kcal_plan = float(plan.get("calories") or 0)
    prot_plan = float((plan.get("macros") or {}).get("protein") or (plan.get("macros") or {}).get("protein_g") or 0)

    # 2. De campos explícitos en health_profile / form_data
    kcal_hp = float(hp.get("target_calories") or hp.get("daily_calories") or hp.get("calories") or 0)
    prot_hp = float(hp.get("target_protein") or hp.get("protein") or 0)

    # 3. Del nutrition_calculator si hay datos antropométricos suficientes
    kcal_calc, prot_calc = 0, 0
    _REQUERIDOS = ("gender", "age", "height", "weight", "activityLevel", "mainGoal")
    if hp and all(hp.get(c) for c in _REQUERIDOS):
        try:
            from nutrition_calculator import get_nutrition_targets
            t = get_nutrition_targets(hp) or {}
            kcal_calc = float(t.get("target_calories") or 0)
            prot_calc = float((t.get("macros") or {}).get("protein_g") or 0)
        except Exception:
            pass

    target_kcal = int(round(kcal_plan or kcal_hp or kcal_calc or 0))
    target_prot = int(round(prot_plan or prot_hp or prot_calc or 0))

    # Fallback para proteína basado en peso (1.6 g/kg) si no hay meta explícita
    if target_prot <= 0 and hp.get("weight"):
        try:
            w_lb = float(hp["weight"])
            w_kg = w_lb * 0.453592 if hp.get("weightUnit", "lb").lower() == "lb" else w_lb
            target_prot = int(round(w_kg * 1.6))
        except Exception:
            pass

    return target_kcal, target_prot


def evaluar_brecha_macros_cena(
    consumed: Optional[List[Dict[str, Any]]],
    health: Optional[Dict[str, Any]],
    plan_vigente: Optional[Dict[str, Any]] = None,
) -> MacroGapResult:
    """Evalúa la brecha de macros del usuario para el tick proactivo de las 7:00 PM (Cena)."""
    filas = [m for m in (consumed or []) if isinstance(m, dict)]
    consumed_kcal = int(round(sum(float(m.get("calories") or 0) for m in filas)))
    consumed_prot = int(round(sum(float(m.get("protein") or 0) for m in filas)))

    target_kcal, target_prot = _obtener_metas(health, plan_vigente)

    # Sin metas calóricas confiables: degradar suavemente a modo estándar sin directivas restrictivas
    if target_kcal <= 0:
        return MacroGapResult(
            mode="desconocido",
            target_kcal=0,
            target_protein=target_prot,
            consumed_kcal=consumed_kcal,
            consumed_protein=consumed_prot,
            remaining_kcal=0,
            remaining_protein=0,
            pct_kcal_consumed=0.0,
            instruccion_prompt="",
            alimentos_magros_filtrados=[],
        )

    remaining_kcal = target_kcal - consumed_kcal
    remaining_prot = max(0, target_prot - consumed_prot) if target_prot > 0 else 0
    pct_consumed = consumed_kcal / float(target_kcal) if target_kcal > 0 else 0.0

    consumidos_set = _extraer_alimentos_consumidos(filas)
    magros_filtrados = _filtrar_fuentes_magras(consumidos_set)

    pct_prot_consumed = consumed_prot / float(target_prot) if target_prot > 0 else 1.0

    # ─────────────────────────────────────────────────────────────
    # CASO 1A [BB-029]: Micro-Rescate Proteico (kcal >= 92% y proteína <= 65%)
    # Colisión BB-020 vs BB-024: el presupuesto calórico está casi lleno pero hay déficit proteico severo.
    # Prohibido felicitar y mandar a dormir con agua (riesgo muscular); prohibido sugerir cena pesada (rompería 110%).
    # Sugiérele exclusivamente un micro-rescate ultra-aislado de ~100-120 kcal (25 g prot limpia en claras o whey en agua).
    # ─────────────────────────────────────────────────────────────
    if (pct_consumed >= 0.92 or remaining_kcal <= 120) and pct_prot_consumed <= 0.65 and target_prot > 0:
        micro_magros = _filtrar_fuentes_micro_rescate(consumidos_set)
        sug_txt = ", ".join(micro_magros[:2])
        prompt_txt = (
            f"\n⚡ REGLA OBLIGATORIA DE MICRO-RESCATE PROTEICO [BB-029] (Lleva {consumed_kcal} de {target_kcal} kcal · {pct_consumed:.0%}, pero solo {consumed_prot} de {target_prot} g de proteína · {pct_prot_consumed:.0%}):\n"
            f"- El presupuesto calórico diario está prácticamente agotado (le quedan solo ~{max(0, remaining_kcal)} kcal), pero su proteína está en déficit severo (faltan ~{remaining_prot} g).\n"
            "- TERMINANTEMENTE PROHIBIDO proponer platos calóricos, comidas pesadas, carbohidratos (arroz, mangú, pan, víveres) o grasas adicionales: rompería el 110% de calorías y perdería la Barra Verde Esmeralda (BB-024).\n"
            "- TERMINANTEMENTE PROHIBIDO enviar solo felicitación y agua: el usuario necesita salvar su requerimiento muscular antes de dormir.\n"
            f"- Acción obligatoria: adviértele con empatía que sus calorías están casi al tope pero su proteína quedó corta, y sugiérele EXCLUSIVAMENTE una opción ultra-aislada de ~100–120 kcal con ~25 g de proteína limpia (ej. {sug_txt}).\n"
            "- Explícale que esta opción suma apenas ~5% de calorías manteniéndolo dentro del margen de la barra verde (en déficit) mientras rescata su masa muscular."
        )
        return MacroGapResult(
            mode="micro_rescate_proteico",
            target_kcal=target_kcal,
            target_protein=target_prot,
            consumed_kcal=consumed_kcal,
            consumed_protein=consumed_prot,
            remaining_kcal=remaining_kcal,
            remaining_protein=remaining_prot,
            pct_kcal_consumed=pct_consumed,
            instruccion_prompt=prompt_txt,
            alimentos_magros_filtrados=micro_magros,
        )

    # ─────────────────────────────────────────────────────────────
    # CASO 1B: Cierre Limpio e Hidratación (kcal >= 95% o <= 100 kcal restantes, CON proteína adecuada)
    # ─────────────────────────────────────────────────────────────
    if pct_consumed >= 0.95 or remaining_kcal <= 100:
        prompt_txt = (
            f"\n🌙 REGLA OBLIGATORIA DE CIERRE DE DÍA / HIDRATACIÓN (Lleva {consumed_kcal} de {target_kcal} kcal · {pct_consumed:.0%}):\n"
            f"- El usuario ya cubrió prácticamente la totalidad de su presupuesto calórico diario (le quedan solo ~{max(0, remaining_kcal)} kcal) con proteína adecuada ({consumed_prot} de {target_prot} g · {pct_prot_consumed:.0%}).\n"
            "- TERMINANTEMENTE PROHIBIDO proponer platos calóricos, comidas pesadas, recetas completas o empujarle a cenar comida sólida.\n"
            "- Mensaje adecuado: salúdalo con calidez, felicítale por su jornada y pregúntale cómo se siente. Si tiene un apetito o antojo ligero antes de dormir, "
            "sugiérele exclusivamente un vaso de agua, una infusión o un té caliente sin azúcar, o anímale a descansar con satisfacción."
        )
        return MacroGapResult(
            mode="cierre_hidratacion",
            target_kcal=target_kcal,
            target_protein=target_prot,
            consumed_kcal=consumed_kcal,
            consumed_protein=consumed_prot,
            remaining_kcal=remaining_kcal,
            remaining_protein=remaining_prot,
            pct_kcal_consumed=pct_consumed,
            instruccion_prompt=prompt_txt,
            alimentos_magros_filtrados=[],
        )

    # ─────────────────────────────────────────────────────────────
    # CASO 2: Rescate de Proteína Magra (pocas kcal < 350, pero mucha proteína pendiente >= 25g)
    # ─────────────────────────────────────────────────────────────
    if remaining_kcal < 350 and remaining_prot >= 25:
        sug_txt = ", ".join(magros_filtrados[:3])
        prompt_txt = (
            f"\n⚡ REGLA OBLIGATORIA DE PROTEÍNA MAGRA PURA (Quedan ~{remaining_kcal} kcal y faltan ~{remaining_prot} g de proteína):\n"
            f"- Al usuario le queda muy poco margen calórico (~{remaining_kcal} kcal) pero todavía tiene una brecha alta de proteína (~{remaining_prot} g).\n"
            "- PROHIBIDO sugerir carbohidratos densos (arroz, mangú, víveres, pan, avena, pasta) o grasas adicionales (aguacate, queso amarillo, frituras).\n"
            f"- Sugiere EXCLUSIVAMENTE una opción de proteína magra de alta densidad que no consumió hoy (ej. {sug_txt}).\n"
            "- Explícale brevemente que esta opción le permite cerrar su proteína limpia sin sobrepasar sus calorías del día."
        )
        return MacroGapResult(
            mode="proteina_magra",
            target_kcal=target_kcal,
            target_protein=target_prot,
            consumed_kcal=consumed_kcal,
            consumed_protein=consumed_prot,
            remaining_kcal=remaining_kcal,
            remaining_protein=remaining_prot,
            pct_kcal_consumed=pct_consumed,
            instruccion_prompt=prompt_txt,
            alimentos_magros_filtrados=magros_filtrados,
        )

    # ─────────────────────────────────────────────────────────────
    # CASO 3: Déficit Amplio (>= 500 kcal y >= 35g de proteína)
    # ─────────────────────────────────────────────────────────────
    if remaining_kcal >= 500 and remaining_prot >= 35:
        prompt_txt = (
            f"\n🍽️ CENA COMPLETA DE ALTA PROTEÍNA (Quedan ~{remaining_kcal} kcal y faltan ~{remaining_prot} g de proteína):\n"
            "- El usuario tiene un presupuesto amplio disponible para cerrar su meta del día.\n"
            "- Sugiérele una cena completa, nutritiva y satisfactoria con una fuente de proteína generosa "
            "acompañada de vegetales y carbohidratos saludables moderados, sin repetir lo consumido hoy."
        )
        return MacroGapResult(
            mode="cena_completa",
            target_kcal=target_kcal,
            target_protein=target_prot,
            consumed_kcal=consumed_kcal,
            consumed_protein=consumed_prot,
            remaining_kcal=remaining_kcal,
            remaining_protein=remaining_prot,
            pct_kcal_consumed=pct_consumed,
            instruccion_prompt=prompt_txt,
            alimentos_magros_filtrados=magros_filtrados,
        )

    # ─────────────────────────────────────────────────────────────
    # CASO 4: Estándar Equilibrado
    # ─────────────────────────────────────────────────────────────
    prompt_txt = (
        f"\n🥗 PRESUPUESTO PARA LA CENA: Quedan ~{remaining_kcal} kcal"
        + (f" y ~{remaining_prot} g de proteína" if remaining_prot > 0 else " (proteína ya cubierta)")
        + ". Sugiere una cena ligera y balanceada dentro de este margen, cuidando la variedad culinaria."
    )
    return MacroGapResult(
        mode="estandar",
        target_kcal=target_kcal,
        target_protein=target_prot,
        consumed_kcal=consumed_kcal,
        consumed_protein=consumed_prot,
        remaining_kcal=remaining_kcal,
        remaining_protein=remaining_prot,
        pct_kcal_consumed=pct_consumed,
        instruccion_prompt=prompt_txt,
        alimentos_magros_filtrados=magros_filtrados,
    )
