# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-173 · 2026-09-23] Lo que el revisor médico tiene que LEER en el plan para aprobarlo, por condición.

El revisor médico (un LLM) rechaza con severidad CRÍTICA cuando el plan no DICE lo que la condición exige, aunque el
alimento sea el correcto: cada rechazo cuesta un intento entero (2-3 min y su gasto) y, si reincide, el usuario
recibe el plan de emergencia. Batería real del 23-sep:
  · embarazo — «pescado sin especie», «queso/leche sin pasteurizar» (→ `embarazo_seguro`, lote 172-173);
  · hipertensión — «queso fresco y atún sin especificar versiones bajas en sodio».

Este módulo es la puerta única: `etiquetar(plan, form_data)` aplica, según las reglas activas del perfil
(`condition_rules.detect_active_rules`), las etiquetas de cada condición. Se llama desde la sustitución clínica
(`graph_orchestrator._apply_condition_substitutions`) y al final del escudo (`db_plans._finalize_plan_data_for_insert`),
porque los cerradores añaden alimentos DESPUÉS de la sustitución. Idempotente. Knob `MEALFIT_CLINICAL_LABELS` (True).
tooltip-anchor: P1-PLAN-LOTE-173-ETIQUETAS-CLINICAS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# HTA: el queso fresco/blando y el pescado en lata, «bajo en sodio». Los curados (parmesano, cheddar…) ya son otra
# conversación (el revisor pide moderarlos, no etiquetarlos).
_QUESO_HTA = re.compile(r"\b(?:queso(?!\s+(?:cheddar|parmesano|gouda|provolone|edam|de\s+papa|de\s+bola|amarillo|suizo|"
                        r"manchego|curado|azul))|ricotta|cottage|reques[oó]n|mozzarella)\b[^,;()]*", re.IGNORECASE)
# [P1-PLAN-LOTE-175] + el palmito en conserva (batería real, HTA: «250 g de palmito, si es en conserva… que se enjuague»).
_LATA_HTA = re.compile(r"\b(?:at[uú]n|sardinas?|palmitos?)\b[^,;()]*", re.IGNORECASE)
_YA_BAJO = re.compile(r"bajo\s+en\s+sodio|baja\s+en\s+sodio|sin\s+sal|sin\s+sodio|reducid[oa]\s+en\s+sodio", re.IGNORECASE)


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLINICAL_LABELS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _reglas(form_data) -> set:
    try:
        from condition_rules import detect_active_rules
        return {getattr(r, "id", "") for r in detect_active_rules(form_data or {})}
    except Exception:                                                          # noqa: BLE001
        return set()


def _sufijo(s: str, rx, sufijo: str) -> str:
    if _YA_BAJO.search(s):
        return s
    m = rx.search(s)
    if not m:
        return s
    fin = m.end()
    while fin > m.start() and s[fin - 1] == " ":
        fin -= 1
    return s[:fin] + sufijo + s[fin:]


def _linea_hta(s: str) -> str:
    s = _sufijo(s, _QUESO_HTA, " bajo en sodio")
    sufijo = (" bajas en sodio" if re.search(r"\bsardinas\b", s, re.IGNORECASE)
              else " bajos en sodio" if re.search(r"\bpalmitos\b", s, re.IGNORECASE) else " bajo en sodio")
    return _sufijo(s, _LATA_HTA, sufijo)


def _etiquetar_hta(plan: dict) -> int:
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            cambio = False
            for campo in ("ingredients", "ingredients_raw"):
                lineas = m.get(campo)
                if not isinstance(lineas, list):
                    continue
                nuevas = [(_linea_hta(x) if isinstance(x, str) else x) for x in lineas]
                if nuevas != lineas:
                    m[campo] = nuevas
                    cambio = True
            if cambio:
                m.pop("_display", None)
                m["_hta_labels"] = True
                tocadas += 1
    return tocadas


# [P1-PLAN-LOTE-180 · 2026-09-23] «Ceviche» de CARNE: el revisor rechazó CRÍTICO «el almuerzo del día 3 se describe como
# preparado con pollo crudo al estilo ceviche» (batería real, HTA) — y las recetas guardadas de «ceviche de pollo» SÍ
# cocinan el pollo («verifica la pechuga a 74 °C»), pero el revisor sólo lee nombre, ingredientes y notas de seguridad.
# Para todo usuario (no es una condición): la nota que el resumen del revisor copia, y que además es la instrucción
# correcta. El marisco ya tiene la suya (P1-SEAFOOD-MARINADE-BLANCH). tooltip-anchor: P1-PLAN-LOTE-180-CEVICHE-DE-CARNE
_CEVICHE = re.compile(r"\b(?:ceviche|cebiche)\b", re.IGNORECASE)
_CARNE = re.compile(r"\b(?:pollo|pechugas?|pavo|cerdo|res|carne|chivo)\b", re.IGNORECASE)
_NOTA_CEVICHE = ("⚠️ Seguridad alimentaria: cocina la carne por completo (74 °C por dentro, sin partes rosadas) ANTES de "
                 "marinarla en el limón; el cítrico sólo da sabor, no la cuece.")


def _nota_ceviche_de_carne(plan: dict) -> int:
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict) or not _CEVICHE.search(str(m.get("name") or "")):
                continue
            if not any(isinstance(x, str) and _CARNE.search(x) for x in (m.get("ingredients") or [])):
                continue
            pasos = m.get("recipe")
            if not isinstance(pasos, list) or any("el cítrico sólo da sabor" in str(p) for p in pasos):
                continue
            pasos.append(_NOTA_CEVICHE)
            m.pop("_display", None)
            tocadas += 1
    return tocadas


def etiquetar(plan: dict, form_data) -> int:
    """Devuelve cuántas comidas tocó (sumando condiciones). Muta `plan`."""
    if not (enabled() and isinstance(plan, dict)):
        return 0
    n = _nota_ceviche_de_carne(plan)                        # [P1-PLAN-LOTE-180] para todos, antes de las condiciones
    reglas = _reglas(form_data)
    if not reglas:
        return n
    if "pregnancy" in reglas:
        try:
            import embarazo_seguro
            n += embarazo_seguro.etiquetar(plan, form_data)
        except Exception as e:                                                 # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-173] etiquetas de embarazo no-op: {type(e).__name__}: {e}")
    if "hta" in reglas or "renal" in reglas:
        n_hta = _etiquetar_hta(plan)
        if n_hta:
            logger.info(f"🧂 [P1-PLAN-LOTE-173] HTA: «bajo en sodio» escrito en {n_hta} comida(s)")
        n += n_hta
    return n


def etiquetar_antes_del_revisor(plan, form_data) -> int:
    """[P1-PLAN-LOTE-175 · 2026-09-23] La misma puerta, llamada al ENTRAR en `review_plan_node`, y que no lanza nunca.

    Batería real del 23-sep: embarazo y lactancia perdieron un intento entero (~4 min) —y embarazo, el plan: segundo
    rechazo CRÍTICO y plan de emergencia— porque el revisor leyó «queso blanco fresco» y «ricotta» sin «pasteurizado».
    La etiqueta ya corría en la sustitución clínica, pero después de ella los cerradores y el re-renderizado de las
    líneas vuelven a escribir el alimento sin la palabra. Lo que el revisor lee es lo que hay al entrar en su nodo: ahí
    va la etiqueta. tooltip-anchor: P1-PLAN-LOTE-175-ANTES-DEL-REVISOR"""
    try:
        return etiquetar(plan, form_data)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-175] etiquetas antes del revisor no-op: {type(e).__name__}: {e}")
        return 0


def plan_etiquetado(plan, form_data):
    """El mismo plan, etiquetado (in situ): la forma que cabe en la primera línea del nodo revisor."""
    etiquetar_antes_del_revisor(plan, form_data)
    return plan
