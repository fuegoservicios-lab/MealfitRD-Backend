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
# [P1-PLAN-LOTE-183] + los frutos secos y las semillas, «sin sal» (rd12, HTA: «el maní fileteado y la mantequilla de maní
# no están especificados como sin sal»). tooltip-anchor: P1-PLAN-LOTE-183-SIN-SAL
_FRUTO_SECO_HTA = re.compile(r"\b(?:mantequilla de man[ií]|man[ií]|almendras?|nueces|nuez(?!\s+moscada)|merey|pistachos?|"
                             r"semillas? de (?:girasol|calabaza|auyama))\b[^,;()]*", re.IGNORECASE)
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
    return _sufijo(_sufijo(s, _LATA_HTA, sufijo), _FRUTO_SECO_HTA, " sin sal")   # [P1-PLAN-LOTE-183]


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


# [P1-PLAN-LOTE-182 · 2026-09-23] Habichuelas SECAS: el revisor rechazó CRÍTICO un plan de embarazo por «habichuelas rojas
# secas sin indicar que deben hervirse al menos 10 minutos» (fitohemaglutinina: crudas o a medio cocer son tóxicas, para
# cualquiera). La nota de embarazo que ya existía decía «hasta que estén tiernas» y se omitía si la receta nombraba el
# remojo. Ésta va a todo plato con una habichuela declarada seca, para todos, salvo que la receta ya diga los 10 minutos.
# tooltip-anchor: P1-PLAN-LOTE-182-HABICHUELAS-SECAS
_LEGUMBRE_SECA = re.compile(r"\b(?:habichuelas?|frijol(?:es)?|jud[ií]as?|alubias?|porotos?)\b[^,;()]*\bsec[oa]s?\b",
                            re.IGNORECASE)
_YA_HIERVE_10 = re.compile(r"\bhi[eé]rv\w*[^.]{0,80}?\b(?:10|diez)\s*min", re.IGNORECASE)
_NOTA_HABICHUELAS = ("⚠️ Seguridad alimentaria: remoja las habichuelas secas y hiérvelas a fuego fuerte al menos 10 minutos "
                     "antes de bajar el fuego; crudas o a medio cocer son tóxicas.")


def _nota_habichuelas_secas(plan: dict) -> int:
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            if not any(isinstance(x, str) and _LEGUMBRE_SECA.search(x) for x in (m.get("ingredients") or [])):
                continue
            pasos = m.get("recipe")
            if not isinstance(pasos, list) or any(_YA_HIERVE_10.search(str(p)) for p in pasos):
                continue
            pasos.append(_NOTA_HABICHUELAS)
            m.pop("_display", None)
            tocadas += 1
    return tocadas


# [P1-PLAN-LOTE-210 · 2026-09-24] Alergia a MARISCOS sin alergia a PESCADO: el plan sirve pescado de aleta (es lo que el
# usuario pidió al marcar sólo el chip «Mariscos», que es distinto del chip «Pescado»), pero el revisor médico preguntó
# «confirmar si incluye pescado» en 3 de 4 corridas de ese perfil. La duda se resuelve ESCRITA en el plato: para el
# revisor y para el usuario, que puede marcar también «Pescado» si le hace reacción. Quien declara pescado (o «seafood»,
# que desde este lote cubre las dos clases) no recibe pescado y no necesita la nota. tooltip-anchor: P1-PLAN-LOTE-210-MARISCOS
_NOTA_MARISCOS = ("⚕️ Alergia a mariscos: el pescado de aleta (tilapia, mero, atún, sardina…) no es un marisco y tu "
                  "perfil no lo excluye. Si también te hace reacción, márcalo en tus alergias y lo quitamos.")


def _solo_mariscos(form_data) -> bool:
    fd = form_data if isinstance(form_data, dict) else {}
    decl = []
    for k in ("allergies", "otherAllergies"):
        v = fd.get(k)
        decl.extend(v if isinstance(v, list) else ([v] if isinstance(v, str) and v.strip() else []))
    if not decl:
        return False
    try:
        import graph_orchestrator as go
        exp = go._expand_allergy_declarations(decl)
    except Exception:                                                          # noqa: BLE001
        return False
    return _clase_declarada(exp, "mariscos") and not _clase_declarada(exp, "pescado")


def _clase_declarada(expansion: set, clase: str) -> bool:
    """[P1-PLAN-LOTE-249 · 2026-09-25] ¿La declaración cubre la CLASE entera? Una declaración de la clase (o de uno de
    sus miembros) se expande a TODOS sus sinónimos; un sinónimo compartido no la declara: «bacalaítos» vive en gluten
    (la masa) Y en pescado, y el alérgico al gluten recibía «Aclaración del alergia al pescado» y el revisor le
    rechazaba el pescado. tooltip-anchor: P1-PLAN-LOTE-249-CLASE-DECLARADA"""
    try:
        import graph_orchestrator as go
        from constants import strip_accents
        miembros = {strip_accents(str(t)).lower() for t in go._ALLERGEN_SYNONYMS[clase]}
    except Exception:                                                          # noqa: BLE001
        return False
    return bool(miembros) and miembros <= set(expansion or ())


def _pescado_sin_mariscos(form_data) -> str:
    """[P1-PLAN-LOTE-227 · 2026-09-25] «alergia»/«rechazo» cuando el usuario excluyó el PESCADO pero no los mariscos
    (chips distintos del formulario; lote 210: el marisco no es pescado), «» si no. Lo lee el revisor.
    tooltip-anchor: P1-PLAN-LOTE-227-PESCADO-NO-ES-MARISCO"""
    fd = form_data if isinstance(form_data, dict) else {}

    def _decl(*keys):
        out = []
        for k in keys:
            v = fd.get(k)
            out.extend(v if isinstance(v, list) else ([v] if isinstance(v, str) and v.strip() else []))
        return out
    try:
        import graph_orchestrator as go
        alergia = go._expand_allergy_declarations(_decl("allergies", "otherAllergies"))
        rechazo = set(__import__("rechazos").terminos_de_rechazo({"dislikes": _decl("dislikes", "otherDislikes")}))  # [P1-PLAN-LOTE-258]
    except Exception:                                                          # noqa: BLE001
        return ""
    if _clase_declarada(alergia, "mariscos") or _clase_declarada(rechazo, "mariscos"):
        return ""
    if _clase_declarada(alergia, "pescado"):
        return "alergia"
    if _clase_declarada(rechazo, "pescado"):
        return "rechazo"
    return ""


def notas_para_el_revisor(form_data) -> tuple:
    """[P1-PLAN-LOTE-227 · fuera del grafo en P1-PLAN-LOTE-240] (nota de mariscos, nota de pescado) que el revisor lee
    junto a las alergias y a los rechazos. Cadena vacía cuando no aplica."""
    mariscos = ""
    if _solo_mariscos(form_data):
        mariscos = ("\nAclaración de la alergia a mariscos: son crustáceos y moluscos (camarón, langosta, cangrejo, "
                    "pulpo, calamar, mejillón, lambí). El pescado de aleta (tilapia, mero, atún, sardina, salmón…) NO "
                    "es un marisco, y este usuario NO marcó la alergia a pescado, que tiene su propia opción en el "
                    "formulario: el pescado está PERMITIDO. No lo rechaces ni pidas aclararlo por esta alergia.")
    pescado = ""
    _pn = _pescado_sin_mariscos(form_data)
    if _pn:
        pescado = (f"\nAclaración del {_pn} al pescado: excluye el pescado de aleta (tilapia, mero, atún, sardina, "
                   "salmón, bacalao…). Los MARISCOS (camarón, langosta, cangrejo, pulpo, calamar, lambí) NO son pescado "
                   "y este usuario NO los marcó, que tienen su propia opción en el formulario: están PERMITIDOS. No los "
                   f"rechaces por este {_pn}.")
    return mariscos, pescado


def _pescado_rx():
    import graph_orchestrator as go
    terminos = sorted(set(go._ALLERGEN_SYNONYMS["pescado"]) - set(go._ALLERGEN_SYNONYMS["mariscos"]), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terminos) + r")s?\b", re.IGNORECASE)


def _nota_mariscos_no_pescado(plan: dict, form_data) -> int:
    if not _solo_mariscos(form_data):
        return 0
    try:
        from constants import strip_accents
        rx = _pescado_rx()
    except Exception:                                                          # noqa: BLE001
        return 0
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            if not any(isinstance(x, str) and rx.search(strip_accents(x)) for x in (m.get("ingredients") or [])):
                continue
            pasos = m.get("recipe")
            if not isinstance(pasos, list) or any("Alergia a mariscos:" in str(p) for p in pasos):
                continue
            pasos.append(_NOTA_MARISCOS)
            m.pop("_display", None)
            tocadas += 1
    return tocadas


def _nota_embarazo_recalculada(plan: dict, form_data) -> int:
    """[P1-PLAN-LOTE-227 · 2026-09-25] La nota combinada de embarazo/lactancia se calcula en la capa clínica, y DESPUÉS
    el tope de pescado del embarazo (lote 187) cambia el pescado por pollo/pavo/res: la nota seguía diciendo «cocina el
    pescado y los mariscos» y no decía nada de la carne. El revisor lo leyó así y rechazó CRÍTICO «el pollo no indica
    74 °C; la nota solo menciona pescado y mariscos» (2 de 3 corridas reales de embarazo, 25-sep). La nota se recalcula
    donde se etiqueta: al entrar al revisor y en el escudo, después del tope. tooltip-anchor: P1-PLAN-LOTE-227-NOTA-EMBARAZO"""
    try:
        import graph_orchestrator as go
        return int(go._apply_pregnancy_food_safety_annotations(plan, form_data) or 0)
    except Exception:                                                          # noqa: BLE001
        return 0


def etiquetar(plan: dict, form_data) -> int:
    """Devuelve cuántas comidas tocó (sumando condiciones). Muta `plan`."""
    if not (enabled() and isinstance(plan, dict)):
        return 0
    n = _nota_ceviche_de_carne(plan)                        # [P1-PLAN-LOTE-180] para todos, antes de las condiciones
    n += _nota_habichuelas_secas(plan)                      # [P1-PLAN-LOTE-182] ídem
    n += _nota_mariscos_no_pescado(plan, form_data)        # [P1-PLAN-LOTE-210] mariscos ≠ pescado, escrito
    n += _nota_embarazo_recalculada(plan, form_data)       # [P1-PLAN-LOTE-227] la nota describe el plato FINAL
    reglas = _reglas(form_data)
    if not reglas:
        return n
    if "pregnancy" in reglas:
        try:
            import embarazo_seguro
            n += embarazo_seguro.etiquetar(plan, form_data)
            # [P1-PLAN-LOTE-193] el tope semanal de pescado (187) otra vez aquí, al entrar al revisor: rd19 llegó con
            # 525 g porque los cerradores escalan el pescado DESPUÉS de la sustitución por condición. Idempotente.
            n += __import__("embarazo_pescado").limitar_pescado(plan, form_data)
        except Exception as e:                                                 # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-173] etiquetas de embarazo no-op: {type(e).__name__}: {e}")
    if "dm2" in reglas:
        n += __import__("dm2_seguro").nota_casabe(plan, form_data)   # [P1-PLAN-LOTE-195] lo que el revisor lee del casabe
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
