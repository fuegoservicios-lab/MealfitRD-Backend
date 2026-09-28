# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-26 · 2026-09-12] C5 (primera parte) del plan de pendientes: cultura, horario, equipo, tiempo y básicos
declarados como CONTEXTO del plato — no como dogma ni como dato que se finge.

Tres gaps del backlog culinario (CUL-P1-01, CUL-P1-03, CUL-P1-05) comparten una misma forma: el motor tenía una regla
GENERAL escrita en un sitio (un tope de claras, «arroz nunca de noche», «15 min» de preparación) y la PERSONA había
declarado algo que la contradecía o la completaba (10 claras como básico, arroz en la cena como básico, remojo de una
noche, ninguna licuadora). Este módulo es la única lectura de esas declaraciones para las cinco superficies que las
necesitan; cada una engancha con dos o tres líneas y falla abierta.

## Lo medido antes de escribir (flota + corpus fijo + biblioteca, 2026-09-12)

· Claras entregadas en la flota: 1, 2, 3 y 5 por comida; ningún plan declara la clara como básico; 5 de 6 planes
  vivos ya llevan la política EN VIGOR (`_plan_policy.enforced`), así que el tope por ración (E5) está activo. Lo que
  seguía ciego a la declaración: el techo de COMIDAS con huevo (`max(3, 25 %)` en el gate de variedad y en su autofix)
  y el conteo, que sumaba «1 clara» de aglutinante como otra comida de huevo.
· Tiempos: 5 de 64 comidas del corpus declaran 15-20 min y sus pasos piden horas («marina 2 horas», «la noche
  anterior», «24 horas» — una de ellas es conservación, no espera; se distingue).
· Equipo: el formulario principal NO lo pregunta (decisión de producto `P2-FORM-KITCHEN-EQUIPMENT`, 2026-06-22); sí lo
  captura el panel opt-in de Súper Personalización (`kitchenEquipment`, etiquetas «Horno», «Airfryer», «Licuadora»…)
  y hasta hoy sólo lo leía el prompt del plan. 10 de 64 comidas del corpus y 53 de 193 recetas congeladas piden horno.
· Horario: la rúbrica del juez llevaba una prohibición LITERAL («arroz/pasta como base NUNCA en desayuno ni cena») que
  los países beta heredaban intacta y que ignoraba al usuario que pidió arroz de noche como básico. El gate
  determinista ya era `soft` en la cena y la pasta de noche ya era legítima (decisión 2026-06-27).

## Qué decide este módulo

· `declared_equipment` / `missing_equipment` / `equipment_block`: el equipo declarado (o `None` si no se declaró: se
  dice, no se finge) frente al que los pasos exigen; estufa/sartén/olla se asumen siempre.
· `hidden_wait_minutes` / `check_hidden_time`: horas de espera en los pasos (remojo, marinado, «la noche anterior»)
  que el `prep_time` del plato no cuenta; la conservación («dura 3 días», «consume dentro de 24 horas») no es espera.
· `slot_preference_exempt` / `rice_staple_for_slot`: un básico declarado PARA esa franja no es una violación de
  horario — lo típico (la guía positiva), lo preferido (la declaración) y lo prohibido (una restricción explícita)
  son tres cosas distintas.
· `egg_meal_cap` / `egg_is_binder`: el techo de comidas con huevo honra al básico declarado (una comida por día y
  franja pedida, nunca menos que el de siempre) y una clara de aglutinante no cuenta como plato de huevo.
· `judge_context`: lo que el juez debe saber de la persona antes de acusar (básicos, cocinas, equipo, tiempo).

Knob `MEALFIT_CULINARY_CONTEXT` (default True): apagado, cada enganche devuelve la conducta anterior. Los dos checks
nuevos del escáner (V8a tiempo oculto, V8b equipo no disponible) son de aviso y no dependen del knob.
Puro: sin base ni red; nunca lanza. tooltip-anchor: P1-PLAN-LOTE-26-CULINARY-CONTEXT
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Optional

_STOP = frozenset({"de", "del", "la", "el", "los", "las", "en", "con", "sin", "y", "a", "al", "o", "u", "para", "por"})


def _norm(s) -> str:
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


def context_enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CULINARY_CONTEXT", True)
    except Exception:
        return True


def _superpers(form_data) -> dict:
    """El bloque de Súper Personalización, venga plano, anidado en `health_profile` o ya hidratado en `form_data`."""
    f = form_data if isinstance(form_data, dict) else {}
    sp = f.get("super_personalization")
    if not isinstance(sp, dict):
        hp = f.get("health_profile")
        sp = hp.get("super_personalization") if isinstance(hp, dict) else None
    return sp if isinstance(sp, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Equipo de cocina
# ─────────────────────────────────────────────────────────────────────────────────────────────

#: clave canónica → (etiquetas del panel, normalizadas · regex sobre los pasos normalizados)
EQUIPO = {
    "horno": (("horno",), re.compile(r"\bhorn(?:o|ea|ear|eado|eada|eados|eadas|eandolos?|eandolas?)\b|\bgratin")),
    "airfryer": (("airfryer", "air fryer", "freidora de aire"), re.compile(r"\bair ?fryer\b|freidora de aire")),
    "microondas": (("microondas",), re.compile(r"\bmicroondas\b")),
    "licuadora": (("licuadora",), re.compile(r"\blicu(?:a|as|ar|ado|ada|ados|adas|adora|alo|ala|alos|alas)\b")),
    "batidora": (("batidora",), re.compile(r"\bbatidora\b")),
    "olla_presion": (("olla de presion",), re.compile(r"olla de presion|\ba presion\b|olla express")),
    "olla_arrocera": (("olla arrocera", "arrocera"), re.compile(r"\barrocera\b")),
    "parrilla": (("parrilla", "bbq"), re.compile(r"\bparrilla\b")),
    "sandwichera": (("sandwichera",), re.compile(r"\bsandwichera\b")),
}
#: lo que toda cocina tiene: no se comprueba (declararlo o no da igual)
EQUIPO_BASE = frozenset({"estufa", "sarten", "caldero", "olla", "sarten caldero"})
_EQUIPO_LABELS = {"horno": "horno", "airfryer": "airfryer", "microondas": "microondas", "licuadora": "licuadora",
                  "batidora": "batidora", "olla_presion": "olla de presión", "olla_arrocera": "olla arrocera",
                  "parrilla": "parrilla", "sandwichera": "sandwichera"}


def _equipo_canon(etiqueta) -> Optional[str]:
    n = _norm(etiqueta)
    if not n:
        return None
    for clave, (labels, _rx) in EQUIPO.items():
        if n in labels or any(n == l for l in labels):
            return clave
    for clave, (labels, _rx) in EQUIPO.items():
        if any(l in n for l in labels):
            return clave
    return None


def declared_equipment(form_data) -> Optional[set]:
    """Claves canónicas del equipo declarado; `None` si la persona NO lo declaró (el panel es opcional).

    Lee `kitchenEquipment` plano (algunos transportes lo hidratan) o dentro de Súper Personalización. Una lista vacía
    declarada se trata como no declarada: nadie cocina sin nada."""
    try:
        f = form_data if isinstance(form_data, dict) else {}
        raw = f.get("kitchenEquipment")
        if not isinstance(raw, list):
            raw = _superpers(f).get("kitchenEquipment")
        if not isinstance(raw, list) or not raw:
            return None
        out = set()
        for x in raw:
            c = _equipo_canon(x)
            if c:
                out.add(c)
        return out if out or any(_norm(x) in EQUIPO_BASE for x in raw) else None
    except Exception:
        return None


def declared_equipment_labels(form_data) -> list:
    """Etiquetas legibles («horno», «licuadora») del equipo declarado, para el prompt. `[]` si no se declaró."""
    d = declared_equipment(form_data)
    return sorted(_EQUIPO_LABELS[k] for k in d) if d else []


def equipment_required(pasos) -> set:
    """Claves canónicas del equipo que los pasos EXIGEN (por su texto)."""
    txt = _norm(" . ".join(str(p) for p in (pasos or []))) if isinstance(pasos, (list, tuple)) else _norm(pasos)
    return {clave for clave, (_l, rx) in EQUIPO.items() if rx.search(txt)}


def missing_equipment(meal, declared: Optional[set]) -> list:
    """Equipo que los pasos del plato exigen y la persona no declaró tener. `[]` si no declaró nada o si no falta."""
    try:
        if declared is None or not isinstance(meal, dict):
            return []
        req = equipment_required(meal.get("recipe") or [])
        return sorted(_EQUIPO_LABELS[k] for k in (req - set(declared)))
    except Exception:
        return []


def equipment_block(form_data) -> str:
    """Bloque del prompt del día con el equipo disponible; «» si no se declaró (prompt byte-idéntico)."""
    if not context_enabled():
        return ""
    labels = declared_equipment_labels(form_data)
    if not labels:
        return ""
    faltan = sorted(v for k, v in _EQUIPO_LABELS.items() if k not in (declared_equipment(form_data) or set()))
    aviso = f" NO tienes: {', '.join(faltan)} — no propongas técnicas que los necesiten." if faltan else ""
    return (f"\n• 🍳 EQUIPO DE COCINA DISPONIBLE (además de estufa, sartén y olla): {', '.join(labels)}.{aviso} "
            f"Un plato que pide horno sin horno no es una receta: es una promesa que no se puede cumplir.")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Tiempo oculto
# ─────────────────────────────────────────────────────────────────────────────────────────────

_RE_HORAS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:h\b|hrs?\b|horas?\b)")
_RE_NOCHE = re.compile(r"toda la noche|la noche anterior|desde la vispera|la vispera|de un dia para otro|"
                       r"durante la noche|toda una noche|de la noche a la manana")
#: una cláusula de CONSERVACIÓN habla del plato hecho, no de una espera para hacerlo. [P1-PLAN-LOTE-745 · ronda 1] Se
#: mira sobre el PASO entero, como siempre se miró de hecho (ver `_RE_CORTE_CLAUSULA`).
_RE_ALMACEN = re.compile(r"\bguard|\bconserv|\bdura\b|\bduran\b|se mantien|aguanta|refrigera(?:do|da)?\b.*\bhasta|"
                         r"consume(?:lo|la|los|las)? (?:dentro|antes)|dentro de las?\b|\bcongel|\bsobra")
# [P1-PLAN-LOTE-745 · 2026-09-28 · ronda 1] El LÍMITE de seguridad alimentaria y el PLAZO de conservación no son una
# espera: «no lo dejes a temperatura ambiente más de 2 horas», «refrigera lo que sobre dentro de 2 horas» (4 comidas
# del corpus de 426 planes). Se RECORTAN de la cláusula antes de contar horas, en vez de descartar la cláusula entera:
# «marina el pollo 3 horas en la nevera y refrigera lo que sobre dentro de 2 horas» sigue pidiendo sus 3 horas. Sólo el
# TOPE («… más de»): «reposa a temperatura ambiente 1 hora» es una espera y cuenta. Y «dentro de N horas» sólo es un
# plazo con un verbo de conservación delante: «cocínalo dentro de 24 horas» no conserva nada.
_H_NUM = r"\d+(?: \d+)?\s*(?:h|hrs?|horas?)\b"
_RE_PLAZO_CONSERVACION = re.compile(
    r"temperatura ambiente (?:por |durante )?mas de (?:las? )?" + _H_NUM
    + r"|\b(?:refriger|consum|guard|conserv|congel)\w*(?: \w+){0,5}? dentro de (?:las? )?" + _H_NUM)
# Las exclusiones NUEVAS del lote (fármaco, tope y plazo) se aplican por CLÁUSULA del paso CRUDO: `_norm` borra la
# puntuación, así que partir después de normalizar —lo que hacía el código— no parte nada, y la primera versión del
# lote callaba el paso entero («marina el pollo 2 horas; no lo dejes a temperatura ambiente más de 2 horas» salía 0). El
# punto y los dos puntos entre cifras («1.5 horas», «8:00») no parten.
#
# `_RE_ALMACEN` (la conservación de antes del lote) se queda sobre el PASO entero A PROPÓSITO: pasarlo a cláusula
# destapa, medido sobre los dos corpus, 9 + 135 comidas nuevas —todas la nota «💡 Cocción previa: remoja las habichuelas
# secas 8-12 h y hiérvelas 60-90 min (puedes … guardarla en la nevera hasta 4 días)», callada hoy por su «guarda». Si
# ese remojo de la legumbre SECA opcional es tiempo oculto del plato es una decisión del dueño, no de este lote.
_RE_CORTE_CLAUSULA = re.compile(r"(?<!\d)[.:]|[.:](?!\d)|[;()]")
_RE_ESPERA = re.compile(r"remoj|descongel|marin|adob|repos|macer|ferment|leud|enfri|refriger|cuaj|hidrat|dej")
_RE_MIN = re.compile(r"(\d{1,3})\s*min")
_NOCHE_MIN = 8 * 60
# [P1-PLAN-LOTE-745 · 2026-09-28] Una cláusula que habla de un FÁRMACO habla de separar la dosis, no de esperar a que
# el plato esté listo: «toma la levotiroxina en ayunas y separa estos alimentos … al menos 4 horas de la dosis». Sobre
# las 848 comidas recientes del 28-sep, 36 de los 37 V8a eran esa nota clínica leída como 4 h de espera oculta (el
# único verdadero: la avena de «la noche anterior» con 5 min declarados, que sigue disparando). Es el mismo error que
# P1-PLAN-LOTE-229 cerró en el clamp de tiempos, visto desde el escáner. Y un paso que ABRE con «⚕» es la nota
# clínica entera (la marca de `_is_recipe_safety_note_step`): consejo médico, no cocina. La nota de seguridad (⚠) y la
# de cocción previa (💡) NO se saltan: un «remoja 8-12 h» ahí SÍ es tiempo del plato.
# tooltip-anchor: P1-PLAN-LOTE-745-V8A
# [ronda 1] Se mira por CLÁUSULA del paso crudo (`_RE_CORTE_CLAUSULA`): «remoja las habichuelas 8 horas; si tomas
# levotiroxina, separa este plato 4 horas de la dosis» sigue pidiendo su remojo. «pastilla» excluye la «pastilla de
# caldo», que es cocina.
_RE_MEDICACION = re.compile(r"\bdosis\b|levotiroxina|levothyroxin|\bmedicament|\bmedicin|\bpildora|"
                            r"\bpastilla(?!\s+de\s+caldo)|\bfarmaco|\bcapsula|\beutirox|\bsynthroid"
                            # el consejo de ESTILO DE VIDA tampoco es tiempo del plato: «evita acostarte durante las 2-3
                            # horas posteriores a la cena» (gastritis). La PERSONA que se acuesta, no el filete: «acuesta
                            # los filetes sobre la cebolla y marina 2 horas» es cocina y cuenta.
                            r"|\bacostarte\b|\bacostarse\b|\bte acuestes\b")
_NOTA_CLINICA = "⚕"   # ⚕ — prefijo de la nota clínica por condición
# Un `prep_time` que SUMA la espera ya la declaró: «10 min más reposo nocturno», «10 min (más refrigeración)», «10 min
# + refrigeración» (las formas del corpus de 426 planes son todas aditivas). `declared_prep_minutes` sólo lee minutos,
# así que sin esto se acusaba justo al plato que sí avisó. [ronda 1] Sólo la forma ADITIVA («+», «más» o «y» delante):
# «20 min (sin reposo)», «25 min (no requiere remojo)» o «40 min (guarda en la nevera)» NIEGAN o sólo mencionan la
# espera, y callarlos silenciaba esperas reales de 2-8 h que la base acusaba. El «+» se lee «más» antes de `_norm`,
# que lo borraría.
_RE_PREP_DECLARA_ESPERA = re.compile(
    r"(?<!\bsin )(?<!\bno )\b(?:mas|y)\s+(?:(?:el|la|los|las|un|una|de|del|en|toda|\d+|h|hrs?|horas?)\s+){0,3}"
    r"(?:repos|refriger|nocturn|vispera|remoj|marin|hidrat|ferment|leud|noche)")


def _es_nota_clinica(paso) -> bool:
    return str(paso or "").lstrip().startswith(_NOTA_CLINICA)


def hidden_wait_minutes(pasos) -> tuple:
    """`(minutos, evidencia)` de la espera MÁS larga que los pasos piden en horas (o «toda la noche»). `(0, "")` si
    ninguna. «marina 2 horas» cuenta; «dura 3 días en la nevera» y «consume dentro de 24 horas» son conservación y no
    cuentan (por PASO); una nota de medicación o el tope de seguridad no cuentan (por CLÁUSULA, ver `_RE_CORTE_CLAUSULA`);
    «hornea 1 hora» es cocción activa y sí cuenta (el reloj corre igual)."""
    mejor, evidencia = 0, ""
    for p in (pasos or []):
        if _es_nota_clinica(p):
            continue                                   # [P1-PLAN-LOTE-745] consejo médico, no un paso de cocina
        if _RE_ALMACEN.search(_norm(p)):
            continue                                   # conservación: por PASO, como antes (ver `_RE_CORTE_CLAUSULA`)
        for cruda in _RE_CORTE_CLAUSULA.split(str(p or "")):   # [P1-PLAN-LOTE-745 · ronda 1] partir ANTES de `_norm`
            c = _norm(cruda)
            if not c or _RE_MEDICACION.search(c):
                continue
            c_horas = _RE_PLAZO_CONSERVACION.sub(" ", c)   # el tope/plazo de conservación no es tiempo del plato
            mins = 0
            for m in _RE_HORAS.finditer(c_horas):
                try:
                    mins = max(mins, int(round(float(m.group(1).replace(",", ".")) * 60)))
                except ValueError:
                    continue
            if _RE_NOCHE.search(c_horas):
                mins = max(mins, _NOCHE_MIN)
            if mins > mejor:
                mejor, evidencia = mins, c[:120]
    return mejor, evidencia


def declared_prep_minutes(meal) -> Optional[int]:
    """Minutos que el plato DECLARA en `prep_time` («40 min», «1 h», «1 h 30 min»); `None` si no los declara."""
    try:
        txt = _norm((meal or {}).get("prep_time") or "")
        if not txt:
            return None
        total = 0
        m_h = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:h\b|hrs?\b|horas?\b)", txt)
        if m_h:
            total += int(round(float(m_h.group(1).replace(",", ".")) * 60))
        m_m = _RE_MIN.search(txt)
        if m_m:
            total += int(m_m.group(1))
        if not m_h and not m_m:
            m_n = re.search(r"^(\d{1,3})$", txt)
            total = int(m_n.group(1)) if m_n else 0
        return total or None
    except Exception:
        return None


def check_hidden_time(meal, *, minimo_min: int = 60) -> Optional[dict]:
    """`{"espera_min", "declarado_min", "evidencia"}` si los pasos piden una espera ≥ `minimo_min` que el `prep_time`
    del plato no cubre; `None` si no hay espera oculta. Un plato SIN `prep_time` con una espera larga también sale:
    nadie le dijo al usuario que empiece la víspera."""
    try:
        if not isinstance(meal, dict):
            return None
        espera, ev = hidden_wait_minutes(meal.get("recipe") or [])
        if espera < minimo_min:
            return None
        declarado = declared_prep_minutes(meal)
        if declarado is not None and declarado >= espera:
            return None
        if _RE_PREP_DECLARA_ESPERA.search(_norm(str(meal.get("prep_time") or "").replace("+", " mas "))):
            return None                                # [P1-PLAN-LOTE-745] el plato ya dice que reposa o remoja
        return {"espera_min": espera, "declarado_min": declarado, "evidencia": ev}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Básicos declarados y horario
# ─────────────────────────────────────────────────────────────────────────────────────────────

_SLOT_ALIASES = {"desayuno": "desayuno", "breakfast": "desayuno", "almuerzo": "almuerzo", "lunch": "almuerzo",
                 "comida": "almuerzo", "cena": "cena", "dinner": "cena", "merienda": "merienda", "snack": "merienda",
                 "merienda am": "merienda", "merienda pm": "merienda", "colacion": "merienda"}


def declared_staples(form_data) -> list:
    """`[{"name", "slots": [franjas canónicas] }]`: `stapleAnchors` con sus franjas y `stapleFoods` sin franja (= todas)."""
    out, vistos = [], set()
    try:
        f = form_data if isinstance(form_data, dict) else {}
        for item in (f.get("stapleAnchors") or []) if isinstance(f.get("stapleAnchors"), list) else []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            slots = []
            for sl in (item.get("slots") or []):
                canon = _SLOT_ALIASES.get(_norm(sl))
                if canon and canon not in slots:
                    slots.append(canon)
            n = str(item["name"]).strip()
            if _norm(n) not in vistos:
                vistos.add(_norm(n))
                out.append({"name": n, "slots": slots})
        raw = f.get("stapleFoods") if f.get("stapleFoods") is not None else f.get("staple_foods")
        for s in (raw or []) if isinstance(raw, list) else []:
            n = str(s or "").strip()
            if n and _norm(n) not in vistos:
                vistos.add(_norm(n))
                out.append({"name": n, "slots": []})
    except Exception:
        return out
    return out


def _tokens(nombre) -> set:
    return {t for t in _norm(nombre).split() if t not in _STOP and len(t) >= 3}


def slot_preference_exempt(rule_label: str, slot_key: str, form_data, rules_table: Optional[dict] = None) -> Optional[str]:
    """El nombre del básico declarado que EXIME esta violación de horario, o `None`.

    Exime cuando la persona declaró como básico un alimento de la REGLA (sus tokens: «arroz», «moro», «pasta»…) y
    lo pidió para ESA franja (o sin franja: para todas). «Arroz blanco» declarado para la cena convierte «arroz de
    noche» en lo que pidió; no toca las demás reglas del plato (un sopón con arroz sigue siendo sopón)."""
    try:
        if not context_enabled():
            return None
        tabla = rules_table
        if tabla is None:
            from constants import SLOT_INAPPROPRIATE_FOODS as tabla
        tokens_regla = set()
        for r in (tabla or {}).get(slot_key) or []:
            if r.get("label") == rule_label:
                tokens_regla = {_norm(t) for t in (r.get("tokens") or ())}
                break
        if not tokens_regla:
            return None
        for st in declared_staples(form_data):
            if st["slots"] and slot_key not in st["slots"]:
                continue
            if _tokens(st["name"]) & tokens_regla:
                return st["name"]
        return None
    except Exception:
        return None


def rice_staple_for_slot(form_data, slot_key: str = "cena") -> Optional[str]:
    """El básico de ARROZ declarado para esa franja (o sin franja), o `None`: quien lo pidió no recibe el autofix que se
    lo quita."""
    try:
        if not context_enabled():
            return None
        from constants import _SLOT_RICE_TOKENS
        toks = {_norm(t) for t in _SLOT_RICE_TOKENS}
        for st in declared_staples(form_data):
            if st["slots"] and slot_key not in st["slots"]:
                continue
            if _tokens(st["name"]) & toks:
                return st["name"]
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# El huevo: cuántas comidas y cuál cuenta
# ─────────────────────────────────────────────────────────────────────────────────────────────

_RE_HUEVO = re.compile(r"\b(huevos?|claras?|yemas?)\b")
_RE_CANT = re.compile(r"(\d+(?:[.,]\d+)?|[½¼¾])\s*(?:unidades?\s+(?:de\s+)?)?(huevos?|claras?|yemas?)\b")
_RE_MASA = re.compile(r"\b(masa|mezcla|rebozad|empaniz|apanad|croqueta|torta|tortita|panqueque|pancake|arepita|"
                      r"bollito|albondiga|hamburguesa|budin|bizcocho|magdalena|galleta|waffle|crepe|crepa|yaniqueque|"
                      r"pastelon|relleno|liga)")
_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75}


def egg_pieces(meal) -> float:
    """Piezas de huevo (enteros + claras + yemas) que la lista compra para el plato; 0 si no hay o no hay cifra."""
    total = 0.0
    for ing in (meal or {}).get("ingredients") or []:
        n = _norm(ing)
        for m in _RE_CANT.finditer(n):
            v = m.group(1)
            total += _FRAC.get(v, 0.0) if v in _FRAC else float(v.replace(",", "."))
    return total


def egg_is_binder(meal) -> bool:
    """True si el huevo del plato es AGLUTINANTE (una clara en una masa), no la proteína: no está en el NOMBRE, la
    lista compra ≤ 1 pieza, y el nombre o los pasos hablan de masa/mezcla/rebozado."""
    try:
        if not isinstance(meal, dict) or not context_enabled():
            return False
        nombre = _norm(meal.get("name"))
        if _RE_HUEVO.search(nombre) or re.search(r"revoltillo|revuelto|tortilla|omelet|frittata", nombre):
            return False
        piezas = egg_pieces(meal)
        if piezas <= 0 or piezas > 1.0:
            return False
        txt = nombre + " . " + _norm(" . ".join(str(p) for p in (meal.get("recipe") or [])))
        return bool(_RE_MASA.search(txt))
    except Exception:
        return False


def egg_meal_cap(total_meals: int, n_days: int, egg_staple: bool, default_cap: int) -> int:
    """Techo de comidas con huevo en el plan: el de siempre (`default_cap`), salvo que la persona declarara el huevo
    (en cualquier forma) como básico — entonces al menos una comida por día, y nunca menos que el default."""
    try:
        if not egg_staple or not context_enabled():
            return int(default_cap)
        return max(int(default_cap), int(n_days or 0))
    except Exception:
        return int(default_cap)


def egg_staple_declared(form_data) -> bool:
    try:
        from plan_policy import egg_staple_forms
        return bool(egg_staple_forms(form_data))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Lo que el juez debe saber de la persona
# ─────────────────────────────────────────────────────────────────────────────────────────────

def judge_context(form_data, country: Optional[str] = None) -> dict:
    """`{"contexto": {...}}` para el payload del juez: básicos (con su franja), cocinas elegidas, equipo, tiempo de
    cocina y país de la cocina. Vacío ⇒ `{}` (payload byte-idéntico al de siempre)."""
    try:
        if not context_enabled():
            return {}
        f = form_data if isinstance(form_data, dict) else {}
        ctx: dict = {}
        basicos = [f"{s['name']} ({', '.join(s['slots'])})" if s["slots"] else s["name"] for s in declared_staples(f)]
        if basicos:
            ctx["basicos_declarados"] = basicos[:12]
        cocinas = f.get("cultureProfiles") or _superpers(f).get("cuisines")
        if isinstance(cocinas, list) and cocinas:
            ctx["cocinas_elegidas"] = [str(x) for x in cocinas][:6]
        eq = declared_equipment_labels(f)
        if eq:
            ctx["equipo_disponible"] = eq
        ct = str(f.get("cookingTime") or "").strip()
        if ct:
            ctx["tiempo_de_cocina"] = ct
        if country:
            ctx["pais_de_la_cocina"] = str(country)
        if not ctx:
            return {}
        ctx["lectura"] = ("Un básico declarado para una franja es lo que la persona PIDIÓ: no es `slot_inapropiado`. "
                          "Las cocinas elegidas admiten sus patrones (avena salada, yogur como salsa, pescado con salsa de "
                          "fruta): juzga por técnica y composición, no por asociación de nombres.")
        return {"contexto": ctx}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-FRUIT-SAVORY-CLASH · P1-MENU-COHERENCE-1] Pareo chocante fruta+salado — movido aquí desde graph_orchestrator
# (P1-PLAN-LOTE-26: extraer, no subir el tope). Los nombres se re-exportan allí tal cual.
# ─────────────────────────────────────────────────────────────────────────────────────────────

#: Frutas dulces DOMINANTES (sabor que manda en el plato). Bases saladas: almidones (arroz/moro/pasta…), huevo-salado
#: (revoltillo/revuelto) y crucíferas/berenjena. NO proteínas (pollo con piña / cerdo con guayaba son platos tropicales
#: aceptables). Match SOLO en el NOMBRE del plato (no ingredientes) + word-boundary.
_SWEET_DOMINANT_FRUITS = ("mango", "pina", "lechosa", "papaya", "guayaba", "melon", "sandia", "mamey", "zapote")
_SAVORY_CLASH_TOKENS = ("arroz", "moro", "locrio", "pasta", "espagueti", "macarron", "fideo", "espaguetis",
                        "revoltillo", "revuelto", "coliflor", "brocoli", "berenjena", "huevo", "mangu", "platano verde",
                        "tostones", "mofongo")  # [P1-CLASH-HUEVO-Y-VIVERES]
#: [P1-MENU-COHERENCE-1 · 2026-07-29] Las frutas DE AGUA como ensalada/guarnición de un plato de CARNE o PESCADO sí son
#: pareo chocante — van en desayuno/merienda. Mango/piña/guayaba con proteína siguen siendo aceptables.
_WATER_SWEET_FRUITS = ("lechosa", "papaya", "melon", "sandia", "mamey", "zapote")
_MEAT_MAIN_CLASH_TOKENS = ("chuleta", "cerdo", "brocheta", "pollo", "pavo", "res", "bistec",
                           "conejo", "chivo", "mero", "pescado", "tilapia", "salmon", "atun",
                           "camarones", "pulpo", "calamar", "langosta", "carne")
#: [P1-PLAN-LOTE-26] (CUL-P1-03) Componentes SEPARADOS en el nombre («… con fruta al lado», «acompañado de mango»,
#: «y de postre …»): la fruta junto al plato no es la fruta dentro del plato. Sólo con la separación DICHA en el nombre.
_SEPARATE_COMPONENT_RE = re.compile(r"\bal lado\b|\bacompanad[oa]s? de\b|\bde acompanamiento\b|\baparte\b|"
                                    r"\b(?:y|con) (?:una? )?(?:porcion|taza|bowl|vaso) de\b|\bde postre\b|\bcomo postre\b")


def _name_has_token(token: str, text_low: str) -> bool:
    """True si `token` aparece en `text_low` (ya en minúscula/sin acentos) con frontera de palabra inicial → evita que
    'pina' matchee 'espina' o 'macarron' matchee substrings. [P1-FRUIT-SAVORY-CLASH]"""
    try:
        return re.search(r"\b" + re.escape(token), text_low) is not None
    except Exception:
        return token in text_low


def _meal_has_sweet_savory_clash(meal: dict) -> bool:
    """[P1-FRUIT-SAVORY-CLASH] True si el NOMBRE del plato combina una fruta dulce dominante con una base salada
    (mango+arroz, revoltillo+mango, coliflor+mango), o una fruta DE AGUA con un plato de carne/pescado (brochetas de
    cerdo + ensalada de lechosa — P1-MENU-COHERENCE-1). SSOT del detector per-comida — reusado por
    build_variety_report (S1) y appetibility_fix_for_update (S2/S3). Match word-boundary sobre el nombre. FAIL-SAFE:
    error → False. [P1-PLAN-LOTE-26] Un componente SEPARADO dicho en el nombre («revoltillo con mango al lado») no es
    una mezcla. tooltip-anchor: P1-FRUIT-SAVORY-CLASH"""
    try:
        from constants import strip_accents
        name_low = strip_accents(str((meal or {}).get("name", "")).lower())
        if not name_low:
            return False
        if context_enabled() and _SEPARATE_COMPONENT_RE.search(name_low):
            return False
        if (any(_name_has_token(fr, name_low) for fr in _SWEET_DOMINANT_FRUITS)
                and any(_name_has_token(tok, name_low) for tok in _SAVORY_CLASH_TOKENS)):
            return True
        return (any(_name_has_token(fr, name_low) for fr in _WATER_SWEET_FRUITS)
                and any(_name_has_token(tok, name_low) for tok in _MEAT_MAIN_CLASH_TOKENS))
    except Exception:
        return False


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b)) if b else 0
