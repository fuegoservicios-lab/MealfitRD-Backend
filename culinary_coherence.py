"""[P1-CULINARY-CONTRACT · 2026-07-31] Validador determinista de coherencia
culinaria — SSOT (espejo del rol de shopping_calculator en el guard de lista).

PURO a propósito: sin env vars, sin LLM, sin DB. El catálogo entra como
argumento; los knobs viven en los callers (graph_orchestrator / cron_tasks).
Matching: word-boundary + alias más largo gana + acentos fuera + plural↔singular
BIDIRECCIONAL (lecciones pollo⊂repollo, sal⊂salami, FP tomates↔tomate del
dry-run 2026-07-31). Fail-open POR CHECK: alimento sin metadata ⇒ se salta el
check para ese alimento. El scan JAMÁS muta el plan.
tooltip-anchor: P1-CULINARY-CONTRACT
"""
from __future__ import annotations

import re

from constants import strip_accents

# Vocabulario canónico (el mismo de la migración; el sanity DO $$ lo enforza en DB)
PREP_VOCAB = ("hervir", "plancha", "freir", "hornear", "guisar", "saltear",
              "licuar", "tostar", "crudo", "ninguno")

# fragmento-regex → método canónico. Centraliza (y supera) _COOKING_VERB_RE.
VERB_TO_METHOD = {
    r"hierv\w*|hirv\w*|cuec\w*|coce\w*|cocci[oó]n": "hervir",
    r"plancha|parrilla": "plancha",
    r"fr[ií]e\w*|fre[ií]r": "freir",
    r"hornea\w*|horno|airfryer": "hornear",
    r"guisa\w*": "guisar",
    # [Task-4 RESOLUCIÓN 1 · controller] "sofr[ií]\w*" vive en ESTA alternancia
    # (fusionado con "saltea\w*"), NO bajo "freir" (el brief original lo
    # agrupaba junto con freír). Sofreír cebolla/ají es la base de TODA
    # receta dominicana, y la metadata de Vegetales (migración T3) lleva
    # "saltear" pero NO "freir" en prep_methods — dejarlo bajo freir habría
    # hecho que "Sofríe la cebolla" disparara V1 falso-positivo en recetas
    # legítimas del golden set (T5 lo habría medido como FP). Culinariamente
    # sofreír Y saltear son la misma técnica (grasa caliente, movimiento
    # constante, poco tiempo). [Fix post-review] va FUSIONADO en la misma
    # clave que "saltea\w*" (no en una entrada separada): dos claves
    # distintas resolviendo al mismo método producían DOS entradas
    # duplicadas en `metodos` por paso (p.ej. "Sofríe y saltea..."), y por
    # tanto dos violaciones V1 idénticas para el mismo (food, método).
    # [Task-5 · golden set] "dora\w*" vive AQUÍ (fusionado con saltear/sofreír),
    # NO con "tuesta\w*|tosta\w*". "Agrega la pechuga de pollo en trozos y
    # dora"/"Agrega la carne de res y dora" son el paso de sellado en caliente
    # que abre CADA guiso dominicano (Pechuga de pollo, Carne de res: fresh
    # proteins con 'saltear' y 'freir' en prep_methods pero SIN 'tostar' —
    # tostar es para pan/casabe, dorar carne es sellar en sartén/grasa, la
    # misma técnica que sofreír). Antes de este fix, "dora" resolvía a
    # 'tostar' y el golden set (5/5 buenos con "... y dora." en Locrío/
    # Sancocho) disparaba V1 falso-positivo real contra el catálogo de Neon
    # (Pechuga de pollo / Carne de res sin 'tostar' en prep_methods) —
    # detectado por `test_capa1_cero_fp_sobre_los_buenos`, NO por un test
    # unitario con catálogo sintético.
    r"saltea\w*|sofr[ií]\w*|dora\w*": "saltear",
    r"lic[uú]a\w*": "licuar",
    r"tuesta\w*|tosta\w*": "tostar",
}
_VERB_RES = [(re.compile(rf"\b(?:{frag})", re.IGNORECASE), metodo)
             for frag, metodo in VERB_TO_METHOD.items()]

# Exentos de V3 (T5). UNA lista canónica — criterio del audit real 2026-07-31
# que contó 4/12 huérfanos (condimentos no cuentan).
CONDIMENT_EXEMPT = frozenset({
    "aceite", "sal", "agua", "pimienta", "oregano", "vinagre", "sazon",
    "condimento", "especia", "caldo", "cubito", "ajo en polvo", "canela",
})

_RE_ESTADO = re.compile(r"ya\s+vien[e]?\s+cocid|ya\s+est[aá]\s+cocid", re.IGNORECASE)


def _norm(text: str) -> str:
    return strip_accents(str(text or "").lower())


def step_has_cooking_verb(paso: str) -> bool:
    """True si `paso` contiene algún verbo de cocción del vocabulario
    canónico (`VERB_TO_METHOD`). Export público de lo que antes solo se leía
    vía `_VERB_RES` (privado del módulo) — usado por el path degradado
    (`cron_tasks._build_filtered_edge_recipe_day`) para identificar cuál paso
    degradar a 'Sirve el {food}.' cuando el scan reporta V1/V2.
    tooltip-anchor: P1-CULINARY-CONTRACT"""
    return any(rx.search(_norm(paso)) for rx, _ in _VERB_RES)


def _sing_plural_pattern(word: str) -> str:
    """Patrón que matchea la forma singular Y plural de `word` (bidireccional:
    si word ya viene en plural, también matchea el singular)."""
    w = re.escape(word)
    if word.endswith("es") and len(word) > 4:
        return rf"{re.escape(word[:-2])}(?:e?s)?"
    if word.endswith("s") and len(word) > 3:
        return rf"{re.escape(word[:-1])}s?"
    return rf"{w}(?:e?s)?"


# [IMPORTANT-5 · post-review-final] Matching por TOKEN con word-boundary, no substring
# plano — "sal" ⊂ "Salami"/"Salmón", "agua" ⊂ "Aguacate", "sal" ⊂ "EnSALada" (14ª
# aparición documentada de esta clase de bug en el repo). `\b` no separa DENTRO de una
# palabra continua (agua|cate no tiene borde entre 'a' y 'c'), así que exige la palabra
# COMPLETA — reusa `_sing_plural_pattern` por token para seguir aceptando plurales
# ("sales", "especias") y `\s+` entre tokens para exenciones multi-palabra ("ajo en
# polvo"). Construido a import-time (frozenset CONDIMENT_EXEMPT es estable).
_CONDIMENT_EXEMPT_RES = [
    re.compile(r"\b" + r"\s+".join(_sing_plural_pattern(w) for w in ex.split()) + r"\b")
    for ex in CONDIMENT_EXEMPT
]


def build_culinary_index(catalog: list) -> dict:
    """Índice nombre-normalizado → metadata + regex word-boundary del alias."""
    index = {}
    for row in catalog or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        norm = _norm(name)
        tokens = [_sing_plural_pattern(t) for t in norm.split()]
        rx = re.compile(r"\b" + r"\s+".join(tokens) + r"\b")
        index[norm] = {
            "name": name,
            "prep_methods": row.get("prep_methods"),
            "ready_to_eat": row.get("ready_to_eat"),
            "rx": rx,
        }
    return index


def find_catalog_foods(text: str, index: dict) -> list:
    """Alimentos del catálogo mencionados en `text`. Alias más largo gana:
    los spans ya cubiertos por un match largo no re-matchean con uno corto."""
    blob = _norm(text)
    hits = []          # (start, end, name)
    for norm_name in sorted(index, key=len, reverse=True):
        for m in index[norm_name]["rx"].finditer(blob):
            if any(s <= m.start() < e or s < m.end() <= e for s, e, _ in hits):
                continue     # span ya reclamado por un alias más largo
            hits.append((m.start(), m.end(), index[norm_name]["name"]))
    seen, out = set(), []
    for _, _, name in sorted(hits):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _iter_meals(plan_data: dict):
    for d in (plan_data or {}).get("days") or []:
        if not isinstance(d, dict):
            continue
        for m in d.get("meals") or []:
            if isinstance(m, dict):
                yield d.get("day"), m


def _v1_verbo_alimento(day, meal, index) -> list:
    out = []
    for paso in meal.get("recipe") or []:
        # dict.fromkeys en vez de list comp: cinturón y tirantes contra la
        # próxima clave de VERB_TO_METHOD que alguien añada resolviendo a un
        # método ya cubierto por otra clave (orden de primera aparición
        # preservado; sin esto, dos claves→mismo método duplican `metodos` y
        # por tanto duplican la violación V1 para el mismo (food, método)).
        metodos = list(dict.fromkeys(
            met for rx, met in _VERB_RES if rx.search(_norm(paso))))
        if not metodos:
            continue
        foods = find_catalog_foods(paso, index)
        metas = {food: (index.get(_norm(food)) or {}) for food in foods}
        for met in metodos:
            # [Task-4 RESOLUCIÓN 2 · controller] atribución verbo→alimento por
            # paso: todo verbo del paso se cruza con todo alimento del paso
            # (letra del brief), PERO con salvaguarda — si el paso menciona
            # ≥2 alimentos y ≥1 de ellos SÍ acepta el método, el check se
            # salta los demás alimentos de ese paso para ESE método. Un paso
            # multi-alimento con un destinatario válido del verbo no acusa a
            # los acompañantes: "Hierve el arroz y sirve con casabe" no es
            # cocer el casabe. Si NINGÚN alimento acepta el método, la
            # salvaguarda no aplica y se acusa a todos (no hay "destinatario
            # válido" que lo lea como paso legítimo con acompañante inocente).
            accepting = {f for f, meta in metas.items()
                         if meta.get("prep_methods") is not None
                         and met in meta.get("prep_methods")}
            safeguard = len(foods) >= 2 and len(accepting) >= 1
            for food in foods:
                if food in accepting:
                    continue
                meta = metas[food]
                prep = meta.get("prep_methods")
                if prep is None:
                    continue                      # fail-open: sin metadata no se juzga
                if safeguard:
                    continue                       # acompañante del destinatario válido
                if meta.get("ready_to_eat") is True:
                    out.append(_viol(day, meal, "V1", food,
                                     f"paso aplica '{met}' a un listo-para-comer: {paso[:120]}",
                                     "minor", False))
                else:
                    out.append(_viol(day, meal, "V1", food,
                                     f"'{met}' no está en prep_methods{tuple(prep)}: {paso[:120]}",
                                     "minor", False))
    return out


def _v2_estado_imposible(day, meal, index) -> list:
    out = []
    textos = list(meal.get("recipe") or []) + list(meal.get("ingredients") or [])
    for t in textos:
        if not _RE_ESTADO.search(_norm(t)):
            continue
        for food in find_catalog_foods(t, index):
            meta = index.get(_norm(food)) or {}
            if meta.get("ready_to_eat") is False:      # NULL ⇒ fail-open
                out.append(_viol(day, meal, "V2", food,
                                 f"'(ya viene cocido)' sobre alimento fresco: {str(t)[:120]}",
                                 "high", False))
    return out


def _mencionado_por_prefijo(food: str, pasos_norm: str, comida_foods: list) -> bool:
    """[Task-5 · golden set] Nombres compuestos con calificador final ('Arroz
    blanco', 'Yogurt griego sin azúcar') que la prosa dominicana menciona por
    su forma genérica ('el arroz', 'el yogurt griego') — el calificador
    completo casi nunca se repite en la receta si ya está en `ingredients`.
    Detectado por `test_capa1_cero_fp_sobre_los_buenos` contra el catálogo
    real (NO por los tests unitarios con catálogo sintético, que no tienen
    ningún alimento de 2+ palabras con calificador recortable).

    Prueba prefijos DECRECIENTES del nombre (nunca el nombre completo — eso ya
    lo cubre `food in en_pasos` antes de llamar aquí) y acepta el más largo
    que aparezca en los pasos, EXCEPTO si ese prefijo es ambiguo con otro
    alimento de la MISMA comida que comparte la cabeza pero difiere después
    ('Ají cubanela' vs 'Ají morrón' — el guard existe precisamente porque el
    golden set inyecta 'Ají morrón' huérfano en comidas que sí mencionan 'el
    ají cubanela'; caer a la cabeza sola 'ají' lo habría enmascarado)."""
    tokens = _norm(food).split()
    for k in range(len(tokens) - 1, 0, -1):
        prefijo = tokens[:k]
        ambiguo = any(
            otro != food and _norm(otro).split()[:k] == prefijo
            for otro in comida_foods
        )
        if ambiguo:
            continue
        patron = re.compile(
            r"\b" + r"\s+".join(_sing_plural_pattern(t) for t in prefijo) + r"\b")
        if patron.search(pasos_norm):
            return True
    return False


def _v3_huerfanos(day, meal, index) -> list:
    pasos_blob = " || ".join(meal.get("recipe") or [])
    pasos_norm = _norm(pasos_blob)
    en_pasos = set(find_catalog_foods(pasos_blob, index))
    ingredientes = meal.get("ingredients") or []
    # Alimentos de ESTA comida ya resueltos por ingrediente — solo para el
    # guard de ambigüedad de `_mencionado_por_prefijo` (no cambia qué cuenta
    # como huérfano por sí solo).
    comida_foods = []
    for ing in ingredientes:
        resuelto = find_catalog_foods(ing, index)
        if resuelto:
            comida_foods.append(resuelto[0])

    out = []
    for ing in ingredientes:
        n = _norm(ing)
        if any(rx.search(n) for rx in _CONDIMENT_EXEMPT_RES):
            continue
        foods = find_catalog_foods(ing, index)
        if not foods:
            continue          # no resoluble al catálogo (p.ej. 'picados') ⇒ skip
        food = foods[0]       # el alias más largo/primero del string
        if food in en_pasos:
            continue
        if _mencionado_por_prefijo(food, pasos_norm, comida_foods):
            continue
        out.append(_viol(day, meal, "V3", food,
                         f"listado ('{str(ing)[:60]}') pero ningún paso lo menciona",
                         "minor", True))
    return out


def _viol(day, meal, check, food, detail, severity, repairable):
    return {"day": day, "meal": meal.get("meal") or meal.get("name"),
            "check": check, "food": food, "detail": detail,
            "severity": severity, "repairable": repairable}


def culinary_contract_scan(plan_data: dict, catalog: list) -> list:
    """Escanea el plan completo. Retorna lista de Violations (vacía si todo
    coherente o si no hay datos). Jamás lanza: fail-open total."""
    try:
        index = build_culinary_index(catalog)
        if not index:
            return []
        out = []
        for day, meal in _iter_meals(plan_data):
            out.extend(_v1_verbo_alimento(day, meal, index))
            out.extend(_v2_estado_imposible(day, meal, index))
            out.extend(_v3_huerfanos(day, meal, index))
        return out
    except Exception:
        return []


def scan_coverage(plan_data: dict, catalog: list) -> float:
    """Fracción de alimentos mencionados en el plan que tienen metadata
    (telemetría de cobertura para el rollout warn→block)."""
    try:
        index = build_culinary_index(catalog)
        vistos, con_meta = set(), 0
        for _, meal in _iter_meals(plan_data):
            blob = " | ".join(list(meal.get("ingredients") or []) +
                              list(meal.get("recipe") or []))
            for f in find_catalog_foods(blob, index):
                if f in vistos:
                    continue
                vistos.add(f)
                if (index.get(_norm(f)) or {}).get("prep_methods") is not None:
                    con_meta += 1
        return (con_meta / len(vistos)) if vistos else 1.0
    except Exception:
        return 1.0
