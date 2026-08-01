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
#
# [P1-CULINARY-CONTRACT-FP1 · 2026-08-01, clase A] Primer plan real en prod
# (165dd761) midió 9 violaciones V1, las 6 de esta clase todas en pasos de
# "Montaje:" — el `\w*` genérico tras cada raíz tragaba la forma PARTICIPIAL/
# ADJETIVAL, no solo el imperativo: «el yaniqueque HORNEADO», «pollo
# desmechado SALTEADO», «las TOSTADAS», «almendras TOSTADAS» disparaban V1
# como si el paso estuviera COCINANDO el alimento, cuando en realidad lo
# describía ya cocido (un paso de montaje no cocina — ver además el skip
# explícito en `_v1_verbo_alimento`). Fix: negative-lookahead de participio
# (`d[oa]s?\b` / `t[oa]s?\b` para las irregulares -frito/-sofrito) tras cada
# raíz cuya forma participial existe en español. Verificado LETRA A LETRA
# (reporte `.superpowers/culinary-fp-round1-report.md`, sección "Traza"):
#   - hervir: "hervido" NO matchea 'hierv' (h-e-r-v ≠ h-i-e-r-v) y "cocido"
#     NO matchea 'coce' (c-o-c-i ≠ c-o-c-e) — el gap de orden de letras ya
#     los excluía. SIN cambios.
#   - freir: "frito/frita" tampoco matchea ('fr[ií]e' exige una 'e' justo
#     tras la i/í; 'fre[ií]r' exige una 'r' justo tras la e) — SIN cambios.
#   - hornear/guisar/saltear-sofreír-dorar/licuar/tostar: SÍ colisionaban
#     (hornea+do, guisa+do, saltea+do, sofr[ií]+to, dora+do, licua+do,
#     tosta+da) — lookahead aplicado a las 5 raíces + a la irregular sofrito.
#     'tuesta\w*' no tiene forma participial propia (el participio de
#     "tostar" es "tostado", no "tuestado") — SIN cambios.
VERB_TO_METHOD = {
    r"hierv\w*|hirv\w*|cuec\w*|coce\w*|cocci[oó]n": "hervir",
    r"plancha|parrilla": "plancha",
    r"fr[ií]e\w*|fre[ií]r": "freir",
    # [P1-CULINARY-CONTRACT-FP round 2 · 2026-08-01, FP-D] "apto para horno"
    # / "recipiente apto para horno" describe el ENVASE (compatibilidad del
    # recipiente), no una instrucción de cocción — antes del fix el
    # sustantivo suelto "horno" disparaba V1 igual que "al horno"/"en el
    # horno" (instrucciones reales de cocción). Lookbehind de ancho fijo
    # `(?<!para )` excluye SOLO cuando "horno" viene precedido literalmente
    # de "para " — "al horno"/"en el horno"/"lleva al horno" siguen
    # disparando sin cambios porque no están precedidos de "para ".
    r"hornea(?!d[oa]s?\b)\w*|(?<!para )horno|airfryer": "hornear",
    r"guisa(?!d[oa]s?\b)\w*": "guisar",
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
    r"saltea(?!d[oa]s?\b)\w*|sofr[ií](?!t[oa]s?\b)\w*|dora(?!d[oa]s?\b)\w*": "saltear",
    r"lic[uú]a(?!d[oa]s?\b)\w*": "licuar",
    r"tuesta\w*|tosta(?!d[oa]s?\b)\w*": "tostar",
}
_VERB_RES = [(re.compile(rf"\b(?:{frag})", re.IGNORECASE), metodo)
             for frag, metodo in VERB_TO_METHOD.items()]

# Exentos de V3 (T5). UNA lista canónica — criterio del audit real 2026-07-31
# que contó 4/12 huérfanos (condimentos no cuentan).
CONDIMENT_EXEMPT = frozenset({
    "aceite", "sal", "agua", "pimienta", "oregano", "vinagre", "sazon",
    "condimento", "especia", "caldo", "cubito", "ajo en polvo", "canela", "comino",
})

_RE_ESTADO = re.compile(r"ya\s+vien[e]?\s+cocid|ya\s+est[aá]\s+cocid", re.IGNORECASE)

# [P1-CULINARY-CONTRACT-FP1 · 2026-08-01, clase A refuerzo] Un paso de
# "Montaje:" ensambla componentes YA preparados (el prefijo lo garantiza la
# convención de secciones `_CULINARY_STEP_SECTIONS` en cron_tasks.py) — jamás
# cocina, así que V1 no debe leerlo. Anclado sobre el FP real: «Montaje:
# Sirve el revoltillo sobre el yaniqueque horneado y el Casabe.» V2/V3 SÍ
# siguen leyendo montaje sin cambios (V3 en particular NECESITA las
# menciones de montaje para no marcar huérfanos que solo aparecen ahí).
_RE_MONTAJE_STEP = re.compile(r"^\s*montaje\s*:", re.IGNORECASE)

# [P1-CULINARY-CONTRACT-FP2 · 2026-08-01] Reemplaza el conteo fijo de 4
# palabras (regresión de recall encontrada por el reviewer): el prompt SSOT
# de `day_generator` EXIGE "cocción con AL MENOS un TIEMPO concreto", así que
# el patrón verbo + relleno-mandatado ("a fuego medio por 15 minutos hasta
# ablandar") + objeto es la NORMA de los planes reales, no la excepción — 4
# palabras se comen el relleno antes de llegar al objeto real. La ventana
# ahora corre hasta la FRONTERA DE ORACIÓN (siguiente '.'/';' o fin del
# paso): una frase adverbial de tiempo/fuego no es otra cláusula, así que no
# debe cortar la ventana antes de tiempo.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.;]")


# [P1-CULINARY-CONTRACT-FP round 2 · 2026-08-01, FP-C] La ventana hasta
# frontera de oración (round 1) tiene demasiado alcance: cruza una
# conjunción "y" que introduce un VERBO NUEVO («Tuesta la tortilla integral
# ... y coloca el huevo encima.») y lee el destinatario del verbo SIGUIENTE
# ("huevo", de "coloca") como si fuera el nuestro ("tuesta"). Fix:
# heurística de OBJETO INMEDIATO — tras el verbo, saltar RELLENO canónico
# (artículos, preposiciones, vocabulario de tiempo/fuego/temperatura,
# verbos-de-resultado, y dígitos puros que solo cuantifican relleno como
# "10 minutos") y mirar el PRIMER token de contenido:
#   (a) si abre una mención catalogada  → el objeto es nuestro, resuelve.
#   (b) si es contenido NO catalogado   → el destinatario fue nombrado
#       explícitamente y NO es nuestro — veto, aunque otro alimento
#       catalogado aparezca más adelante en la misma oración (ese es el
#       objeto de OTRO verbo, como "coloca el huevo").
#   (c) si TODA la cola es relleno (sin ningún token de contenido) → no hay
#       destinatario nombrado hacia adelante. Mirar HACIA ATRÁS, acotado a
#       la MISMA cláusula (desde el último '.'/';' antes del verbo, o el
#       inicio del paso si no hay ninguno) — cubre la construcción "Lleva/
#       Mete/Coloca [OBJETO] al horno" donde el alimento se nombra ANTES del
#       disparador "horno" y nada lo sigue salvo relleno de tiempo ("por 10
#       minutos"). Si tampoco hay nada resoluble hacia atrás EN ESA
#       CLÁUSULA («Cuece a fuego lento por 10 minutos; luego agrega el
#       Casabe y sirve.» — Casabe vive en la cláusula SIGUIENTE al ';', no
#       en esta), el veto se mantiene — mismo trade-off fail-open de
#       P1-CULINARY-CONTRACT-FP2 (mejor callar que acusar al alimento
#       equivocado).
_FILLER_TOKENS = frozenset({
    # artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # preposiciones
    "a", "en", "de", "del", "al", "con", "por", "sobre",
    # vocabulario de cocción no-alimento (tiempo/fuego/temperatura)
    "fuego", "lento", "medio", "alto", "bajo", "minuto", "minutos", "min",
    "hora", "horas", "temperatura", "grados", "hasta", "durante",
    "aproximadamente", "cuidadosamente",
    # verbos-de-resultado: describen el EFECTO de la cocción, no el
    # alimento — sin estos, "... hasta ablandar el Casabe" perdía recall
    # (el objeto inmediato tras "ablandar" nunca se alcanzaba a saltar).
    "ablandar", "espesar", "reducir", "integrar", "combinar",
})
_WORD_RE = re.compile(r"\w+")


def _object_inmediato_status(tail: str, index: dict) -> str:
    """'resolves' / 'vetoes' / 'empty' sobre `tail` (ya acotado a la
    cláusula del verbo). Camina token a token saltando `_FILLER_TOKENS` y
    dígitos puros; el primer token de CONTENIDO decide: abre una mención
    catalogada → 'resolves'; es contenido no-catalogado → 'vetoes'; se
    agota la cola sin ningún token de contenido → 'empty' (el llamador
    decide el fallback — ver `_occurrence_resolves`)."""
    spans = _catalog_food_spans(tail, index)
    for m in _WORD_RE.finditer(tail):
        tok = m.group()
        if tok.isdigit() or tok in _FILLER_TOKENS:
            continue
        if any(s == m.start() for s, _, _ in spans):
            return "resolves"
        return "vetoes"
    return "empty"


# [P1-CULINARY-CONTRACT-FP round 3 · 2026-08-01] Plan real 5fba379a (score
# 96.5, corr a046005f) midió 7/7 violaciones V1 FALSO POSITIVAS de UN SOLO
# mecanismo nuevo, distinto de rounds 1/2: la ventana/veto de rounds 1-2
# decidía "¿algún alimento acepta el método, o el objeto inmediato resuelve?"
# UNA VEZ POR MÉTODO agregando TODAS sus ocurrencias en el paso — y si
# CUALQUIER ocurrencia resolvía, `_v1_verbo_alimento` acusaba a TODOS los
# alimentos del PASO ENTERO (`foods = find_catalog_foods(paso, index)`, sin
# acotar), aunque vivieran en una ORACIÓN distinta a la de esa ocurrencia:
#   - Huevo (Desayuno): «Tuesta el pan integral... [oración 2: cuece el
#     huevo]... tuesta las semillas de calabaza.» — la 2ª ocurrencia de
#     'tostar' (semillas de calabaza, legítima) desbloqueaba la acusación
#     para TODO el paso, incluyendo a Huevo en la oración del medio, que no
#     es objeto de NINGUNA ocurrencia de 'tostar'.
#   - Cebolla/Ají cubanela/Ajo/Tomate/Berenjena/Limón (Almuerzo): la ÚNICA
#     ocurrencia de 'tostar' del paso («Tuesta las semillas de calabaza...»)
#     vive en la ÚLTIMA oración; el resto son sofrito/guisado de OTRAS
#     oraciones anteriores, sin relación con 'tostar'.
# Fix: la atribución de alimentos pasa de PASO a CLÁUSULA (oración,
# `_SENTENCE_BOUNDARY_RE`) POR OCURRENCIA — `_clause_bounds` acota la
# oración de cada ocurrencia y `_v1_verbo_alimento` solo considera
# `find_catalog_foods` DENTRO de esa cláusula (no del paso completo) para
# decidir accepting/safeguard/acusación de ESA ocurrencia. La heurística de
# objeto inmediato de round 2 (`_object_inmediato_status`) se conserva sin
# cambios — sigue resolviendo el caso "dos verbos, misma oración, unidos por
# 'y'" (FP-C) que el acotado por cláusula por sí solo NO cubre (ambos objetos
# viven en la MISMA oración ahí, no en oraciones distintas). Traza completa:
# `.superpowers/culinary-fp-round1-report.md`, sección "Ronda 3".
def _clause_bounds(paso_norm: str, pos: int) -> tuple:
    """[start, end) de la cláusula (oración) de `paso_norm` que contiene la
    posición `pos`, acotada por `_SENTENCE_BOUNDARY_RE` ('.'/';') o los
    extremos del paso — la MISMA noción de "oración" que el antiguo
    `_post_verb_resolves` (round 1/2, renombrado/refactorizado a
    `_occurrence_resolves` en round 3) ya usaba para la ventana hacia
    adelante y el fallback hacia atrás, ahora factorizada para además
    acotar qué alimentos cuentan como "del paso" en `_v1_verbo_alimento`
    (round 3)."""
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(paso_norm):
        if m.end() <= pos:
            start = m.end()
        else:
            break
    end_match = _SENTENCE_BOUNDARY_RE.search(paso_norm, pos)
    end = end_match.start() if end_match else len(paso_norm)
    return start, end


def _occurrence_resolves(paso_norm: str, start: int, end: int,
                          clause_start: int, clause_end: int, index: dict) -> bool:
    """[P1-CULINARY-CONTRACT-FP round 1+2+3 · 2026-08-01] True si el objeto
    directo de ESTA ocurrencia del verbo (`start`,`end`) — heurística de
    objeto inmediato (ver bloque de comentario arriba), ventana hacia
    adelante hasta el fin de la cláusula (`clause_end`) con fallback hacia
    atrás hasta el inicio de la cláusula (`clause_start`) — resuelve a un
    alimento del catálogo.

    [round 3] Antes evaluaba TODAS las ocurrencias de un método en el paso y
    bastaba con que UNA sola resolviera para desbloquear la acusación sobre
    el paso ENTERO (`_post_verb_resolves`, plural). Ahora `_v1_verbo_alimento`
    llama esta función UNA VEZ POR OCURRENCIA, con `foods`/`accepting` ya
    acotados a la cláusula de esa ocurrencia — cierra el caso real donde una
    2ª ocurrencia legítima en otra oración "rescataba" el veto y contaminaba
    alimentos de una oración intermedia que no era el objeto de nadie."""
    tail = paso_norm[end:clause_end]
    status = _object_inmediato_status(tail, index)
    if status == "resolves":
        return True
    if status == "vetoes":
        return False
    # status == "empty": nada nombrado hacia adelante en esta cláusula.
    # Fallback (c, round 2): mirar hacia atrás, acotado a la MISMA cláusula.
    head = paso_norm[clause_start:start]
    return bool(head.strip() and find_catalog_foods(head, index))


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


def _catalog_food_spans(text: str, index: dict) -> list:
    """Spans `(start, end, name)` de alimentos del catálogo mencionados en
    `text` (posiciones sobre `_norm(text)`). Alias más largo gana: los spans
    ya cubiertos por un match largo no re-matchean con uno corto. Factored
    fuera de `find_catalog_foods` (P1-CULINARY-CONTRACT-FP round 2) para que
    `_object_inmediato_status` pueda consultar POSICIONES, no solo nombres —
    necesita saber si la mención catalogada arranca exactamente en el primer
    token de contenido tras el verbo, no solo si existe en algún lugar."""
    blob = _norm(text)
    hits = []          # (start, end, name)
    for norm_name in sorted(index, key=len, reverse=True):
        for m in index[norm_name]["rx"].finditer(blob):
            if any(s <= m.start() < e or s < m.end() <= e for s, e, _ in hits):
                continue     # span ya reclamado por un alias más largo
            hits.append((m.start(), m.end(), index[norm_name]["name"]))
    return sorted(hits)


def find_catalog_foods(text: str, index: dict) -> list:
    """Alimentos del catálogo mencionados en `text`. Alias más largo gana:
    los spans ya cubiertos por un match largo no re-matchean con uno corto."""
    seen, out = set(), []
    for _, _, name in _catalog_food_spans(text, index):
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
        paso_norm = _norm(paso)
        if _RE_MONTAJE_STEP.match(paso_norm):
            # [FP1 clase A refuerzo] paso de montaje: nunca cocina, V1 no aplica.
            continue
        # met_spans: método → spans (start, end) de cada ocurrencia del verbo
        # que resuelve a ese método en ESTE paso. dict en vez de un simple
        # `set`/list de métodos: cinturón y tirantes contra la próxima clave
        # de VERB_TO_METHOD que alguien añada resolviendo a un método ya
        # cubierto por otra clave. El `start` (además del `end`) hace falta
        # para `_clause_bounds`/`_occurrence_resolves` (round 1-3).
        met_spans = {}
        for rx, met in _VERB_RES:
            spans = [(m.start(), m.end()) for m in rx.finditer(paso_norm)]
            if spans:
                met_spans.setdefault(met, []).extend(spans)
        if not met_spans:
            continue
        # [P1-CULINARY-CONTRACT-FP round 3] `accused` deduplica por
        # (food, método) a través de OCURRENCIAS del mismo método en el mismo
        # paso (p.ej. "Sofríe y saltea..." dispara 2 ocurrencias de 'saltear'
        # — sin esto, si ambas resolvieran, el mismo alimento se acusaría
        # dos veces vía dos ocurrencias distintas).
        accused = set()
        for met, spans in met_spans.items():
            for start, end in spans:
                # [P1-CULINARY-CONTRACT-FP round 3] atribución por CLÁUSULA
                # (oración), no por paso completo — ver bloque de comentario
                # sobre `_clause_bounds`. Cada ocurrencia del verbo solo
                # "ve" los alimentos de SU PROPIA oración; una ocurrencia
                # legítima en otra oración del mismo paso ya no desbloquea
                # la acusación sobre alimentos de oraciones ajenas.
                clause_start, clause_end = _clause_bounds(paso_norm, start)
                clause_text = paso_norm[clause_start:clause_end]
                foods = find_catalog_foods(clause_text, index)
                if not foods:
                    continue
                metas = {food: (index.get(_norm(food)) or {}) for food in foods}
                # [Task-4 RESOLUCIÓN 2 · controller] atribución verbo→alimento
                # dentro de la CLÁUSULA de esta ocurrencia (letra del brief
                # era "por paso"; round 3 la acota a oración — ver arriba),
                # con salvaguarda — si la cláusula menciona ≥2 alimentos y
                # ≥1 de ellos SÍ acepta el método, el check se salta los
                # demás alimentos de esa cláusula para ESE método. Un paso
                # multi-alimento con un destinatario válido del verbo no
                # acusa a los acompañantes: "Hierve el arroz y sirve con
                # casabe" no es cocer el casabe. Si NINGÚN alimento acepta
                # el método, la salvaguarda no aplica y se acusa a todos (no
                # hay "destinatario válido" que lo lea como paso legítimo
                # con acompañante inocente).
                accepting = {f for f, meta in metas.items()
                             if meta.get("prep_methods") is not None
                             and met in meta.get("prep_methods")}
                safeguard = len(foods) >= 2 and len(accepting) >= 1
                # [FP1 clase B] si NADIE en la cláusula acepta el método
                # (accepting vacío) Y el objeto directo inmediato de ESTA
                # ocurrencia (ver `_occurrence_resolves`) tampoco resuelve a
                # un alimento del catálogo, el destinatario real es un
                # no-catalogado (p.ej. "almendras") — no acusar a los DEMÁS
                # alimentos de la cláusula por un verbo que no era para
                # ellos. Si accepting NO está vacío (hay destinatario válido
                # en la cláusula), este veto no aplica — la salvaguarda de
                # arriba ya decide caso a caso.
                if not accepting and not _occurrence_resolves(
                        paso_norm, start, end, clause_start, clause_end, index):
                    continue
                for food in foods:
                    if food in accepting or (food, met) in accused:
                        continue
                    meta = metas[food]
                    prep = meta.get("prep_methods")
                    if prep is None:
                        continue                      # fail-open: sin metadata no se juzga
                    if safeguard:
                        continue                       # acompañante del destinatario válido
                    accused.add((food, met))
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
