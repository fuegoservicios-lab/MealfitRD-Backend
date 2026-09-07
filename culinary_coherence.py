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

import hashlib
import json
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
    # [P1-CULINARY-HASTA-DORAR · 2026-08-19] `(?<!hasta )` delante de "dora": «hasta
    # dorar» / «hasta dorarlas» describen el PUNTO de cocción, no ordenan saltear. El
    # patrón previo excluía "dorado/dorada/dorados/doradas" pero NO el infinitivo, así
    # que «Hornea las papas hasta dorar» acusaba de salteado a todo alimento del paso
    # que no tuviera 'saltear'. Medido sobre 33 planes REALES de prod (2026-08-19):
    # 12 de 63 violaciones V1 eran esto — 19% de ruido, y contra quien menos toca el
    # fuego (Aceite de oliva, Miel, Vainilla, Mango, Linaza, Plátano maduro), porque
    # el acusado es cualquier alimento nombrado en un paso largo multi-cláusula.
    # Mismo mecanismo que el `(?<!para )horno` de aquí arriba, por la misma razón:
    # una palabra que describe el envase o el punto no es una instrucción de cocción.
    # Se descartaron dos alternativas MEDIDAS, no intuidas: excluir solo el infinitivo
    # desnudo (`|r\b`) caza la mitad (6 de 12) y deja pasar «hasta dorarlas»; añadir
    # `(?<!\ba )` encima no cambia NI UNA violación sobre datos reales.
    # El imperativo sigue disparando intacto: «Dora la cebolla», «Dóralo por ambos
    # lados» y «Sofríe el ajo» — el sellado que abre cada guiso dominicano (Task-5).
    r"saltea(?!d[oa]s?\b)\w*|sofr[ií](?!t[oa]s?\b)\w*|(?<!hasta )dora(?!d[oa]s?\b)\w*": "saltear",
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

# [V4 · 2026-08-01] Tolerancia de divergencia de gramaje entre `ingredients` y
# los pasos, sobre `|N_ingrediente − N_paso| / max(N)`. 25% es GENEROSO a
# propósito: redondeos de lonjas/tazas/piezas a un gramaje "bonito" son
# legítimos (1¾ lonjas ≈ 45 g de un ingrediente cuya línea de compra dice
# "30 g" es 33% de divergencia — SÍ dispara; una lonja de más/menos entre 100
# g y 110 g es 9% — NO dispara). Caso real que motiva el check: plan
# 5f4bb17e, día 2 — ingrediente "30 g de queso" vs Mise en place "desmenuza
# 1¾ lonjas/pedazos de queso de hoja (45 g)": el usuario ve dos números para
# el MISMO alimento y no sabe a cuál creer.
V4_TOLERANCIA = 0.25

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
def clause_bounds(texto: str) -> list:
    """[P1-SUBST-STALE-STEP · 2026-08-01, ronda 4b] Lista de `[start, end)` de TODAS las
    cláusulas (oraciones) de `texto`, acotadas por `_SENTENCE_BOUNDARY_RE` ('.'/';'). Export
    PÚBLICO de la lógica de fronteras que `_clause_bounds` (privado, ronda 3, ver abajo — ahora
    un wrapper de ESTA función) ya usaba para resolver la cláusula de UNA sola posición; se
    generaliza aquí a la partición completa del texto porque `graph_orchestrator._substitute_
    blended_raw_egg` (P1-SUBST-STALE-STEP) la necesita para reescribir SOLO la cláusula
    ofensora de un paso ("Hierve el yogur griego...") sin perder una acción legítima de OTRO
    alimento en una cláusula vecina del mismo paso ("...; mientras tanto, tuesta el pan
    integral..." — finding de review, ninguna evidencia en prod aún). Cláusulas vacías o
    solo-whitespace (dos boundaries adyacentes, o el remanente tras la última boundary cuando
    el texto termina en '.'/';') se OMITEN — un caller que reconstruya el texto original a
    partir de esta lista debe recuperar el separador desde `texto[end]` (el carácter de
    boundary), no asumir que las cláusulas devueltas son contiguas byte a byte."""
    bounds = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(texto):
        if texto[start:m.start()].strip():
            bounds.append((start, m.start()))
        start = m.end()
    if texto[start:].strip():
        bounds.append((start, len(texto)))
    return bounds


def _clause_bounds(paso_norm: str, pos: int) -> tuple:
    """[start, end) de la cláusula (oración) de `paso_norm` que contiene la
    posición `pos`, acotada por `_SENTENCE_BOUNDARY_RE` ('.'/';') o los
    extremos del paso — la MISMA noción de "oración" que el antiguo
    `_post_verb_resolves` (round 1/2, renombrado/refactorizado a
    `_occurrence_resolves` en round 3) ya usaba para la ventana hacia
    adelante y el fallback hacia atrás, ahora factorizada para además
    acotar qué alimentos cuentan como "del paso" en `_v1_verbo_alimento`
    (round 3).

    [P1-SUBST-STALE-STEP · 2026-08-01, ronda 4b] Wrapper delegando a `clause_bounds` (público,
    lista completa) — busca la cláusula cuyo span contiene `pos`. Equivalente al algoritmo
    original para cualquier posición dentro de una cláusula NO vacía (el único caso real: `pos`
    siempre es el offset de una ocurrencia de verbo, nunca cae en whitespace puro entre dos
    boundaries). Fallback `(0, len(paso_norm))` si `pos` no cae en ninguna cláusula devuelta
    (defensivo — no debería ocurrir con texto real)."""
    for start, end in clause_bounds(paso_norm):
        if start <= pos <= end:
            return start, end
    return 0, len(paso_norm)


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


# [P1-CULINARY-CONTRACT-YOGUR · 2026-08-01] "yogur" y "yogurt" son la MISMA palabra en es-DO —
# el catálogo usa la forma con 't' ("Yogurt griego sin azúcar", migración P2-3) pero la prosa
# generada (y el propio swap huevo→yogur de graph_orchestrator.py, `_egg_step_subs`) usa la forma
# SIN 't' ("el yogur griego"). Defecto real medido (plan de producción 97.2,
# 5f4bb17e-14cb-4db3-8d97-79933af690cf, día 2 Desayuno): «Hierve el yogur griego en agua durante
# 8 minutos hasta que estén firmes; pélalos y córtalos en trozos.» — huevo duro aplicado al
# lácteo, y V1 nunca lo vio: "yogur" no resolvía contra el índice construido sobre "yogurt", así
# que `find_catalog_foods` no encontraba NINGÚN alimento en ese paso y el check ni evaluaba el
# método (el scan solo disparó V3 huérfano, ciego a la violación real). Normalizado en `_norm` —
# el único punto por el que pasan TANTO los nombres del catálogo (`build_culinary_index`) COMO el
# texto escaneado (`_catalog_food_spans`/`step_has_cooking_verb`) — cubre las 4 variantes
# (yogur/yogurt/yogures/yogurts) en AMBAS direcciones sin duplicar la normalización en cada
# callsite. `\byogurt(s)?\b` → "yogur"/"yogurs"; combinado con `_sing_plural_pattern("yogur")`
# ("yogur(?:e?s)?") en el índice, las 4 formas convergen al mismo match.
_YOGUR_T_RE = re.compile(r"\byogurt(s)?\b")


def _norm(text: str) -> str:
    s = strip_accents(str(text or "").lower())
    return _YOGUR_T_RE.sub(lambda m: "yogur" + (m.group(1) or ""), s)


# [P1-CULINARY-CONTRACT-BATTER · 2026-08-01] Clase FP "mezclas horneadas" —
# única clase que aún bloqueaba el reloj de F2 (necesita ≥7 días de warn
# limpio; una torta de avena horneada dispara en CADA plan con desayuno
# horneado). Caso real (plan 054a43c2, día 3 Desayuno, "El Toque de Fuego:
# Precalienta el horno a 180°C. Integra la avena, la leche descremada, el
# polvo de hornear, la canela y la vainilla..."): V1 acusaba a los 5
# alimentos integrados porque NINGUNO lista 'hornear' individualmente en
# `prep_methods` — cierto, pero irrelevante: el nombre del alimento "Polvo
# de hornear" contiene LITERALMENTE la palabra "hornear", así que el propio
# regex de `VERB_TO_METHOD` lo lee como una ocurrencia real del verbo dentro
# de la misma cláusula que lo integra ("Integra la avena, ..., el polvo de
# hornear, ..."). El plato (baked oats) es culinariamente NORMAL: lo que se
# hornea es LA MEZCLA resultante de integrar los ingredientes, no cada
# componente por separado — nadie espera que "Polvo de hornear" por sí solo
# acepte 'hornear' como método, igual que nadie espera que "Avena" cruda
# acepte 'hornear': el batter entero sí, sus partes no.
#
# Regla: si una cláusula integra (verbo de MEZCLA_VERBOS) a ≥3 alimentos
# resolubles, los verbos de COCCIÓN que caigan en ESA MISMA cláusula juzgan
# a la mezcla resultante, no a sus componentes → V1 se salta esos alimentos
# para esa ocurrencia. Scope deliberadamente ACOTADO a la cláusula de
# integración (no "el paso completo", no "cláusulas adyacentes") — un paso
# que integra 2 alimentos (bajo el umbral) y LICÚA un tercero en una
# oración aparte NO debe exentar al tercero: "mantén simple" > generalizar
# de más sobre un solo caso real. El caso real en sí no necesita mirar la
# oración anterior ("Precalienta el horno", clause 1) porque esa cláusula
# no menciona NINGÚN alimento — sin alimentos no hay acusación posible ahí,
# con o sin esta regla (ver `if not foods: continue` más abajo).
#
# Verbos en forma imperativa (la receta SIEMPRE instruye en 2ª persona) con
# el mismo anti-participio de VERB_TO_METHOD — "ya integrado"/"bien
# mezclado" describe algo YA HECHO, no una instrucción de integrar ahora.
# 'bate' y 'une' NO usan el patrón raíz+comodín de los demás: "bat\w*"
# colisionaría con "Batata" (alimento real del catálogo) y "un\w*"
# colisionaría con los artículos "un"/"una"/"unos"/"unas" — el mismo modo de
# fallo documentado repetidas veces en este archivo (bug de 14 apariciones:
# "sal"⊂"salami"/"sal"⊂"ensalada"). Ambos usan alternancia de formas EXACTAS
# (imperativo + conjugaciones comunes de receta) en vez de raíz abierta.
# 'integr' además excluye explícitamente el adjetivo "integral"/"integrales"
# (pan/arroz/tortilla integral — calificador COMÚN en la prosa dominicana,
# ya protagonista de un FP real: round 2 caso FP-C, "Tuesta la tortilla
# integral..."): sin el lookahead `al(?:es)?\b`, "integral" matchearía como
# si fuera una conjugación del verbo "integrar".
MEZCLA_VERBOS = frozenset({
    "integra", "mezcla", "combina", "bate", "incorpora", "une", "revuelve",
})
_MEZCLA_RE = re.compile(
    r"\b(?:"
    r"integr(?!al(?:es)?\b|ad[oa]s?\b)\w*"
    r"|mezcl(?!ad[oa]s?\b)\w*"
    r"|combin(?!ad[oa]s?\b)\w*"
    r"|incorpor(?!ad[oa]s?\b)\w*"
    r"|revuelv\w*"
    r"|bat(?:e|es|imos|en|an|id)\b"
    r"|un(?:e|en|imos)\b"
    r")", re.IGNORECASE)

# Umbral mínimo de alimentos "aplicado a ≥3 alimentos resolubles" (letra del
# diseño) — por debajo de esto, un paso que integra 1-2 alimentos y cocina
# uno de ellos individualmente sigue siendo un caso normal de V1 (p.ej.
# "Integra el huevo y bate; cuece a fuego lento" con un único alimento no
# debe volverse inmune solo por la presencia del verbo "bate").
_MEZCLA_MIN_ALIMENTOS = 3


def _es_clausula_mezcla(clause_text: str, foods: list) -> bool:
    """True si `clause_text` (ya `_norm`, delimitada por `_clause_bounds`)
    integra ≥`_MEZCLA_MIN_ALIMENTOS` alimentos resolubles bajo un verbo de
    `MEZCLA_VERBOS` — en ese caso los verbos de cocción de ESTA MISMA
    cláusula juzgan a la mezcla resultante, no a cada componente (ver
    bloque de comentario arriba). `foods` ya viene calculado por el caller
    (`find_catalog_foods` sobre `clause_text`) para no re-escanear dos
    veces por ocurrencia."""
    return len(foods) >= _MEZCLA_MIN_ALIMENTOS and bool(_MEZCLA_RE.search(clause_text))


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
                # [P1-CULINARY-CONTRACT-BATTER] salvaguarda "mezcla": esta
                # cláusula integra ≥3 alimentos resolubles bajo un verbo de
                # MEZCLA_VERBOS — el verbo de cocción de ESTA ocurrencia
                # juzga a la mezcla resultante, no a cada componente (caso
                # real: "Integra la avena, la leche descremada, el polvo de
                # hornear, la canela y la vainilla..." — 'hornear' matchea
                # DENTRO del nombre "Polvo de hornear"). Ver bloque de
                # comentario sobre `_es_clausula_mezcla` más arriba.
                if _es_clausula_mezcla(clause_text, foods):
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


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · V4 · 2026-08-01] Consistencia de cantidades
# ingredientes↔Mise en place. Caso real (plan 5f4bb17e, capturas del owner):
# `ingredients` dice "30 g de queso" pero el paso de Mise en place dice
# "desmenuza 1¾ lonjas/pedazos de queso de hoja (45 g)" — 30≠45, ambos en
# GRAMOS del MISMO alimento, y el usuario no sabe a cuál creer.
#
# REGLAS DURAS:
#   (a) SOLO compara gramos con gramos — nunca inventa una conversión
#       taza/cdta/unidad → gramos. Si un lado no declara "N g" explícito,
#       se salta esa comparación en silencio (no es una violación "no
#       comparable", simplemente no aplica).
#   (b) El alimento se resuelve con el matcher canónico del módulo
#       (`find_catalog_foods`/`_catalog_food_spans`, word-boundary + alias
#       más largo gana) — jamás substring.
#   (c) Si un alimento tiene gramaje explícito en varios pasos, manda la
#       PRIMERA mención de Mise en place; si Mise en place no lo declara,
#       cae al primer paso (en orden) que sí lo declare.
#   (d) Fail-open total — hereda el try/except de `culinary_contract_scan`.
# ---------------------------------------------------------------------------

_V4_GRAMS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?)\b", re.IGNORECASE)
_RE_MISE_STEP = re.compile(r"^\s*mise en place\s*:", re.IGNORECASE)

# [V4-FIX3 · 2026-08-01] Aproximación declarada (≈/~) NO es un contrato de cantidad — es
# honestidad del propio sistema ("más o menos", no "exactamente"). Caso real: el humanizador
# anota hints parentéticos aproximados sobre unidades vagas (lonja/pedazo/porción) vía
# `append_gram_hint` (`humanize_ingredients.py`) — "21.5 molondrones medianos (≈322 g)" — y sin
# este skip, V4 comparaba ese número aproximado contra el gramaje EXACTO del otro lado y disparaba
# un falso positivo contra un hint que el propio sistema generó, no contra un desacuerdo real.
# Ancla al final de la porción `[≈~]\s*$` (marcador inmediatamente antes del número, con o sin
# espacio de por medio: "≈20 g" / "≈ 20 g" / "~20 g") — SOLO descarta la mención marcada, la
# comparación del alimento se salta EN SILENCIO (mismo criterio que la regla dura (a): no
# inventar, no acusar donde no hay contrato). El caso real 30↔45 g (plan 5f4bb17e, ambos números
# EXACTOS, sin ≈/~) sigue disparando sin cambios — solo lo aproximado se exime.
_V4_APPROX_LEAD_RE = re.compile(r"[≈~]\s*$")


def _v4_grams_by_food(text_norm: str, index: dict) -> dict:
    """{food: gramos} de TODAS las menciones con gramaje explícito de
    `text_norm` (ya normalizado, `_norm`). Empareja cada número "N g" con el
    alimento catalogado MÁS CERCANO dentro de la MISMA cláusula (oración,
    `clause_bounds`) — nunca con el primero que matchee en todo el texto:
    "mide 15 g de merey y 10 g de granola" debe emparejar 15↔merey y
    10↔granola por PROXIMIDAD, no ambos con el primer alimento que aparezca.
    Si un alimento tiene ≥2 menciones con gramaje en el mismo texto, se queda
    con la PRIMERA (orden de aparición, cláusula por cláusula). Menciones
    precedidas de ≈/~ (aproximación declarada) se DESCARTAN — ver
    `_V4_APPROX_LEAD_RE`."""
    out = {}
    for c_start, c_end in clause_bounds(text_norm):
        clause = text_norm[c_start:c_end]
        foods = _catalog_food_spans(clause, index)
        if not foods:
            continue
        grams = [(m.start(), m.end(), float(m.group(1).replace(",", ".")))
                 for m in _V4_GRAMS_RE.finditer(clause)
                 if not _V4_APPROX_LEAD_RE.search(clause[:m.start()])]
        for g_start, g_end, val in grams:
            best_food, best_dist = None, None
            for f_start, f_end, f_name in foods:
                if g_start >= f_end:
                    dist = g_start - f_end
                elif g_end <= f_start:
                    dist = f_start - g_end
                else:
                    dist = 0
                if best_dist is None or dist < best_dist:
                    best_dist, best_food = dist, f_name
            if best_food is not None and best_food not in out:
                out[best_food] = val
    return out


def _v4_cantidad_inconsistente(day, meal, index) -> list:
    out = []
    ingredientes = meal.get("ingredients") or []
    pasos = meal.get("recipe") or []

    # Lado ingrediente: convención del repo es 1 alimento resoluble por
    # renglón ("CADA CONDIMENTO EN SU PROPIO RENGLÓN") — se resuelve el
    # primer alimento del renglón (mismo criterio que V3, `foods[0]`) y se
    # busca SU gramaje dentro de ESE MISMO renglón.
    ing_grams = {}
    for ing in ingredientes:
        n = _norm(str(ing))
        foods = find_catalog_foods(n, index)
        if not foods:
            continue
        food = foods[0]
        if food in ing_grams:
            continue          # renglón duplicado del mismo alimento: se queda con el primero
        pares = _v4_grams_by_food(n, index)
        if food in pares:
            ing_grams[food] = pares[food]

    if not ing_grams:
        return out             # ningún ingrediente declara gramaje explícito ⇒ nada que comparar

    # Lado pasos: prioriza Mise en place (regla c); si un alimento no
    # declara gramaje ahí, cae al primer paso (en orden) que sí lo declare.
    mise_grams, primer_grams = {}, {}
    for paso in pasos:
        n = _norm(str(paso))
        pares = _v4_grams_by_food(n, index)
        es_mise = bool(_RE_MISE_STEP.match(n))
        for food, val in pares.items():
            if food not in primer_grams:
                primer_grams[food] = val
            if es_mise and food not in mise_grams:
                mise_grams[food] = val

    for food, ing_val in ing_grams.items():
        paso_val = mise_grams.get(food, primer_grams.get(food))
        if paso_val is None:
            continue           # ningún paso declara gramaje explícito para este alimento ⇒ skip
        denom = max(ing_val, paso_val)
        if denom <= 0:
            continue
        if abs(ing_val - paso_val) / denom > V4_TOLERANCIA:
            out.append(_viol(day, meal, "V4", food,
                             f"ingrediente declara {ing_val:g} g, pasos declaran {paso_val:g} g",
                             "minor", False))
    return out



# ---------------------------------------------------------------------------
# V5 — el espejo de V3: el paso USA algo que la lista NO trae
# ---------------------------------------------------------------------------
# [P1-CULINARY-V5-GHOST-STEP · 2026-09-06] V3 pregunta «¿hay un ingrediente que ningún paso
# menciona?». Nadie preguntaba lo contrario, y es la categoría más frecuente del juez culinario
# (`paso_incoherente`, 96 de 227 comidas señaladas): «coloca el cilantro por encima» en un plato
# cuya lista trae orégano, «añade la piña» a un cottage con manzana. El usuario compra la lista y
# la receta le manda usar algo que no tiene.
#
# Cinco filtros, y cada uno nació de un falso positivo MEDIDO sobre 1.186 comidas vivas. El detector
# ingenuo daba 460 acusaciones; éste da 11, con 10 reales juzgadas a mano:
#
#   460 → 364  el índice devuelve el alias corto Y el largo: «yogurt griego» casaba también `Yogur`
#   364 → 287  las notas de seguridad hablan de CLASES en abstracto («el pollo/cerdo debe cocinarse»)
#   287 → 241  la lista y el paso nombran el mismo alimento con alias distintos
#   241 → 100  el índice no resuelve «1½ filetes de pescado», y eso NO significa que no esté
#   100 →  11  un paso que USA lo nombra tras un verbo de entrada; uno que lo PRODUCE, no
#
# El cuarto es el que más enseña: sin él el detector medía el recall del CATÁLOGO y acusaba al plan
# de su propia ceguera — un ceviche con su pescado en la lista salía acusado de no tenerlo.
#
# Severidad `minor` y `repairable=False`: es telemetría. No escala a bloqueo sin un golden set
# humano; subirlo a partir de la tasa del propio juez sería el overfitting que ya se pagó en agosto.

#: Notas de seguridad: hablan de clases de alimento en abstracto, no de los ingredientes del plato.
_V5_NOTA = re.compile("seguridad alimentaria|riesgo de salmonella|^\\s*\u26a0", re.IGNORECASE)

#: Un paso que NIEGA un alimento no lo está usando: «se reemplazó el huevo crudo por yogur».
_V5_NEGACION = re.compile(
    r"(se reemplaz|reemplaza|en lugar de|en vez de|sustituy|se omit|se retir|no uses?|"
    r"sin\s+(?:el|la|los|las)\s|se elimin|se quit)", re.IGNORECASE)

#: Verbos de ENTRADA. Un paso que consume un ingrediente lo nombra como su objeto; uno que lo
#: produce lo nombra como resultado («hasta formar el sofrito», «hasta que cuajada»).
_V5_ENTRADA = re.compile(
    r"\b(?:mide|midiendo|anade|anadir|agrega|agregar|incorpora|incorporar|coloca|colocar|pon|poner|"
    r"echa|echar|sirve|servir|acompana|distribuye|reparte|espolvorea|unta|corta|cortar|pica|picar|"
    r"pela|pelar|lava|lavar|ralla|rallar|trocea|exprime|escurre|bate|batir|mezcla|mezclar|licua|"
    r"hidrata|porciona|reserva|vierte|cubre|termina con)\b")


def _v5_mas_especifico(food: str) -> list:
    return [w for w in re.split(r"[^a-z0-9]+", food) if len(w) >= 4]


def _v5_resueltos(texto: str, index: dict) -> set:
    """Alimentos de `texto`, quedandose SOLO con el mas especifico.

    `find_catalog_foods` devuelve el alias corto y el largo: «yogurt griego sin azucar» casa `Yogur`
    y `Yogurt griego sin azucar`. La linea de ingrediente resuelve al largo, asi que el corto se
    convertia en fantasma — 128 de las 460 acusaciones de la primera version."""
    fs = [_norm(f) for f in find_catalog_foods(texto, index)]
    return {f for f in fs if not any(f != o and f in o for o in fs)}


def _v5_en_texto(food: str, crudo_norm: str) -> bool:
    """¿Alguna palabra significativa del alimento aparece LITERALMENTE en la lista cruda?

    Deliberadamente permisivo: este check solo debe disparar cuando NADA en la lista se parece. Sin
    el, «1½ filetes de pescado» —que el indice no resuelve a `Filete de pescado blanco`— acusaba a
    un ceviche de no llevar pescado. Un detector que confunde «no lo encuentro» con «no esta» acusa
    al plan de su propia ceguera."""
    palabras = _v5_mas_especifico(food)
    if not palabras:
        return True                       # nombre demasiado corto para afirmar nada: no se acusa
    return any(re.search(r"\b" + re.escape(w) + r"(?:s|es)?\b", crudo_norm)
               for w in palabras)


def _v5_paso_mas_especifico(pnorm: str, pos: int, crudo_norm: str) -> bool:
    """¿El paso nombra el alimento con MÁS detalle que la lista? «pica la chuleta de cerdo» cuando la
    lista dice «½ chuleta»: es el mismo alimento, no un fantasma.

    Se mira una ventana corta alrededor del match. Ensancharla o bajar el umbral de palabra parece
    inofensivo y NO lo es: probado sobre las 1.186 comidas, con palabras de 3 letras el filtro se
    tragaba «con», «las» y «una» —presentes en toda lista— y el detector caía a CERO, llevandose por
    delante los hallazgos reales. Un filtro que descarta todo no es preciso, es ciego."""
    if pos < 0:
        return False
    ventana = pnorm[max(0, pos - 28):pos + 28]
    for w in re.split(r"[^a-z0-9]+", ventana):
        if len(w) >= 5 and re.search(r"\b" + re.escape(w) + r"(?:s|es)?\b", crudo_norm):
            return True
    return False


def _v5_paso_usa_lo_que_no_esta(day, meal, index) -> list:
    """[P1-CULINARY-V5-GHOST-STEP] El espejo de V3. Fail-open total."""
    out = []
    try:
        ings = [str(x) for x in (meal.get("ingredients") or [])]
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not ings or not pasos:
            return out
        lista = set()
        for ing in ings:
            lista |= _v5_resueltos(ing, index)
        # El NOMBRE del plato tambien declara: «Vaso de yogur» no tiene que repetirlo en la lista.
        lista |= _v5_resueltos(str(meal.get("name") or ""), index)
        crudo = _norm(" | ".join(ings) + " | " + str(meal.get("name") or ""))

        for paso in pasos:
            if _V5_NOTA.search(paso) or _V5_NEGACION.search(paso):
                continue
            pnorm = _norm(paso)
            for food in _v5_resueltos(paso, index):
                if any(food in l or l in food for l in lista):
                    continue                       # el mismo alimento con otro alias
                if any(rx.search(food) for rx in _CONDIMENT_EXEMPT_RES):
                    continue                       # condimentos: reusa CONDIMENT_EXEMPT
                cabeza = (_v5_mas_especifico(food) or [""])[0]
                if not cabeza:
                    continue
                m = re.search(r"\b" + re.escape(cabeza) + r"(?:s|es)?\b", pnorm)
                if not m:
                    continue                       # el match era un verbo, no el alimento
                if _v5_en_texto(food, crudo):
                    continue                       # el indice no lo resolvio, pero SI esta
                if _v5_paso_mas_especifico(pnorm, m.start(), crudo):
                    continue                       # «chuleta de cerdo» cuando la lista dice «chuleta»
                if not _V5_ENTRADA.search(pnorm[max(0, m.start() - 60):m.start()]):
                    continue                       # la receta lo PRODUCE, no lo consume
                out.append(_viol(day, meal, "V5", food,
                                 f"el paso lo usa pero la lista no lo trae: {paso[:110]}",
                                 "minor", False))
    except Exception:
        return []
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# V6 — el paso PIDE MÁS de lo que la lista compra, en unidades contables
#
# [P1-CULINARY-V6-STEP-OVERASK · 2026-09-06] V4 compara GRAMOS. Nadie miraba las piezas: la lista
# dice «½ diente de ajo» y el paso «pica 1 diente de ajo»; «3 rebanadas de pan» y «mide 4
# rebanadas»; «½ cda de aceite» y «mide 1 cda». Medido sobre 96 planes: 36 comidas (3,0 %), 13
# alimentos, y el patrón es uno solo — **el modelo redondea las fracciones hacia arriba al
# recitar la lista en el «Mise en place»**.
#
# Dos decisiones que hacen esto medible y no ruidoso:
#
# 1. **Solo se acusa cuando el paso pide MÁS.** Un paso que usa MENOS que el total puede estar
#    repartiendo el ingrediente entre pasos («calienta 1 cda» de las 2 que compra, y el resto
#    después) — legítimo, y contarlo castigaría a la receta bien escrita. Uno que pide más no
#    tiene de dónde sacarlo. La dirección es la que distingue el defecto del reparto.
# 2. **La unidad es obligatoria.** Se probó admitir la mención sin unidad («2 guineítos» contra
#    «½ guineíto») y sube de 36 a 153 hallazgos con ruido demostrable: sin unidad que ancle el
#    número al alimento, se le pega cualquier cifra vecina — «coloca el Batata como base» heredó
#    el «3» de otra frase, y un «huevo 4.0» salió de «2 minutos por lado». Descartado MEDIDO.
#
# `g`/`ml` se reconocen para que «355 g de lechosa» no caiga al cubo de las piezas, pero NO se
# comparan: la coherencia en gramos ya es de V4 (P1-STEP-GRAM-HINT-STALE).
#
# V6 no decide de qué LADO está el error, y no debe: en «Canoas de repollo» la lista pedía ½ hoja
# y el paso 6 hojas — ahí la equivocada era la lista, porque con media hoja no hay canoas. Lo que
# afirma es que los dos se contradicen.
_V6_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_V6_CONTABLE = (r"unidades?|dientes?|rebanadas?|lonjas?|tazas?|cucharadas?|cucharaditas?|cdas?|"
                r"cdtas?|gajos?|ramitas?|filetes?|pedazos?|hojas?|tallos?|latas?|paquetes?")
_V6_MASA = r"gramos?|g|kg|mililitros?|ml|litros?|l|onzas?|oz|lb|libras?"
_V6_MASA_RE = re.compile(_V6_MASA, re.IGNORECASE)
_V6_RE = re.compile(r"(\d+(?:[.,]\d+)?|[½¼¾⅓⅔⅛]|\d[½¼¾⅓⅔])\s*"
                    r"(" + _V6_CONTABLE + "|" + _V6_MASA + r")\s*(?:de\s+)?([a-zñ ]{3,28})")


def _v6_valor(txt: str):
    """«½» → 0.5, «1½» → 1.5, «0,33» → 0.33. None si no es un número que entienda."""
    t = (txt or "").strip()
    if t in _V6_FRAC:
        return _V6_FRAC[t]
    m = re.match(r"^(\d+)([½¼¾⅓⅔])$", t)
    if m:
        return float(m.group(1)) + _V6_FRAC.get(m.group(2), 0)
    try:
        return float(t.replace(",", "."))
    except Exception:
        return None


def _v6_cuentas(texto: str, index: dict) -> dict:
    """{alimento: {(unidad_singular, valor)}} de las menciones CONTABLES del texto.

    La clave es el NOMBRE DEL CATÁLOGO («Ajo»), no su forma normalizada: V1-V4 reportan así, y una
    agrupación aguas abajo que viera «Ajo» y «ajo» los contaría como dos alimentos. La forma
    normalizada se usa solo para el filtro de especificidad, igual que en V5."""
    out: dict = {}
    for m in _V6_RE.finditer(_norm(texto)):
        val = _v6_valor(m.group(1))
        if val is None or _V6_MASA_RE.fullmatch(m.group(2) or ""):
            continue                                   # la masa se reconoce y se descarta: es V4
        uni = re.sub(r"e?s$", "", m.group(2))
        # se queda el alias MÁS específico: «yogurt griego sin azúcar» casa también `Yogur`
        crudos = list(find_catalog_foods(m.group(3), index))
        normas = {f: _norm(f) for f in crudos}
        for f in crudos:
            if any(f != o and normas[f] in normas[o] for o in crudos):
                continue
            out.setdefault(f, set()).add((uni, round(val, 3)))
    return out


def _v6_paso_pide_mas_que_la_lista(day, meal, index) -> list:
    """[P1-CULINARY-V6-STEP-OVERASK] Fail-open total."""
    out = []
    try:
        ings = [str(x) for x in (meal.get("ingredients") or [])]
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not ings or not pasos:
            return []
        en_lista: dict = {}
        for ing in ings:
            for food, pares in _v6_cuentas(ing, index).items():
                en_lista.setdefault(food, set()).update(pares)
        for paso in pasos:
            for food, pares in _v6_cuentas(paso, index).items():
                if food not in en_lista:
                    continue                           # eso es V5, no V6
                for uni, val in pares:
                    vals = {v for u, v in en_lista[food] if u == uni}
                    if not vals:
                        continue                       # otra unidad: no comparable sin densidad
                    total = max(vals)
                    # tolerancia: «⅓ taza» y «0.33 taza» son la misma cantidad
                    if val <= total + max(0.06, 0.05 * total):
                        continue                       # usa lo mismo o menos: puede repartir
                    out.append(_viol(day, meal, "V6", food,
                                     f"el paso pide {val:g} {uni} y la lista compra "
                                     f"{min(vals):g}: {paso[:100]}",
                                     "minor", False))
    except Exception:
        return []
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
            out.extend(_v4_cantidad_inconsistente(day, meal, index))
            out.extend(_v5_paso_usa_lo_que_no_esta(day, meal, index))
            out.extend(_v6_paso_pide_mas_que_la_lista(day, meal, index))
        return out
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────────────────────
# La huella de lo que el juez MIRÓ
#
# [P1-JUDGE-REVISION-STAMP · 2026-09-06] `_culinary_judge_history` guardaba `{ts, model,
# violations, action_taken}` — nada que atara una entrada a una VERSIÓN del plan. Con eso, «el juez
# se quejó y lo arreglamos» y «se quejó y lo entregamos» son indistinguibles, y su tasa se lee como
# si fuera de defectos ENTREGADOS.
#
# No es una sospecha. Medido el 2026-09-06 sobre 96 planes: de las 37 quejas juzgables del tipo
# «X no aparece en la lista», **6 nombraban algo que SÍ está en el plan entregado** (almendras,
# pistachos, guineítos verdes, queso cottage). Y el sub-patrón más citado —«los ingredientes dicen
# 4¾ lonjas/pedazos de queso» cuando el plato lleva cottage— aparece **0 veces de 23** líneas de
# queso con lonja en planes vivos: la reparación ya lo había convertido antes de entregar.
#
# La huella cubre NOMBRE, INGREDIENTES y PASOS porque eso es exactamente lo que el juez lee.
# **No se reutiliza `services.compute_plan_hash`** pese a que se declara «fuente única de verdad
# para detectar si un plan cambió»: hashea ingredientes y suplementos, y el bucket más grande del
# juez (`paso_incoherente`) es de PASOS. Un paso reparado dejaría ese hash quieto y la comparación
# diría «es el mismo plan» justo en los casos que más importan — una huella que no cubre lo que se
# juzgó reintroduce la misma ambigüedad, sólo que más difícil de ver.
#
# Devuelve `None` cuando no puede calcularla, y quien la consuma DEBE tratar ese `None` como
# **desconocido**, jamás como «coincide»: las entradas anteriores a este P-fix no la llevan, y
# colapsar lo desconocido hacia cualquiera de los dos lados es cómo se fabrica una cifra falsa.
# tooltip-anchor: P1-JUDGE-REVISION-STAMP
def judged_fingerprint(plan_data: dict) -> "str | None":
    """SHA-256 truncado de (día, franja, nombre, ingredientes, pasos) de cada comida."""
    try:
        filas = []
        for di, d in enumerate((plan_data or {}).get("days") or [], 1):
            for m in (d.get("meals") or []):
                if not isinstance(m, dict):
                    continue
                filas.append([di, str(m.get("meal") or ""), str(m.get("name") or ""),
                              [str(x) for x in (m.get("ingredients") or [])],
                              [str(x) for x in (m.get("recipe") or [])]])
        if not filas:
            return None
        crudo = json.dumps(filas, sort_keys=False, ensure_ascii=False)
        return hashlib.sha256(crudo.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def judgment_covers_delivered(entry: dict, plan_data: dict) -> "bool | None":
    """¿La entrada del historial juzgó el plan que se entregó? `None` = no se puede saber.

    Tres estados, no dos, y el tercero es el importante: una entrada sin `judged_fingerprint`
    (todas las anteriores a este P-fix) no dice ni que sí ni que no. Devolver `False` ahí
    convertiría «no lo sé» en «se reparó», que es precisamente la confusión que este P-fix cierra.
    """
    try:
        sello = (entry or {}).get("judged_fingerprint")
        if not sello:
            return None
        actual = judged_fingerprint(plan_data)
        return None if not actual else (sello == actual)
    except Exception:
        return None


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
