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
from pathlib import Path

from constants import strip_accents

import logging
# [P1-PLAN-LOTE-6 · F5] los guards de este módulo se tragaban su fallo sin rastro: ahora hay a quién contárselo.
logger = logging.getLogger(__name__)

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
#
# [P1-PLAN-LOTE-52 · 2026-09-15] El punto ENTRE DOS CIFRAS es un decimal, no un fin de oración. Con `[.;]`, «0.23 g de
# Sal» se partía en «0» | «23 g de sal» y V4 y el reparador leían 23 g (×100); «57.2 g» se leía como 2 g. La coma nunca
# fue frontera: por eso «57,2 g» sí se leía bien. Medido sobre los 11 planes de los últimos 21 días: 69 de 172 comidas
# tienen alguna mención así, y el scan y el reparador dan lo MISMO con una frontera y con otra. Hoy es inerte, porque los
# pasos casi nunca citan gramos de un condimento, pero `formatear_cantidad` escribe los decimales con punto. «Añade 2.
# Luego…» sigue partiendo: basta con que falte la cifra a un lado.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<!\d)\.|\.(?!\d)|;")


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


def _index_entry(name: str, norm: str, row: dict) -> dict:
    tokens = [_sing_plural_pattern(t) for t in norm.split()]
    return {
        "name": name,
        "prep_methods": row.get("prep_methods"),
        "ready_to_eat": row.get("ready_to_eat"),
        "rx": re.compile(r"\b" + r"\s+".join(tokens) + r"\b"),
    }


def build_culinary_index(catalog: list) -> dict:
    """Índice nombre-normalizado → metadata + regex word-boundary del alias.

    [P1-CULINARY-ALIAS-INDEX · 2026-09-07] Carga TAMBIÉN `master_ingredients.aliases`.
    Hasta hoy leía sólo `row["name"]` — mientras el docstring de esta función decía
    «regex del ALIAS» y `find_catalog_foods` decía «alias más largo gana». El código
    hablaba de alias por todas partes y no cargaba ninguno: el vocabulario de TODA la
    capa culinaria eran los 348 nombres canónicos. Medido sobre la flota, **597 de 1.194
    comidas (50 %)** mencionaban al menos un alimento que ninguna capa podía ver —
    `Clara de huevo` (+107), `Yema de huevo` (+116), `Yogurt griego entero` (+87),
    `Queso blanco` (+73). Un detector no puede acusar lo que no sabe nombrar.

    Tres reglas de seguridad, porque ampliar vocabulario es la vía clásica al falso
    positivo (19 colisiones por subcadena documentadas en este repo):

    1. **El nombre canónico SIEMPRE gana.** Se indexan los 348 nombres primero; ningún
       alias de otro alimento puede desplazar a uno (`repollo morado` es alias de
       `Repollo` y nombre de `Repollo morado`: manda el segundo).
    2. **Un alias ambiguo se DESCARTA, no se reparte.** `mariscos` lo reclaman Pulpo,
       Calamar y Mejillones; `nueces`, Nueces mixtas y Almendras fileteadas. Elegir uno
       por orden de fila sería inventar una identidad que el dato no tiene.
    3. Los `\\b` + `_sing_plural_pattern` que ya usaba el nombre valen igual para el
       alias, así que `res` (→ Carne de res) no casa dentro de «queso f-res-co» ni `sal`
       dentro de «ensalada». Es la defensa que faltaba en las 19 colisiones, no una nueva.

    Efecto colateral que es CORRECCIÓN, no regresión: `Yogurt` pierde 68 comidas y `Sal`
    8, porque «yogurt griego entero» y «mantequilla sin sal» pasan a resolver al alimento
    largo — hoy resuelven al corto, que es el alimento equivocado (el yogurt normal tiene
    3,47 g de proteína; el griego, 8,78).
    """
    index = {}
    filas = [r for r in (catalog or []) if isinstance(r, dict) and str(r.get("name") or "").strip()]

    for row in filas:                                   # regla 1: canónicos primero
        name = str(row["name"]).strip()
        norm = _norm(name)
        if norm:
            index[norm] = _index_entry(name, norm, row)

    # Regla 3: una palabra que aparece como TOKEN en el nombre canónico de dos alimentos
    # distintos no identifica a ninguno. `platano` es token de `Plátano maduro` y de
    # `Plátano verde`; `soya`, de `Salsa de soya` y `Soya texturizada`. Sin esto, un paso que
    # dice «maja el plátano» en una receta de plátano MADURO añadía un span de `Plátano verde`
    # que desplazaba al objeto real del verbo — así se perdió la captura de V1 sobre unas
    # lentejas horneadas. La ambigüedad se mide contra los nombres canónicos, no solo entre
    # alias: dos alias que chocan entre sí son el caso fácil.
    tok_duenos: dict = {}
    for row in filas:
        nombre_canon = str(row["name"]).strip()
        for t in _norm(nombre_canon).split():
            tok_duenos.setdefault(t, set()).add(nombre_canon)
    tokens_ambiguos = {t for t, d in tok_duenos.items() if len(d) > 1}

    duenos: dict = {}                                   # regla 2: alias → alimentos que lo reclaman
    for row in filas:
        name = str(row["name"]).strip()
        al = row.get("aliases")
        if not isinstance(al, list):
            continue
        for a in al:
            na = _norm(str(a))
            if not na or na in index:                   # el canónico ya ganó: ni se mira
                continue
            # Regla 4: un alias de UNA palabra que nombra una FORMA no identifica un
            # alimento. `Harina de trigo` lleva el alias literal «harina» y `Pasta integral`
            # el alias «pasta»; con ellos dentro, «muele la avena hasta obtener una harina
            # fina» daba por usada la harina de trigo. Es LA MISMA ceguera que
            # `_V3_FORMA_GENERICA` cerró en `_mencionado_por_prefijo` (P1-CULINARY-V7), y
            # entró por otra puerta: aquel guard filtra PREFIJOS del nombre canónico, y un
            # alias llega como clave entera, así que jamás pasaba por él. Medido: sin esta
            # regla, V3 perdía sus tres mejores capturas («la harina de trigo queda sin
            # utilizar»), y «harina de avena» resolvía a `Harina de trigo`.
            if " " not in na and (na in _V3_FORMA_GENERICA or na in tokens_ambiguos):
                continue
            duenos.setdefault(na, {})[name] = row

    for na, reclamantes in duenos.items():
        if len(reclamantes) != 1:
            continue                                    # ambiguo ⇒ fuera, no se reparte
        name, row = next(iter(reclamantes.items()))
        index[na] = _index_entry(name, na, row)

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


# [P1-CULINARY-V7 · 2026-09-07] Sustantivos de FORMA: lo que un alimento PUEDE LLEGAR A SER, no
# lo que es. La prosa de una receta los produce sola («hasta obtener una harina», «hasta formar una
# pasta»), así que como prefijo de una palabra no prueban que el alimento esté mencionado.
_V3_FORMA_GENERICA = frozenset({
    "harina", "polvo", "pasta", "crema", "pure", "masa", "salsa", "caldo", "jugo", "mezcla",
    "aderezo", "trozos", "tiras", "cubos", "hojuelas",
})


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
        # [P1-CULINARY-V7 · 2026-09-07] Un prefijo de UNA palabra que nombra una FORMA —no una
        # identidad— lo produce la prosa a partir de cualquier ingrediente, y aceptarlo ciega a V3.
        #
        # Caso real del golden set (028ad9ed64): el paso dice «muele la avena hasta obtener una
        # HARINA fina» y esto daba por mencionada la «Harina de trigo», que estaba genuinamente
        # huérfana — 40 g comprados y jamás usados. El guard de ambigüedad de abajo no lo veía
        # porque la colisión no es con otro alimento de la comida, es con una palabra que la prosa
        # FABRICA al describir una técnica.
        #
        # «Arroz blanco» → «arroz» sigue funcionando: `arroz` es una identidad, no una forma.
        if k == 1 and prefijo[0] in _V3_FORMA_GENERICA:
            continue
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


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-25 · 2026-09-12] (C4 · H8) La receta CONGELADA trae su propia contabilidad
#
# Los pasos de la biblioteca no llevan cantidades de ingrediente a propósito, así que V6/V7a/V7e —que leen NÚMEROS del
# texto— callan sobre ellos, y V3 sólo puede preguntar «¿lo nombra algún paso?». Medido materializando las 193 recetas
# como plato: el escáner de hoy emite 6 hallazgos (V1 4, V3 1, V5 1) y ninguno de cantidad — «la mitad del aceite» …
# «la otra mitad del aceite» … «la otra mitad del aceite» pasa limpio porque nadie podía mirarlo.
#
# `recipe_usage` asigna a cada paso los constituyentes que usa y con qué FRACCIÓN de lo comprado, atado al texto por
# hash: con asignación vigente, estos cuatro checks leen las CUENTAS (Σ por constituyente) en vez de adivinar. «Usa lo
# mismo o menos: puede repartir» deja de ser una excusa — se sabe cuánto reparte cada paso y si la suma cierra.
# Knob `MEALFIT_RECIPE_USAGE_EXACT` (default True): apagado ⇒ la heurística de siempre, sin redeploy.
#
# La CADENA de checks de `culinary_contract_scan` no cambia (los tests la leen como texto): cada check se cede a sí mismo
# desde dentro. Fail-open: cualquier excepción devuelve `None` y corre la heurística. tooltip-anchor: P1-PLAN-LOTE-25-SCAN-EXACT

def _cuentas_exactas(meal):
    """Las cuentas de `recipe_usage` para una comida de receta congelada con asignación vigente; `None` si no aplica
    (comida del LLM, receta reescrita en el plato, knob apagado, biblioteca sin asignación, o cualquier excepción)."""
    try:
        from recipe_usage import cuentas_para_comida
        return cuentas_para_comida(meal)
    except Exception:
        return None


def _violaciones_exactas(day, meal, cuentas, checks: tuple) -> list:
    """Las violaciones que la contabilidad exacta sostiene para ESTOS checks, con la forma de `_viol` (minor, no reparable)."""
    try:
        from recipe_usage import hallazgos_para_scanner
        return [_viol(day, meal, check, food, detail, "minor", False)
                for check, food, detail in hallazgos_para_scanner(cuentas) if check in checks]
    except Exception:
        return []


def _v3_huerfanos(day, meal, index) -> list:
    _ex = _cuentas_exactas(meal)                     # [P1-PLAN-LOTE-25] receta congelada: cuentas (Σ = 0), no texto
    if _ex is not None:
        return _violaciones_exactas(day, meal, _ex, ("V3",))
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
            # [P1-PLAN-LOTE-23 · 2026-09-12] (C2) por GRAMÁTICA, no por cercanía: ver `grams_owner`.
            best_food = grams_owner(clause, g_start, g_end, foods)
            if best_food is not None and best_food not in out:
                out[best_food] = val
    return out


_GRAMS_SIGUE_RE = re.compile(r"^\s*(?:de\s+|del\s+|de\s+l[ao]s?\s+)?$")
_GRAMS_PRECEDE_RE = re.compile(r"^\s*[:(]?\s*$")


def grams_owner(clause: str, g_start: int, g_end: int, foods: list) -> "str | None":
    """[P1-PLAN-LOTE-23 · 2026-09-12] (C2 · CUL-P0-03) A qué alimento pertenece un «N g» dentro de su cláusula, por
    GRAMÁTICA: el alimento que SIGUE al número («70 g de nabo, 265 g de tomate» → 70↔nabo, 265↔tomate; con «de»/«del»
    opcional) o, si no hay, el que lo PRECEDE pegado («yogur natural (90 g)», con «(» o «:» entre medio). Si no se da
    ninguna de las dos formas, `None`: sin dueño no hay comparación.

    Antes se elegía el alimento MÁS CERCANO en caracteres, y en «corta 70 g de nabo, 265 g de tomate» el «265» estaba a
    3 caracteres del nabo y a 9 del tomate. Medido en el corpus fijo del 09-12: 2 de los 4 V4 eran esta atribución
    (los 265 g del tomate al nabo; los 90 g del yogur al maní), y el reparador del contrato final (`recipe_contract`)
    los habría REESCRITO sobre el alimento equivocado. Un medidor que atribuye por distancia no puede alimentar un
    reparador. `foods` son spans `(ini, fin, nombre)` sobre la cláusula normalizada. tooltip-anchor: P1-PLAN-LOTE-23-GRAMS-OWNER"""
    try:
        siguen = [(f_ini, f_name) for f_ini, f_fin, f_name in foods
                  if f_ini >= g_end and _GRAMS_SIGUE_RE.match(clause[g_end:f_ini])]
        if siguen:
            return min(siguen)[1]
        preceden = [(f_fin, f_name) for f_ini, f_fin, f_name in foods
                    if f_fin <= g_start and _GRAMS_PRECEDE_RE.match(clause[f_fin:g_start])]
        if preceden:
            return max(preceden)[1]
    except Exception:
        return None
    return None


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
        n = _norm(_texto_de_consumo(paso))   # [P1-PLAN-LOTE-26] el almacenaje no es consumo
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


#: [P1-PLAN-LOTE-26 · 2026-09-12] (CUL-P1-05) Una cláusula de ALMACENAJE («congela porciones de 140 g», «guarda el resto
#: en la nevera») no consume: V4/V6/V7a/V7d/V7e leen el paso sin esas cláusulas. Medido en el corpus fijo: 1 de 64 comidas
#: las traía; en la flota, congelar por porciones es lo que el ciclo de 30 días pide, así que crecerá.
_ALMACENAJE_RE = re.compile(r"\b(congel|guard|conserv|reserva para|refrigera (?:el|la|los|las) rest|porciones? de\s*\d)", re.IGNORECASE)


def _texto_de_consumo(paso) -> str:
    """El paso sin sus cláusulas de almacenaje (separadas por . ; ,)."""
    try:
        partes = re.split(r"([.;,])", str(paso or ""))
        out = []
        for k in range(0, len(partes), 2):
            cl = partes[k]
            sep = partes[k + 1] if k + 1 < len(partes) else ""
            if _ALMACENAJE_RE.search(cl):
                continue
            out.append(cl + sep)
        return "".join(out)
    except Exception:
        return str(paso or "")


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
    _ex = _cuentas_exactas(meal)                     # [P1-PLAN-LOTE-25] receta congelada: cuentas (Σ > 1), no texto
    if _ex is not None:
        return _violaciones_exactas(day, meal, _ex, ("V6",))
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
            for food, pares in _v6_cuentas(_texto_de_consumo(paso), index).items():   # [P1-PLAN-LOTE-26]
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


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-CULINARY-V7 · 2026-09-07] Las tres clases que el golden set humano destapó.
#
# De las 25 comidas que NINGUNA capa marcó, el dueño encontró defecto en 19 — recall ponderado
# 12,3 % (determinista) y 15,2 % (juez). Sus notas, escritas en 79 de 80 casos, agrupan esos
# defectos ciegos en clases, y estas tres son las mecanizables:
#
#   V7a  la lista compra N piezas y los pasos usan MENOS  (6 casos)
#        «declara dos tortillas, pero el procedimiento solo utiliza una y prepara un burrito»
#        Es el ESPEJO de V6, que solo mira el exceso (`paso > lista`). El defecto contrario —lo
#        que se compra y sobra— no lo veía nadie.
#
#   V7b  el mismo alimento dos veces con unidades INCOMPATIBLES  (3 casos)
#        «½ ají y 50 g de ají cubanela»; «el calabacín duplicado en unidades y gramos». No es que
#        sobre: es que no se sabe cuánto comprar. V4 no lo ve porque compara gramos CON gramos.
#
#   V7c  legumbre declarada SECA que ningún paso remoja ni hierve  (5 casos)
#        «Las habichuelas figuran secas, pero el procedimiento las trata como cocidas.» Servir
#        habichuelas crudas no es un defecto de estilo.
#
# La cuarta clase que las notas destapan —usado A MEDIAS: «la mitad del ajo queda sin usar»,
# «falta asignar la mitad restante del aceite»— NO se implementa aquí a propósito: exige seguir
# cantidades repartidas ENTRE pasos, y meterla de prisa haría ruido en las tres que sí son
# nítidas. Queda anotada, no olvidada.
#
# Las tres nacen en `warn`, como V5 y V6 y por el mismo motivo: su precisión no está medida contra
# un golden set INDEPENDIENTE. El de hoy dejó de serlo en cuanto se leyeron sus notas para diseñar
# esto — medir aquí sería medir cuánto me aprendí las respuestas, que es el sobreajuste que este
# proyecto ya tiene documentado con el juez al 89 %.

_V7_MEDIDA_RE = re.compile(r"\b(" + _V6_CONTABLE + "|" + _V6_MASA + r")\b", re.IGNORECASE)
# `_V6_MASA_RE` NO sirve aquí: no lleva límites de palabra porque V6 lo usa con `.fullmatch()`.
# Con `.search()`, la `l` de «cubane_l_a» casa como «litro» y «½ ají cubanela» se clasificaba como
# MASA — con lo que V7b veía una sola familia y callaba. Lo cazó su propio test unitario a los dos
# minutos de escribirlo. 19ª colisión por subcadena del proyecto, y van dos mías hoy.
_V7_MASA_RE = re.compile(r"\b(" + _V6_MASA + r")\b", re.IGNORECASE)
_V7_CANT = r"(\d+(?:[.,]\d+)?|[½¼¾⅓⅔⅛]|\d[½¼¾⅓⅔])"
_V7_PIEZA_RE = re.compile(_V7_CANT + r"\s+((?:de\s+)?[a-zñ]+(?:\s+[a-zñ]+){0,3})")
# Verbos que convierten una pieza en MASA. Tras uno de ellos el singular es COLECTIVO («ralla el
# tomate» con 2 tomates en la lista) y el número gramatical deja de informar.
#
# Lo enseñó un guard preexistente: `test_trampa_fp_plural_singular` tiene un fixture commiteado
# EXACTAMENTE para este par —«2½ tomates» vs «Ralla el tomate»— identificado como falso positivo
# el 2026-07-31. Se reintrodujo aquí por no haber buscado antes lo que el repo ya sabía.
#
# Se elige el VERBO y no una lista de alimentos a propósito: «rellena la tortilla» sí significa una
# tortilla, y una lista de «alimentos masificables» habría que mantenerla a mano para siempre.
_V7_MASIFICA_RE = re.compile(
    r"\brall|\bpica\b|\bpicad|\bpique|\btritur|\bmaja\b|\bmajad|\blicu|\bmuele\b|\bmolid|"
    r"\bmachac|\bdesmenuz|\bpure\b|\bpuré\b|\bhaz una pasta", re.IGNORECASE)
_V7_SECO_RE = re.compile(r"\bsec[oa]s?\b", re.IGNORECASE)
# ACCIONES de cocción, y `cocid*` NO está entre ellas a propósito.
#
# La primera versión la incluía y por eso V7c no disparó ni una vez teniendo cinco casos: el paso
# de 01c22c6847 dice «escurre las lentejas y las habichuelas negras COCIDAS» — y eso no prueba que
# se cocieran, es exactamente la contradicción que se busca. El ingrediente las declara SECAS y el
# paso las da por cocidas sin que ningún paso las cueza.
#
# Usar el síntoma como coartada es cómo un detector se ciega a sí mismo.
_V7_COCCION_RE = re.compile(r"\bremoj|\bhierv|\bhervi|\bcoce|\bcocin|\bcuece|\bhidrat", re.IGNORECASE)
# El estado declarado, que es señal en la dirección CONTRARIA: si aparece sin una acción, refuerza.
_V7_ESTADO_COCIDO_RE = re.compile(r"\bcocid[oa]s?\b", re.IGNORECASE)
# Solo lo que CAMBIA de peso y de comestibilidad al cocerse. Una lechuga «seca» no es esto.
_V7_SECABLES_RE = re.compile(
    r"\b(habichuela|frijol|lenteja|garbanzo|guandul|gandul|haba|soya|arroz|quinoa|cebada|"
    r"bulgur|avena|pasta|espagueti|fideo|codito|macarr)", re.IGNORECASE)


_V7_RANGO_RE = re.compile(_V7_CANT + r"\s*(?:-|–|—|\ba\b)\s*$")


def _v7_piezas(texto: str, index: dict, *, agregar: str = "suma") -> dict:
    """{alimento: total de PIEZAS} de «N <alimento>» — sin unidad de medida por medio.

    [P1-PLAN-LOTE-31 · 2026-09-13] `agregar="max"`: en vez de SUMAR las menciones del texto, se queda con la mayor.
    Es lo que V7e necesita dentro de UN paso — «separa 6 claras de huevo reservando 6 claras en un bol» son las
    mismas seis claras, no doce: sumadas daban «el paso pide 12 y la lista compra 6», tres de los diez V7e de los
    planes recién generados del bench real (los otros siete eran de verdad). La lista y V7a siguen sumando: dos
    líneas del mismo alimento compran la suma, y en la dirección «la lista compra de más» sumar sólo reduce el hueco.

    «2 tortillas de trigo» sí; «2 cucharadas de cilantro» NO — eso lo mide V6, y contar la
    cucharada como pieza convertiría cada especia en un falso positivo.

    [P1-CULINARY-V7A-CORTE-RANGO · 2026-09-07] Dos correcciones que salieron de medir el
    detector contra 60 comidas retenidas que el dueño etiquetó a ciegas:

    **La medida detrás del alimento describe el CORTE, no el conteo.** «corta 2 ciruelas
    medianas EN GAJOS» son dos ciruelas partidas en gajos, no «2 gajos»; el guard veía
    `gajos` en la cola y descartaba la pieza entera. Por eso V7a no vio su propio caso de
    manual (lista «3 ciruelas», paso «corta 2 ciruelas»). La medida solo invalida el conteo
    cuando va ANTES del alimento, que es donde una medida de verdad vive: «2 cucharadas de
    cilantro». Se compara la POSICIÓN, no la presencia.

    **El extremo alto de un rango no fija una cantidad.** «1–2 mandarinas» con un paso que
    dice «pela la mandarina» no se contradice: el límite inferior autoriza el singular. Era
    el único falso positivo del detector en toda la muestra. Un rango no es un conteo."""
    out: dict = {}
    blob = _norm(texto)
    for m in _V7_PIEZA_RE.finditer(blob):
        val = _v6_valor(m.group(1))
        cola = m.group(2) or ""
        if val is None:
            continue
        if _V7_RANGO_RE.search(blob[:m.start()]):
            continue                                   # es el techo de «N–M»: no fija nada
        medida = _V7_MEDIDA_RE.search(cola)
        crudos = list(find_catalog_foods(cola, index))
        if len(crudos) != 1:
            continue                                   # ambiguo o nada: no se cuenta
        if medida:
            spans = _catalog_food_spans(cola, index)
            if not spans or medida.start() < spans[0][0]:
                continue                               # medida ANTES del alimento ⇒ es medida
        if agregar == "max":
            out[crudos[0]] = max(out.get(crudos[0], 0.0), val)
        else:
            out[crudos[0]] = out.get(crudos[0], 0.0) + val
    return out


def _v7a_lista_compra_de_mas(day, meal, index) -> list:
    """La lista compra N piezas y los pasos, sumados, usan menos. Fail-open total."""
    _ex = _cuentas_exactas(meal)                     # [P1-PLAN-LOTE-25] receta congelada: cuentas (0 < Σ < 1), no texto
    if _ex is not None:
        return _violaciones_exactas(day, meal, _ex, ("V7a",))
    out = []
    try:
        ings = [str(x) for x in (meal.get("ingredients") or [])]
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not ings or not pasos:
            return []
        en_lista: dict = {}
        for ing in ings:
            for food, n in _v7_piezas(ing, index).items():
                en_lista[food] = en_lista.get(food, 0.0) + n
        usado: dict = {}
        for paso in pasos:
            for food, n in _v7_piezas(_texto_de_consumo(paso), index).items():   # [P1-PLAN-LOTE-26]
                usado[food] = usado.get(food, 0.0) + n
        pasos_norm = [_norm(p) for p in pasos]
        for food, comprado in en_lista.items():
            if comprado <= 1:
                continue
            gastado = usado.get(food)
            if gastado is not None:
                if gastado >= comprado - 0.05:
                    continue
                out.append(_viol(day, meal, "V7a", food,
                                 f"la lista compra {comprado:g} y los pasos usan {gastado:g}",
                                 "minor", True))
                continue
            # Sin cifra en los pasos, la señal es el NÚMERO GRAMATICAL, que es lo que el humano
            # usó: «la cantidad de pan no coincide con el plural "tostadas"», «declara dos
            # tortillas pero el procedimiento solo utiliza una». Exigir una cifra dejaba fuera los
            # seis casos de esta clase, porque la prosa dice «coloca la tortilla», no «1 tortilla».
            #
            # Solo dispara si el alimento SÍ aparece en los pasos —si no, es huérfano y lo ve V3—
            # y SIEMPRE en singular. Una sola mención en plural basta para callarlo: el reparto
            # entre pasos es legítimo y no hay por qué adivinarlo.
            # El número gramatical solo es evidencia con conteos ENTEROS ≥ 2. «1½ guineos» y un
            # paso que dice «el guineo» no se contradicen — media pieza no tiene plural. Sin este
            # guard, ese caso era el único falso positivo del detector.
            if comprado < 2 or abs(comprado - round(comprado)) > 0.01:
                continue
            cabeza = _norm(food).split()[0]
            sing = re.compile(r"\b" + re.escape(cabeza) + r"\b")
            plur = re.compile(r"\b" + re.escape(cabeza) + r"(?:e?s)\b")
            menciones = [pn for pn in pasos_norm if sing.search(pn) or plur.search(pn)]
            if not menciones or any(plur.search(pn) for pn in menciones):
                continue
            # Si algún paso lo convierte en MASA, el singular es colectivo y no dice nada. Se mira
            # la cláusula, no el paso entero: «ralla el queso; coloca las tortillas» ralla el
            # queso, no las tortillas.
            masificado = False
            for pn in menciones:
                for mm in re.finditer(r"\b" + re.escape(cabeza) + r"\w*", pn):
                    ini, fin = _clause_bounds(pn, mm.start())
                    if _V7_MASIFICA_RE.search(pn[ini:fin]):
                        masificado = True
                        break
                if masificado:
                    break
            if masificado:
                continue
            out.append(_viol(day, meal, "V7a", food,
                             f"la lista compra {comprado:g} y los pasos hablan de una sola "
                             f"({cabeza}, siempre en singular)", "minor", True))
    except Exception:
        return []
    return out


def _v7b_duplicado_incompatible(day, meal, index) -> list:
    """El mismo alimento en dos líneas con familias de unidad distintas (pieza y masa)."""
    out = []
    try:
        familias: dict = {}
        for ing in [str(x) for x in (meal.get("ingredients") or [])]:
            crudos = list(find_catalog_foods(ing, index))
            if len(crudos) != 1:
                continue
            n = _norm(ing)
            fam = "masa" if _V7_MASA_RE.search(n) else ("pieza" if _V7_PIEZA_RE.search(n) else None)
            if not fam:
                continue
            familias.setdefault(crudos[0], {}).setdefault(fam, []).append(str(ing)[:44])
        for food, fams in familias.items():
            if len(fams) < 2:
                continue
            muestras = " / ".join(v[0] for v in fams.values())
            out.append(_viol(day, meal, "V7b", food,
                             f"aparece en dos unidades incompatibles: {muestras}",
                             "minor", False))
    except Exception:
        return []
    return out


def _v7c_seco_sin_coccion(day, meal, index) -> list:
    """Legumbre o grano declarado SECO que ningún paso remoja ni hierve."""
    out = []
    try:
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not pasos:
            return []
        pasos_norm = [_norm(p) for p in pasos]
        for ing in [str(x) for x in (meal.get("ingredients") or [])]:
            n = _norm(ing)
            if not _V7_SECO_RE.search(n) or not _V7_SECABLES_RE.search(n):
                continue
            crudos = list(find_catalog_foods(ing, index))
            if not crudos:
                continue
            food = crudos[0]
            cabeza = _norm(food).split()[0]
            # Se busca un paso que nombre ESE alimento Y lo cueza. Basta el sustantivo cabeza: la
            # prosa dice «las habichuelas», no «las habichuelas blancas», y exigir el nombre
            # completo produciría el mismo falso positivo que `_mencionado_por_prefijo` documenta.
            # La acción y el alimento tienen que estar en la MISMA CLÁUSULA, no solo en el mismo
            # paso. «El Toque de Fuego: calienta la plancha; cocina la berenjena y el ají; añade
            # las lentejas» cuece la berenjena, no las lentejas — y comprobando por paso, ese
            # `cocina` daba por cocidas unas lentejas que nadie coció. Es el mismo error de
            # alcance que `_occurrence_resolves` ya resuelve para V1, con su misma herramienta.
            cocido = False
            for pn in pasos_norm:
                for m in re.finditer(r"\b" + re.escape(cabeza) + r"\w*", pn):
                    ini, fin = _clause_bounds(pn, m.start())
                    if _V7_COCCION_RE.search(pn[ini:fin]):
                        cocido = True
                        break
                if cocido:
                    break
            if cocido:
                continue
            # Si además algún paso lo da por COCIDO sin haberlo cocido, no es un olvido de
            # redacción: es una contradicción entre la lista y el procedimiento.
            contradice = any(
                _V7_ESTADO_COCIDO_RE.search(pn) and re.search(r"\b" + re.escape(cabeza) + r"\w*", pn)
                for pn in pasos_norm)
            detalle = (f"declarado seco ('{str(ing)[:44]}') y un paso lo da por COCIDO sin cocerlo"
                       if contradice else
                       f"declarado seco ('{str(ing)[:44]}') y ningún paso lo remoja ni lo hierve")
            out.append(_viol(day, meal, "V7c", food, detalle,
                             "major" if contradice else "minor", True))
    except Exception:
        return []
    return out


# [P1-CULINARY-V7D-MASA · 2026-09-07] El espejo en MASA que faltaba.
#
# V6 cubre «el paso pide MÁS que la lista». V7a cubre lo contrario —la lista compra de más—
# pero SOLO en piezas contables, porque `_v7_piezas` descarta a propósito todo lo que lleve
# unidad de medida («2 cucharadas de cilantro» son cucharadas, no piezas). El resultado es un
# hueco exacto: «420 ml de leche» en la lista y «250 ml de leche» en el paso no lo ve nadie.
#
# Medido sobre la flota antes de escribirlo: 10 de 1.194 comidas (0,8 %), y siete son la misma
# forma —leche de avena comprada a 340-545 ml y usada a 200-250—, lo que apunta a un sesgo del
# generador, no a ruido. Prevalencia parecida a la de V7b (7 de 1.194), que ya está desplegado.
_V7D_MASA_RE = re.compile(
    r"(\d+(?:[.,]\d+)?|[½¼¾⅓⅔⅛]|\d[½¼¾⅓⅔])\s*"
    r"(ml|mililitros?|l|litros?|g|gr|gramos?|kg|kilos?)\b\s*"
    r"(?:de\s+)?([a-zñ]+(?:\s+[a-zñ]+){0,3})", re.IGNORECASE)
_V7D_A_GRAMOS = {"ml": 1.0, "mililitro": 1.0, "mililitros": 1.0, "l": 1000.0, "litro": 1000.0,
                 "litros": 1000.0, "g": 1.0, "gr": 1.0, "gramo": 1.0, "gramos": 1.0,
                 "kg": 1000.0, "kilo": 1000.0, "kilos": 1000.0}
# Por debajo de esto es redacción, no un sobrante: hace falta que falle EN PROPORCIÓN y además
# que el hueco valga una compra. Un solo umbral dejaba pasar «100 g -> 75 g» (25 g no cambian
# nada) o marcaba «40 g -> 30 g» como si importara.
_V7D_TOLERANCIA = 0.25
_V7D_MIN_GRAMOS = 30.0
#: [P1-PLAN-LOTE-26 · 2026-09-12] (CUL-P1-05) ml → g SOLO con densidad respaldada (g/ml, USDA/BEDCA orientativos). Antes
#: todo ml valía 1 g: la miel (1,42) o el aceite (0,92) se comparaban con una conversión inventada. Un líquido que no esté
#: aquí no se compara en ml (no es un hallazgo: es «no comparable», como V4 con las tazas).
_V7D_DENSIDAD = {"agua": 1.0, "leche": 1.03, "caldo": 1.0, "jugo": 1.04, "zumo": 1.04, "vinagre": 1.01, "aceite": 0.92,
                 "yogur": 1.05, "yogurt": 1.05, "salsa de tomate": 1.05, "pure de tomate": 1.05, "crema": 1.0,
                 "leche de coco": 0.97, "leche de almendra": 1.02, "leche de avena": 1.02, "miel": 1.42, "sirope": 1.33,
                 "vino": 0.99, "cerveza": 1.01, "cafe": 1.0, "te": 1.0, "kefir": 1.03, "bebida": 1.02}


def _densidad_respaldada(food: str):
    n = _norm(food)
    mejor = None
    for k, d in _V7D_DENSIDAD.items():
        if re.search(r"\b" + re.escape(k) + r"s?\b", n) and (mejor is None or len(k) > len(mejor[0])):
            mejor = (k, d)
    return mejor[1] if mejor else None


_V7D_VOLUMEN = ("ml", "mililitro", "mililitros", "l", "litro", "litros")


def _v7d_masas(texto: str, index: dict) -> dict:
    """{alimento: gramos} de las masas EXPLÍCITAS del texto (ml se cuenta 1:1, como siempre). Sin unidad, no cuenta.

    Suma a lo largo de todo el texto a propósito: repartir un ingrediente entre dos pasos
    («250 ml ahora, 170 ml al final») es legítimo y no debe disparar."""
    return {food: fam.get("g", 0.0) + fam.get("ml", 0.0) for food, fam in _v7d_masas_por_familia(texto, index).items()}


def _v7d_masas_por_familia(texto: str, index: dict) -> dict:
    """[P1-PLAN-LOTE-26 · 2026-09-12] (CUL-P1-05) `{alimento: {"g": gramos, "ml": mililitros}}`: la familia de la unidad
    viaja con la cantidad. ml con ml y g con g se comparan sin convertir nada; cruzar familias exige una densidad
    respaldada (`_V7D_DENSIDAD`) — «400 ml de avena» contra «100 ml de avena» es comparable; «400 ml de avena» contra
    «120 g de avena» no lo es sin saber cuánto pesa un ml de avena, y nadie aquí lo sabe."""
    out: dict = {}
    for m in _V7D_MASA_RE.finditer(_norm(texto)):
        val = _v6_valor(m.group(1))
        uni = m.group(2).lower()
        fac = _V7D_A_GRAMOS.get(uni)
        if val is None or fac is None:
            continue
        hits = find_catalog_foods(m.group(3), index)
        if len(hits) != 1:
            continue                                   # ambiguo o desconocido: no se cuenta
        fam = "ml" if uni in _V7D_VOLUMEN else "g"
        d = out.setdefault(hits[0], {})
        d[fam] = d.get(fam, 0.0) + val * fac
    return out


def _v7d_comparables(lista: dict, usado: dict, food: str):
    """`(g_lista, g_paso, unidad)` en una MISMA familia, o `None` si no hay forma respaldada de compararlas."""
    a, b = lista.get(food) or {}, usado.get(food) or {}
    for fam in ("g", "ml"):
        if fam in a and fam in b and not (set(a) - {fam}) and not (set(b) - {fam}):
            return a[fam], b[fam], fam
    # familias cruzadas: sólo con densidad conocida del alimento (g/ml)
    dens = _densidad_respaldada(food)
    if dens is None:
        return None
    ga = a.get("g", 0.0) + a.get("ml", 0.0) * dens
    gb = b.get("g", 0.0) + b.get("ml", 0.0) * dens
    return ga, gb, "g"


def _v7e_paso_pide_mas_piezas(day, meal, index) -> list:
    """[P1-CULINARY-V7E-PIEZAS · 2026-09-07] Un paso pide MÁS PIEZAS de las que la lista compra.

    `V6` ya cubre esta dirección, pero exige una palabra de UNIDAD: `_v6_cuentas('½ diente de
    ajo')` da `{('diente', 0.5)}` y `_v6_cuentas('½ guineo verde')` da `{}`. Con un conteo
    DESNUDO los dos lados salen vacíos y no hay nada que comparar. `_v7_piezas` sí lo lee —y ya
    rechaza «2 tazas» como medida, que es lo que hacía peligrosa esta comparación.

    El hueco es el más ancho medido hasta hoy: **127 de 1.194 comidas (10,6 %)**, trece veces la
    prevalencia de V7d. Y sobre las 140 etiquetadas por el dueño dispara 16 veces, **las 16 en
    comidas que él marcó con defecto**, con sus notas describiendo esta acusación palabra por
    palabra: «¾ de unidad en ingredientes frente a 2 unidades en la preparación», «Corregir pera:
    se declara ½ y se utiliza 1», «Unificar ajíes —1 frente a 2—, cebolla —1 frente a ½—».

    **Se compara POR PASO, nunca la suma.** En la dirección contraria (V7a, la lista compra de
    más) sumar los pasos es seguro: más menciones sólo reducen el hueco. Aquí sumar sería un
    falso positivo garantizado — «corta 1 tomate» y luego «añade el tomate» daría 2 contra 1. La
    pregunta correcta es si ALGÚN paso, por sí solo, pide más de lo que hay comprado.

    [P1-PLAN-LOTE-31 · 2026-09-13] Y DENTRO del paso tampoco se suma: la mención mayor manda
    (`_v7_piezas(..., agregar="max")`). «Separa 6 claras de huevo reservando 6 claras en un bol»
    son las mismas seis claras; sumadas daban 12 contra 6 — tres de los diez V7e «de fábrica» del
    bench real eran esto. Medido en el corpus fijo: 38 con suma, 38 con máximo (ningún coste allí).
    tooltip-anchor: P1-PLAN-LOTE-31-V7E-MAX

    No solapa con V6: cuando el texto trae unidad («1 diente de ajo»), `_v7_piezas` ve la medida
    delante del alimento y no cuenta la pieza, así que ese caso lo sigue reportando V6 y sólo V6.
    """
    if _cuentas_exactas(meal) is not None:           # [P1-PLAN-LOTE-25] receta congelada: la dirección «pide más» es V6 exacto
        return []
    out = []
    try:
        ings = [str(x) for x in (meal.get("ingredients") or [])]
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not ings or not pasos:
            return []
        en_lista: dict = {}
        for ing in ings:
            for food, n in _v7_piezas(ing, index).items():
                en_lista[food] = en_lista.get(food, 0.0) + n
        for paso in pasos:
            for food, n in _v7_piezas(_texto_de_consumo(paso), index, agregar="max").items():   # [P1-PLAN-LOTE-26] [P1-PLAN-LOTE-31]
                total = en_lista.get(food)
                if total is None:
                    continue                           # no está en la lista: eso es V5
                # Misma tolerancia que V6: «⅓» y «0.33» son la misma cantidad, y el suelo
                # absoluto evita que un redondeo de redacción dispare sobre cantidades chicas.
                if n <= total + max(0.06, 0.05 * total):
                    continue
                out.append(_viol(
                    day, meal, "V7e", food,
                    f"el paso pide {n:g} y la lista compra {total:g}: {paso[:100]}",
                    "minor", False))
    except Exception:
        return []
    return out


def _v7d_masa_sobrante(day, meal, index) -> list:
    """La lista compra N gramos y los pasos, sumados, usan bastantes menos. Fail-open total."""
    out = []
    try:
        ings = [str(x) for x in (meal.get("ingredients") or [])]
        pasos = [str(x) for x in (meal.get("recipe") or [])]
        if not ings or not pasos:
            return []
        en_lista = _v7d_masas_por_familia(" ".join(ings), index)
        usado = _v7d_masas_por_familia(" ".join(_texto_de_consumo(p) for p in pasos), index)   # [P1-PLAN-LOTE-26]
        for food in en_lista:
            # Un paso que NO cuantifica no contradice a la lista: «cocina la avena con la leche»
            # es una instrucción normal, no una declaración de cantidad. Exigir la cifra en los
            # DOS lados es lo que separa esta capa de V3, que ya cubre el ingrediente ausente.
            if food not in usado:
                continue
            comp = _v7d_comparables(en_lista, usado, food)     # [P1-PLAN-LOTE-26] misma familia, o densidad respaldada
            if comp is None:
                continue
            g_lista, g_paso, uni = comp
            if g_paso >= g_lista * (1 - _V7D_TOLERANCIA):
                continue
            if (g_lista - g_paso) < _V7D_MIN_GRAMOS:
                continue
            out.append(_viol(
                day, meal, "V7d", food,
                f"la lista compra {g_lista:g} {uni} y los pasos usan {g_paso:g} {uni}",
                "minor", True))
    except Exception:
        return []
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-26 · 2026-09-12] (C5 · CUL-P1-05) V8a tiempo oculto · V8b equipo no disponible
#
# V8a: los pasos piden una espera en HORAS (remojo, marinado, «la noche anterior») que el `prep_time` del plato no cubre.
# Medido en el corpus fijo: 5 de 64 comidas declaran 15-20 min y piden horas; una era conservación («consume dentro de
# 24 horas»), que no es espera y no dispara. Un plato de «10 minutos» no puede esconder una víspera.
# V8b: la receta exige un equipo (horno, airfryer, licuadora, microondas, olla de presión…) que la persona declaró NO
# tener en Súper Personalización. Sin declaración no se evalúa (`estado["contexto"]["equipo"] = "no_declarado"`): el
# formulario principal no lo pregunta (decisión de producto P2-FORM-KITCHEN-EQUIPMENT) y fingir un dato es peor que
# decir que falta. Las dos nacen `minor` y no reparables: el reparador de tiempos/equipo es CUL-P1-04.
# tooltip-anchor: P1-PLAN-LOTE-26-V8

def _v8a_tiempo_oculto(day, meal, index) -> list:
    try:
        from culinary_context import check_hidden_time
        h = check_hidden_time(meal)
        if not h:
            return []
        decl = (f"{h['declarado_min']} min declarados" if h.get("declarado_min") is not None else "sin tiempo declarado")
        return [_viol(day, meal, "V8a", "tiempo",
                      f"los pasos piden {h['espera_min'] / 60:g} h de espera («{h['evidencia'][:80]}») y el plato dice {decl}",
                      "minor", False)]
    except Exception:
        return []


def _v8b_equipo_no_disponible(day, meal, declared) -> list:
    try:
        if declared is None:
            return []
        from culinary_context import missing_equipment
        return [_viol(day, meal, "V8b", eq, f"la receta pide {eq} y la persona declaró no tenerlo", "minor", False)
                for eq in missing_equipment(meal, declared)]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-27 · 2026-09-12] (C5 · CUL-P1-02) V9 estructura del plato — el contrato ligero de `dish_structure`
# (familia, componentes, relaciones de cantidades sensibles) emite sus tres hallazgos mecanizables con la evidencia de la
# biblioteca curada: `crema_sin_espesante`, `wrap_desproporcionado`, `tortilla_vegetales_crudos`. `minor`, no reparable.
# Medido: 0 falsos positivos sobre las 193 recetas curadas y 0 sobre el corpus fijo; los tres casos del backlog disparan.
# tooltip-anchor: P1-PLAN-LOTE-27-V9

def _v9_estructura(day, meal, index) -> list:
    try:
        from dish_structure import contract
        k = contract(meal)
        return [_viol(day, meal, "V9", r.get("tipo"), f"{r.get('detalle')} — evidencia: {r.get('evidencia')}", "minor", False)
                for r in (k.get("relaciones") or [])]
    except Exception:
        return []


def _viol(day, meal, check, food, detail, severity, repairable):
    return {"day": day, "meal": meal.get("meal") or meal.get("name"),
            "check": check, "food": food, "detail": detail,
            "severity": severity, "repairable": repairable}


#: Los checks de la capa 1, en el orden en que corren.
CHECKS_CAPA1 = ("V1", "V2", "V3", "V4", "V5", "V6", "V7a", "V7b", "V7c", "V7d", "V7e", "V8a", "V8b", "V9")

#: [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Versión del ESQUEMA de hallazgo: desde aquí cada violación (capa 1 y
#: juez) lleva `meal_index`, la posición de la comida en su día. Cambia cuando cambie la forma del hallazgo.
FINDING_SCHEMA_VERSION = "2026-09-12.certeza"   # [P1-PLAN-LOTE-28] + certeza/componente/intencion en el juez

_RULES_FP: "str | None" = None


def rules_fingerprint() -> "str | None":
    """[P1-PLAN-LOTE-22 · 2026-09-12] (C1) sha256[:16] del fuente de ESTE fichero: la versión de las reglas que produjo un
    hallazgo. Mismo cálculo que `culinary_corpus.huella_reglas` (texto utf-8, saltos normalizados) para que las dos
    huellas coincidan. Calculada una vez por proceso; `None` si no se puede leer (jamás lanza)."""
    global _RULES_FP
    if _RULES_FP is None:
        try:
            _RULES_FP = hashlib.sha256(Path(__file__).resolve().read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
        except Exception:
            return None
    return _RULES_FP


def _iter_meals_idx(plan_data: dict):
    """[P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Como `_iter_meals`, más la POSICIÓN de la comida en su día: la
    identidad por ocurrencia. Dos meriendas el mismo día son dos comidas y sus hallazgos no se mezclan."""
    for d in (plan_data or {}).get("days") or []:
        if not isinstance(d, dict):
            continue
        for mi, m in enumerate(d.get("meals") or []):
            if isinstance(m, dict):
                yield d.get("day"), m, mi


def culinary_contract_scan(plan_data: dict, catalog: list, _estado: "dict | None" = None, form_data=None) -> list:
    """Escanea el plan completo. Retorna lista de Violations (vacía si todo
    coherente o si no hay datos). Jamás lanza: fail-open total.

    [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Cada violación lleva `meal_index` (posición de la comida en su día:
    identidad por OCURRENCIA, no por franja) y `meal_seal` (contenido de esa comida). Si el caller pasa `_estado` (un
    dict) se rellena con el ESTADO del scan — `culinary_contract_scan_status` es la forma cómoda de pedirlo: `[]`
    significaba «coherente», «sin catálogo» y «reventó», y las tres se persistían igual. La cadena de checks vive AQUÍ,
    literal, porque cinco tests la leen como texto: una capa que existe y nadie invoca es el modo de fallo de P1-G."""
    estado = _estado if isinstance(_estado, dict) else {}
    estado.update({"status": "error", "checks": list(CHECKS_CAPA1), "meals": 0, "violations": 0, "error": None,
                   "reglas_huella": rules_fingerprint(), "schema": FINDING_SCHEMA_VERSION, "exactas": 0})
    # [P1-PLAN-LOTE-26] (CUL-P1-05) el equipo declarado llega con el formulario (o dentro del plan persistido); si no, se dice
    _declared = None
    try:
        from culinary_context import declared_equipment as _de
        _fd = form_data if isinstance(form_data, dict) else (plan_data or {}).get("form_data")
        _declared = _de(_fd)
    except Exception:
        _declared = None
    estado["contexto"] = {"equipo": "declarado" if _declared is not None else "no_declarado"}
    try:
        index = build_culinary_index(catalog)
        if not index:
            estado["status"] = "no_catalog"
            estado["checks"] = []
            return []
        out = []
        n = 0
        for day, meal, mi in _iter_meals_idx(plan_data):
            n += 1
            start = len(out)
            if _cuentas_exactas(meal) is not None:
                estado["exactas"] += 1               # [P1-PLAN-LOTE-25] V3/V6/V7a/V7e por cuentas, no por texto
            out.extend(_v1_verbo_alimento(day, meal, index))
            out.extend(_v2_estado_imposible(day, meal, index))
            out.extend(_v3_huerfanos(day, meal, index))
            out.extend(_v4_cantidad_inconsistente(day, meal, index))
            out.extend(_v5_paso_usa_lo_que_no_esta(day, meal, index))
            out.extend(_v6_paso_pide_mas_que_la_lista(day, meal, index))
            out.extend(_v7a_lista_compra_de_mas(day, meal, index))
            out.extend(_v7b_duplicado_incompatible(day, meal, index))
            out.extend(_v7c_seco_sin_coccion(day, meal, index))
            out.extend(_v7d_masa_sobrante(day, meal, index))
            out.extend(_v7e_paso_pide_mas_piezas(day, meal, index))
            out.extend(_v8a_tiempo_oculto(day, meal, index))
            out.extend(_v8b_equipo_no_disponible(day, meal, _declared))
            out.extend(_v9_estructura(day, meal, index))
            for v in out[start:]:
                v.setdefault("meal_index", mi)
                v.setdefault("meal_seal", meal_seal(meal))
        estado["status"] = "scanned" if n else "no_meals"
        estado["meals"] = n
        estado["violations"] = len(out)
        return out
    except Exception as _f5e:
        logger.warning(f"[P1-PLAN-LOTE-6] culinary_contract_scan: `build_culinary_index` tragado sin rastro ({type(_f5e).__name__}: {_f5e})")
        estado["error"] = f"{type(_f5e).__name__}: {_f5e}"[:240]
        return []


def culinary_contract_scan_status(plan_data: dict, catalog: list, form_data=None) -> "tuple[list, dict]":
    """[P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) El scan con su ESTADO: `(violations, estado)`.

        estado = {"status": "scanned" | "no_meals" | "no_catalog" | "error", "checks": [...], "meals": n,
                  "violations": n, "error": None | "Tipo: mensaje", "reglas_huella": ..., "schema": ...}

    Antes un catálogo vacío en producción aprobaba todos los planes en silencio (medido en P1-CULINARY-METADATA-BETA:
    cobertura 100 % → 59 % con los tests en verde). Fail-open se conserva: jamás lanza, y con `no_catalog`/`error`
    devuelve `[]` — pero ahora lo DICE. tooltip-anchor: P1-PLAN-LOTE-22-SCAN-STATUS"""
    estado: dict = {}
    return culinary_contract_scan(plan_data, catalog, _estado=estado, form_data=form_data), estado


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


def scan_coverage(plan_data: dict, catalog: list) -> "float | None":
    """Fracción de alimentos mencionados en el plan que tienen metadata.

    [P1-COVERAGE-UNKNOWN-NOT-PERFECT · 2026-09-07] Devuelve **`None` cuando no pudo medir**
    —excepción, o ningún alimento reconocido— y ya NO `1.0`. Ese `1.0` decía «cobertura perfecta»
    justo en los dos casos en que la función no había mirado nada, que es la telemetría con la
    que se decide el rollout `warn → block`.

    No es hipotético: en `P1-CULINARY-METADATA-BETA` la cobertura real cayó de 100 % a 59 % con
    141 filas nuevas sin `prep_methods`, la capa 1 quedó en fail-open y **los tests siguieron en
    verde** porque ninguno miraba el DATO. Un medidor que al fallar informa del mejor valor
    posible convierte su propia ceguera en una buena noticia.

    Quien la consuma debe tratar `None` como *desconocido*, nunca como 1,0 ni como 0,0.
    """
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
        return (con_meta / len(vistos)) if vistos else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Identidad, versión y estado EXPLÍCITO de evaluación
#
# Cuatro cosas que antes se confundían con «aprobado»: el juez que no llegó a juzgar, el scan sin catálogo, el juicio
# de OTRA versión del plan y una comida que hereda el hallazgo de su hermana de franja. Las funciones de abajo no
# deciden nada (siguen sin mutar el plan): dan nombre a cada estado para que el medidor y el orquestador no tengan
# que adivinarlo a partir de una lista vacía.
# tooltip-anchor: P1-PLAN-LOTE-22-EVAL-STATE

def slot_norm(text) -> str:
    """Franja normalizada para comparar «Merienda» con «merienda» o «Almuerzo» con «almuerzo»."""
    return _norm(text).strip()


def meal_seal(meal: dict) -> "str | None":
    """Identidad de CONTENIDO de una comida: sha256[:16] de (franja, nombre, ingredientes, pasos) — lo mismo que lee el
    juez, SIN el día ni la posición. Por qué hace falta además de `judged_fingerprint` (sello del plan entero, que lleva
    la posición del día): el shift archiva y RENUMERA días, así que el sello del plan declara obsoleto todo lo juzgado
    aunque la comida entregada sea byte a byte la juzgada — medido en el corpus fijo del 09-12: 5 de 5 planes
    «juzgado_obsoleto», juez sobre lo entregado 0 de 64 comidas. El sello por comida sobrevive al shift y a la
    reparación de OTRA comida. Jamás lanza. tooltip-anchor: P1-PLAN-LOTE-22-MEAL-SEAL"""
    try:
        crudo = json.dumps([str(meal.get("meal") or ""), str(meal.get("name") or ""),
                            [str(x) for x in (meal.get("ingredients") or [])],
                            [str(x) for x in (meal.get("recipe") or [])]], ensure_ascii=False)
        return hashlib.sha256(crudo.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def meal_seals_index(plan: dict) -> dict:
    """sello → (day, meal_index) de cada comida ENTREGADA. Dos comidas idénticas comparten sello: gana la primera."""
    out = {}
    try:
        for day, m, mi in _iter_meals_idx(plan):
            s = meal_seal(m)
            if s and s not in out:
                out[s] = (day, mi)
    except Exception:
        pass
    return out


def scan_coverage_detail(plan_data: dict, catalog: list) -> dict:
    """Tres coberturas, no una. `scan_coverage` da una sola cifra (alimentos con `prep_methods` / alimentos vistos) y
    con ella se decide `warn → block`; pero un 59 % puede ser «el catálogo no tiene metadata» o «el índice no reconoce
    la mitad de las líneas», y la reparación es distinta. Devuelve, cada una `None` cuando no es medible:

      · `reconocimiento`: líneas de ingredientes en las que el índice encontró algún alimento (el PARSER);
      · `catalogo`: alimentos reconocidos con `prep_methods` (lo que V1 necesita; = `scan_coverage`);
      · `ready_to_eat`: alimentos reconocidos con `ready_to_eat` declarado (lo que V2 necesita);
      · `por_check`: qué cobertura gobierna a cada check (V1 → catálogo, V2 → ready_to_eat, resto → reconocimiento).

    `estado` ∈ {medida, sin_catalogo, sin_alimentos, error}. Jamás lanza."""
    out = {"estado": "medida", "lineas": 0, "lineas_reconocidas": 0, "alimentos": 0, "con_prep_methods": 0,
           "con_ready_to_eat": 0, "reconocimiento": None, "catalogo": None, "ready_to_eat": None, "por_check": {},
           "error": None}
    try:
        index = build_culinary_index(catalog)
        if not index:
            out["estado"] = "sin_catalogo"
            return out
        vistos = set()
        for _, meal in _iter_meals(plan_data):
            lineas = list(meal.get("ingredients") or [])
            for ln in lineas:
                out["lineas"] += 1
                if find_catalog_foods(str(ln), index):
                    out["lineas_reconocidas"] += 1
            blob = " | ".join(lineas + list(meal.get("recipe") or []))
            vistos.update(find_catalog_foods(blob, index))
        out["alimentos"] = len(vistos)
        for f in vistos:
            fila = index.get(_norm(f)) or {}
            if fila.get("prep_methods") is not None:
                out["con_prep_methods"] += 1
            if fila.get("ready_to_eat") is not None:
                out["con_ready_to_eat"] += 1
        if out["lineas"]:
            out["reconocimiento"] = round(out["lineas_reconocidas"] / out["lineas"], 3)
        if vistos:
            out["catalogo"] = round(out["con_prep_methods"] / len(vistos), 3)
            out["ready_to_eat"] = round(out["con_ready_to_eat"] / len(vistos), 3)
        else:
            out["estado"] = "sin_alimentos"
        out["por_check"] = {c: (out["catalogo"] if c == "V1" else out["ready_to_eat"] if c == "V2" else out["reconocimiento"])
                            for c in CHECKS_CAPA1}
        return out
    except Exception as e:
        out["estado"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"[:240]
        return out


def judge_payload_meals(plan: dict) -> list:
    """Lo que el juez LEE — día, franja, nombre, ingredientes y pasos— más `idx`, la posición de la comida en su día,
    para que pueda devolver `meal_index` y dos meriendas del mismo día dejen de ser indistinguibles."""
    return [{"day": d.get("day"), "idx": mi, "slot": m.get("meal"), "name": m.get("name"),
             "ingredients": m.get("ingredients"), "recipe": m.get("recipe")}
            for d in ((plan or {}).get("days") or []) if isinstance(d, dict)
            for mi, m in enumerate(d.get("meals") or []) if isinstance(m, dict)]


def resolve_judge_violations(plan: dict, violations: list) -> list:
    """Ata cada queja del juez a UNA ocurrencia. El juez habla en `(day, meal=franja)` y, si la rúbrica se lo pide,
    `meal_index`; dos meriendas el mismo día comparten franja. Resolución, en `resolucion`:

      · `por_sello`: la queja ya trae `meal_seal` (se selló al juzgar) y esa comida sigue ENTREGADA, quizá en otro día
        tras el shift — `ocurrencia_actual = [day, meal_index]` dice dónde está hoy; `day`/`meal_index` quedan como
        se juzgaron;
      · `declarada`: trae `meal_index` válido y la franja de esa comida coincide;
      · `unica`: una sola comida del día tiene esa franja (todo lo anterior a este P-fix cae aquí);
      · `ambigua`: dos o más — el hallazgo NO se reparte: `meal_index = None`;
      · `sin_comida`: ninguna comida con esa franja en ese día (día archivado por el shift, franja renombrada).

    Las quejas resueltas a una comida quedan SELLADAS (`meal_seal`) si no lo estaban: es lo que permite, más tarde,
    saber si la comida juzgada es la entregada sin depender del día. Devuelve copias; jamás lanza (fail-open: la lista
    de entrada)."""
    try:
        sellos = meal_seals_index(plan)
        por_dia = {}
        for pos, d in enumerate((plan or {}).get("days") or [], 1):
            if isinstance(d, dict):
                por_dia.setdefault(d.get("day") if d.get("day") is not None else pos, d)
        out = []
        for v in violations or []:
            if not isinstance(v, dict):
                continue
            v2 = dict(v)
            if v2.get("meal_seal") and v2["meal_seal"] in sellos:
                v2["resolucion"] = "por_sello"
                v2["ocurrencia_actual"] = list(sellos[v2["meal_seal"]])
                out.append(v2)
                continue
            meals = list(((por_dia.get(v2.get("day")) or {}).get("meals") or []))
            franja = slot_norm(v2.get("meal"))
            cand = [i for i, m in enumerate(meals) if isinstance(m, dict) and slot_norm(m.get("meal")) == franja]
            mi = v2.get("meal_index")
            if isinstance(mi, int) and not isinstance(mi, bool) and 0 <= mi < len(meals) and isinstance(meals[mi], dict) \
                    and (not franja or slot_norm(meals[mi].get("meal")) == franja):
                v2["resolucion"] = "declarada"
            elif len(cand) == 1:
                v2["meal_index"], v2["resolucion"] = cand[0], "unica"
            elif len(cand) > 1:
                v2["meal_index"], v2["resolucion"] = None, "ambigua"
            else:
                v2["meal_index"], v2["resolucion"] = None, "sin_comida"
            if v2.get("meal_index") is not None and not v2.get("meal_seal"):
                v2["meal_seal"] = meal_seal(meals[v2["meal_index"]])
            out.append(v2)
        return out
    except Exception:
        return list(violations or [])


def judge_context(*, country=None, model=None, guard=None, rubric=None, plan=None) -> dict:
    """El CONTEXTO en que se juzgó, al lado del sello de QUÉ se juzgó (`judged_fingerprint`). Un cambio de rúbrica, de
    modelo o de país invalida la comparación entre dos juicios igual que un cambio de pasos; sin esto «el juez mejoró»
    y «cambiamos la rúbrica» son indistinguibles. Jamás lanza."""
    ctx = {"country": country, "model": model, "guard": guard, "rubric_fingerprint": None, "meals": None,
           "schema": FINDING_SCHEMA_VERSION, "reglas_huella": rules_fingerprint()}
    try:
        if rubric:
            ctx["rubric_fingerprint"] = hashlib.sha256(str(rubric).encode("utf-8")).hexdigest()[:16]
        if plan is not None:
            ctx["meals"] = sum(1 for _ in _iter_meals(plan))
    except Exception:
        pass
    return ctx


#: Los estados del juez sobre el plan ENTREGADO. Sólo `juzgado_vigente` puede aprobar.
ESTADOS_JUEZ = ("juzgado_vigente", "juzgado_obsoleto", "no_disponible", "no_evaluado", "desconocido")


def judge_evaluation_state(plan_data: dict) -> dict:
    """El estado EXPLÍCITO del juez sobre la versión que se entrega:

      · `no_evaluado`: sin historial (knob `off`, o nunca corrió);
      · `juzgado_vigente`: alguna entrada juzgó ESTA versión (sello igual) — se toma la última de ellas;
      · `no_disponible`: la última entrada no llegó a juzgar (timeout/error/breaker) y ninguna vigente;
      · `juzgado_obsoleto`: juzgó OTRA versión (el plan cambió después);
      · `desconocido`: entradas sin sello (anteriores a P1-JUDGE-REVISION-STAMP).

    `aprobado` sólo cuando es vigente y sin hallazgos: timeout, error, off, catálogo vacío, obsoleto y desconocido
    NO cuentan como aprobados. Jamás lanza."""
    try:
        hist = [h for h in ((plan_data or {}).get("_culinary_judge_history") or []) if isinstance(h, dict)]
        if not hist:
            return {"estado": "no_evaluado", "aprobado": False, "hallazgos": 0, "entrada": None}
        sellos = meal_seals_index(plan_data)

        def _por_sello(h):
            vs = [v for v in (h.get("violations") or []) if isinstance(v, dict) and v.get("meal_seal")]
            return {"con_sello": len(vs), "entregadas": sum(1 for v in vs if v["meal_seal"] in sellos)}

        vigentes = [i for i, h in enumerate(hist) if judgment_covers_delivered(h, plan_data) is True]
        if vigentes:
            i = vigentes[-1]
            n = len(hist[i].get("violations") or [])
            return {"estado": "juzgado_vigente", "aprobado": n == 0, "hallazgos": n, "entrada": i, "sellos": _por_sello(hist[i])}
        h = hist[-1]
        n = len(h.get("violations") or [])
        if h.get("status") == "unavailable":
            estado = "no_disponible"
        else:
            estado = "juzgado_obsoleto" if judgment_covers_delivered(h, plan_data) is False else "desconocido"
        return {"estado": estado, "aprobado": False, "hallazgos": n, "entrada": len(hist) - 1, "sellos": _por_sello(h)}
    except Exception:
        return {"estado": "desconocido", "aprobado": False, "hallazgos": 0, "entrada": None}


def contract_evaluation_state(plan_data: dict, catalog: list) -> dict:
    """El estado del contrato determinista sobre el plan, con el mismo vocabulario: `evaluado` (scan corrió sobre ≥1
    comida), `no_evaluable` (sin catálogo o reventó), `no_evaluado` (sin comidas). `aprobado` sólo si evaluado y sin
    hallazgos. Jamás lanza."""
    viol, est = culinary_contract_scan_status(plan_data, catalog)
    if est["status"] == "scanned":
        return {"estado": "evaluado", "aprobado": not viol, "hallazgos": len(viol), "scan": est}
    if est["status"] == "no_meals":
        return {"estado": "no_evaluado", "aprobado": False, "hallazgos": 0, "scan": est}
    return {"estado": "no_evaluable", "aprobado": False, "hallazgos": 0, "scan": est}

