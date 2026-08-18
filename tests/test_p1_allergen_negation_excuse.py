# [P1-ALLERGEN-NEGATION-EXCUSE · 2026-08-09] El scanner de alérgenos castigaba el CUMPLIMIENTO:
# el generador emitía «avena certificada sin gluten» / «quinoa certificada sin gluten» para el
# alérgico a gluten (obedeciendo la directiva) y `\bgluten\b` matcheaba DENTRO de la negación →
# rechazo crítico + retry envenenado (7+ ocurrencias en las corridas N=20, viva en la corrida 2:
# corr=abb71a1d 2026-08-09 03:50). El mismo matcher alimenta el SKELETON ALLERGEN SCRUB, que le
# QUITABA al day-gen la «Avena sin gluten» que el planner asignó correctamente — un FP, dos
# superficies. La excusa es de PREFIJO (la plant-adj existente mira el sufijo): un token cuyo
# prefijo inmediato es negación («sin X», «libre de X», «cero X», «no contiene X») está declarando
# AUSENCIA. Solo se absuelve el token negado — «leche sin lactosa» sigue violando 'lácteos' (la
# proteína láctea está presente; solo 'lactosa' quedaría negada).
#
# [fix-round 1 · P1-COUNTRY-SYSTEM-F2 T4 review · 2026-08-17] `graph_orchestrator.py` ganó una
# excusa FORWARD hermana (`_GLUTEN_FORWARD_EXCUSE_RX`, scoped a gluten únicamente) para poder sumar
# 'avena' bare a `_ALLERGEN_SYNONYMS['gluten']` sin reintroducir este mismo FP («avena certificada
# sin gluten»: la negación SIGUE al término, esta excusa de PREFIJO no la alcanza). Delta medido y
# aceptado a propósito: «pan sin gluten» (pan real GF, mismo claim que avena/quinoa certificadas)
# pasa de violar-vía-'pan' a excusado — ver `test_pan_sin_gluten_ya_no_viola_fix_round_1` abajo
# (reemplaza a `test_pan_sin_gluten_sigue_flagged_por_pan`, que anclaba el comportamiento VIEJO).
# El sesgo a sobre-detectar sigue intacto para pan SIN el claim GF — ver
# `test_pan_integral_sin_claim_gf_sigue_violando`.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

import graph_orchestrator as go  # noqa: E402


def _plan(*ings):
    return {"days": [{"meals": [{"name": "plato", "ingredients": list(ings)}]}]}


def _viols(ings, allergies):
    return go._scan_allergen_violations(_plan(*ings), allergies)


def test_avena_certificada_sin_gluten_no_viola():
    # el caso medido (corr=abb71a1d y 6+ más): cumplimiento castigado
    assert _viols(["20 g de avena certificada sin gluten"], ["Gluten"]) == []


def test_quinoa_sin_gluten_no_viola():
    assert _viols(["15 g de quinoa certificada sin gluten en crudo"], ["Gluten"]) == []


def test_libre_de_gluten_no_viola():
    assert _viols(["30 g de harina de arroz libre de gluten"], ["Gluten"]) == []


def test_gluten_real_sigue_violando():
    # `forbidden` es un SET: con 'trigo' Y 'harina de trigo' matcheando la misma línea, CUÁL se
    # reporta depende del orden de iteración (hash aleatorio por proceso) — anclar uno concreto
    # hizo flaky la 1ª versión de este test (falló en la suite completa, pasaba solo). La
    # aserción honesta: HAY violación y el término es un sinónimo de gluten.
    v = _viols(["15 g de harina de trigo"], ["Gluten"])
    assert v and "trigo" in v[0][2], "el trigo real DEBE seguir flagged"
    v2 = _viols(["seitan (gluten de trigo)"], ["Gluten"])
    assert v2, "gluten AFIRMADO (no negado) debe seguir flagged"


def test_leche_sin_lactosa_sigue_violando_lacteos():
    # la negación de 'lactosa' NO absuelve 'leche': la proteína láctea (caseína/whey) está
    # presente — para el alérgico a lácteos es violación REAL. Absolverla sería el error
    # simétrico (fail-open) que el sesgo de sobre-detección prohíbe.
    v = _viols(["200 ml de leche sin lactosa"], ["Lácteos"])
    assert v and v[0][2] == "leche"


def test_pan_sin_gluten_ya_no_viola_fix_round_1():
    # [fix-round 1 · reemplaza a test_pan_sin_gluten_sigue_flagged_por_pan] Delta VERIFICADO
    # antes (violaba vía 'pan') y después (excusado) de sumar `_GLUTEN_FORWARD_EXCUSE_RX`: 'pan'
    # es un término de la categoría gluten como cualquier otro, y «pan sin gluten» es el MISMO
    # claim de cumplimiento que «avena certificada sin gluten» — mismo mecanismo, misma confianza
    # en el texto. No es un ensanchamiento ad-hoc: es la consecuencia natural de aplicar la excusa
    # forward de manera uniforme a TODA la categoría (la alternativa — excusar solo 'avena' con
    # una lista de excepciones por-término — arrastra complejidad sin beneficio de seguridad real,
    # ver reporte T4 §Fix round 1).
    v = _viols(["1 rebanada de pan sin gluten"], ["Gluten"])
    assert v == [], "pan sin gluten' es CUMPLIMIENTO (pan real GF) — no debe violar tras fix-round 1"


def test_pan_integral_sin_claim_gf_sigue_violando():
    # sobre-detección intencional SIGUE intacta para pan SIN claim de ausencia: 'pan integral' no
    # tiene ninguna negación cerca (ni prefijo ni forward) — la excusa forward exige la palabra
    # literal 'gluten' tras una negación dentro de la ventana corta, y aquí no hay ninguna.
    v = _viols(["1 rebanada de pan integral"], ["Gluten"])
    assert v, "pan integral (sin claim GF) debe seguir violando — sobre-detección intacta"


def test_pool_scrub_ya_no_roba_la_avena_sin_gluten():
    # segunda superficie envenenada (mismo SSOT): el scrub del skeleton le quitaba al day-gen
    # la 'Avena sin gluten' que el planner asignó CORRECTAMENTE para el alérgico.
    #
    # [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, i)] El paréntesis "(panqueques)" original
    # colisiona con una expansión LEGÍTIMA y no-relacionada del vocabulario: T7 (altas de catálogo
    # PR/US, el mismo día) sumó 'panqueque'/'panqueques' a `_ALLERGEN_SYNONYMS['gluten']` (son un
    # alimento de trigo real). Esta fixture ya fallaba en HEAD antes de esta task (reproducido
    # contra 5458c85, la base de Task 9) — el fallo es ortogonal a lo que este test verifica (la
    # excusa de NEGACIÓN, no el vocabulario de 'panqueques'). Cambiado el paréntesis para que la
    # fixture vuelva a probar SOLO lo que su nombre promete, sin tocar el vocabulario de alérgenos
    # (fuera de scope de (i) — la whitelist es el mecanismo BACKWARD, no el vocabulario forward).
    assert go._allergen_pool_item_banned("Avena sin gluten (para el desayuno)", ["Gluten"]) is False
    assert go._allergen_pool_item_banned("Pan integral", ["Gluten"]) is True


# [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, i · T4-parked, backward mirror del forward
# fix-round 2)] `_ALLERGEN_NEGATION_PREFIX_RX` (el mecanismo BACKWARD, preexistente desde
# 2026-08-09 — NUNCA tocado por T4) tenía un filler `(?:\w+\s+)?` SIN restricción: en «Sin gluten
# trigo» (2 términos de gluten distintos, SIN conjunción siquiera), al escanear 'trigo' el filler
# se tragaba 'gluten' como relleno genérico y excusaba 'trigo' — que no tiene claim propio. MISMA
# clase de leak que el forward (`_GLUTEN_FORWARD_EXCUSE_RX`) tuvo en su fix-round 1 y cerró en su
# fix-round 2 con una whitelist evidence-derived. Búsqueda de evidencia backward (los 8 tests de
# arriba + `master_ingredients`/`supermarket_products` en vivo, `ILIKE '%sin %'`/`%libre%`):
# NEGATIVA — ningún test ni fila de catálogo evidencia un relleno de 1 palabra en esta dirección
# ('trazas', el ejemplo del comentario original de 2026-08-09, nunca apareció en ningún test ni
# catálogo). Dirección fail-safe: el filler se ELIMINÓ (no se blanqueó) — mismo criterio que cerró
# el hueco simétrico del forward cuando la evidencia también fue negativa.

def test_i_sin_gluten_trigo_ya_no_se_excusa():
    """RED-first (reproducido contra graph_orchestrator.py PRE-fix): 'Sin gluten trigo' devolvía
    `[]` — el filler sin restricción se tragaba 'gluten' y excusaba 'trigo'."""
    v = _viols(["Sin gluten trigo"], ["Gluten"])
    assert v and v[0][2] == "trigo", "trigo' (sin claim propio) DEBE seguir violando"


def test_i_no_contiene_gluten_trigo_ya_no_se_excusa():
    v = _viols(["No contiene gluten trigo"], ["Gluten"])
    assert v and v[0][2] == "trigo"


def test_i_libre_de_gluten_trigo_ya_no_se_excusa():
    """Las 4 formas de negación ('sin'/'libre de'/'cero'/'no contiene') comparten el mismo regex
    — ancla las otras 2 no cubiertas por los tests anteriores."""
    v = _viols(["Libre de gluten trigo"], ["Gluten"])
    assert v and v[0][2] == "trigo"


def test_i_cero_gluten_trigo_ya_no_se_excusa():
    v = _viols(["Cero gluten trigo"], ["Gluten"])
    assert v and v[0][2] == "trigo"


def test_i_leche_sin_lactosa_sigue_violando_tras_el_fix():
    """Control de regresión: el caso YA anclado por
    test_leche_sin_lactosa_sigue_violando_lacteos sigue verde tras quitar el filler (0-relleno,
    nunca dependió de él)."""
    v = _viols(["200 ml de leche sin lactosa"], ["Lácteos"])
    assert v and v[0][2] == "leche"


def test_i_los_8_casos_de_negation_excuse_siguen_verdes_0_relleno():
    """Los 4 claims GF legítimos (0-relleno o excusados vía FORWARD, nunca vía el filler
    backward) permanecen excusados — el filler eliminado NUNCA los sostenía."""
    assert _viols(["20 g de avena certificada sin gluten"], ["Gluten"]) == []
    assert _viols(["15 g de quinoa certificada sin gluten en crudo"], ["Gluten"]) == []
    assert _viols(["30 g de harina de arroz libre de gluten"], ["Gluten"]) == []
    assert _viols(["1 rebanada de pan sin gluten"], ["Gluten"]) == []


def test_i_mutacion_filler_sin_restriccion_reproduce_el_leak():
    """MUTACIÓN bidireccional: reconstruye el regex PRE-fix (filler `(?:\\w+\\s+)?` sin
    restricción) y confirma EN VIVO que sí excusaba 'trigo' en 'Sin gluten trigo' — la evidencia
    de que el fix real (no solo el comentario) cerró el hueco."""
    import re as _re_mutation
    _rx_legacy = _re_mutation.compile(
        r"(?:\bsin|\blibres?\s+de|\bcero|\bno\s+contienen?)\s+(?:\w+\s+)?$"
    )
    s = "sin gluten trigo"
    idx_trigo = s.index("trigo")
    window = s[max(0, idx_trigo - 24): idx_trigo]
    assert _rx_legacy.search(window), "el regex legacy debía excusar 'trigo' (reproduce el leak)"
    assert not go._ALLERGEN_NEGATION_PREFIX_RX.search(window), (
        "el regex re-anclado (sin filler) NO debe excusar 'trigo'"
    )


def test_i_trazas_nunca_fue_evidenciado():
    """Documenta la búsqueda de evidencia (no solo el código): 'trazas' — el ejemplo ilustrativo
    del comentario de diseño original (2026-08-09) — no aparece en NINGÚN archivo de producción
    relacionado con el vocabulario de alérgenos ni en el catálogo, confirmando que era
    especulativo, nunca evidencia real."""
    src = (os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"))
    with open(src, encoding="utf-8") as f:
        text = f.read()
    # 'trazas' solo debe aparecer en PROSA de comentario (documentando la decisión de no
    # whitelistearlo) — nunca como token vivo dentro de una whitelist/regex compilado.
    assert "_GLUTEN_FORWARD_FILLER_WHITELIST = (\"certificada\",)" in text, (
        "la whitelist forward sigue siendo SOLO 'certificada' — si 'trazas' se añadió, "
        "debe venir con su propio test de evidencia"
    )
