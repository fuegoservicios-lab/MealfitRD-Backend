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
    assert go._allergen_pool_item_banned("Avena sin gluten (panqueques)", ["Gluten"]) is False
    assert go._allergen_pool_item_banned("Pan integral", ["Gluten"]) is True
