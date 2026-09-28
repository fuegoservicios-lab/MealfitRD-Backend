# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-746 · 2026-09-28] El revisor no rechaza con confirmaciones.

Producción, 29-ago → 28-sep (journal de `mealfit-backend`): de las 38 razones de rechazo que escribió el revisor LLM, 7
no señalaban ningún defecto. Cinco llegaron juntas en una sola revisión (25-sep 04:33:03, bloque 92328ff7 semana 2),
con severidad «minor»: el plan se regeneró dos veces y esas frases viajaron a la directiva del reintento como
«RESTRICCIONES ACUMULADAS»:

    · «No se detectan alérgenos declarados (paciente sin alergias).»
    · «No se detectan violaciones de condiciones médicas (paciente sin condiciones ni medicamentos).»
    · «Dieta 'balanced' respetada; no hay restricciones vegetarianas/veganas/sin gluten declaradas.»
    · «El plan contiene 7 días (Día 4 a Día 7) pero el plan solicitado es de 3 días; esto es una inconsistencia
      estructural, no un riesgo médico.»  (el bloque era de 4 días, 4-7: el revisor no recibe el número pedido)

El lote 227 rebaja lo que se niega con su CONCLUSIÓN («…por lo que no hay violación»); esto cubre la otra forma: la
razón ENTERA es un parte de cumplimiento. Solo se descarta cuando CADA cláusula es una de estas tres, anclada de punta a
punta (lo que no casa entero, se queda):

    1. negación de algo MALO — «no se detecta(n) / no hay / sin / ningún» + alérgeno, violación, incumplimiento, riesgo,
       interacción, contraindicación, conflicto, problema, alimento/ingrediente rechazado o prohibido, restricción
       declarada. Negar algo BUENO («no se detectan fuentes de hierro», «no hay suficiente proteína») es un defecto y
       no casa;
    2. cumplimiento sin negación ni matiz — «dieta 'X' respetada», «el plan respeta las restricciones declaradas»;
    3. el número de días del plan comparado con el «solicitado», cerrado por «esto es una inconsistencia estructural,
       no un riesgo médico» (el revisor no recibe el número de días pedido; lo que cierra es su propia conclusión);
    4. y, como la cola del 227, una última cláusula que concluye «(sin problema en este punto)».

Y nunca si la razón trae un giro que afirma algo más: pero, sin embargo, excepto, salvo, si, aunque, sino, solo…, un
matiz que insinúa lo menor (grave, crítica, evidente, relevante…), un parcial (parcialmente, casi, en general) o un
verbo de acción (debe, conviene, reemplazar, reducir, verificar, confirmar…). Pregunta-y-respuesta sin conclusión
(«¿Incluye hígado? No.») se queda: depende de QUÉ se pregunta.

Los guards DETERMINISTAS (alérgeno, dieta, rechazos, piso de proteína, tiramina, pomelo) corren después sobre el plan y
conservan la última palabra: lo que aquí se descarta es TEXTO del revisor, no una comprobación.

Se engancha en `revisor_no_defectos._downgrade_reviewer_non_issues` (la razón pasa a aviso, `_reviewer_advisories`).
Knob `MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD` (True), leído en cada llamada: apagarlo deja solo las reglas del 227.
tooltip-anchor: P1-PLAN-LOTE-746-CONFIRMACIONES
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_I = re.IGNORECASE

# Giros que afirman algo MÁS que la confirmación: con cualquiera de ellos la razón se queda entera.
_GIROS = re.compile(
    r"\b(?:pero|sin embargo|no obstante|aunque|excepto|salvo|a excepci[oó]n|sino|s[oó]lo|solamente|aparte de|"
    r"m[aá]s all[aá] de|si|siempre que|mientras|cuando|incluso|tambi[eé]n|adem[aá]s|"
    r"graves?|cr[ií]tic[oa]s?|mayor(?:es)?|importantes?|significativ\w*|relevantes?|seri[oa]s?|evidentes?|"
    r"aparent\w*|a primera vista|en principio|pendientes?|por confirmar|"
    r"parcial\w*|casi|mayormente|en general|en su mayor[ií]a|a medias|"
    r"verific\w*|confirm\w*|asegur\w*|revis\w*|deb\w*|convien\w*|recomend\w*|suger\w*|reemplaz\w*|sustitu\w*|"
    r"cambi\w*|corrig\w*|correg\w*|ajust\w*|evit\w*|reduc\w*|elimin\w*|quit\w*|añad\w*|agreg\w*|aument\w*|"
    r"falta\w*|insuficient\w*|exced\w*|exces\w*|super[ae]\w*|sobrepas\w*|por debajo|d[eé]ficit|alt[oa]s?|elevad\w*)\b",
    _I)
_NEGACION = re.compile(r"\b(?:no|nunca|jam[aá]s|ni)\b", _I)

# 1. negación de algo malo
_NEG = (r"(?:(?:el|este)\s+plan\s+)?"
        r"(?:no\s+se\s+(?:detecta|detectan|detect[oó]|detectaron|encuentra|encuentran|encontr[oó]|encontraron|"
        r"observa|observan|observ[oó]|identifica|identifican|identific[oó]|identificaron|evidencia|evidencian|"
        r"aprecia|aprecian|halla|hallan|hall[oó]|registra|registran|incluye|incluyen)|"
        r"no\s+(?:hay|existe|existen|presenta|contiene|incluye|tiene|aparecen?|figuran?)|sin|ning[uú]n|ninguna|ninguno)"
        r"(?:\s+(?:ning[uú]n|ninguna|ninguno|una?|el|la|los|las|otr[oa]s?))?")
_MALO = (r"(?:al[eé]rgen[oa]s?|violaci[oó]n(?:es)?|incumplimientos?|infracci[oó]n(?:es)?|riesgos?|"
         r"interacci[oó]n(?:es)?|contraindicaci[oó]n(?:es)?|conflictos?|problemas?|"
         r"(?:alimentos?|ingredientes?|productos?)\s+(?:rechazad|prohibid|vetad|excluid)[oa]s?|"
         r"restricci[oó]n(?:es)?(?:\s+[^\s,;:()]+){0,4}?\s+declarad[oa]s?)")
_COLA = (r"(?:\s+declarad[oa]s?)?"
         r"(?:\s+(?:de|del|en|para|por|con|a|al|sobre)\s+[^,;:()]{1,60}?)?"
         r"(?:\s+(?:aparecen?|figuran?|se\s+incluyen?|est[aá]n?\s+presentes?)(?:\s+en\s+el\s+plan)?)?"
         r"(?:\s*\([^()]{0,100}\))?")
_NIEGA_ALGO_MALO = re.compile(_NEG + r"\s+" + _MALO + _COLA, _I)

# 2. cumplimiento
_NOMBRE = r"(?:['\"«“‘][^'\"»”’]{1,30}['\"»”’]|[a-záéíóúñü-]+)"
_CUMPLE = re.compile(
    r"(?:(?:la\s+)?dieta\s+" + _NOMBRE + r"\s+(?:(?:es|fue|est[aá]|queda)\s+)?"
    r"(?:respetad[oa]|cumplid[oa]|se\s+respeta|se\s+cumple)(?:\s+en\s+[^,;:()]{1,40})?"
    r"|(?:(?:el|este)\s+plan|el\s+men[uú])\s+(?:respeta|cumple(?:\s+con)?)\s+"
    r"(?:la|las|los|el|sus|su|todas?\s+las|todos\s+los)\s+[^,;:()]{1,60}"
    r"|(?:(?:el|este)\s+plan|el\s+men[uú])\s+es\s+(?:segur[oa]|adecuad[oa]|apt[oa]|compatible)"
    r"(?:\s+(?:para|con)\s+[^,;:()]{1,60})?)"
    r"(?:\s*\([^()]{0,100}\))?",
    _I)

# 3. el número de días, cerrado por «no un riesgo médico»
_DIAS = re.compile(
    r"(?:el|este)\s+(?:plan|bloque)\s+(?:contiene|tiene|incluye|abarca|cubre|presenta)\s+\d+\s+d[ií]as"
    r"(?:\s*\([^()]{0,40}\))?\s*,?\s+(?:pero|mientras\s+que|y)\s+(?:el\s+(?:plan|bloque)\s+)?"
    r"(?:solicitad|pedid|requerid|esperad|indicad)[oa]s?\s+(?:es|era|son|eran)\s+(?:de\s+)?\d+(?:\s+d[ií]as)?",
    _I)
_SIN_RIESGO_MEDICO = re.compile(
    r"(?:esto|eso|lo\s+cual|lo\s+que)\s+(?:es|constituye|representa|supone)\s+una?\s+"
    r"(?:inconsistencia|discrepancia|diferencia|incoherencia)\s+(?:estructural|de\s+estructura|de\s+formato|formal)"
    r"\s*,\s*(?:y\s+)?no\s+(?:es\s+)?(?:un|ning[uú]n)\s+riesgo\s+(?:m[eé]dico|cl[ií]nico|para\s+la\s+salud)",
    _I)


# 4. la conclusión explícita «(sin problema en este punto)» cierra la última cláusula (batería: «El paciente declaró que
#    NO le gusta la berenjena; no aparece berenjena en el plan (sin problema en este punto).»). Misma regla que la cola
#    del 227 («…por lo que este punto se cumple»): la conclusión manda, y solo sin giros.
_SIN_PROBLEMA_FINAL = re.compile(
    r"(?:\(\s*sin\s+(?:ning[uú]n\s+)?problemas?(?:\s+en\s+este\s+punto)?\s*\)"
    r"|[,;—–-]\s*sin\s+(?:ning[uú]n\s+)?problemas?\s+en\s+este\s+punto)$",
    _I)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _clausulas(texto: str) -> list:
    partes = re.split(r"(?<=[.;!?])\s+|;\s*", texto)
    return [p.strip().rstrip(".;!? ").strip() for p in partes if p.strip(" .;!?")]


def motivo(texto) -> "str | None":
    """Qué regla descarta la razón (`confirmacion` / `dias_sin_riesgo_medico`), o None si se queda. Puro; nunca lanza."""
    try:
        if not isinstance(texto, str) or not activo():
            return None
        t = " ".join(texto.split())
        cl = _clausulas(t)
        if not cl:
            return None
        if len(cl) == 2 and _DIAS.fullmatch(cl[0]) and _SIN_RIESGO_MEDICO.fullmatch(cl[1]):
            return "dias_sin_riesgo_medico"
        if _GIROS.search(t):
            return None
        if _SIN_PROBLEMA_FINAL.search(cl[-1]):
            return "concluye_sin_problema"
        for c in cl:
            if _NIEGA_ALGO_MALO.fullmatch(c):
                continue
            if _CUMPLE.fullmatch(c) and not _NEGACION.search(c):
                continue
            return None
        return "confirmacion"
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-746] no-op: {type(e).__name__}: {e}")
        return None


def es_confirmacion_sin_defecto(texto) -> bool:
    """True si la razón del revisor no señala ningún defecto (ver el docstring del módulo)."""
    m = motivo(texto)
    if m:
        logger.info(f"📋 [P1-PLAN-LOTE-746] razón del revisor sin defecto ({m}) → aviso: {str(texto)[:140]}")
    return bool(m)


__all__ = ["es_confirmacion_sin_defecto", "motivo", "activo"]
