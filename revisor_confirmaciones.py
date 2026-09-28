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
razón ENTERA es un parte de cumplimiento. Solo se descarta cuando CADA cláusula es una de estas formas, anclada de punta a
punta (lo que no casa entero, se queda):

    1. negación de algo MALO — «no se detecta(n) / no hay / sin / ningún» + alérgeno, violación, incumplimiento, riesgo,
       interacción, contraindicación, conflicto, problema, alimento/ingrediente rechazado o prohibido, restricción
       declarada. Negar algo BUENO («no se detectan fuentes de hierro», «no hay suficiente proteína») es un defecto y
       no casa;
    2. cumplimiento sin negación ni matiz — «dieta 'X' respetada», «el plan respeta las restricciones declaradas»;
    3. el número de días del plan comparado con el «solicitado», cerrado por «esto es una inconsistencia estructural,
       no un riesgo médico» (el revisor no recibe el número de días pedido; lo que cierra es su propia conclusión);
    4. y, como la cola del 227, una última cláusula que concluye «(sin problema en este punto)» y niega la presencia de
       algo («no aparece berenjena en el plan»), con cada cláusula anterior confirmación (1-2) o declaración pura del
       paciente («El paciente declaró que NO le gusta la berenjena»).

Y nunca si la razón trae un giro que afirma algo más: pero, sin embargo, excepto, salvo, si, aunque, sino, solo…, un
matiz que insinúa lo menor (grave, crítica, evidente, relevante…), un parcial (parcialmente, casi, en general) o un
verbo de acción (debe, conviene, reemplazar, reducir, verificar, confirmar…). Pregunta-y-respuesta sin conclusión
(«¿Incluye hígado? No.») se queda: depende de QUÉ se pregunta.

Los guards DETERMINISTAS (alérgeno, dieta, rechazos, piso de proteína, tiramina, pomelo) corren después sobre el plan y
conservan la última palabra: lo que aquí se descarta es TEXTO del revisor, no una comprobación.

Se engancha en `revisor_no_defectos._downgrade_reviewer_non_issues` (la razón pasa a aviso, `_reviewer_advisories`).
Knob `MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD` (True), leído en cada llamada: apagarlo deja solo las reglas del 227.
DEPENDE del knob del 227: con `MEALFIT_REVIEWER_NON_ISSUES_ADVISORY` apagado, `_downgrade_reviewer_non_issues` sale
antes del bucle y esta regla no corre en el revisor, aunque su propio knob siga en True.

[P1-PLAN-LOTE-746 · 2026-09-28] Tres huecos que aprobaban defectos reales con severidad `critical`, cerrados: la regla 4
exige que TODA cláusula anterior sea confirmación o declaración pura del paciente (y «(sin problemas)» sin «en este
punto» ya no es la conclusión); el texto libre no admite verbos que afirmen contenido, días, comidas ni cantidades, y el
único paréntesis es el del paciente («(paciente sin alergias)»); «restricción … declarada» solo con negación de
EXISTENCIA («no hay», «sin» sin artículo), nunca con incluye/contiene/presenta/tiene.
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

# [P1-PLAN-LOTE-746 · 2026-09-28] El TEXTO LIBRE (la cola «de/en/para…», el objeto de «el plan respeta…», el nombre de la
# dieta, el paréntesis) no puede llevar un hallazgo: nunca un verbo conjugado que afirme contenido o un cambio
# («…(la merienda del Día 3 contiene maní)», «ningún alimento prohibido… se retiró del plan»), ni un día, una comida o
# una cantidad concretos. Lo que no casa se queda (un reintento de más; al revés, un defecto aprobado).
_NO_LIBRE = (r"(?:contien\w*|conten\w*|inclu\w*|aport\w*|combin\w*|tien\w*|tuv\w*|llev\w*|us[aeoó]\w*|sirv\w*|"
             r"retir\w*|mantien\w*|mantuv\w*|pon\w*|pus\w*|mezcl\w*|acompañ\w*|dej\w*|conserv\w*|persist\w*|sigu\w*|"
             r"contin[uú]\w*|permanec\w*|aparec\w*|figur\w*|trae\w*|traj\w*|lleg\w*|hay|hubo|existe\w*|exist[ií]\w*|"
             r"qued\w*|fue|fueron|es|son|era|eran|est[aá](?:n|s|ba\w*|r\w*)?|ha|han|hab[ií]a\w*|se|que|"
             r"d[ií]as?|desayunos?|almuerzos?|cenas?|meriendas?|\d\w*)")
_LIBRE = r"(?:(?!\b" + _NO_LIBRE + r"\b)[^,;:()])"          # un carácter de texto libre nominal
_PALABRA = r"(?:(?!\b" + _NO_LIBRE + r"\b)[^\s,;:()])+"     # una palabra de texto libre nominal
# El único paréntesis admitido es el que describe al PACIENTE, no al plan: «(paciente sin alergias)», «(el paciente no
# reporta alergias)». Propuesta de la revisión, con el mismo texto libre nominal dentro.
_PAREN = (r"(?:\s*\(\s*(?:el\s+|la\s+)?paciente\s+(?:sin|no\s+(?:reporta|report[oó]|declara|declar[oó]|tiene|presenta))"
          r"(?:(?!\b" + _NO_LIBRE + r"\b)[^();]){0,80}\))?")

# 1. negación de algo malo
_NEG = (r"(?:(?:el|este)\s+plan\s+)?"
        r"(?:no\s+se\s+(?:detecta|detectan|detect[oó]|detectaron|encuentra|encuentran|encontr[oó]|encontraron|"
        r"observa|observan|observ[oó]|identifica|identifican|identific[oó]|identificaron|evidencia|evidencian|"
        r"aprecia|aprecian|halla|hallan|hall[oó]|registra|registran|incluye|incluyen)|"
        r"no\s+(?:hay|existe|existen|presenta|contiene|incluye|tiene|aparecen?|figuran?)|sin|ning[uú]n|ninguna|ninguno)"
        r"(?:\s+(?:ning[uú]n|ninguna|ninguno|una?|el|la|los|las|otr[oa]s?))?")
_MALO = (r"(?:al[eé]rgen[oa]s?|violaci[oó]n(?:es)?|incumplimientos?|infracci[oó]n(?:es)?|riesgos?|"
         r"interacci[oó]n(?:es)?|contraindicaci[oó]n(?:es)?|conflictos?|problemas?|"
         r"(?:alimentos?|ingredientes?|productos?)\s+(?:rechazad|prohibid|vetad|excluid)[oa]s?)")
# [P1-PLAN-LOTE-746] «restricción … declarada» es algo BUENO que debe estar: solo es una confirmación negada su EXISTENCIA
# («no hay / no se detectan / sin / ninguna restricción… declarada»). Con verbo de posesión («el plan no incluye/contiene/
# presenta/tiene la restricción de sodio declarada») o artículo («sin la restricción…») dice que FALTA: un defecto.
_NEG_EXISTE = (r"(?:no\s+se\s+(?:detecta|detectan|detect[oó]|detectaron|encuentra|encuentran|encontr[oó]|encontraron|"
               r"observa|observan|observ[oó]|identifica|identifican|identific[oó]|identificaron|registra|registran)|"
               r"no\s+(?:hay|existe|existen)|sin|ning[uú]n|ninguna|ninguno)"
               r"(?:\s+(?:ning[uú]n|ninguna|ninguno|otr[oa]s?))?")
_RESTRICCION = r"restricci[oó]n(?:es)?(?:\s+" + _PALABRA + r"){0,4}?\s+declarad[oa]s?"
_COLA = (r"(?:\s+declarad[oa]s?)?"
         r"(?:\s+(?:de|del|en|para|por|con|a|al|sobre)\s+" + _LIBRE + r"{1,60}?)?"
         r"(?:\s+(?:aparecen?|figuran?|se\s+incluyen?|est[aá]n?\s+presentes?)(?:\s+en\s+el\s+plan)?)?"
         + _PAREN)
_NIEGA_ALGO_MALO = re.compile(
    r"(?:" + _NEG + r"\s+" + _MALO + r"|" + _NEG_EXISTE + r"\s+" + _RESTRICCION + r")" + _COLA, _I)

# 2. cumplimiento
_NOMBRE = r"(?:['\"«“‘](?:(?!\b" + _NO_LIBRE + r"\b)[^'\"»”’]){1,30}['\"»”’]|[a-záéíóúñü-]+)"
_CUMPLE = re.compile(
    r"(?:(?:la\s+)?dieta\s+" + _NOMBRE + r"\s+(?:(?:es|fue|est[aá]|queda)\s+)?"
    r"(?:respetad[oa]|cumplid[oa]|se\s+respeta|se\s+cumple)(?:\s+en\s+" + _LIBRE + r"{1,40})?"
    r"|(?:(?:el|este)\s+plan|el\s+men[uú])\s+(?:respeta|cumple(?:\s+con)?)\s+"
    r"(?:la|las|los|el|sus|su|todas?\s+las|todos\s+los)\s+" + _LIBRE + r"{1,60}"
    r"|(?:(?:el|este)\s+plan|el\s+men[uú])\s+es\s+(?:segur[oa]|adecuad[oa]|apt[oa]|compatible)"
    r"(?:\s+(?:para|con)\s+" + _LIBRE + r"{1,60})?)"
    + _PAREN,
    _I)

# 3. el número de días, cerrado por «no un riesgo médico» (el paréntesis solo puede ser el rango: «(Día 4 a Día 7)»)
_DIAS = re.compile(
    r"(?:el|este)\s+(?:plan|bloque)\s+(?:contiene|tiene|incluye|abarca|cubre|presenta)\s+\d+\s+d[ií]as"
    r"(?:\s*\(\s*d[ií]as?\s+\d+\s*(?:a|al|-|–|y|hasta)\s*(?:el\s+)?(?:d[ií]a\s+)?\d+\s*\))?"
    r"\s*,?\s+(?:pero|mientras\s+que|y)\s+(?:el\s+(?:plan|bloque)\s+)?"
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
#    [P1-PLAN-LOTE-746 · 2026-09-28] La conclusión absuelve SU punto, no la razón entera: «Anemia: el desayuno del Día 2
#    combina leche con avena… Hidratación adecuada (sin problemas)» salía aprobada con severidad `critical`. Ahora:
#      · la conclusión es «(sin problema(s) en este punto)» — sin «en este punto» es el veredicto de UNA línea;
#      · cada cláusula ANTERIOR es una confirmación (reglas 1-2) o una declaración pura del paciente («El paciente
#        declaró que NO le gusta la berenjena»), que no menciona el plan;
#      · y el cuerpo de la última cláusula niega la PRESENCIA de algo («no aparece berenjena en el plan») o es una
#        confirmación (reglas 1-2). Un hallazgo («el Día 3 incluye hígado…») con la coletilla se queda.
_SIN_PROBLEMA_FINAL = re.compile(
    r"\s*(?:\(\s*sin\s+(?:ning[uú]n\s+)?problemas?\s+en\s+este\s+punto\s*\)"
    r"|[,;—–-]\s*sin\s+(?:ning[uú]n\s+)?problemas?\s+en\s+este\s+punto)$",
    _I)
_DECLARACION = re.compile(
    r"(?:el|la)\s+(?:paciente|usuari[oa])\s+(?:declar[oó]|indic[oó]|report[oó]|rechaz[oó]|dijo|señal[oó]|mencion[oó])"
    r"(?:\s+que)?(?:\s+" + _PALABRA + r"){1,8}",
    _I)
_MENCIONA_PLAN = re.compile(
    r"\b(?:plan|planes|men[uú]|comidas?|recetas?|platos?|incluid[oa]s?|servid[oa]s?|presentes?)\b", _I)
_NO_APARECE = re.compile(
    r"(?:" + _PALABRA + r"\s+){0,4}"
    r"no\s+(?:aparece|aparecen|figura|figuran|se\s+incluye|se\s+incluyen|est[aá]n?\s+presentes?|se\s+usa|se\s+usan)"
    r"(?:\s+" + _PALABRA + r"){0,4}?(?:\s+en\s+(?:el|este|ning[uú]n)\s+(?:plan|men[uú]))?",
    _I)


def _confirma(c: str) -> bool:
    """Reglas 1-2 sobre una cláusula."""
    return bool(_NIEGA_ALGO_MALO.fullmatch(c) or (_CUMPLE.fullmatch(c) and not _NEGACION.search(c)))


def _declaracion_pura(c: str) -> bool:
    return bool(_DECLARACION.fullmatch(c)) and not _MENCIONA_PLAN.search(c)


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
        fin = _SIN_PROBLEMA_FINAL.search(cl[-1])
        if fin:                                                                # regla 4 (revisión 1: ver arriba)
            cuerpo = cl[-1][:fin.start()].strip()
            if (all(_confirma(c) or _declaracion_pura(c) for c in cl[:-1])
                    and cuerpo and (_confirma(cuerpo) or _NO_APARECE.fullmatch(cuerpo))):
                return "concluye_sin_problema"
            return None
        if all(_confirma(c) for c in cl):
            return "confirmacion"
        return None
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
