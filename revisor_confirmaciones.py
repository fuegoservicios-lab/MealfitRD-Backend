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
       LO QUE EL PACIENTE DECLARÓ rechazar («no aparece berenjena en el plan»), con cada cláusula anterior confirmación
       (1-2) o declaración de rechazo/alergia del paciente («El paciente declaró que NO le gusta la berenjena»).

Todo con FORMAS CERRADAS (revisión 2, abajo): el alcance, el objeto y el paréntesis se enumeran; no hay texto libre.

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

[P1-PLAN-LOTE-746 · 2026-09-28] Revisión 1 — tres huecos que aprobaban defectos reales con severidad `critical`,
cerrados: la regla 4 exige que TODA cláusula anterior sea confirmación o declaración del paciente (y «(sin problemas)»
sin «en este punto» ya no es la conclusión); el único paréntesis es el del paciente («(paciente sin alergias)»);
«restricción … declarada» solo con negación de EXISTENCIA («no hay», «sin» sin artículo), nunca con incluye/contiene/
presenta/tiene.

[P1-PLAN-LOTE-746 · 2026-09-28] Revisión 2 — FORMAS CERRADAS. La ronda 1 acotó el texto libre con una lista de palabras
PROHIBIDAS y se le escapó el alcance parcial: «La dieta renal se cumple en la mitad de las comidas», «El plan respeta la
restricción de potasio en la semana uno», «No hay interacción con la warfarina en la mayoría de las comidas», «El plan
es adecuado para un adulto sano sin diabetes» salían aprobadas con severidad `critical` (en main rechazaban; renal,
sodio, potasio y warfarina solo los ve el LLM). Una lista de lo prohibido siempre olvida algo: ahora se enumera lo
PERMITIDO y todo lo demás se queda —
  · alcance: «en (todo) el plan / el menú», «en todas las comidas», «en todos los días», «en ninguna comida»;
  · objeto: condiciones médicas, restricciones, alergias, medicamentos, preferencias, rechazos, intolerancias o «la
    dieta X», con «declaradas / del paciente»; «es seguro/adecuado/compatible para el paciente / con sus restricciones /
    con la dieta X»;
  · paréntesis: «(paciente sin alergias / condiciones ni medicamentos…)», solo con esos sustantivos;
  · «restricción … declarada» SIN cola: «… declaradas en el plan» se lee como que el plan no la aplica;
  · regla 4: lo negado es LO MISMO que el paciente declaró rechazar (o aquello a lo que es alérgico), y la declaración
    es de rechazo o alergia, nunca de una condición («indicó anemia ferropénica; no aparece hierro hemo»: negar algo
    BUENO es un defecto). Los cuantificadores y «severa» pasan a giros.
`oraciones_sin_hallazgo` le da la misma inversión a la regla del 227 (sin listas de verbos): ver `revisor_no_defectos`.
tooltip-anchor: P1-PLAN-LOTE-746-CONFIRMACIONES
tooltip-anchor: P1-PLAN-LOTE-746-FORMAS-CERRADAS
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_I = re.IGNORECASE

# Giros que afirman algo MÁS que la confirmación: con cualquiera de ellos la razón se queda entera. [revisión 2] con los
# cuantificadores del alcance parcial («la mayoría», «la mitad», «algunas», «la primera»…) y «severa».
_GIROS = re.compile(
    r"\b(?:pero|sin embargo|no obstante|aunque|excepto|salvo|a excepci[oó]n|sino|s[oó]lo|solamente|[uú]nicamente|"
    r"exclusivamente|aparte de|m[aá]s all[aá] de|si|siempre que|mientras|cuando|incluso|tambi[eé]n|adem[aá]s|"
    r"graves?|sever\w*|cr[ií]tic[oa]s?|mayor(?:es)?|mayor[ií]a|mitad|importantes?|significativ\w*|relevantes?|"
    r"seri[oa]s?|evidentes?|aparent\w*|a primera vista|en principio|pendientes?|por confirmar|"
    r"parcial\w*|casi|mayormente|en general|en su mayor[ií]a|a medias|algun[oa]s|poc[oa]s|ciert[oa]s|vari[oa]s|"
    r"primer[oa]?|[uú]ltim[oa]s?|"
    r"verific\w*|confirm\w*|asegur\w*|revis\w*|deb\w*|convien\w*|recomend\w*|suger\w*|reemplaz\w*|sustitu\w*|"
    r"cambi\w*|corrig\w*|correg\w*|ajust\w*|evit\w*|reduc\w*|elimin\w*|quit\w*|añad\w*|agreg\w*|aument\w*|"
    r"falta\w*|insuficient\w*|exced\w*|exces\w*|super[ae]\w*|sobrepas\w*|por debajo|d[eé]ficit|alt[oa]s?|elevad\w*)\b",
    _I)
_NEGACION = re.compile(r"\b(?:no|nunca|jam[aá]s|ni)\b", _I)

# Lo que el nombre de una dieta entre comillas o las palabras de «restricciones … declaradas» no pueden llevar: un verbo
# que afirme contenido o un cambio, ni un día, una comida, una cantidad o un cuantificador.
_NO_LIBRE = (r"(?:contien\w*|conten\w*|inclu\w*|aport\w*|combin\w*|tien\w*|tuv\w*|llev\w*|us[aeoó]\w*|sirv\w*|"
             r"retir\w*|mantien\w*|mantuv\w*|pon\w*|pus\w*|mezcl\w*|acompañ\w*|dej\w*|conserv\w*|persist\w*|sigu\w*|"
             r"contin[uú]\w*|permanec\w*|aparec\w*|figur\w*|trae\w*|traj\w*|lleg\w*|hay|hubo|existe\w*|exist[ií]\w*|"
             r"qued\w*|fue|fueron|es|son|era|eran|est[aá](?:n|s|ba\w*|r\w*)?|ha|han|hab[ií]a\w*|se|que|"
             r"d[ií]as?|desayunos?|almuerzos?|cenas?|meriendas?|comidas?|platos?|plan|men[uú]|semanas?|jornadas?|"
             r"mayor[ií]a|mitad|parte|algun\w*|poc[oa]s|ciert[oa]s|vari[oa]s|primer\w*|segund\w*|[uú]ltim\w*|"
             r"[uú]nicamente|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|mg|kcal|\d\w*)")
_PALABRA = r"(?:(?!\b" + _NO_LIBRE + r"\b)[^\s,;:()])+"     # una palabra de texto libre nominal
# «dieta 'balanced'», «dieta 'baja en sodio'», «dieta vegetariana». [revisión 2] Entre comillas solo letras, espacio, «/»
# y «-», sin conjunciones que sumen un alimento («'vegetariana con pollo'», «'vegetariana y pescado'»).
_NOMBRE = (r"(?:['\"«“‘](?:(?!\b(?:" + _NO_LIBRE + r"|con|y|o|m[aá]s|menos|pero|adem[aá]s)\b)[a-záéíóúñü /-]){1,30}"
           r"['\"»”’]|(?!\b" + _NO_LIBRE + r"\b)[a-záéíóúñü-]+)")

# ── formas CERRADAS [revisión 2] ──
_DECLARADO = (r"(?:\s+(?:declarad|registrad|conocid|reportad)[oa]s?(?:\s+(?:por|de)\s+(?:el\s+|la\s+)?paciente)?)?"
              r"(?:\s+del\s+paciente)?")
_OBJ_CLIN = (r"(?:(?:las|los|sus?|la|el)\s+)?(?:condiciones\s+m[eé]dicas|condici[oó]n\s+m[eé]dica|"
             r"restricciones(?:\s+diet[eé]ticas)?|alergias?(?:\s+alimentarias)?|medicamentos|medicaci[oó]n|"
             r"preferencias|rechazos|intolerancias?|dieta\s+" + _NOMBRE + r")" + _DECLARADO)
_ALCANCE = (r"(?:\s+en\s+(?:(?:todo\s+)?(?:el|este)\s+(?:plan|men[uú])|todas\s+las\s+comidas|"
            r"todos\s+los\s+(?:d[ií]as|platos)|ninguna\s+(?:de\s+las\s+)?comidas?|ning[uú]n\s+(?:d[ií]a|plato)|"
            r"ninguno\s+de\s+los\s+(?:d[ií]as|platos)))?")
# El único paréntesis es el que describe al PACIENTE con esos sustantivos: «(paciente sin alergias)», «(el paciente no
# reporta alergias)», «(paciente sin condiciones ni medicamentos)». «(paciente sin alergias, maní en el postre)» se queda.
_PAC_NOMBRE = (r"(?:alergias?(?:\s+alimentarias)?|condiciones(?:\s+m[eé]dicas)?|condici[oó]n\s+m[eé]dica|medicamentos|"
               r"medicaci[oó]n|restricciones(?:\s+diet[eé]ticas)?|intolerancias?|enfermedades|patolog[ií]as|rechazos)")
_PAREN = (r"(?:\s*\(\s*(?:el\s+|la\s+)?paciente\s+(?:sin|no\s+(?:reporta|report[oó]|declara|declar[oó]|tiene|presenta))"
          r"\s+(?:ning[uú]n[oa]?\s+)?" + _PAC_NOMBRE + r"(?:(?:\s*,\s*|\s+(?:ni|y|o)\s+)" + _PAC_NOMBRE + r")*"
          r"(?:\s+(?:declarad|conocid|registrad|reportad)[oa]s?)?\s*\))?")

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
_MALO_X = _MALO + _DECLARADO + r"(?:\s+(?:de|con|para|por)\s+" + _OBJ_CLIN + r")?"
_LISTA_MALO = (_MALO_X + r"(?:(?:\s*,\s*|\s+(?:ni|y|o)\s+)(?:(?:ning[uú]n[oa]?|otr[oa]s?|una?|el|la|los|las)\s+)?"
               + _MALO_X + r")*")                                          # «alérgenos ni violaciones de …»
# «restricción … declarada» es algo BUENO que debe estar: solo es una confirmación negada su EXISTENCIA («no hay / no se
# detectan / sin / ninguna restricción… declarada»). Con verbo de posesión o artículo («el plan no incluye / sin LA
# restricción de sodio declarada») dice que FALTA; y con cola («… declaradas en el plan») que el plan no la aplica.
_NEG_EXISTE = (r"(?:no\s+se\s+(?:detecta|detectan|detect[oó]|detectaron|encuentra|encuentran|encontr[oó]|encontraron|"
               r"observa|observan|observ[oó]|identifica|identifican|identific[oó]|identificaron|registra|registran)|"
               r"no\s+(?:hay|existe|existen)|sin|ning[uú]n|ninguna|ninguno)"
               r"(?:\s+(?:ning[uú]n|ninguna|ninguno|otr[oa]s?))?")
_RESTRICCION = (r"restricci[oó]n(?:es)?(?:\s+" + _PALABRA + r"){0,4}?\s+declarad[oa]s?"
                r"(?:\s+(?:por|de)\s+(?:el\s+|la\s+)?paciente)?")
_NIEGA_ALGO_MALO = re.compile(
    r"(?:" + _NEG + r"\s+" + _LISTA_MALO + r"(?:\s+(?:aparecen?|figuran?|se\s+incluyen?|est[aá]n?\s+presentes?))?"
    + _ALCANCE + r"|" + _NEG_EXISTE + r"\s+" + _RESTRICCION + r")" + _PAREN, _I)

# 2. cumplimiento
_SUJETO_PLAN = r"(?:(?:el|este)\s+plan|el\s+men[uú])"
_OBJ_CUMPLE = (r"(?:(?:todas\s+)?(?:las|sus)\s+(?:restricciones(?:\s+diet[eé]ticas)?|preferencias|alergias|"
               r"condiciones\s+m[eé]dicas|intolerancias)|(?:todos\s+)?(?:los|sus)\s+(?:rechazos|medicamentos)|"
               r"(?:la|su)\s+dieta\s+" + _NOMBRE + r")" + _DECLARADO)
_OBJ_APTO = (r"(?:(?:el|la)\s+paciente|(?:las|sus)\s+(?:restricciones|condiciones\s+m[eé]dicas|alergias)" + _DECLARADO
             + r"|(?:la|su)\s+dieta(?:\s+" + _NOMBRE + r")?)")
_CUMPLE = re.compile(
    r"(?:(?:la\s+)?dieta\s+" + _NOMBRE + r"\s+(?:(?:es|fue|est[aá]|queda)\s+)?"
    r"(?:respetad[oa]|cumplid[oa]|se\s+respeta|se\s+cumple)" + _ALCANCE
    + r"|" + _SUJETO_PLAN + r"\s+(?:respeta|cumple(?:\s+con)?)\s+" + _OBJ_CUMPLE
    + r"(?:\s+y\s+" + _OBJ_CUMPLE + r")?" + _ALCANCE
    + r"|" + _SUJETO_PLAN + r"\s+es\s+(?:segur[oa]|adecuad[oa]|apt[oa]|compatible)"
    r"(?:\s+(?:para|con)\s+" + _OBJ_APTO + r")?" + _ALCANCE + r")" + _PAREN,
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
#    [revisión 1] La conclusión absuelve SU punto, no la razón entera: la conclusión es «(sin problema(s) en este
#    punto)», y cada cláusula anterior es confirmación (1-2) o declaración del paciente.
#    [revisión 2] Y lo NEGADO es lo mismo que el paciente declaró rechazar: «La paciente indicó anemia ferropénica; no
#    aparece hierro hemo en el plan (sin problema…)» o «El paciente rechazó el hígado…; no aparece berenjena…» se quedan.
_SIN_PROBLEMA_FINAL = re.compile(
    r"\s*(?:\(\s*sin\s+(?:ning[uú]n\s+)?problemas?\s+en\s+este\s+punto\s*\)"
    r"|[,;—–-]\s*sin\s+(?:ning[uú]n\s+)?problemas?\s+en\s+este\s+punto)$",
    _I)

# Un ALIMENTO nombrado (en una declaración o en una ausencia): palabras sueltas, nunca un verbo, un día, una comida, una
# cantidad, un «presente/servido/incluido» ni un «suficiente» (lo que afirma que algo ESTÁ o que falta algo bueno).
_STOP_ALIMENTO = (r"(?:" + _NO_LIBRE + r"|y|o|ni|no|sin|con|en|del|al|de|a|por|para|el|la|los|las|un|una|lo|le|"
                  r"presentes?|servid\w*|incluid\w*|detectad\w*|encontrad\w*|visibles?|añadid\w*|agregad\w*|usad\w*|"
                  r"utilizad\w*|rechaz\w*|prohibid\w*|pacientes?|usuari[oa]s?|suficientes?|pero|tambi[eé]n|"
                  r"adem[aá]s|s[oó]lo|solamente|gust\w*|ningun\w*|ning[uú]n)")
_PALABRA_ALIMENTO = r"['\"«“‘]?(?!" + _STOP_ALIMENTO + r"\b)[a-záéíóúñü][a-záéíóúñü-]*['\"»”’]?"
_ALIMENTO = (r"(?:(?:el|la|los|las|un|una)\s+)?" + _PALABRA_ALIMENTO
             + r"(?:\s+(?:de\s+)?" + _PALABRA_ALIMENTO + r"){0,2}")
_ALIMENTOS = _ALIMENTO + r"(?:(?:\s*,\s*|\s+(?:y|ni|o)\s+)" + _ALIMENTO + r")*"

# Declaración del paciente: SOLO de rechazo o alergia (lo que el plan debe NO tener). Una condición («indicó anemia»)
# no es una declaración: negar su tratamiento es un defecto.
_SUJ_PAC = r"(?:el|la)\s+(?:paciente|usuari[oa])"
_ALERGICO = r"es\s+al[eé]rgic[oa](?:\s+(?:a|al))?"
_DECLARACION = re.compile(
    r"(?:" + _SUJ_PAC + r"\s+(?:declar[oó]|indic[oó]|report[oó]|dijo|señal[oó]|mencion[oó]|expres[oó]|refiri[oó])\s+"
    r"que\s+(?:no\s+le\s+gustan?|no\s+(?:come|consume|tolera|quiere)|rechaza|" + _ALERGICO + r")"
    r"|" + _SUJ_PAC + r"\s+(?:rechaz[oó]|rechaza|no\s+(?:come|consume|tolera)|" + _ALERGICO + r"|"
    r"(?:declar[oó]|report[oó]|tiene)\s+(?:alergia|rechazo)(?:\s+(?:a|al))?)"
    r"|(?:los|sus)\s+(?:rechazos|alimentos\s+rechazados)(?:\s+(?:declarados|del\s+paciente))?\s+son"
    r"|(?:las|sus)\s+alergias(?:\s+(?:declaradas|del\s+paciente))?\s+son(?:\s+(?:a|al))?"
    r")\s+(?P<items>" + _ALIMENTOS + r")",
    _I)
# Ausencia: «la berenjena no aparece en el plan», «no aparece berenjena», «no aparece en el plan» / «ninguno aparece…»
# (remiten a lo declarado), «el plan no contiene pescado ni berenjena».
_PRESENCIA = (r"(?:aparecen?|figuran?|se\s+incluyen?|est[aá]n?\s+presentes?|se\s+usan?|se\s+encuentran?|"
              r"se\s+detectan?|se\s+detect[oó]|se\s+encontr[oó])")
_EN_PLAN = (r"(?:\s+en\s+(?:(?:todo\s+)?(?:el|este)\s+(?:plan|men[uú])|ninguna\s+(?:de\s+las\s+)?comidas?|"
            r"ning[uú]n\s+(?:d[ií]a|plato)|los\s+ingredientes))?")
_AUSENCIA = re.compile(
    r"(?:(?P<suj>" + _ALIMENTOS + r")\s+)?no\s+" + _PRESENCIA + r"(?:\s+(?P<obj>" + _ALIMENTOS + r"))?" + _EN_PLAN
    + r"|ning[uú]n[oa]?(?:\s+de\s+(?:ellos|ellas|los\s+dos|las\s+dos|ambos|ambas))?\s+"
    r"(?:aparece|figura|se\s+incluye|est[aá]\s+presente)" + _EN_PLAN
    + r"|" + _SUJETO_PLAN + r"\s+no\s+(?:contiene|incluye|tiene|lleva|usa|presenta)\s+(?P<obj2>" + _ALIMENTOS + r")"
    + _EN_PLAN,
    _I)
_SUBCLAUSULA = re.compile(
    r"\s*;\s*|\s*[—–]\s*|,\s*y\s+|\s+y\s+(?=(?:no|ni|ning[uú]n[oa]?)\b)|,\s+(?=(?:no|ni|ning[uú]n[oa]?|sin|respetando)\b)",
    _I)
# «…, respetando los rechazos del paciente» describe SU cláusula: lo ausente en ella ES lo declarado («El plan no
# contiene pescado ni berenjena, respetando los rechazos del paciente», corpus de baterías). No absuelve otra cláusula
# («El Día 2 incluye pescado…, y el resto se generó respetando los rechazos» se queda).
_RESPETANDO = re.compile(
    r"respetando\s+(?:los|las|sus)\s+(?:rechazos|restricciones|preferencias)" + _DECLARADO, _I)

# El veredicto LOCAL del 227 («…contiene queso — sin alergia a lácteos declarada, no es violación») absuelve SU oración,
# y solo si lo que le sigue en su cláusula es cerrado: «no hay violaciones en las comidas principales» / «… de la dieta
# vegetariana en la primera semana» no es un veredicto (ya en main: misma clase que el alcance parcial).
_VEREDICTO_LOCAL = re.compile(
    r"no\s+(?:es|constituye|representa|supone|implica)\s+(?:una\s+|ninguna\s+)?violaci[oó]n(?:es)?"
    r"|no\s+hay\s+(?:ninguna\s+)?violaci[oó]n(?:es)?",
    _I)
_RESTO_VEREDICTO = re.compile(
    r"(?:\s+(?:declarad[oa]s?|del\s+paciente))*"
    r"(?:\s+(?:de|por|para|con|en\s+cuanto\s+a)\s+(?:" + _OBJ_CLIN + r"|(?:la\s+)?alergia\s+(?:a|al)\s+" + _ALIMENTO
    + r"|" + _ALIMENTO + r"))?" + _PAREN + r"\s*",
    _I)
_CORTE_VEREDICTO = re.compile(r"[,;:—–.!?]")


def _confirma(c: str) -> bool:
    """Reglas 1-2 sobre una cláusula."""
    return bool(_NIEGA_ALGO_MALO.fullmatch(c) or (_CUMPLE.fullmatch(c) and not _NEGACION.search(c)))


def _norm_alimento(s: str) -> tuple:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    ws = [w for w in re.findall(r"[a-z]+", s) if w not in ("el", "la", "los", "las", "un", "una", "de", "del")]
    return tuple(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws)


def _alimentos(txt) -> set:
    return {n for n in (_norm_alimento(x) for x in re.split(r"\s*,\s*|\s+(?:y|ni|o)\s+", txt or "", flags=_I)) if n}


def _subclausulas(texto: str) -> list:
    return [p.strip().rstrip(".;!?, ").strip() for p in _SUBCLAUSULA.split(texto or "") if p and p.strip(" .;!?,")]


def _neutras(clausulas) -> bool:
    """[revisión 2] Cada cláusula es confirmación (reglas 1-2), declaración de rechazo/alergia del paciente o AUSENCIA; y
    si hay ausencias, lo ausente es exactamente lo declarado (lo que no remite a una declaración puede ser algo bueno
    que falta: «el hierro hemo no aparece en el plan»)."""
    declarados, ausentes, remite, hay_ausencia, respeta = set(), set(), False, False, False
    for c in clausulas:
        if not c or _GIROS.search(c):
            return False
        if _confirma(c):
            continue
        if _RESPETANDO.fullmatch(c):
            respeta = True                    # lo ausente ES lo declarado
            continue
        m = _DECLARACION.fullmatch(c)
        if m:
            declarados |= _alimentos(m.group("items"))
            continue
        m = _AUSENCIA.fullmatch(c)
        if not m:
            return False
        hay_ausencia = True
        if m.group("suj") and m.group("obj"):
            return False
        txt = m.group("suj") or m.group("obj") or m.group("obj2")
        if txt:
            ausentes |= _alimentos(txt)
        else:
            remite = True                     # «no aparece en el plan» / «ninguno aparece…»: lo declarado
    if not hay_ausencia:
        return True
    if respeta:
        declarados |= ausentes
    return bool(declarados) and ausentes <= declarados and (remite or declarados <= ausentes)


def _veredicto_local(oracion: str) -> bool:
    for m in _VEREDICTO_LOCAL.finditer(oracion):
        resto = oracion[m.end():]
        corte = _CORTE_VEREDICTO.search(resto)
        if _RESTO_VEREDICTO.fullmatch(resto[:corte.start()] if corte else resto):
            return True
    return False


def oraciones_sin_hallazgo(texto) -> bool:
    """[P1-PLAN-LOTE-746 · revisión 2] La regla del 227, invertida: la razón no afirma ningún defecto si CADA oración
    trae su propio veredicto («…, no es violación») o CADA cláusula suya es confirmación, declaración de rechazo/alergia
    del paciente o la ausencia de lo declarado. Sin listas de verbos: «Hay pollo en la cena del Día 2… El pescado no
    aparece en el plan» se queda porque «Hay pollo…» no es ninguna de esas formas. Puro; nunca lanza."""
    try:
        if not isinstance(texto, str):
            return False
        resto = []
        for s in re.split(r"(?<=[.!?])\s+", " ".join(texto.split())):
            s = s.strip().rstrip(".!? ").strip()
            if not s or _veredicto_local(s):
                continue
            resto.extend(_subclausulas(s))
        return _neutras(resto)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-746] oraciones_sin_hallazgo no-op: {type(e).__name__}: {e}")
        return False


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
    """Qué regla descarta la razón (`confirmacion` / `dias_sin_riesgo_medico` / `concluye_sin_problema`), o None si se
    queda. Puro; nunca lanza."""
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
        if fin:                                                                # regla 4 (revisiones 1 y 2)
            cuerpo = _subclausulas(cl[-1][:fin.start()])
            previas = [s for c in cl[:-1] for s in _subclausulas(c)]
            if (cuerpo and not all(_DECLARACION.fullmatch(s) for s in cuerpo)
                    and _neutras(previas + cuerpo)):
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


__all__ = ["es_confirmacion_sin_defecto", "motivo", "activo", "oraciones_sin_hallazgo"]
