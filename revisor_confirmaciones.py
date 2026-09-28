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

[P1-PLAN-LOTE-746 · 2026-09-28] Revisión 3 —
  · EL VEREDICTO Y SU COLA (importante, ya en main): «…no es violación de alergia, pero es un ingrediente prohibido
    temporalmente…», «No hay violación de la alergia, pero el Día 3 incluye maní…», «…no es violación; el Día 3 incluye
    hígado encebollado, que el paciente rechazó» salían aprobadas con severidad `critical`: el veredicto local miraba
    su texto hasta el primer signo y absolvía la oración ENTERA. Ahora lo que queda tras su objeto cerrado es otra
    cláusula: con un contraste (pero, aunque, sin embargo, no obstante, excepto, salvo, sino, mientras que…) se queda;
    si no, cada cláusula suya pasa por la misma regla (confirmación, declaración, ausencia de lo declarado u otro
    veredicto local). Solo dos colas cerradas se aceptan tal cual: «solo riesgo de seguridad alimentaria genérico ya
    advertido en el plan» (n.º 20) y «pero se recomienda verificar tolerancia» (n.º 30 del corpus). Igual tras la
    conclusión del 227 («…, por lo que no es una violación, pero…»): `conclusion_cerrada`.
  · ALCANCE: uno afirmativo para cumplir («en todas las comidas») y uno negativo para negar algo malo («en ninguna
    comida»): «se cumple en ningún día» no es un cumplimiento y «no se detectan alérgenos en todas las comidas» es
    ambiguo. «ningún/ninguna» cuentan como negación.
  · NOMBRES CERRADOS: el nombre de la dieta sale del SSOT `constants.canonicalize_diet_type` (vía `diet_type_aliases`),
    no de texto libre entre comillas; «restricciones … declaradas» solo con calificativos de una lista cerrada; el
    paciente del «es seguro para…» solo con un calificativo de otra («renal», «embarazada»…).
  · el veredicto con SU PROPIO sujeto juzga su cláusula, no la de antes: «Hay pollo en la cena del Día 2; el queso no
    es violación» se queda (`_antes_del_veredicto`); solo tras un preámbulo cerrado («— sin alergia a lácteos
    declarada, no es violación») juzga la cláusula anterior, que es el caso del 227.
  · RECHAZOS DE MÁS frente a main (la inversión del 227), cerrados donde se puede sin texto libre: «El plan es seguro
    para el paciente renal / la paciente embarazada» (calificativo cerrado), «X, rechazados por el paciente, no
    aparecen en el plan» y «Los alimentos rechazados (X) no aparecen en el plan» (ORACIÓN ENTERA: lo ausente es lo
    declarado en ella). COSTE ACEPTADO — un reintento de más, ninguno en el fixture ni en el corpus de baterías —:
    «El plan no incluye pescado ni berenjena. El plan es seguro.» y «… ninguno aparece en el plan» sin declaración
    (la ausencia de lo no declarado puede ser algo BUENO que falta: el hierro hemo); «Día 2 | Desayuno: huevo revuelto.
    Sin alergia al huevo declarada, no es violación.» (el veredicto en OTRA oración es la forma del hueco del 227
    entre oraciones); y una dieta fuera del SSOT («Dieta 'baja en sodio' respetada», «DASH», «renal»).
  · SIGUE ABIERTO (ya en main): el veredicto que el LLM da sobre su propia cláusula («Día 2: pollo en la cena, no es
    violación de la dieta vegetariana»; «El Día 3 incluye hígado…, que el paciente rechazó; no hay violación de la
    dieta vegetariana») y la conclusión del 227 sobre las oraciones ANTERIORES (la decisión del 25-sep).

[P1-PLAN-LOTE-746 · 2026-09-28] Revisión 4 —
  · LA CLÁUSULA PROPIA DEL VEREDICTO (bloqueante, ya en main; el punto de la revisión 3 en el otro sentido): lo dicho
    en la revisión 3 («un veredicto con su propio sujeto ya no absuelve la cláusula anterior») solo era cierto tras
    «;», «—» o «, y». Con una coma o dos puntos delante, o un contraste, el veredicto seguía absolviendo la oración
    entera con severidad `critical`: «…plátano maduro, prohibido temporalmente en el perfil de gustos, pero no es
    violación de alergia», «…inhibe la absorción de hierro en la anemia, pero no es violación…», «Día 3: 4200 mg de
    potasio para paciente renal, el plátano no constituye violación». Ahora la cláusula propia va tras la última
    frontera de CUALQUIER tipo: con un contraste o una concesión (si bien, pese a, a pesar de…) se queda; con un sujeto
    propio, todo lo anterior queda pendiente salvo una etiqueta de lugar («Día 1 | Cena:», que ya no se rechaza tras
    «—»). Un pronombre que remite atrás («…, lo cual no es violación») no es un sujeto propio; un sujeto con
    determinante tras el que va un preámbulo («…, el queso, sin alergia declarada, no es violación») sí. COSTE
    ACEPTADO: una lista con artículo delante del preámbulo («queso, el yogur griego, sin alergia…») se lee así.
    SIGUE ABIERTO (ya en main): sin conector ni sujeto el veredicto juzga su cláusula entera aunque lleve comas
    («Día 2: plátano, prohibido temporalmente, no es violación de alergia»; también con «pero sin alergia declarada,»
    delante) — partirla por comas rompería la n.º 20.
  · LO DECLARADO NO PUEDE ESTAR EN LO JUZGADO (menor, ya en main): «Día 2: hígado — no hay violación de la dieta
    vegetariana; el paciente rechazó el hígado» se aprobaba porque una declaración sola es neutra. Ahora se guarda la
    cláusula que cada veredicto absuelve y la declaración no puede nombrar un alimento de ella (en la oración o en otra;
    también tras la conclusión del 227).
  · «(paciente sin ninguna alergia)» / «(el paciente no reporta alergias)»: la negación se busca fuera del paréntesis
    del paciente (rechazo de más desde la revisión 3, cuando «ningún/ninguna» entró en `_NEGACION`).
tooltip-anchor: P1-PLAN-LOTE-746-CONFIRMACIONES
tooltip-anchor: P1-PLAN-LOTE-746-FORMAS-CERRADAS
tooltip-anchor: P1-PLAN-LOTE-746-VEREDICTO-Y-SU-COLA
tooltip-anchor: P1-PLAN-LOTE-746-CLAUSULA-PROPIA
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
_NEGACION = re.compile(r"\b(?:no|nunca|jam[aá]s|ni|ning[uú]n[oa]?)\b", _I)   # [revisión 3] + ningún/ninguna

# Lo que el nombre de un alimento (regla 4) no puede ser: un verbo que afirme contenido o un cambio, ni un día, una
# comida, una cantidad o un cuantificador.
_NO_LIBRE = (r"(?:contien\w*|conten\w*|inclu\w*|aport\w*|combin\w*|tien\w*|tuv\w*|llev\w*|us[aeoó]\w*|sirv\w*|"
             r"retir\w*|mantien\w*|mantuv\w*|pon\w*|pus\w*|mezcl\w*|acompañ\w*|dej\w*|conserv\w*|persist\w*|sigu\w*|"
             r"contin[uú]\w*|permanec\w*|aparec\w*|figur\w*|trae\w*|traj\w*|lleg\w*|hay|hubo|existe\w*|exist[ií]\w*|"
             r"qued\w*|fue|fueron|es|son|era|eran|est[aá](?:n|s|ba\w*|r\w*)?|ha|han|hab[ií]a\w*|se|que|"
             r"d[ií]as?|desayunos?|almuerzos?|cenas?|meriendas?|comidas?|platos?|plan|men[uú]|semanas?|jornadas?|"
             r"mayor[ií]a|mitad|parte|algun\w*|poc[oa]s|ciert[oa]s|vari[oa]s|primer\w*|segund\w*|[uú]ltim\w*|"
             r"[uú]nicamente|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|mg|kcal|\d\w*)")


def _vocabulario_de_dietas() -> str:
    """[revisión 3] Los nombres de dieta que reconoce el SSOT (`constants.canonicalize_diet_type`, vía
    `diet_type_aliases`: vegano/a, vegetariano/a, pescetariano/a… y `balanced`), los largos primero. Si el SSOT no
    carga, ninguno: ninguna «dieta X» es entonces una confirmación (falla hacia rechazar)."""
    try:
        from constants import diet_type_aliases
        nombres = {n for c in ("vegan", "vegetarian", "pescatarian", "balanced") for n in diet_type_aliases(c)}
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-746] sin el SSOT de dietas ninguna «dieta X» se descarta: {type(e).__name__}")
        nombres = set()
    if not nombres:
        return r"(?!)"
    return r"(?:" + "|".join(re.escape(n) for n in sorted(nombres, key=len, reverse=True)) + r")"


# «dieta 'balanced'», «dieta vegetariana». [revisión 3] VOCABULARIO CERRADO del SSOT, con o sin comillas: ni texto libre
# entre comillas («'vegetariana hasta el miércoles'», «'renal sin control de potasio'») ni una palabra cualquiera
# («la dieta pollo»). Una dieta fuera del SSOT («'baja en sodio'», «DASH») se queda: un reintento de más, a sabiendas.
_DIETAS = _vocabulario_de_dietas()
_NOMBRE = r"(?:['\"«“‘]" + _DIETAS + r"['\"»”’]|" + _DIETAS + r"\b)"

# ── formas CERRADAS [revisión 2] ──
_DECLARADO = (r"(?:\s+(?:declarad|registrad|conocid|reportad)[oa]s?(?:\s+(?:por|de)\s+(?:el\s+|la\s+)?paciente)?)?"
              r"(?:\s+del\s+paciente)?")
_OBJ_CLIN = (r"(?:(?:las|los|sus?|la|el)\s+)?(?:condiciones\s+m[eé]dicas|condici[oó]n\s+m[eé]dica|"
             r"restricciones(?:\s+diet[eé]ticas)?|alergias?(?:\s+alimentarias)?|medicamentos|medicaci[oó]n|"
             r"preferencias|rechazos|intolerancias?|dieta\s+" + _NOMBRE + r")" + _DECLARADO)
# [revisión 3] Dos alcances: el AFIRMATIVO para cumplir («se respeta en todas las comidas»; «se cumple en ningún día» es
# que no se cumple) y el NEGATIVO para negar algo malo («no se detectan alérgenos en ninguna comida»; «… en todas las
# comidas» es ambiguo: puede que en alguna sí). «(todo) el plan» vale en los dos.
_ALCANCE_TODO = (r"(?:\s+en\s+(?:(?:todo\s+)?(?:el|este)\s+(?:plan|men[uú])|todas\s+las\s+comidas|"
                 r"todos\s+los\s+(?:d[ií]as|platos)))?")
_ALCANCE_NINGUNO = (r"(?:\s+en\s+(?:(?:todo\s+)?(?:el|este)\s+(?:plan|men[uú])|ninguna\s+(?:de\s+las\s+)?comidas?|"
                    r"ning[uú]n\s+(?:d[ií]a|plato)|ninguno\s+de\s+los\s+(?:d[ií]as|platos)))?")
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
# [revisión 3] Sin palabras libres: los calificativos de la restricción son una lista CERRADA — los nombres de dieta del
# SSOT («vegetarianas/veganas»), «sin gluten/lactosa», «de sodio/potasio/…» y «dietéticas/alimentarias/…». «No hay
# restricciones de sodio respetadas declaradas» se queda.
_CALIF_RESTRICCION = (r"(?:" + _DIETAS + r"s?\b|sin\s+(?:gluten|lactosa|l[aá]cteos|az[uú]car)|"
                      r"de\s+(?:gluten|lactosa|l[aá]cteos|sodio|sal|potasio|f[oó]sforo|az[uú]car(?:es)?|purinas|"
                      r"grasas?|carbohidratos|alergias?)|diet[eé]ticas|alimentarias|m[eé]dicas|cl[ií]nicas|especiales|"
                      r"adicionales|particulares)")
_RESTRICCION = (r"restricci[oó]n(?:es)?(?:\s+" + _CALIF_RESTRICCION + r"(?:(?:\s*/\s*|\s*,\s*|\s+(?:y|ni|o)\s+)"
                + _CALIF_RESTRICCION + r"){0,5})?\s+declarad[oa]s?(?:\s+(?:por|de)\s+(?:el\s+|la\s+)?paciente)?")
_NIEGA_ALGO_MALO = re.compile(
    r"(?:" + _NEG + r"\s+" + _LISTA_MALO + r"(?:\s+(?:aparecen?|figuran?|se\s+incluyen?|est[aá]n?\s+presentes?))?"
    + _ALCANCE_NINGUNO + r"|" + _NEG_EXISTE + r"\s+" + _RESTRICCION + r")" + _PAREN, _I)

# 2. cumplimiento
_SUJETO_PLAN = r"(?:(?:el|este)\s+plan|el\s+men[uú])"
_OBJ_CUMPLE = (r"(?:(?:todas\s+)?(?:las|sus)\s+(?:restricciones(?:\s+diet[eé]ticas)?|preferencias|alergias|"
               r"condiciones\s+m[eé]dicas|intolerancias)|(?:todos\s+)?(?:los|sus)\s+(?:rechazos|medicamentos)|"
               r"(?:la|su)\s+dieta\s+" + _NOMBRE + r")" + _DECLARADO)
# [revisión 3] «el paciente renal», «la paciente embarazada»: el calificativo del paciente, de una lista CERRADA (nunca
# «sin insuficiencia renal», «sano», «con 4200 mg de potasio»).
_PAC_CALIF = (r"(?:renal(?:es)?|diab[eé]tic[oa]|hipertens[oa]|embarazada|gestante|lactante|cel[ií]ac[oa]|an[eé]mic[oa]|"
              r"card[ií]ac[oa]|" + _DIETAS + r")\b")
_OBJ_APTO = (r"(?:(?:el|la)\s+paciente(?:\s+" + _PAC_CALIF + r")?|(?:las|sus)\s+(?:restricciones|condiciones\s+m[eé]dicas|"
             r"alergias)" + _DECLARADO + r"|(?:la|su)\s+dieta(?:\s+" + _NOMBRE + r")?)")
_CUMPLE = re.compile(
    r"(?:(?:la\s+)?dieta\s+" + _NOMBRE + r"\s+(?:(?:es|fue|est[aá]|queda)\s+)?"
    r"(?:respetad[oa]|cumplid[oa]|se\s+respeta|se\s+cumple)" + _ALCANCE_TODO
    + r"|" + _SUJETO_PLAN + r"\s+(?:respeta|cumple(?:\s+con)?)\s+" + _OBJ_CUMPLE
    + r"(?:\s+y\s+" + _OBJ_CUMPLE + r")?" + _ALCANCE_TODO
    + r"|" + _SUJETO_PLAN + r"\s+es\s+(?:segur[oa]|adecuad[oa]|apt[oa]|compatible)"
    r"(?:\s+(?:para|con)\s+" + _OBJ_APTO + r")?" + _ALCANCE_TODO + r")" + _PAREN,
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
                  r"adem[aá]s|s[oó]lo|solamente|gust\w*|ningun\w*|ning[uú]n|"
                  # [revisión 3] ni lo que no es un alimento: «no hay violaciones de la dieta pollo» no es el
                  # veredicto sobre el alimento «dieta pollo»
                  r"dietas?|restricci\w*|condici\w*|alergi\w*|intoleranci\w*|medicament\w*|medicaci\w*|violaci\w*)")
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
# [revisión 3] La declaración y la ausencia en UNA oración: «El pescado y la berenjena, rechazados por el paciente, no
# aparecen en el plan», «Los alimentos rechazados (pescado, berenjena) no aparecen en el plan». Solo como oración
# ENTERA: el apósito solo («Hígado encebollado, rechazado por el paciente.») señala un plato, no lo declara ausente.
_RECHAZADOS_AUSENTES = re.compile(
    r"(?:" + _ALIMENTOS + r"\s*,\s*(?:rechazad|vetad|excluid)[oa]s?\s+por\s+(?:el|la)\s+(?:paciente|usuari[oa])\s*,"
    r"|(?:los|las|sus)\s+(?:alimentos|ingredientes)\s+rechazad[oa]s(?:\s+(?:por\s+(?:el|la)\s+paciente|"
    r"del\s+paciente|declarad[oa]s))?\s*\(\s*" + _ALIMENTOS + r"\s*\))"
    r"\s+no\s+" + _PRESENCIA + _EN_PLAN,
    _I)

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
# [revisión 3] Lo que sigue al objeto cerrado del veredicto es OTRA cláusula. Con un contraste se queda: «no es
# violación de alergia, pero es un ingrediente prohibido temporalmente…».
_CONTRASTE = re.compile(
    r"\b(?:pero|aunque|sin\s+embargo|no\s+obstante|excepto|salvo|a\s+excepci[oó]n|sino|mientras\s+que|en\s+cambio|"
    r"por\s+el\s+contrario|aun\s+as[ií]|con\s+todo)\b", _I)
# Las únicas colas que se aceptan TAL CUAL (cerradas, sin texto libre): la n.º 20 de producción («…, solo riesgo de
# seguridad alimentaria genérico ya advertido en el plan») y la n.º 30 del corpus de baterías («…, por lo que no es
# una violación, pero se recomienda verificar tolerancia»). Ninguna afirma un defecto.
_COLA_SIN_HALLAZGO = re.compile(
    r"(?:s[oó]lo|solamente|[uú]nicamente)\s+(?:un\s+)?riesgo\s+(?:de\s+)?(?:seguridad|inocuidad)\s+alimentaria"
    r"(?:\s+gen[eé]ric[oa])?\s+ya\s+(?:advertid|señalad|indicad|mencionad)[oa]\s+en\s+el\s+(?:plan|men[uú])"
    r"|pero\s+se\s+recomienda\s+(?:verificar|confirmar|vigilar|evaluar)\s+(?:la\s+)?tolerancia",
    _I)
# La cláusula del veredicto termina en la última frontera anterior; si es solo este preámbulo, el veredicto juzga la
# cláusula de antes («X — sin alergia a lácteos declarada, no es violación»). Un preámbulo más laxo solo absolvería lo
# mismo que antes de la revisión 3: nunca más.
_FRONTERA_VEREDICTO = re.compile(r"\s*[;—–]\s*|,\s*y\s+", _I)
# [revisión 4] La cláusula PROPIA del veredicto va tras la última frontera de CUALQUIER tipo — también «,», «:» y
# « y el/la…» (un sujeto nuevo) —: «…prohibido temporalmente en el perfil, pero no es violación de alergia», «…para
# paciente renal, el plátano no constituye violación» absolvían la oración entera (ya en main).
_FRONTERA_PROPIA = re.compile(
    r"\s*[,;:—–]\s*|\s+(?:y|e)\s+(?=(?:el|la|los|las|un|una|este|esta|estos|estas|ese|esa|su|sus)\b)", _I)
# Concesiones que `_CONTRASTE` no nombra: delante del veredicto dicen lo mismo que «aunque».
_CONCESION = re.compile(
    r"\b(?:si\s+bien|pese\s+a|a\s+pesar\s+de|aun\s+cuando|aun\s+as[ií]|aunque|empero|mas|con\s+la\s+salvedad)\b", _I)
# Un pronombre o un conector de consecuencia NO es un sujeto propio: remite a la cláusula anterior, que es la que el
# veredicto juzga («El plan incluye queso en la cena, lo cual no es violación»).
_ANAFORA = re.compile(
    r"(?:lo|el|la|los|las)\s+cual(?:es)?|lo\s+que|que|esto|eso|ello|lo\s+anterior|"
    r"por\s+(?:ende|tanto|consiguiente)|en\s+consecuencia|de\s+ah[ií]\s+que", _I)
# Una ETIQUETA de lugar («Día 1 | Cena», «Día 1 | Merienda y Día 2 | Merienda», «Desayuno del Día 3») no afirma nada:
# delante de un sujeto propio («Día 1 | Cena: el queso no es violación») no queda pendiente.
_COMIDA = r"(?:desayuno|almuerzo|comida|cena|merienda|snack|colaci[oó]n|media\s+mañana)s?"
# Sin ambigüedad (la repetición anidada no puede leer «, Día 2» de dos maneras: sin backtracking exponencial): tras la
# coma o la «y» interna va un NÚMERO; tras la externa, una palabra («Día», «Cena»).
_UNA_ETIQUETA = (r"(?:d[ií]as?\s+\d+(?:\s*(?:,|y|a|al|-|–)\s*\d+)*(?:\s*[|\-–]?\s*" + _COMIDA + r")?"
                 r"|" + _COMIDA + r"(?:\s+del?\s+d[ií]a\s+\d+)?|nota|observaci[oó]n|aclaraci[oó]n)")
_ETIQUETA = re.compile(_UNA_ETIQUETA + r"(?:\s*(?:,|y|e)\s+" + _UNA_ETIQUETA + r")*", _I)
# Sin sujeto propio tras la última frontera, pero con un PREÁMBULO explícito delante del veredicto («…, el queso, sin
# alergia a lácteos declarada, no es violación»): el veredicto juzga la cláusula anterior al preámbulo, y si esa empieza
# con su propio determinante («el queso»), lo que va antes queda pendiente. Solo con «,;:—–» (una lista «queso y la
# crema» no se parte) y nunca tras un contraste («…, pero los ingredientes son vegetarianos, por lo que…»: la n.º 20).
_FRONTERA_CLAUSULA = re.compile(r"\s*[,;:—–]\s*")
_SUJETO_NUEVO = re.compile(r"(?:el|la|los|las|un|una|este|esta|estos|estas|ese|esa|su|sus)\s", _I)
_PREAMBULO_VEREDICTO = re.compile(
    r"(?:sin\s+(?:ninguna\s+)?alergias?(?:\s+(?:a|al)\s+[^\s,;]+(?:\s+[^\s,;]+)?)?(?:\s+declarad[oa]s?)?"
    r"|(?:el|la)\s+paciente\s+no\s+(?:declar[oó]|report[oó]|tiene|indic[oó])\s+(?:ninguna\s+)?alergias?"
    r"(?:\s+(?:a|al)\s+[^\s,;]+(?:\s+[^\s,;]+)?)?(?:\s*,?\s*(?:por\s+lo\s+que|as[ií]\s+que|de\s+modo\s+que|entonces))?"
    r"|por\s+lo\s+(?:tanto|que)|as[ií]\s+que|entonces|de\s+modo\s+que)?\s*,?",
    _I)


def _confirma(c: str) -> bool:
    """Reglas 1-2 sobre una cláusula. [revisión 4] La negación se busca SIN el paréntesis del paciente (`_PAREN`, el
    único que `_CUMPLE` admite): «Dieta 'balanced' respetada (paciente sin ninguna alergia)» es un cumplimiento."""
    return bool(_NIEGA_ALGO_MALO.fullmatch(c)
                or (_CUMPLE.fullmatch(c) and not _NEGACION.search(re.sub(r"\s*\([^()]*\)\s*$", "", c))))


def _norm_alimento(s: str) -> tuple:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    ws = [w for w in re.findall(r"[a-z]+", s) if w not in ("el", "la", "los", "las", "un", "una", "de", "del")]
    return tuple(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws)


def _alimentos(txt) -> set:
    return {n for n in (_norm_alimento(x) for x in re.split(r"\s*,\s*|\s+(?:y|ni|o)\s+", txt or "", flags=_I)) if n}


def _subclausulas(texto: str) -> list:
    return [p.strip().rstrip(".;!?, ").strip() for p in _SUBCLAUSULA.split(texto or "") if p and p.strip(" .;!?,")]


def _palabras(txt) -> set:
    """Las palabras de un texto normalizadas como las de un alimento (sin tildes, artículos ni plural)."""
    return {w for w in _norm_alimento(txt) if len(w) > 2}


def _neutras(clausulas, juzgados=()) -> bool:
    """[revisión 2] Cada cláusula es confirmación (reglas 1-2), declaración de rechazo/alergia del paciente o AUSENCIA; y
    si hay ausencias, lo ausente es exactamente lo declarado (lo que no remite a una declaración puede ser algo bueno
    que falta: «el hierro hemo no aparece en el plan»).
    [revisión 4] `juzgados`: las cláusulas que un veredicto local absolvió («Día 2: hígado — no hay violación…»). Lo
    que el paciente declaró rechazar o a lo que es alérgico no puede aparecer en ellas: «…; el paciente rechazó el
    hígado» dice que el plan TRAE lo rechazado."""
    declarados = _neutras_y_declarados(clausulas)
    if declarados is None:
        return False
    en_juzgados = set().union(*(_palabras(j) for j in juzgados)) if juzgados else set()
    return not ({w for d in declarados for w in d if len(w) > 2} & en_juzgados)


def _neutras_y_declarados(clausulas):
    """Lo declarado (conjunto de alimentos normalizados) si las cláusulas son neutras; None si no."""
    declarados, ausentes, remite, hay_ausencia, respeta = set(), set(), False, False, False
    for c in clausulas:
        if not c or _GIROS.search(c):
            return None
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
            return None
        hay_ausencia = True
        if m.group("suj") and m.group("obj"):
            return None
        txt = m.group("suj") or m.group("obj") or m.group("obj2")
        if txt:
            ausentes |= _alimentos(txt)
        else:
            remite = True                     # «no aparece en el plan» / «ninguno aparece…»: lo declarado
    if not hay_ausencia:
        return declarados
    if respeta:
        declarados |= ausentes
    ok = bool(declarados) and ausentes <= declarados and (remite or declarados <= ausentes)
    return declarados if ok else None


def _cola(resto: str):
    """Lo que sigue al OBJETO CERRADO de un veredicto («… de alergia (paciente sin alergias)»), sin su signo: la cola
    de la oración ("" si no queda nada), o None si el objeto no es cerrado — entonces no es un veredicto («no hay
    violaciones en las comidas principales»)."""
    corte = _CORTE_VEREDICTO.search(resto)
    if not _RESTO_VEREDICTO.fullmatch(resto[:corte.start()] if corte else resto):
        return None
    return re.sub(r"^(?:y|e)\s+", "", resto[corte.start():].strip(" ,;:—–.!?"), flags=_I) if corte else ""


def _pendientes_de_cola(cola: str, prof: int, juzgados=None):
    if not cola or _COLA_SIN_HALLAZGO.fullmatch(cola):
        return []
    if _CONTRASTE.search(cola) or prof >= 12:          # tope de veredictos encadenados: pasado, se queda
        return None
    return _pendientes(cola, prof + 1, juzgados)


def _sin_etiquetas(clausulas) -> list:
    return [c for c in clausulas if not _ETIQUETA.fullmatch(c.strip(" |:"))]


def _antes_del_veredicto(prefijo: str, juzgados=None):
    """[revisión 3] Lo que PRECEDE al veredicto y él no juzga. Juzga SU cláusula (tras la última frontera «;», «—» o
    «, y»); si su cláusula es solo un preámbulo cerrado (nada, «sin alergia a lácteos declarada,», «el paciente no
    declaró alergia al maní, por lo que»), juzga la cláusula ANTERIOR — el caso del 227: «…contiene queso mozzarella —
    sin alergia a lácteos declarada, no es violación». Las demás quedan pendientes: en «Hay pollo en la cena del Día 2;
    el queso no es violación» el veredicto es del queso, no del pollo.
    [revisión 4 · CLÁUSULA PROPIA] La revisión 3 solo veía «;», «—» y «, y»: con una coma o dos puntos delante («…para
    paciente renal, el plátano no constituye violación») o un contraste («…prohibido temporalmente en el perfil, pero
    no es violación de alergia») el veredicto absolvía la oración entera (ya en main). Ahora su cláusula propia va tras
    la última frontera de CUALQUIER tipo (`_FRONTERA_PROPIA`): con un contraste o una concesión → None (se queda); con
    un sujeto propio (ni preámbulo ni pronombre que remita atrás) → juzga solo ese sujeto y TODO lo anterior queda
    pendiente, salvo una etiqueta de lugar («Día 1 | Cena:»). Con un preámbulo o un pronombre entre el sujeto y el
    veredicto («…, el queso, sin alergia a lácteos declarada, no es violación»), lo mismo si la cláusula anterior al
    preámbulo abre con su propio determinante (`_SUJETO_NUEVO`). Si no, como en la revisión 3. Sigue abierto (ya en
    main): sin conector ni sujeto, el veredicto juzga su cláusula entera aunque lleve comas o un contraste antes del
    preámbulo («Día 2: plátano, prohibido temporalmente, [pero sin alergia declarada,] no es violación de alergia»):
    partirla por comas rompería la n.º 20 («…crudo, pero los ingredientes son vegetarianos…; no es violación») y el
    caso del 227. `juzgados` recibe la cláusula que el veredicto absuelve (ver `_neutras`)."""
    fronteras = list(_FRONTERA_PROPIA.finditer(prefijo))
    propia = (prefijo[fronteras[-1].end():] if fronteras else prefijo).strip()
    if _CONTRASTE.search(propia) or _CONCESION.search(propia):
        return None
    if (fronteras and propia and not _PREAMBULO_VEREDICTO.fullmatch(propia)
            and not _ANAFORA.fullmatch(propia)):
        if juzgados is not None:
            juzgados.append(propia)
        return _sin_etiquetas(_subclausulas(prefijo[:fronteras[-1].start()]))
    cortes = list(_FRONTERA_CLAUSULA.finditer(prefijo))
    ini, fin = [0] + [c.end() for c in cortes], [c.start() for c in cortes] + [len(prefijo)]
    k, explicito = len(ini) - 1, False
    while k > 0:                                       # se salta el preámbulo explícito que precede al veredicto
        t = prefijo[ini[k]:fin[k]].strip()
        if t and not (_PREAMBULO_VEREDICTO.fullmatch(t) or _ANAFORA.fullmatch(t)):
            break
        explicito, k = explicito or bool(t), k - 1
    if explicito and k > 0 and _SUJETO_NUEVO.match(prefijo[ini[k]:fin[k]].strip()):
        if juzgados is not None:
            juzgados.append(prefijo[ini[k]:fin[k]])
        return _sin_etiquetas(_subclausulas(prefijo[:fin[k - 1]]))
    partes = _FRONTERA_VEREDICTO.split(prefijo)
    previas, juzgada = partes[:-1], partes[-1]
    if previas and _PREAMBULO_VEREDICTO.fullmatch(partes[-1].strip()):
        previas, juzgada = previas[:-1], previas[-1]
    if juzgados is not None:
        juzgados.append(juzgada)
    return _sin_etiquetas([c for p in previas for c in _subclausulas(p)])


def _pendientes(oracion: str, prof: int = 0, juzgados=None):
    """[revisión 3] Las cláusulas de la oración que su veredicto local NO absuelve ([] = ninguna), o None si lo que
    sigue al veredicto trae un contraste. El veredicto absuelve la cláusula que juzga (`_antes_del_veredicto`; sigue
    abierto: el LLM juzga su propia cláusula, «Día 2: pollo en la cena, no es violación de la dieta vegetariana»); lo
    que le SIGUE es otra cláusula y pasa por la misma regla — puede traer su propio veredicto.
    [revisión 4] También None si lo que lo PRECEDE en su cláusula propia es un contraste («…, pero no es violación»)."""
    for m in _VEREDICTO_LOCAL.finditer(oracion):
        cola = _cola(oracion[m.end():])
        if cola is not None:
            antes = _antes_del_veredicto(oracion[:m.start()], juzgados)
            despues = None if antes is None else _pendientes_de_cola(cola, prof, juzgados)
            return None if despues is None else antes + despues
    return _subclausulas(oracion)


def conclusion_cerrada(resto, juzgado="") -> bool:
    """[P1-PLAN-LOTE-746 · revisión 3] Para la conclusión del 227 («…, por lo que no es una violación» / «… este punto
    se cumple»): lo que le sigue en su oración es su objeto cerrado y, detrás, nada o una cola sin contraste que sea
    confirmación, declaración o ausencia de lo declarado. «…, por lo que no es una violación, pero el Día 3 aporta
    4200 mg de potasio» se queda. [revisión 4] `juzgado`: lo que la conclusión absuelve (lo que la precede en su
    oración); una declaración en la cola no puede nombrar lo que aparece ahí. Puro; nunca lanza."""
    try:
        cola = _cola(str(resto or ""))
        if cola is None:
            return False
        juzgados = [str(juzgado or "")]
        p = _pendientes_de_cola(cola, 0, juzgados)
        return p is not None and _neutras(p, juzgados)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-746] conclusion_cerrada no-op: {type(e).__name__}: {e}")
        return False


def oraciones_sin_hallazgo(texto) -> bool:
    """[P1-PLAN-LOTE-746 · revisión 2] La regla del 227, invertida: la razón no afirma ningún defecto si CADA oración
    trae su propio veredicto («…, no es violación») o CADA cláusula suya es confirmación, declaración de rechazo/alergia
    del paciente o la ausencia de lo declarado. Sin listas de verbos: «Hay pollo en la cena del Día 2… El pescado no
    aparece en el plan» se queda porque «Hay pollo…» no es ninguna de esas formas. [revisión 3] Lo que sigue al
    veredicto en su oración no queda absuelto: ver `_pendientes`. Puro; nunca lanza."""
    try:
        if not isinstance(texto, str):
            return False
        resto, juzgados = [], []
        for s in re.split(r"(?<=[.!?])\s+", " ".join(texto.split())):
            s = s.strip().rstrip(".!? ").strip()
            if not s or _RECHAZADOS_AUSENTES.fullmatch(s):
                continue
            p = _pendientes(s, 0, juzgados)
            if p is None:
                return False
            resto.extend(p)
        return _neutras(resto, juzgados)          # [revisión 4] lo declarado no puede estar en lo juzgado
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


__all__ = ["es_confirmacion_sin_defecto", "motivo", "activo", "oraciones_sin_hallazgo", "conclusion_cerrada"]
