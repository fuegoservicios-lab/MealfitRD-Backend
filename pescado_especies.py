# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Las especies de pescado que el mapa de proteínas principales no veía.

`graph_orchestrator._MAIN_PROTEIN_ALIASES['pescado']` alimenta el contador de monotonía cross-día
(`cross_day_proteins`), el gate same-day del revisor, `reeleccion_dia` y el armador determinista. Su lista curada
es dominicana (tilapia, mero, chillo…) y no conocía trucha (CO), boquerones (ES), sardina, mojarra, bagre,
huachinango, lenguado ni caballa: en G24 CO la trucha de D1 y D3 no existía para ningún contador, y las sardinas
del almuerzo más la trucha de la cena del mismo día pasaban el gate. Y contaba «dorado», que en el plato es un
adjetivo: G24 CO D1 sumaba pescado por «Plátano Maduro Dorado» y un plan vivo de RD por «queso blanco fresco
dorado» — la lección de P1-DORADO-NO-ES-PEZ (`constants.PROTEIN_SYNONYMS`), que este mapa nunca heredó.

No es otra tabla de peces: las especies salen del vocabulario de pescado del escáner (`_ALLERGEN_SYNONYMS
['pescado']`, que ya incluye `vocabulario_mar.PESCADOS_EXTRA` y las altas de catálogo beta). De ahí se quita lo
que el escáner lista por SEGURIDAD y no es un pez que se sirva de principal (condimentos, fondos, adornos, una
imitación), los homónimos, y todo alias que choque con la curación de otra etiqueta (el mismo guard de
P2-PROTEIN-ALIASES-SSOT: así el atún conserva su etiqueta). Las frases inequívocas («filete de dorado/a») se
quedan. Knob `MEALFIT_BETA_FISH_SPECIES_COUNT` (False ⇒ el mapa exacto de antes). Puro; nunca lanza.
tooltip-anchor: P1-PLAN-LOTE-857-PESCADO-ESPECIES

Ronda 4 (revisión r3). (a) Contar sardinas y truchas despertó al autofix de proteína repetida, que DETECTA y
REESCRIBE con este mismo mapa: tilapia + «Ensalada de sardinas en lata» salía «Ensalada de Pechuga de pollo en lata» /
«1 lata de pechuga de pollo en lata» (el fallo que P2-PROTEIN-LADDER-GAPS cerró para el atún). (b) La escalera no
miraba la dieta: con dieta pescetariana, tilapia + mero ya salía «Pechuga de pollo…» en la base;
`destino_apto_para_la_dieta` veta el destino con la guarda de dieta del revisor. (c) Las anchoas cuentan (datos abajo).

Ronda 5 (revisión r4). En la comida ligera, el pescetariano vuelve a recibir queso como en la base.

Ronda 6 (revisión r5, cambio de estrategia). Las rondas 4-5 intentaban que el autofix REESCRIBIERA bien la conserva o el
crudo de las especies nuevas, y cada una dejaba otro caso límite con pollo crudo. Ahora las especies nuevas sólo sirven
para DETECTAR; el autofix no toca su comida ni ninguna con un pez listo, crudo o frío (`motivo_para_no_reescribir`, abajo).

Ronda 7 (revisión r6). El día cuya repetición de pescado incluye una especie nueva se salta ENTERO (`especie_nueva_en`), y
el filtro de crudo/frío por patrones cede a la regla positiva de V7f: el pez se reescribe sólo si una cláusula que lo nombra
lo cuece (detalle abajo). Las reescrituras del lote son un subconjunto de las de la base, medido en un replay congelado.

Ronda 8 (revisión r7). Cinco vetos más sobre la regla positiva, cada uno con su sonda de `probe7.py`: «cocido» en la línea
sin la firma del cerrador, «crudo» junto al pez, el plato crudo dicho en un paso, el pez ya listo en un paso (ahumado,
curado, sobrante, ya horneado…) y la cocción de otro alimento tras «mientras». No es un «nunca»: quedan los residuos que
lista el bloque de abajo.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from knobs import _env_bool

logger = logging.getLogger(__name__)

BETA_FISH_SPECIES_COUNT = _env_bool("MEALFIT_BETA_FISH_SPECIES_COUNT", True)

# Del escáner, pero no son un pez de plato: la alergia necesita verlos, la variedad no.
# [ronda 4] Las anchoas SÍ son pez de plato: la única comida del corpus que las lleva (ES, «Coca de vegetales con
# anchoas») pone 40 g, la mitad de la proteína del plato, y el catálogo las tiene en «Proteínas». Como adorno (César,
# salsa inglesa) no aparecen en ningún plan de producción ni de G24; y anchoas + boquerones el mismo día son el mismo pez.
NO_ES_PEZ_DE_PLATO = frozenset((
    "salsa de pescado", "salsa inglesa", "worcestershire",   # condimentos (llevan anchoa)
    "cesar",                                                 # el aderezo César
    "fumet",                                                 # el fondo de pescado
    "caviar", "hueva",                                       # huevas
    "surimi",                                                # la imitación de cangrejo
))
# Homónimos con la frontera de palabra inicial del resolvedor (`culinary_context._name_has_token`).
HOMONIMOS = frozenset((
    "dorado",   # «plátano maduro dorado», «queso dorado» — P1-DORADO-NO-ES-PEZ
    "carpa",    # «carpaccio»
))
# La frase inequívoca que sí es el pez (la del dorado ya viene de `constants.PROTEIN_SYNONYMS`).
FRASES_INEQUIVOCAS = ("filete de dorada",)


def _norm(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower().strip()).encode("ascii", "ignore").decode("ascii")


def extender_pescado(alias_por_etiqueta: dict, vocabulario_pescado) -> tuple:
    """Muta `alias_por_etiqueta['pescado']` en su sitio: + especies del vocabulario del escáner, − homónimos.
    Devuelve `(añadidos, quitados)`. Con el knob apagado no toca nada y devuelve `([], [])`."""
    if not BETA_FISH_SPECIES_COUNT:
        return [], []
    try:
        pez = alias_por_etiqueta.get("pescado")
        if not isinstance(pez, list):
            return [], []
        quitados = [a for a in pez if _norm(a) in HOMONIMOS]
        for a in quitados:
            pez.remove(a)
        otras = {_norm(a) for lbl, als in alias_por_etiqueta.items() if lbl != "pescado" for a in (als or ())}
        tengo = {_norm(a) for a in pez}
        anadidos = []
        for cand in (*(vocabulario_pescado or ()), *FRASES_INEQUIVOCAS):
            c = _norm(cand)
            if (len(c) < 3 or c in tengo or c in NO_ES_PEZ_DE_PLATO or c in HOMONIMOS
                    or any(c.startswith(t) for t in tengo if " " not in t)          # «boquerones» ya es «boqueron»
                    or any(c in o or o in c for o in otras)):                       # guard de P2-PROTEIN-ALIASES-SSOT
                continue
            pez.append(c)
            tengo.add(c)
            anadidos.append(c)
        logger.info(f"🐟 [P1-PLAN-LOTE-857] pescado: +{len(anadidos)} especies del escáner, "
                    f"−{len(quitados)} homónimos ({', '.join(quitados) or '—'})")
        return anadidos, quitados
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] no se pudo extender el pescado ({type(e).__name__}: {e}) — el mapa "
                       f"sigue como estaba")
        return [], []


# ── rondas 6-8 · detectar sí; reescribir sólo el pez de antes, y sólo si la receta lo cuece ───────────────────────────────
# Cinco rondas intentando que `_protein_repeat_autofix` REESCRIBIERA bien las especies nuevas dejaban pollo crudo en casos
# límite, y la base ya lo hacía con el atún («Ensalada de atún… Sirve frío» salía «Escurre pechuga de pollo y mézclala…
# Sirve frío»). La ronda 6 filtraba lo crudo o frío por patrones y la revisión r6 encontró los que se le escapaban
# («Marina la cebolla, el ají y el mero», «se marina», «sírvelo helado», «camarones ya cocidos»). Ronda 7:
#   (1) Las especies nuevas cuentan para DETECTAR (contador cross-día, gate same-day, `variety_report`), pero un día cuya
#       repetición de pescado incluye una especie nueva se salta ENTERO en el autofix (`especie_nueva_en`, motivo
#       `especie_nueva`): decide el gate, que regenera el día. Con la especie nueva DELANTE, ella se quedaba de guardiana
#       y el autofix reescribía el otro pez, uno de la base, en una repetición que la base no veía (d8b10b05 D2,
#       «Espaguetis con sardinas»: la tilapia pasaba a pollo). Saltando el día, las reescrituras del lote son un
#       subconjunto de las de la base (`tests/test_p1_plan_lote_857_replay.py`). El reescritor usa los alias de antes.
#       El subconjunto se mide sobre el CORPUS, no es universal. La guardiana (la comida que se queda) es la de la base:
#       la primera con la proteína en el nombre, o la primera aparición. La base contaba «dorado» como pez, así que con
#       «Plátano maduro dorado» en el desayuno, tilapia y mero, la guardiana era el plátano y reescribía los dos peces
#       (tilapia a pollo, mero a pavo); la rama deja la tilapia de guardiana y pasa el mero a pollo, no a pavo. Es mejor
#       (el día conserva un pescado), pero es una reescritura distinta de la base; el corpus no tiene ningún día así.
#   (2) Para cualquier etiqueta del mar (la que la guarda de dieta veta al vegetariano y no al pescetariano: pescado, atún,
#       camarones…), la REGLA POSITIVA de V7f: el autofix reescribe el pez sólo si una cláusula que lo nombra (o su
#       enclítico en la cláusula siguiente: «…; hornéalo 20 minutos») lo cuece (`culinary_coherence._v7f_evidencia`: verbo
#       de cocción que no sea participio, o fuego y tiempo). El blanqueo antes de marinar no cuenta; una nota de seguridad
#       tampoco, ni lo que va después de «mientras» (ronda 8). Además no reescribe la conserva (`pez_en_conserva`; en un
#       paso también «ahumado»/«curado» junto al pez), el precocido («ya cocido», «precocido», «cocidos» junto al pez en un
#       paso, «sobrante», «ya horneado/asado/hervido/frito/cocinado», y «cocido» en la línea sin la firma del cerrador:
#       `pez_precocido`) ni el crudo (el plato crudo en el nombre o en un paso, o «crudo»/«en crudo» hasta dos palabras
#       después del pez: `pez_crudo`).
#       Excepciones: sin pasos no hay cláusula que deje el pez crudo (se reescribe como en la base), y en la comida ligera
#       o dulce el único destino es el queso, que no se cuece (el arreglo de la ronda 5).
#   RESIDUOS (se siguen reescribiendo, a sabiendas; `tests/test_p1_plan_lote_857.py::_RESIDUOS`, xfail estricto): otro
#       alimento cocido en la misma frase que el pez sin «mientras» (el sándwich: «…y tuesta el pan en la sartén»); los
#       tiempos que cuecen un pez y no un ave (el sellado de 30 s, el caldo que reposa 1 minuto, «cocina 1 minuto más»,
#       el escabeche frito 3 minutos por lado); y la comida con dos peces, donde la cocción de uno cuenta para el otro. El
#       tiempo mínimo del ave que hereda los tiempos del pez es de la sesión de PLATO (`ave_a_74` sólo sube los °C).
# SSOT: la conserva es la lista del cerrador (`graph_orchestrator._PRECOOKED_PROTEIN_HINT`, con la que escribe «ya viene
# cocido»), la cocción es la de V7f. Knob `MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD` (2).
# tooltip-anchor: P1-PLAN-LOTE-857-NO-REESCRIBIR
PEZ_LISTO_GUARD = _env_bool("MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD", True)

# Además de la lista del cerrador: los dos peces que el mercado sólo vende listos para comer (la anchoa fresca es el
# boquerón; la mojama es atún curado en sal), y la lata o «ya viene cocido» dichos en la línea o en la frase del pez.
SIEMPRE_EN_CONSERVA = ("anchoa", "mojama")
_LATA_RE = re.compile(r"\b(?:latas?|enlatad\w*|conservas?)\b|\bya\s+viene\s+cocid")
_SARDINA_FRESCA_RE = re.compile(r"\bsardinas?\s+fresc")
# El producto listo junto al pez: «atún al natural», «atún en salmuera», «bacalao en aceite» (en el nombre o la línea; en
# un paso «el pescado en aceite caliente» es freírlo, así que ahí sólo cuentan «al natural» y «en salmuera»).
_PRODUCTO_LINEA = r"(?:al\s+natural|en\s+salmuera|en\s+aceite)\b"
_PRODUCTO_PASO = r"(?:al\s+natural|en\s+salmuera)\b"
_JUNTO = r"\w*(?:\s+\w+){0,2}?\s+"
# Precocido junto al pez: «el pescado ya cocido y desmenuzado», «camarones precocidos», y en un PASO también «agrega los
# camarones cocidos». Entre medias sólo un tamaño o estado («camarones grandes cocidos»), nunca un verbo («hasta que el
# pescado esté cocido»).
_PEZ_HASTA_COCIDO = r"\w*\s+(?:(?:grandes?|pequen\w*|median\w*|pelad\w*|limpi\w*)\s+){0,2}"
_PRECOCIDO_LINEA = r"(?:ya\s+(?:pre)?cocid|precocid)"
_PRECOCIDO_PASO = r"(?:ya\s+)?(?:pre)?cocid"
_PRECOCIDO_ANTES = r"\bprecocid\w*\s+(?:de\s+)?"
# [ronda 8 · (1)] En la LÍNEA de ingrediente, «cocido» a secas («120 g de Pulpo cocido») es precocido, salvo que un paso
# lleve la FIRMA del cerrador de proteína: éste escribe el peso cocido («30 g de arenque cocido») y en el paso «Cocina
# arenque a la plancha o hervido y sírvelo como proteína del plato» (prod-1461aeca D3, prod-92328ff7 D9: se reescriben). Sin
# la firma, «Calienta el pescado en la sartén 1 minuto» sólo calienta lo que ya venía cocido. Sólo las líneas, no el nombre.
_COCIDO_EN_LA_LINEA = r"\w*\s+(?:\w+\s+){0,2}cocid"
_FIRMA_DEL_CERRADOR_RE = re.compile(r"a\s+la\s+plancha\s+o\s+hervid|como\s+proteina\s+del\s+plato")
# [ronda 8 · (4)] El pez ya listo dicho en un PASO: «el salmón ahumado», «la tilapia sobrante», «el pescado ya horneado».
_LISTO_EN_EL_PASO = r"\w*\s+(?:ya\s+(?:hornead|asad|hervid|frit|cocinad)|sobrante)"
_CURADO_EN_EL_PASO = r"\w*\s+(?:ahumad|curad)"
# Crudo por el nombre del plato (se blanquee o no); [ronda 8 · (3)] también dicho en un PASO («Prepara el cebiche: …»).
_PLATO_CRUDO_RE = re.compile(r"\b(?:ceviche|cebiche|tiradito|aguachile|tartar|tartare|carpaccio|sashimi|sushi|poke|"
                             r"tataki)\b")
# [ronda 8 · (2)] «crudo», «cruda» o «en crudo» hasta dos palabras después del pez: «sírvelo con el mero crudo en láminas».
_CRUDO_JUNTO = r"\w*(?:\s+\w+){0,2}?\s+(?:crud[oa]s?|en\s+crudo)\b"
# [ronda 8 · (5)] Lo que sigue a «mientras» es OTRA preparación: «marínala 20 minutos mientras sofríes la cebolla».
_MIENTRAS_RE = re.compile(r"\bmientras\b")
# Un blanqueo breve antes de marinar no es una cocción (lo inyecta `_inject_blanch_for_citrus_marinade`).
_BLANQUEO_RE = re.compile(r"\bblanque\w*|\bescald\w*|\bantes\s+de\s+marinar")
# «sirve frío»: sin acentos, V7f lee «frio» como «frío» de freír (`fri[eo]`); en una receta es el adjetivo.
_FRIO_RE = re.compile(r"\bfri[oa]s?\b")


def _lista(valor) -> list:
    return list(valor) if isinstance(valor, (list, tuple)) else [valor]


def _rx_alias(alias) -> "re.Pattern | None":
    """Frontera de palabra inicial, como el detector (`culinary_context._name_has_token`)."""
    als = sorted({_norm(a) for a in (alias or ()) if _norm(a)}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(a) for a in als) + r")") if als else None


def especie_nueva_en(comidas, especies_nuevas):
    """El nombre de la primera comida de una repetición de pescado (≥2 comidas) que lleva una especie NUEVA en el nombre o
    las líneas, o None. Con uno así, el autofix salta el día entero. Knob apagado o sin especies nuevas ⇒ None."""
    try:
        comidas = [m for m in (comidas or ()) if isinstance(m, dict)]
        if not BETA_FISH_SPECIES_COUNT or not especies_nuevas or len(comidas) < 2:
            return None
        nueva = _rx_alias(especies_nuevas)
        for m in comidas:
            if any(nueva.search(_norm(t)) for t in [m.get("name")] + _lista(m.get("ingredients")) if t):
                return str(m.get("name") or "—")
        return None
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] especie_nueva_en: {type(e).__name__}: {e} — el día va al gate")
        return "sin_verificar"


def _en_conserva(lineas, pasos, pez, precocido) -> bool:
    fresca = any(_SARDINA_FRESCA_RE.search(t) for t in lineas + pasos)
    pistas = [_norm(h) for h in (precocido or ()) if not (fresca and _norm(h) == "sardina")] + list(SIEMPRE_EN_CONSERVA)
    producto_linea = re.compile(pez.pattern + _JUNTO + _PRODUCTO_LINEA)
    for t in lineas:                                   # el nombre y las líneas: la lista entera del cerrador
        if pez.search(t) and (any(h in t for h in pistas) or _LATA_RE.search(t) or producto_linea.search(t)):
            return True
    import culinary_coherence as cc
    producto_paso = re.compile(pez.pattern + _JUNTO + _PRODUCTO_PASO)
    for paso in pasos:                                 # en los pasos, la lata en la frase del pez o el producto junto a él
        for a, b in cc.clause_bounds(paso):
            cl = paso[a:b]
            if pez.search(cl) and (_LATA_RE.search(cl) or producto_paso.search(cl)):
                return True
    curado = re.compile(pez.pattern + _CURADO_EN_EL_PASO)             # [ronda 8 · (4)] «el salmón ahumado» en un paso
    return any(curado.search(p) for p in pasos)


def _precocido(ingredientes, lineas, pasos, pez) -> bool:
    antes = _PRECOCIDO_ANTES + pez.pattern
    linea = re.compile(pez.pattern + _PEZ_HASTA_COCIDO + _PRECOCIDO_LINEA + "|" + antes)
    paso = re.compile(pez.pattern + _PEZ_HASTA_COCIDO + _PRECOCIDO_PASO + "|" + antes
                      + "|" + pez.pattern + _LISTO_EN_EL_PASO)                    # [ronda 8 · (4)] sobrante, ya horneado…
    if any(linea.search(t) for t in lineas) or any(paso.search(t) for t in pasos):
        return True
    cocido = re.compile(pez.pattern + _COCIDO_EN_LA_LINEA)                        # [ronda 8 · (1)] sin la firma del cerrador
    return (any(cocido.search(t) for t in ingredientes)
            and not any(_FIRMA_DEL_CERRADOR_RE.search(p) for p in pasos))


def _servido_crudo(nombre, lineas, pasos, pez) -> bool:
    """El plato crudo por su nombre o dicho en un paso, o «crudo»/«en crudo» junto al pez (ronda 8 · (2) y (3))."""
    if _PLATO_CRUDO_RE.search(nombre) or any(_PLATO_CRUDO_RE.search(p) for p in pasos):
        return True
    junto = re.compile(pez.pattern + _CRUDO_JUNTO)
    return any(junto.search(t) for t in lineas + pasos)


def _tramos(clausula) -> list:
    """[ronda 8 · (5)] La cláusula cortada en cada «mientras»: lo que sigue es otra preparación y su cocción no es la del
    pez («Marina la tilapia 20 minutos mientras hierve el arroz»)."""
    cortes = [0] + [m.start() for m in _MIENTRAS_RE.finditer(clausula)] + [len(clausula)]
    return [clausula[x:y] for x, y in zip(cortes, cortes[1:]) if clausula[x:y].strip()]


def _lo_cuece(pasos, pez) -> bool:
    """¿Algún tramo de cláusula que nombra el pez (o su enclítico en la cláusula siguiente) lo cuece, con el criterio de
    V7f? El blanqueo antes de marinar y las notas de seguridad no cuentan, ni lo que va después de «mientras»."""
    import culinary_coherence as cc
    for paso in pasos:
        if cc._V5_NOTA.search(paso):
            continue
        previa = False
        for a, b in cc.clause_bounds(paso):
            cl = paso[a:b]
            nombra = bool(pez.search(cl))
            if not _BLANQUEO_RE.search(cl):
                for tramo in _tramos(cl):
                    if ((pez.search(tramo) or (previa and cc._V7F_ENCLITICO_RE.search(tramo)))
                            and cc._v7f_evidencia(_FRIO_RE.sub(" ", tramo))):
                        return True
            previa = nombra
    return False


def motivo_para_no_reescribir(etiqueta, meal, alias_etiqueta, precocido, vetado, ligero=False):
    """¿Por qué el autofix de proteína repetida NO debe reescribir `meal` (una comida con la etiqueta `etiqueta`)?
    `pez_en_conserva` | `pez_precocido` | `pez_crudo` | `pez_sin_coccion` | `sin_verificar` (falló la lectura: no se
    reescribe) | None (se reescribe como en la base). `precocido` es `graph_orchestrator._PRECOOKED_PROTEIN_HINT`;
    `vetado`, `_diet_pool_item_banned` (qué etiqueta es del mar); `ligero`, que el único destino posible es el queso
    (merienda, desayuno ligero o plato dulce). Pura; nunca lanza."""
    if not isinstance(meal, dict):
        return None
    try:
        if not PEZ_LISTO_GUARD or not _del_mar(etiqueta, vetado):
            return None
        pez = _rx_alias(alias_etiqueta) or _rx_alias([etiqueta])
        nombre = _norm(meal.get("name"))
        ingredientes = [_norm(x) for k in ("ingredients", "ingredients_raw") for x in _lista(meal.get(k)) if x]
        lineas = [nombre] + ingredientes
        pasos = [_norm(x) for x in _lista(meal.get("recipe")) if isinstance(x, str) and x.strip()]
        if _en_conserva(lineas, pasos, pez, precocido):
            return "pez_en_conserva"
        if _precocido(ingredientes, lineas, pasos, pez):
            return "pez_precocido"
        if _servido_crudo(nombre, lineas, pasos, pez):
            return "pez_crudo"
        if pasos and not ligero and not _lo_cuece(pasos, pez):
            return "pez_sin_coccion"
        return None
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] motivo_para_no_reescribir: {type(e).__name__}: {e} — no se reescribe")
        return "sin_verificar"


# ── ronda 4 · (b) la escalera respeta la dieta (ronda 5: el queso de la comida ligera) ────────────────────────────────
def _del_mar(etiqueta, vetado) -> bool:
    """Proteína del mar según la guarda de dieta: la veta el vegetariano y no el pescetariano (`_DIET_SEAFOOD_TERMS`)."""
    return bool(vetado(etiqueta, "vegetarian")) and not vetado(etiqueta, "pescatarian")


def destino_apto_para_la_dieta(fuente, destino, forma_destino, dieta, vetado, ligero=False) -> bool:
    """¿Puede el autofix de proteína repetida cambiar `fuente` por `destino` (etiquetas) con la dieta declarada?

    `vetado(item, dieta)` es `graph_orchestrator._diet_pool_item_banned`: la guarda de dieta del revisor aplicada a un
    ingrediente (dieta canónica por `constants.canonicalize_diet_type`, términos `_DIET_*_TERMS`), no otra tabla.
    (1) Nunca un destino que la dieta veta: pollo o pavo a un pescetariano, pescado a un vegetariano, queso a un vegano.
    (2) Pescetariano: una proteína del mar sólo se cambia por otra del mar. Otro pez no cierra nada (es la misma etiqueta
    del gate) y el catálogo no tiene otra etiqueta del mar servible como plato fresco (el atún sólo está «en agua»), así
    que con el pescado no se reescribe y decide el gate; la legumbre o el queso de respaldo le quitarían al pescetariano
    la proteína que eligió. [ronda 5] Salvo `ligero` (merienda, desayuno ligero o plato dulce): ahí el autofix sólo admite
    queso, y la base ya pasaba «Casabe con tilapia desmenuzada» a queso, que la dieta permite — la regla quitaba ese
    arreglo sin nada a cambio. Omnívoro ⇒ True sin mirar nada (la conducta de antes). Duda ⇒ False (decide el gate)."""
    try:
        from constants import canonicalize_diet_type
        canon = canonicalize_diet_type(dieta)
        if canon == "balanced":
            return True
        if vetado(forma_destino, dieta):
            return False
        if canon == "pescatarian" and _del_mar(fuente, vetado) and not _del_mar(destino, vetado):
            return bool(ligero) and destino == "queso"
        return True
    except Exception:                                                  # noqa: BLE001
        return False
