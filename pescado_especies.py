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

Ronda 5 (revisión r4). Reescribir la conserva ENTERA daba pollo CRUDO (receta fría, o un recalentado de 2-3 minutos):
la conserva ya no se reescribe; se queda de guardiana y se cambia el otro pez del día (`guardiana_y_conservas`). En la
comida ligera, el pescetariano vuelve a recibir queso como en la base.
"""
from __future__ import annotations

import logging
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


# ── ronda 5 · (a) la conserva se queda; se cambia el pez que se cuece ─────────────────────────────────────────────────
# Revisión r4 (bloquea, seguridad alimentaria): reescribir la conserva ENTERA (ronda 4) daba pechuga de pollo CRUDA —
# «Escurre pechuga de pollo y mézclalas… sirve frío», o el ceviche de G24 CO con «marina pechuga de pollo 5 minutos con
# el jugo de limón» (el plato sí tenía fuego: el del plátano). Y con fuego tampoco: la receta de una conserva está
# escrita para un producto listo para comer, así que su fuego es un RECALENTADO («Calienta las sardinas en la sartén
# 3 minutos» → pollo crudo a los 3 minutos; «Abre la lata de sardinas…» → «Abre pechuga de pollo…»). La conserva no se
# reescribe nunca: se queda ELLA de guardiana y el autofix cambia el otro pez del día, el que la receta sí cuece. Sin
# otro pez que cambiar, la impotencia queda en el log (`conserva_sin_fuego`) y decide el gate. Eso retira las 2 078
# frases de conserva del reescritor, su filtro de coste y la limpieza de pasos que rompía «(de lata, enjuagados y
# escurridos)» de los garbanzos del mismo paso.
#
# Conserva o curado LISTO PARA COMER pegado a la especie: sólo cuentan las palabras que la siguen sin otro alimento en
# medio («tilapia con garbanzos de lata» o «tilapia con pimentón ahumado» no son una conserva).
_LISTO = r"(?:en\s+lata|de\s+lata|enlatad\w*|en\s+conserva|en\s+salmuera|en\s+vinagre|en\s+escabeche|escabechad\w*|ahumad\w*)"
# «En aceite», «en salsa de tomate», «en agua», «al natural»: la LATA sólo en los peces que se venden así, y sólo en la
# lista o el nombre (en un paso, «hierve las sardinas en agua» es una cocción; «merluza en salsa de tomate» es un guiso).
PECES_DE_LATA = frozenset(("sardina", "caballa", "anchoa", "melva", "bonito", "arenque"))
_LISTO_DE_LATA = r"(?:en\s+aceite|en\s+salsa\s+de\s+tomate|en\s+tomate|en\s+agua|al\s+natural)"
# Peces que el catálogo y el mercado sólo venden listos para comer: «Anchoas» es la lata de 50 g (los frescos son
# boquerones) y la mojama es atún curado en sal que se come en lonchas.
SIEMPRE_EN_CONSERVA = frozenset(("anchoa", "mojama"))
# Hasta dos palabras entre la especie y la conserva («bonito del norte en aceite», «sardinas marinadas en vinagre»),
# ninguna de las que introducen OTRO alimento.
_ENTRE = r"(?:\W+(?!(?:con|y|e|o|u|sobre|junto|mas|acompanad\w*)\b)\w+){0,2}?\W+"


def _especies(alias_pescado) -> list:
    """Las especies del mapa, sin acentos, las largas primero (para que «filete de tilapia» gane a «tilapia»)."""
    return sorted({_norm(a) for a in (alias_pescado or ()) if _norm(a)}, key=len, reverse=True)


def pez_en_conserva(meal, alias_pescado) -> bool:
    """¿`meal` lleva un pez de `alias_pescado` en conserva o curado listo para comer? Mira el nombre, la lista, los crudos
    y los pasos (en los pasos, sólo lata/conserva/vinagre/escabeche/ahumado). Pura, sin estado; nunca lanza (duda ⇒
    False, la conducta de antes). Knob `MEALFIT_BETA_FISH_SPECIES_COUNT` apagado ⇒ False."""
    if not BETA_FISH_SPECIES_COUNT or not isinstance(meal, dict):
        return False
    try:
        import re
        especies = _especies(alias_pescado)
        if not especies:
            return False
        pez = r"\b(" + "|".join(re.escape(e) for e in especies) + r")(?:s|es)?\b"
        de_lata = [e for e in especies if e in PECES_DE_LATA]
        rx_listo = re.compile(pez + _ENTRE + _LISTO + r"|\blatas?\W+de\W+(?:\w+\W+)?" + pez)
        rx_siempre = re.compile(r"\b(?:" + "|".join(sorted(SIEMPRE_EN_CONSERVA)) + r")s?\b")
        rx_de_lata = (re.compile(r"\b(" + "|".join(re.escape(e) for e in de_lata) + r")(?:s|es)?\b" + _ENTRE
                                 + _LISTO_DE_LATA) if de_lata else None)
        lista = [meal.get("name")]
        for clave in ("ingredients", "ingredients_raw"):
            valor = meal.get(clave)
            lista += list(valor) if isinstance(valor, list) else [valor]
        pasos = meal.get("recipe")
        pasos = list(pasos) if isinstance(pasos, list) else [pasos]
        for texto in (_norm(t) for t in lista if t):
            if rx_listo.search(texto) or (rx_de_lata and rx_de_lata.search(texto)):
                return True
            if any(e in SIEMPRE_EN_CONSERVA for e in especies) and rx_siempre.search(texto):
                return True
        return any(rx_listo.search(_norm(t)) for t in pasos if t)
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] pez_en_conserva no-op ({type(e).__name__}: {e})")
        return False


# Preparación CRUDA del pez (la misma clase de fallo sin lata, en la sonda del revisor r4: «Sardinas marinadas» + «Sirve
# las sardinas marinadas al limón» salía «Sirve pechuga de pollo marinadas al limón»). Señal positiva: el tipo de plato
# crudo en el NOMBRE, o el pez PEGADO a «marinado/crudo» («sardinas marinadas», «boquerones crudos», «marina la trucha»;
# no «agrega el arroz, que se pesa en crudo» en el paso del locrio de sardinas, réplica 63eedc6b); y ninguna cláusula
# que lo nombre lo cuece (`culinary_coherence._v7f_evidencia`, el mismo criterio de V7f), ni la siguiente con un
# enclítico («…; hornéala 20 minutos»), ni el nombre declara la cocción («al horno», «a la plancha», «guisado»).
_PLATO_CRUDO = r"\b(?:ceviche|cebiche|tiradito|aguachile|tartar|carpaccio|sashimi|sushi|poke)\b"
_CRUDO_DESPUES = r"(?:marinad\w*|crud[oa]s?)\b"
_MARINA_ANTES = r"\bmarin(?:a|ar|ala|alo|alas|alos)\W+(?:\w+\W+){0,2}?"


def pez_crudo(meal, alias_pescado) -> bool:
    """¿`meal` sirve un pez de `alias_pescado` crudo (ceviche, marinado sin fuego, tartar…)? Pura; nunca lanza (duda ⇒
    False, la conducta de antes). Knob `MEALFIT_BETA_FISH_SPECIES_COUNT` apagado ⇒ False."""
    if not BETA_FISH_SPECIES_COUNT or not isinstance(meal, dict):
        return False
    try:
        import re
        import culinary_coherence as cc
        especies = _especies(alias_pescado)
        if not especies:
            return False
        pez_rx = r"\b(?:" + "|".join(re.escape(e) for e in especies) + r")(?:s|es)?\b"
        pez = re.compile(pez_rx)
        pegado = re.compile(pez_rx + _ENTRE + _CRUDO_DESPUES + "|" + _MARINA_ANTES + pez_rx)
        nombre = _norm(meal.get("name"))
        piezas = [nombre]
        for clave in ("ingredients", "ingredients_raw"):
            valor = meal.get(clave)
            piezas += [_norm(x) for x in (valor if isinstance(valor, list) else [valor]) if x]
        pasos = [_norm(x) for x in (meal.get("recipe") or []) if isinstance(x, str) and not cc._V5_NOTA.search(x)]
        crudo = re.search(_PLATO_CRUDO, nombre) or any(pegado.search(t) for t in piezas + pasos)
        if not crudo:
            return False
        if cc._V7F_DECLARADO_COCIDO_RE.search(nombre) or cc._V7F_FUEGO_RE.search(nombre):
            return False                                       # «Tilapia marinada al horno»
        for paso in pasos:
            previa = False
            for a, b in cc.clause_bounds(paso):
                cl = paso[a:b]
                nombra = bool(pez.search(cl))
                if (nombra or (previa and cc._V7F_ENCLITICO_RE.search(cl))) and cc._v7f_evidencia(cl):
                    return False                               # una cláusula lo cuece
                previa = nombra
        return True
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] pez_crudo no-op ({type(e).__name__}: {e})")
        return False


def guardiana_y_conservas(etiqueta, meals, alias_pescado, guardiana) -> tuple:
    """Para el autofix de proteína repetida: `(guardiana, intocables)`. `guardiana` es el índice de la comida que conserva
    la etiqueta (por defecto, la primera con la proteína en el nombre); `intocables`, `{índice: motivo}` de las comidas
    que NO se reescriben porque su receta no cuece carne cruda: `conserva_sin_fuego` (`pez_en_conserva`) o
    `crudo_sin_fuego` (`pez_crudo`). Si la guardiana no es intocable, pasa a serlo la primera intocable, y el autofix
    cambia el otro pez. Otra etiqueta, knob apagado o duda ⇒ `(guardiana, {})`, la conducta de antes."""
    if etiqueta != "pescado" or not BETA_FISH_SPECIES_COUNT:
        return guardiana, {}
    try:
        intocables = {}
        for i, m in enumerate(meals or ()):
            if pez_en_conserva(m, alias_pescado):
                intocables[i] = "conserva_sin_fuego"
            elif pez_crudo(m, alias_pescado):
                intocables[i] = "crudo_sin_fuego"
        if intocables and guardiana not in intocables:
            i0 = min(intocables)
            logger.info(f"🥫 [P1-PLAN-LOTE-857] '{str(meals[i0].get('name'))[:48]}' se queda ({intocables[i0]}): su "
                        f"receta no cuece carne cruda — se cambia el otro pez")
            guardiana = i0
        return guardiana, intocables
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] guardiana_y_conservas no-op ({type(e).__name__}: {e})")
        return guardiana, {}


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
