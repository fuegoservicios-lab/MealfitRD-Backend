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
«1 lata de pechuga de pollo en lata» (el fallo que P2-PROTEIN-LADDER-GAPS cerró para el atún). La conserva y el curado
se reescriben ENTEROS: `compuestos_de_conserva` los genera desde estas mismas especies para
`_PROTEIN_SOURCE_COMPOUNDS['pescado']`, y `preparar_reescritura` deja al reescritor sólo los alias que están en el plato
(miles de compuestos costaban 2,9 s por comida) y quita de los pasos «ya escurridas» y «el líquido de la lata». (b) La
escalera no miraba la dieta: con dieta pescetariana, tilapia + mero ya salía «Pechuga de pollo…» en la base;
`destino_apto_para_la_dieta` veta el destino con la guarda de dieta del revisor. (c) Las anchoas cuentan (datos abajo).
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


# ── ronda 4 · (a) la conserva y el curado se reescriben ENTEROS ─────────────────────────────────────────────────
# Formas que en ningún pez son una preparación del fresco: el sustituto (pollo, pavo…) no viene en lata, ni en vinagre,
# ni ahumado de fábrica. Género explícito: en los PASOS el reescritor tolera el plural (`(?:es|s)?`), no el género.
CONSERVA = ("en lata", "enlatado", "enlatada", "en conserva", "en salmuera", "en vinagre", "en salazon",
            "ahumado", "ahumada", "salado", "salada")
# «En aceite» o «en salsa de tomate» son la LATA sólo en los peces que se venden así; en el fresco son una preparación
# que el sustituto hereda bien («merluza en salsa de tomate» → «pechuga de pollo en salsa de tomate», como antes).
# Datos: catálogo «Sardinas en lata» y «Anchoas» (lata de 50 g); caballa, melva y bonito son las conservas del mercado
# ES; el arenque del catálogo es ahumado o salado.
PECES_DE_LATA = frozenset(("sardina", "caballa", "anchoa", "melva", "bonito", "arenque"))
EN_ACEITE_O_SALSA = ("en aceite de oliva", "en aceite", "en salsa de tomate", "en tomate")
LATA = ("lata de", "latas de")
TRAS_LA_LATA = ("en lata", "enlatado", "enlatada")   # «1 lata de Sardinas en lata (120 g)»: la fila del catálogo

_CONSERVAS: set = set()   # los compuestos generados: si el plato trae uno, era una conserva


def _plural(especie: str):
    """Plural español de una especie de una palabra; None si es compuesta o ya termina en «s»."""
    if not especie or " " in especie or "-" in especie or especie.endswith("s"):
        return None
    if especie[-1] in "aeiou":
        return especie + "s"
    return especie[:-1] + "ces" if especie.endswith("z") else especie + "es"


def compuestos_de_conserva(alias_pescado) -> tuple:
    """Las frases de conserva y curado de cada especie de `alias_pescado` (el mapa ya extendido), para
    `_PROTEIN_SOURCE_COMPOUNDS['pescado']`: el reescritor del autofix las sustituye ENTERAS y largo-primero, así que
    «1 lata de Sardinas en lata (120 g)» pasa a «1 pechuga de pollo (120 g)» y no a «1 lata de pechuga de pollo en lata».
    Sin la especie en frase («filete de…», «bacalao salado» ya curado): nada. Knob apagado ⇒ `()`."""
    if not BETA_FISH_SPECIES_COUNT:
        return ()
    try:
        out = []
        for alias in alias_pescado or ():
            e = _norm(alias)
            if not e or e.startswith("filete") or "salad" in e:
                continue
            de_lata = e in PECES_DE_LATA
            sufijos = CONSERVA + (EN_ACEITE_O_SALSA if de_lata else ())
            tras_la_lata = TRAS_LA_LATA + (EN_ACEITE_O_SALSA if de_lata else ())
            for forma in (e, _plural(e)):
                if forma:
                    out += [f"{forma} {s}" for s in sufijos]
                    out += [f"{lata} {forma} {s}" for lata in LATA for s in tras_la_lata]
            out += [f"{lata} {e}" for lata in LATA]      # «lata de sardina»: el `\w*` y el plural cubren «sardinas»
        compuestos = tuple(dict.fromkeys(out))
        _CONSERVAS.clear()
        _CONSERVAS.update(compuestos)
        logger.info(f"🥫 [P1-PLAN-LOTE-857] pescado: {len(compuestos)} frases de conserva/curado se reescriben enteras")
        return compuestos
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] sin compuestos de conserva ({type(e).__name__}: {e})")
        return ()


def preparar_reescritura(aliases, meal) -> tuple:
    """Lo que `_protein_repeat_autofix._rewrite_meal` necesita antes de reescribir `meal`.

    (1) Sólo los alias que ESTÁN en el plato (subcadena sin acentos de nombre, ingredientes, crudos y pasos: el patrón
    del reescritor exige ese mismo texto, así que el resultado es idéntico; lo que cambia es el coste) y SIEMPRE el
    primero, del que la concordancia de los pasos lee el alimento viejo. (2) Si el plato traía un pez en conserva, los
    pasos sin «ya escurridas» ni «el líquido de la lata» (`pasos_sustitucion.limpiar_pasos_enlatado`, el mismo del
    cambio enlatado→fresco del tope de sodio). Knob apagado ⇒ los alias tal cual y los pasos intactos."""
    als = tuple(aliases or ())
    if not BETA_FISH_SPECIES_COUNT or not isinstance(meal, dict):
        return als
    try:
        partes = [meal.get("name")]
        for clave in ("ingredients", "ingredients_raw", "recipe"):
            valor = meal.get(clave)
            partes += list(valor) if isinstance(valor, list) else [valor]
        texto = _norm(" ".join(str(p) for p in partes if p))
        presentes = tuple(a for i, a in enumerate(als) if i == 0 or _norm(a) in texto)
        if any(_norm(a) in _CONSERVAS for a in presentes):
            import pasos_sustitucion
            pasos_sustitucion.limpiar_pasos_enlatado(meal)
        return presentes
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] preparar_reescritura no-op ({type(e).__name__}: {e})")
        return als


# ── ronda 4 · (b) la escalera respeta la dieta ────────────────────────────────────────────────────────────────
def _del_mar(etiqueta, vetado) -> bool:
    """Proteína del mar según la guarda de dieta: la veta el vegetariano y no el pescetariano (`_DIET_SEAFOOD_TERMS`)."""
    return bool(vetado(etiqueta, "vegetarian")) and not vetado(etiqueta, "pescatarian")


def destino_apto_para_la_dieta(fuente, destino, forma_destino, dieta, vetado) -> bool:
    """¿Puede el autofix de proteína repetida cambiar `fuente` por `destino` (etiquetas) con la dieta declarada?

    `vetado(item, dieta)` es `graph_orchestrator._diet_pool_item_banned`: la guarda de dieta del revisor aplicada a un
    ingrediente (dieta canónica por `constants.canonicalize_diet_type`, términos `_DIET_*_TERMS`), no otra tabla.
    (1) Nunca un destino que la dieta veta: pollo o pavo a un pescetariano, pescado a un vegetariano, queso a un vegano.
    (2) Pescetariano: una proteína del mar sólo se cambia por otra del mar. Otro pez no cierra nada (es la misma etiqueta
    del gate) y el catálogo no tiene otra etiqueta del mar servible como plato fresco (el atún sólo está «en agua»), así
    que con el pescado no se reescribe y decide el gate; la legumbre o el queso de respaldo le quitarían al pescetariano
    la proteína que eligió. Omnívoro ⇒ True sin mirar nada (la conducta de antes). Duda ⇒ False (decide el gate)."""
    try:
        from constants import canonicalize_diet_type
        canon = canonicalize_diet_type(dieta)
        if canon == "balanced":
            return True
        if vetado(forma_destino, dieta):
            return False
        return not (canon == "pescatarian" and _del_mar(fuente, vetado) and not _del_mar(destino, vetado))
    except Exception:                                                  # noqa: BLE001
        return False
