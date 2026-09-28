"""[P1-PLAN-LOTE-225 · 2026-09-24] Los alimentos del catálogo en los 5 idiomas: para LEER y para ENTENDER lo escrito.

El catálogo (`master_ingredients`) tiene UN nombre por alimento, el canónico en español, y es el identificador con
el que resuelven `pantry_names_match`, el guard de coherencia y el backstop de alergias: eso no se toca. Hasta este
lote, además, sólo tenía un gloss inglés (`name_en`), así que todo lo que el usuario ESCRIBE en su idioma se perdía:

  - una alergia libre fuera de las clases modeladas («Strawberry», «Fraise», «Fragola», «Morango») pasaba literal al
    escáner, que busca dentro de platos escritos en español, y no casaba con «Fresas». MEDIDO con
    `clinical_backstop_for_meal` sobre platos que SÍ llevaban el alimento: 32 de 60 declaraciones en inglés, francés,
    italiano o portugués NO bloqueaban (fresa, tomate, piña, aguacate, maíz, coco, ajo, mango…);
  - los buscadores sólo entendían español e inglés, y en portugués, francés e italiano no encontraban nada.

Este módulo es la mitad que faltaba: `data/food_names_i18n.json` da a cada nombre canónico su nombre en `en-US`,
`pt-BR`, `fr-FR` e `it-IT`, y `canonicos_para_texto` traduce lo escrito AL canónico antes de que llegue al motor.
La frontera de P1-I18N-DASHBOARD sigue intacta: el motor sólo ve nombres españoles; lo que cambia es que ahora
entiende a quien no escribe en español.

tooltip-anchor: P1-PLAN-LOTE-225
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LOCALES = ("en-US", "pt-BR", "fr-FR", "it-IT")
_RUTA = Path(__file__).resolve().parent / "data" / "food_names_i18n.json"

# Palabras que no nombran el alimento («Huile d'olive», «Carne de res», «Coscia di pollo»): fuera antes de comparar,
# o «de» casaría media lista.
_VACIAS = frozenset(
    "a al alla all allo and au aux com con d da de del della dei degli delle des di do dos du e em en et in l la las "
    "le les los na no o of os the y".split()
)
_LIGADURAS = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


@lru_cache(maxsize=1)
def _datos() -> dict:
    """El archivo entero. Sin él (o roto), {}: nada se traduce y todo sigue en español."""
    try:
        with open(_RUTA, encoding="utf-8") as f:
            datos = json.load(f)
        return datos if isinstance(datos, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def nombres() -> dict:
    """{nombre canónico: {locale: nombre}}. Sin el archivo, {}: nada se traduce y todo sigue en español."""
    alimentos = _datos().get("alimentos") or {}
    return {str(k): dict(v) for k, v in alimentos.items() if isinstance(v, dict)}


def _variantes(seccion: str) -> dict:
    """[P1-PLAN-LOTE-623] {canónico: [nombres del mismo alimento en otro país hispano]} de `seccion`."""
    crudo = _datos().get(seccion) or {}
    return {str(k): [str(n) for n in v if isinstance(n, str) and n.strip()]
            for k, v in crudo.items() if isinstance(v, list)} if isinstance(crudo, dict) else {}


def nombre_para(canonico: str, locale: str | None) -> str | None:
    """El nombre del alimento en `locale`, o None si no hay (o si el locale es el base): quien lo pinta usa el canónico."""
    if not canonico or not locale or locale not in LOCALES:
        return None
    fila = nombres().get(str(canonico))
    if not fila:
        return None
    valor = str(fila.get(locale) or "").strip()
    return valor or None


def _norm(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto or "").translate(_LIGADURAS))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _raiz(palabra: str) -> str:
    """Raíz común a singular y plural en los 5 idiomas: «strawberries»/«strawberry», «fraises»/«fraise»,
    «fragole»/«fragola», «morangos»/«morango», «tomatoes»/«tomato», «uova»/«uovo», «limões»/«limão».

    Quita UNA vocal final, no todas: con todas, «laitue» (lechuga) y «lait» (leche) daban la misma raíz y declarar
    leche arrastraba la lechuga."""
    w = palabra
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("oes"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    if w.endswith("ao"):
        w = w[:-2] + "o"
    if len(w) > 3 and w[-1] in "aeiou":
        w = w[:-1]
    return w


def _raices(texto: str) -> tuple:
    return tuple(_raiz(w) for w in re.split(r"[^a-z]+", _norm(texto)) if len(w) >= 2 and w not in _VACIAS)


@lru_cache(maxsize=1)
def _indice() -> tuple:
    """(canónico, raíces) por cada forma de escribirlo: el canónico, sus nombres en los 4 idiomas y [P1-PLAN-LOTE-623]
    sus nombres inequívocos en otros países hispanos («Melocotón» → Duraznos, «Boniato» → Batata)."""
    filas = []
    regionales = _variantes("variantes_regionales")
    for canonico, por_locale in nombres().items():
        for forma in (canonico, *[por_locale.get(loc) for loc in LOCALES], *regionales.get(canonico, ())):
            r = _raices(forma or "")
            if r:
                filas.append((canonico, frozenset(r)))
    return tuple(filas)


@lru_cache(maxsize=1)
def _indice_solo_alergias() -> tuple:
    """[P1-PLAN-LOTE-623] (canónico, raíces) de los nombres AMBIGUOS entre países: «Plátano» es la banana (Guineo) en
    España y México y el plátano de cocinar en RD. Solo amplían una alergia —bloquear de más es la dirección segura—;
    en un rechazo o un buscador le quitarían la banana al dominicano que no quiere plátano."""
    return tuple((c, frozenset(r)) for c, formas in _variantes("variantes_solo_alergias").items()
                 if c in nombres() for r in (_raices(f) for f in formas) if r)


def _singular(termino: str) -> str:
    """[P1-PLAN-LOTE-623] El singular español de un canónico normalizado («duraznos» → «durazno», «habichuelas negras»
    → «habichuela negra», «camarones» → «camaron», «nueces» → «nuez»). El patrón del escáner tolera el plural del
    término, no el singular, y el canónico del catálogo suele ir en plural: sin esto «Melocotón» no alcanzaba
    «1 durazno fresco»."""
    def una(w: str) -> str:
        if len(w) > 4 and w.endswith("ces"):
            return w[:-3] + "z"
        if len(w) > 4 and w.endswith("es") and w[-3] in "lnrdj":
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and w[-2] in "aeiou":
            return w[:-1]
        return w
    return " ".join(w if w in _VACIAS else una(w) for w in termino.split(" "))


@lru_cache(maxsize=2048)
def canonicos_para_texto(texto: str) -> tuple:
    """Los nombres canónicos que `texto` nombra, en cualquiera de los 5 idiomas.

    Primero los alimentos cuyo nombre ENTERO está en el texto («strawberry» → Fresas, «allergie aux fraises» →
    Fresas, «coconut milk» → Leche de coco y Coco). Si ninguno, los que CONTIENEN el texto entero («corn» → Maíz
    dulce en granos, Harina de maíz precocida, Tortilla de maíz…): una palabra suelta que en el catálogo sólo
    existe dentro de nombres compuestos.

    El sesgo es el del backstop de alergias: ante la duda, de más. Cada canónico devuelto se busca en el plato como
    si el usuario lo hubiera escrito en español, así que «Tomate» también alcanza «Salsa de tomate».
    """
    raices = frozenset(_raices(texto))
    if not raices:
        return ()
    dentro = sorted({c for c, r in _indice() if r <= raices})
    if dentro:
        return tuple(dentro)
    return tuple(sorted({c for c, r in _indice() if raices <= r}))


def canonicos_que_el_literal_no_alcanza(a_low: str, patron_de) -> list:
    """[P1-PLAN-LOTE-225 · 2026-09-24] Los nombres canónicos del catálogo que la declaración de alergia nombra en
    cualquiera de los 5 idiomas («strawberry», «fraise», «fragola», «morango» → «fresas»), normalizados como el resto de
    términos del escáner. `patron_de(termino)` es el patrón del escáner (`graph_orchestrator._patron_termino_alergeno`,
    se pasa para no importar el grafo aquí). Sin léxico (archivo ausente o roto) devuelve [] y la conducta es la de
    antes. Extraído del grafo (tope de líneas, roadmap 2.5 §11)."""
    try:
        from constants import strip_accents
        out = []
        # [P1-PLAN-LOTE-623] …y los nombres ambiguos entre países, que sólo valen para esto (ver `_indice_solo_alergias`)
        raices = frozenset(_raices(a_low))
        ambiguos = sorted({c for c, r in _indice_solo_alergias() if r <= raices}) if raices else []
        for c in (*canonicos_para_texto(a_low), *ambiguos):
            canon = strip_accents(str(c).lower())
            # El canónico que el literal ya alcanza («fresa» → «fresas»: el escáner tolera el plural) no suma nada.
            # El criterio es el PATRÓN del escáner, no una raíz: «tomato» y «tomate» comparten raíz y el patrón de
            # «tomato» no encuentra «Tomate».
            # [P1-PLAN-LOTE-623] el singular también: el patrón de «duraznos» no encuentra «1 durazno fresco».
            for termino in dict.fromkeys((canon, _singular(canon))):
                if re.search(patron_de(a_low), termino) is None and termino not in out:
                    out.append(termino)
        return out
    except Exception:
        return []
