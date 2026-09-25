"""[P1-PLAN-LOTE-224 · 2026-09-24] Los alimentos del catálogo en los 5 idiomas: para LEER y para ENTENDER lo escrito.

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

tooltip-anchor: P1-PLAN-LOTE-224
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
def nombres() -> dict:
    """{nombre canónico: {locale: nombre}}. Sin el archivo, {}: nada se traduce y todo sigue en español."""
    try:
        with open(_RUTA, encoding="utf-8") as f:
            datos = json.load(f)
        alimentos = datos.get("alimentos") or {}
        return {str(k): dict(v) for k, v in alimentos.items() if isinstance(v, dict)}
    except Exception:
        return {}


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
    """(canónico, raíces) por cada forma de escribirlo: el canónico y sus nombres en los 4 idiomas."""
    filas = []
    for canonico, por_locale in nombres().items():
        for forma in (canonico, *[por_locale.get(loc) for loc in LOCALES]):
            r = _raices(forma or "")
            if r:
                filas.append((canonico, frozenset(r)))
    return tuple(filas)


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


def canonicos_de_declaracion(declaracion: str, patron_del_escaner) -> list:
    """[P1-PLAN-LOTE-224 · 2026-09-24] Los nombres canónicos del catálogo que una alergia o un disgusto nombra en
    cualquiera de los 5 idiomas («strawberry», «fraise», «fragola», «morango» → «fresas»), normalizados (minúsculas,
    sin acentos) como los términos de `graph_orchestrator._expand_allergy_declarations`, que es quien los usa.

    Las clases de alérgenos cubren los 14 del Reglamento UE; esto cubre el resto del catálogo, que es donde una alergia
    en otro idioma caía literal y no casaba con nada. `patron_del_escaner` es el patrón con que el escáner busca un
    término en el plato (`_patron_termino_alergeno`): el canónico que el literal ya alcanza («fresa» → «fresas»: el
    escáner tolera el plural) no suma nada. El criterio es ese PATRÓN, no una raíz: «tomato» y «tomate» comparten raíz
    y el patrón de «tomato» no encuentra «Tomate». Sin léxico (fichero ausente o roto) devuelve [] y la conducta es la
    de antes. Vivía en `graph_orchestrator.py`: salió aquí para no hacer crecer el god file, cuyo tope baja con cada
    extracción (`test_p3_shopping_projection_pkg`).
    """
    try:
        from constants import strip_accents
        out = []
        for c in canonicos_para_texto(declaracion):
            canon = strip_accents(str(c).lower())
            if re.search(patron_del_escaner(declaracion), canon) is None:
                out.append(canon)
        return out
    except Exception:
        return []
