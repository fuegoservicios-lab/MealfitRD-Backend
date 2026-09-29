"""[P1-PLAN-LOTE-853 · 2026-09-29] La palabra de cada país al LEER el plan, sin tocar el plan.

G24 (29-sep, 6 planes reales, código P1-PLAN-LOTE-815): «guineo», «lechosa», «auyama», «ají morrón», «queso blanco»,
«habichuelas» y «funda» salían en nombres, descripciones, pasos y lista de España, México, Colombia y EE. UU. No es
el prompt: son NOMBRES DEL CATÁLOGO, el identificador con que resuelven la Nevera (`pantry_names_match`), el guard de
coherencia y el backstop de alergias, y el modelo está obligado a copiarlos. Por eso el plan no se toca: la palabra
se cambia al PINTAR, por país de MERCADO (`country_for_form_data`: es donde se compra y se lee la etiqueta; la cocina,
I16, no decide el vocabulario del súper).

Lo que hace, con el léxico como DATA (`data/lexico_vista_pais.json`):
  - sustituye la palabra en todas sus apariciones, con su caja («Ají Morrón» → «Pimiento Morrón»);
  - si el género cambia (habichuelas→frijoles) concuerda el determinante de delante («las»→«los», «de la»→«del») y
    los adjetivos de detrás, también coordinados («rojas cocidas»→«rojos cocidos», «cocidas y escurridas»→«cocidos y
    escurridos»); si el RESTO —hasta que el texto vuelve a nombrar la palabra, pasos siguientes incluidos— sigue
    hablando de ella en femenino («májalas», «hasta cubrirlas», «no las revuelvas», «estén blandas», «, previamente
    remojadas») o un vecino no sabe concordar, no sustituye: GLOSA, como el 649; también si entre un determinante que
    cambia y la palabra hay un número o un invariable («las 2 habichuelas», «las demás habichuelas»). Es una
    HEURÍSTICA con límites conocidos y medidos (docs/lexico_vista_pais.md → «Límites conocidos»);
  - «funda» solo como envase de la lista, tras la cantidad: en un paso es el verbo («que el queso funda»);
  - en la frase, lo que el léxico no cubre lo sigue glosando el lote 649 en la MISMA pasada (así «guineo verde» →
    «plátano verde» no vuelve a casar con el «Plátano verde» dominicano y acaba glosado como plátano macho).

La implementación que PINTA es la del frontend (`src/utils/lexicoDelPais.js`, llamada desde `mealDisplay`,
`glossShoppingItemName` y `glossShoppingQty`: pantalla y PDF). Esta es la referencia: los `casos` del JSON los corren
las dos, y es la que usan el replay sin IA y cualquier superficie del backend que pinte texto del plan a un lector en
español. Nada de lo que devuelve vuelve a `plan_data`: `comida_para_leer` devuelve una copia.

Knob `MEALFIT_COUNTRY_DISPLAY_LEXICON` (True). Apagado ⇒ la conducta del 649 (solo la glosa). El frontend lleva el
suyo, `VITE_COUNTRY_DISPLAY_LEXICON`. Doc: docs/lexico_vista_pais.md.

tooltip-anchor: P1-PLAN-LOTE-853
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from knobs import _env_bool

logger = logging.getLogger(__name__)

_RUTA = Path(__file__).resolve().parent / "data" / "lexico_vista_pais.json"

# Una letra (sin dígitos ni «_»), y los bordes de palabra que usa el glosador del frontend: ni letra ni número al lado.
_LETRA = r"[^\W\d_]"
_BORDE_IZQ = r"(?<![^\W_])"
_BORDE_DER = r"(?![^\W_])"
_ACENTOS = "áéíóú"
# Para leer el RESTO en fichas: número (con las fracciones de la receta), palabra o signo. Las fracciones no son letras
# (en Python `\w` las cuenta como alfanuméricas): así «½» es un número aquí y en el frontend (`\p{N}`).
_FRACCIONES = "½¼¾⅓⅔⅕⅛⅜⅝⅞¹²³"
_FICHA = re.compile(r"(\d+(?:[.,]\d+)?|[" + _FRACCIONES + r"])|([^\W\d_" + _FRACCIONES + r"]+)|([^\w\s])")
# El enlace de una coordinación tras el sintagma: «, cocidas» o «y escurridas».
_ENLACE = re.compile(r"\s*(,|" + _LETRA + r"+)\s+(" + _LETRA + r"+)")
# Entre un paso y el siguiente, un corte de frase (lo que empieza el paso no se atribuye al anterior).
_ENTRE_PASOS = "\n.\n"
# La palabra como complemento de otro nombre («tortitas de habichuela», «la masa de la habichuela»).
_COMPLEMENTO = re.compile(r"(?<![^\W_])de(?:\s+(?:la|una|esta|esa))?\s+$", re.IGNORECASE)
# Lo último antes de una posición: un número (grupo 1, con las fracciones de la receta) o una palabra (grupo 2).
_FINAL_NUM_O_PALABRA = re.compile(
    r"(?:(\d+(?:[.,]\d+)?[" + _FRACCIONES + r"]?|[" + _FRACCIONES + r"])|([^\W\d_" + _FRACCIONES + r"]+))\s+$")


def activo() -> bool:
    return _env_bool("MEALFIT_COUNTRY_DISPLAY_LEXICON", True)


@lru_cache(maxsize=1)
def _datos() -> dict:
    """El archivo entero. Sin él (o roto), {}: nada se sustituye y queda la glosa del 649."""
    try:
        with open(_RUTA, encoding="utf-8") as f:
            datos = json.load(f)
        return datos if isinstance(datos, dict) else {}
    except Exception as e:  # pragma: no cover - el test exige el archivo
        logger.warning(f"[P1-PLAN-LOTE-853] léxico de vista ilegible: {e}")
        return {}


def _pais(pais) -> str:
    return str(pais or "").strip().upper()


def filas(pais) -> list:
    """Las entradas del léxico de `pais` (vacío para RD, sin país o país desconocido)."""
    t = (_datos().get("paises") or {}).get(_pais(pais))
    return list(t) if isinstance(t, list) else []


def _concordancia(clave: str) -> dict:
    c = (_datos().get("concordancia") or {}).get(clave)
    return c if isinstance(c, dict) else {}


@lru_cache(maxsize=64)
def _patron(forma: str) -> "re.Pattern":
    return re.compile(_BORDE_IZQ + re.escape(forma) + _BORDE_DER, re.IGNORECASE)


@lru_cache(maxsize=16)
def _patrones(pais: str, ambito: str) -> tuple:
    """(forma, número, fila) del ámbito, de la forma más larga a la más corta («plátano verde» antes que «plátano»)."""
    out = []
    for fila in filas(pais):
        if (fila.get("ambito") or "texto") != ambito:
            continue
        for n in (0, 1):
            out.append((fila["de"][n], n, fila))
    out.sort(key=lambda x: -len(x[0]))
    return tuple(out)


def _con_caja(original: str, destino: str) -> str:
    """`destino` con la caja de `original`: MAYÚSCULAS, Título De Cada Palabra, Inicial o minúsculas."""
    letras = [c for c in original if c.isalpha()]
    if len(letras) > 1 and all(c.isupper() for c in letras):
        return destino.upper()
    palabras = original.split()
    if len(palabras) > 1 and all(p[:1].isupper() for p in palabras):
        return " ".join(w[:1].upper() + w[1:] for w in destino.split(" "))
    if original[:1].isupper():
        return destino[:1].upper() + destino[1:]
    return destino


def _solapa(ini: int, fin: int, ocupados: list) -> bool:
    return any(ini < b and fin > a for a, b in ocupados)


def _palabra_previa(texto: str, hasta: int):
    m = re.search("(" + _LETRA + r"+)(\s+)$", texto[:hasta])
    return (m.group(1), m.start(1), m.end(1)) if m else None


def _palabra_siguiente(texto: str, desde: int):
    m = re.match(r"(\s+)(" + _LETRA + r"+)", texto[desde:])
    return (m.group(2), desde + m.start(2), desde + m.end(2)) if m else None


def _adjetivo_concordado(palabra: str, num: str, c: dict):
    w = palabra.lower()
    fijo = (c.get("adjetivos") or {}).get(num, {}).get(w)
    if fijo:
        return _con_caja(palabra, fijo)
    for suf, nuevo in (c.get("sufijos") or {}).get(num, []):
        if len(w) > len(suf) + 1 and w.endswith(suf):
            return _con_caja(palabra, w[: -len(suf)] + nuevo)
    return None


def _femenina(palabra: str, num: str, c: dict) -> bool:
    """¿`palabra` parece concordar en femenino con el número `num`? (lo que no sabemos reescribir)."""
    w = palabra.lower()
    if w in (c.get("neutras") or []):
        return False
    return w.endswith("as") if num == "pl" else w.endswith("a")


_LISTAS: dict = {}


def _lista(c: dict, clave: str) -> frozenset:
    """La lista `clave` de la concordancia como conjunto, leída una vez (`c` sale del JSON cacheado en `_datos`)."""
    k = (id(c), clave)
    if k not in _LISTAS:
        _LISTAS[k] = frozenset(c.get(clave) or [])
    return _LISTAS[k]


def _fichas(texto: str) -> list:
    """[(tipo, texto)] con tipo «n» (número), «w» (palabra) o «p» (signo); los espacios no cuentan."""
    return [("n" if m.group(1) else "w" if m.group(2) else "p", m.group(0)) for m in _FICHA.finditer(texto)]


def _adjetivo_femenino(palabra: str, num: str, c: dict) -> bool:
    """¿`palabra` es un adjetivo o participio femenino que sabemos concordar (y no un nombre que lo parece)?"""
    w = palabra.lower()
    if w in _lista(c, "sustantivos") or w in _lista(c, "neutras"):
        return False
    return _adjetivo_concordado(palabra, num, c) is not None


def _determinantes(c: dict) -> set:
    """Artículos, demostrativos, cuantificadores y números en letra, de los dos géneros y números."""
    dets = set(_lista(c, "determinantes_otros"))
    for n in ("sg", "pl"):
        d = (c.get("determinantes") or {}).get(n, {})
        dets |= set(d) | set(d.values())
    return dets


def _es_adverbio(w: str, c: dict) -> bool:
    return w in _lista(c, "adverbios") or w.endswith("mente")


def _atribuible(previa: str, num: str, c: dict) -> bool:
    """¿El adjetivo que va justo detrás de `previa` es de `previa` (un nombre) y no de la palabra sustituida?
    En plural el adjetivo concuerda en número: su nombre acaba en -s («zanahorias ralladas»); tras un verbo
    («cocina tapadas»), un adverbio («previamente remojadas») o un infinitivo no hay nombre al que pegarlo."""
    w = previa.lower()
    if (w in _lista(c, "conjunciones") or w in _lista(c, "adverbios") or w in _lista(c, "copulas")
            or w in _lista(c, "preposiciones") or w.endswith("mente")):
        return False
    # un nombre acabado en -o/-os es masculino: un femenino no es suyo («trigo rellena»)
    if num == "pl":
        return w.endswith("s") and not w.endswith("os")
    return not re.search(r"[aeií]r$", w) and not w.endswith("o")


def _resto_inseguro(resto: str, num: str, c: dict) -> bool:
    """¿El resto vuelve sobre la palabra en femenino? Entonces sustituirla deja la frase agramatical.

    - pronombre pegado al verbo, con tilde o sin ella: «májalas», «hasta cubrirlas», «mezclándolas»;
    - pronombre delante del verbo: «no las revuelvas», «se las»;
    - atributo tras un verbo copulativo, también coordinado: «hasta que estén blandas», «quedar suaves y cremosas»;
    - un adjetivo o participio femenino que no es de un nombre vecino: «, previamente remojadas», «, escurridas»,
      «cocina tapadas». Es de un nombre si va justo detrás de él («zanahorias ralladas») o coordinado con otro que
      lo era («uvas frescas y jugosas»); tras un número o un determinante es un nombre («2 cucharadas»)."""
    fichas = _fichas(resto)
    sufijo = "las" if num == "pl" else "la"
    encl = set((c.get("encliticos") or {}).get(num, []))
    no_encl = _lista(c, "no_encliticos")
    procl = _lista(c, "procliticos_previos")
    copulas, adverbios = _lista(c, "copulas"), _lista(c, "adverbios")
    conj, preps = _lista(c, "conjunciones"), _lista(c, "preposiciones")
    dets = _determinantes(c)
    tras_verbo = re.compile(r"(?:[aeií]r|ndo)" + sufijo + "$")
    atribuidos = set()
    for i, (tipo, t) in enumerate(fichas):
        if tipo != "w":
            continue
        w = t.lower()
        previa = fichas[i - 1] if i else None
        pw = previa[1].lower() if previa and previa[0] == "w" else None
        if w not in no_encl and (w in encl or tras_verbo.search(w)
                                 or (len(w) >= 5 and w.endswith(sufijo) and any(a in w for a in _ACENTOS))):
            return True
        if w == sufijo and pw in procl:
            return True
        if w in copulas:
            vistas = 0
            for tipo2, t2 in fichas[i + 1:]:
                if tipo2 == "p" and t2 == ",":
                    continue
                if tipo2 != "w":
                    break
                w2 = t2.lower()
                if w2 in adverbios or w2 in conj:
                    continue
                if w2 in dets or w2 in preps:
                    break
                if _femenina(w2, num, c):
                    return True
                vistas += 1
                if vistas >= 4:
                    break
        if _adjetivo_femenino(t, num, c):
            if previa and (previa[0] == "n" or pw in dets):
                continue  # «2 cucharadas», «las picadas»: un nombre
            # el nombre del que es: el de justo antes, saltando adverbios («cebolla muy fina», «parte más gruesa»)
            j = i - 1
            while j >= 0 and fichas[j][0] == "w" and _es_adverbio(fichas[j][1].lower(), c):
                j -= 1
            ancla = fichas[j] if j >= 0 else None
            aw = ancla[1].lower() if ancla and ancla[0] == "w" else None
            if aw is not None and _atribuible(aw, num, c):
                atribuidos.add(i)
                continue
            if ancla and (ancla[1] == "," or aw in conj) and (j - 1) in atribuidos:
                atribuidos.add(i)
                continue
            return True
    return False


def _inicio_de_invariables(texto: str, hasta: int, c: dict):
    """Dónde empieza la racha de números y de palabras de `previos_invariables` justo delante de `hasta` («las 2
    habichuelas», «las demás habichuelas», «las otras dos»), o None si no hay ninguno. [ronda 2 del revisor]"""
    inv = _lista(c, "previos_invariables")
    inicio = None
    for _ in range(4):
        m = _FINAL_NUM_O_PALABRA.search(texto[:hasta])
        if not m or (m.group(2) is not None and m.group(2).lower() not in inv):
            break
        hasta = inicio = m.start()
    return inicio


def _proxima_mencion(texto: str, desde: int, fila: dict):
    """Dónde vuelve a nombrarse la palabra de `fila` (cualquier número) a partir de `desde`, o None."""
    mejor = None
    for forma in fila["de"]:
        m = _patron(forma).search(texto, desde)
        if m and (mejor is None or m.start() < mejor):
            mejor = m.start()
    return mejor


def _resto_hasta_la_proxima(texto: str, desde: int, fila: dict, siguientes) -> str:
    """El texto del que depende la concordancia: hasta que se vuelve a nombrar la palabra (desde ahí un pronombre es
    de esa mención, que hace su propia comprobación), siguiendo por los pasos siguientes si en este no vuelve."""
    corte = _proxima_mencion(texto, desde, fila)
    if corte is not None:
        return texto[desde:corte]
    partes = [texto[desde:]]
    for sig in siguientes or ():
        if not isinstance(sig, str):
            continue
        k = _proxima_mencion(sig, 0, fila)
        partes.append(sig if k is None else sig[:k])
        if k is not None:
            break
    return _ENTRE_PASOS.join(partes)


def _concordar(texto: str, ini: int, fin: int, num_idx: int, fila: dict, ocupados: list, siguientes=()):
    """Las ediciones de concordancia para sustituir [ini, fin) por un nombre de otro género, o None si no es seguro.

    Devuelve (ediciones, adjetivos_concordados, fin_del_sintagma), o («glosa», adjetivos, posición) si el resto vuelve
    sobre la palabra en femenino: la glosa lleva los adjetivos pegados al nombre, no los coordinados."""
    c = _concordancia(f"{fila['genero'][0]}>{fila['genero'][1]}")
    if not c:
        return None
    num = "pl" if num_idx else "sg"
    dets = (c.get("determinantes") or {}).get(num, {})
    ediciones = []
    # Un número o un invariable entre el determinante y el nombre («las 2 habichuelas», «las demás habichuelas»): el
    # determinante no es el vecino y no se concuerda; si es de los que cambian, no es seguro sustituir.
    salto = _inicio_de_invariables(texto, ini, c)
    if salto is not None:
        antes_de_salto = _palabra_previa(texto, salto)
        if antes_de_salto and antes_de_salto[0].lower() in dets:
            return None
    # Delante: el determinante («las»→«los»), el de antes («todas las»→«todos los») y la contracción («de la»→«del»).
    prev = _palabra_previa(texto, ini)
    if prev:
        palabra, p_ini, p_fin = prev
        nuevo = dets.get(palabra.lower())
        if nuevo:
            antes = _palabra_previa(texto, p_ini)
            contr = (c.get("contracciones") or {}).get(antes[0].lower()) if (antes and num == "sg" and palabra.lower() == "la") else None
            if contr:
                ediciones.append((antes[1], p_fin, _con_caja(antes[0], contr)))
            else:
                ediciones.append((p_ini, p_fin, _con_caja(palabra, nuevo)))
                if antes and dets.get(antes[0].lower()):
                    ediciones.append((antes[1], antes[2], _con_caja(antes[0], dets[antes[0].lower()])))
        elif _femenina(palabra, num, c):
            return None
    # Detrás: hasta 4 adjetivos seguidos (y la glosa que ya traiga el texto); el primer vecino que no sabemos concordar y parece femenino, se renuncia.
    adjetivos = []
    pos = fin
    for _ in range(4):
        # «habichuelas negras (frijoles negros) cocidas»: la glosa que ya traía el texto sobra y se sigue concordando
        ya = _glosa_ya_escrita(texto, pos, " ".join([fila["a"][num_idx], *[a.lower() for a in adjetivos]]))
        if ya:
            ediciones.append((pos, pos + ya, ""))
            pos += ya
        sig = _palabra_siguiente(texto, pos)
        if not sig:
            break
        nuevo = _adjetivo_concordado(sig[0], num, c)
        if not nuevo:
            if _femenina(sig[0], num, c):
                return None
            break
        ediciones.append((sig[1], sig[2], nuevo))
        adjetivos.append(nuevo)
        pos = sig[2]
    pegados, pos_pegados = list(adjetivos), pos
    # Coordinados: «cocidas y escurridas», «rojas, cocidas,». Tras «y» un femenino que no sabemos concordar y no es un
    # alimento («y blanditas») no se deja a medias: se glosa. Tras la coma, lo que no es adjetivo es otro elemento.
    conj = _lista(c, "conjunciones")
    for _ in range(4):
        m = _ENLACE.match(texto, pos)
        if not m:
            break
        enlace, x = m.group(1).lower(), m.group(2)
        if enlace != "," and (enlace not in conj or not adjetivos):
            break
        if x.lower() in _determinantes(c) or x.lower() in _lista(c, "preposiciones"):
            break  # «y una tostada», «y de postre»: otro sintagma
        if not _adjetivo_femenino(x, num, c):
            if enlace != "," and _femenina(x, num, c) and x.lower() not in _lista(c, "sustantivos"):
                return ("glosa", pegados, pos_pegados)
            break
        nuevo = _adjetivo_concordado(x, num, c)
        ediciones.append((m.start(2), m.end(2), nuevo))
        adjetivos.append(nuevo)
        pos = m.end(2)
    if any(_solapa(a, b, ocupados) for a, b, _ in ediciones):
        return None
    # En singular, la palabra también puede ser parte de un plural coordinado («…la habichuela, pica la cebolla y
    # mézclalas»): el resto se lee en los dos números.
    # Si es complemento («tortitas de habichuela apiladas; cocínalas»), el plural es del núcleo: no se lee.
    resto = _resto_hasta_la_proxima(texto, pos, fila, siguientes)
    plural_coordinado = num == "sg" and not _COMPLEMENTO.search(texto[:ini])
    if any(_resto_inseguro(resto, n, c) for n in ((num, "pl") if plural_coordinado else (num,))):
        return ("glosa", pegados, pos_pegados)
    return (ediciones, adjetivos, pos)


def _glosa_ya_escrita(texto: str, desde: int, destino: str) -> int:
    """Largo de « (destino)» justo en `desde` (0 si no está): el texto ya traía la glosa de lo que se sustituye."""
    g = re.match(r"\s*\(" + re.escape(destino) + r"\)", texto[desde:], re.IGNORECASE)
    return g.end() if g else 0


def _ediciones(texto: str, pais: str, ambito: str = "texto", siguientes=()):
    """(ediciones, inserciones, ocupados): lo que el léxico cambia en `texto`, sin aplicarlo."""
    ediciones, inserciones, ocupados = [], [], []
    glosadas = set()
    for forma, n, fila in _patrones(pais, ambito):
        for m in _patron(forma).finditer(texto):
            ini, fin = m.span()
            if _solapa(ini, fin, ocupados):
                continue
            destino = fila["a"][n]
            if fila["genero"][0] == fila["genero"][1]:
                # «ají morrón (pimiento)» → «pimiento»: la glosa que ya traía el texto sobra
                fin_total = fin + _glosa_ya_escrita(texto, fin, destino)
                # «queso blanco fresco» → «queso fresco», no «queso fresco fresco»
                sig = _palabra_siguiente(texto, fin_total)
                ultima = destino.split(" ")[-1].lower()
                if sig and " " in destino and sig[0].lower() == ultima:
                    fin_total = sig[2]
                ediciones.append((ini, fin_total, _con_caja(m.group(0), destino)))
                ocupados.append((ini, fin_total))
                continue
            r = _concordar(texto, ini, fin, n, fila, ocupados, siguientes)
            if r is None or r[0] == "glosa":
                # no es seguro sustituir: se glosa la primera vez, tras el sintagma («habichuelas negras (frijoles negros)»)
                ocupados.append((ini, fin))
                adjetivos, pos_glosa = (r[1], r[2]) if r is not None else ([], fin)
                if fila["de"][n] not in glosadas:
                    glosadas.add(fila["de"][n])
                    glosa = " ".join([destino, *[a.lower() for a in adjetivos]])
                    if not re.match(r"\s*\(" + re.escape(glosa) + r"\)", texto[pos_glosa:], re.IGNORECASE):
                        inserciones.append((pos_glosa, f" ({glosa})"))
                continue
            eds, _adjetivos, pos = r
            ediciones.append((ini, fin, _con_caja(m.group(0), destino)))
            ediciones.extend(eds)
            ocupados.append((min([ini] + [a for a, _, _ in eds]), max([pos] + [b for _, b, _ in eds])))
    return ediciones, inserciones, ocupados


def _inserciones_649(texto: str, pais: str, ocupados: list) -> list:
    """La glosa del lote 649 («guineo (plátano)»): primera aparición de cada canónico, fuera de lo ya ocupado.
    Espejo de `glosarTexto` (frontend `nombresDelPais.js`)."""
    try:
        from food_names_i18n import nombres_por_pais
        tabla = nombres_por_pais().get(pais) or {}
    except Exception:
        return []
    out = []
    for canon, local in sorted(tabla.items(), key=lambda kv: -len(kv[0])):
        for m in _patron(canon).finditer(texto):
            ini, fin = m.span()
            if _solapa(ini, fin, ocupados):
                continue
            ocupados.append((ini, fin))
            if not re.match(r"\s*\(" + re.escape(local) + r"\)", texto[fin:], re.IGNORECASE):
                out.append((fin, f" ({local})"))
            break
    return out


def _aplicar(texto: str, ediciones: list, inserciones: list) -> str:
    # Del final al principio; a igual posición, la sustitución antes que la inserción (la glosa queda delante).
    ops = [(a, 1, b, s) for a, b, s in ediciones] + [(p, 0, p, s) for p, s in inserciones]
    out = texto
    for a, _tipo, b, s in sorted(ops, key=lambda o: (o[0], o[1]), reverse=True):
        out = out[:a] + s + out[b:]
    return out


def localizar_texto(texto, pais):
    """Solo el léxico del país (sin la glosa del 649): sustituciones y, donde no es seguro, su glosa."""
    if not isinstance(texto, str) or not texto or not activo():
        return texto
    ediciones, inserciones, _ = _ediciones(texto, _pais(pais))
    if not ediciones and not inserciones:
        return texto
    return _aplicar(texto, ediciones, inserciones)


def texto_para_leer(texto, pais, siguientes=()):
    """El texto de un plato (nombre, descripción, ingrediente o paso) como lo lee un hispanohablante de `pais`:
    el léxico del país y, en la misma pasada, la glosa del 649 para lo que el léxico no cubre. `siguientes`: los
    pasos que vienen detrás (un paso puede volver sobre la palabra del anterior: «Májalas con un tenedor»)."""
    if not isinstance(texto, str) or not texto:
        return texto
    p = _pais(pais)
    ediciones, inserciones, ocupados = _ediciones(texto, p, "texto", siguientes) if activo() else ([], [], [])
    inserciones = inserciones + _inserciones_649(texto, p, list(ocupados))
    if not ediciones and not inserciones:
        return texto
    return _aplicar(texto, ediciones, inserciones)


def nombre_de_lista_para_leer(nombre, pais):
    """El nombre de un alimento de la lista de compras (sin glosa: la lista lleva la suya, `display_gloss_es`)."""
    if not isinstance(nombre, str) or not nombre or not activo():
        return nombre
    ediciones, inserciones, _ = _ediciones(nombre, _pais(pais))
    if not ediciones:
        return nombre
    return _aplicar(nombre, ediciones, [])


_CANTIDAD_Y_ENVASE = re.compile(r"^(\s*\d+(?:[.,]\d+)?(?:\s*[½¼¾⅓⅔])?\s+|\s*[½¼¾⅓⅔]\s+)(" + _LETRA + "+)")


def envase_para_leer(cantidad, pais):
    """La cantidad de la lista con el envase del país («1 funda (1 Lb)» → «1 bolsa (1 Lb)»). Solo el sustantivo que
    sigue a la cantidad, como `glossShoppingQty`: dentro del paréntesis viven marcas y rótulos reales."""
    if not isinstance(cantidad, str) or not cantidad or not activo():
        return cantidad
    envases = {}
    for fila in filas(pais):
        if fila.get("ambito") == "envase":
            for n in (0, 1):
                envases[fila["de"][n].lower()] = fila["a"][n]
    if not envases:
        return cantidad
    m = _CANTIDAD_Y_ENVASE.match(cantidad)
    if not m or m.group(2).lower() not in envases:
        return cantidad
    return m.group(1) + _con_caja(m.group(2), envases[m.group(2).lower()]) + cantidad[m.end():]


_CAMPOS_DE_LECTURA = ("name", "desc", "description", "ingredients", "recipe")
# Los pasos se leen encadenados (uno puede volver sobre la palabra del anterior); los ingredientes son renglones sueltos.
_CAMPOS_ENCADENADOS = ("recipe",)


def comida_para_leer(meal, pais):
    """Una COPIA de `meal` con los campos que se pintan leídos para `pais`; `meal` tal cual si no cambia nada.
    Solo para pintar: nunca se persiste ni se devuelve al motor (`ingredients_raw` y el resto no se tocan)."""
    if not isinstance(meal, dict):
        return meal
    cambios = {}
    for campo in _CAMPOS_DE_LECTURA:
        v = meal.get(campo)
        if isinstance(v, str):
            nuevo = texto_para_leer(v, pais)
        elif isinstance(v, list):
            encadenado = campo in _CAMPOS_ENCADENADOS
            nuevo = [texto_para_leer(x, pais, v[i + 1:] if encadenado else ()) for i, x in enumerate(v)]
        else:
            continue
        if nuevo != v:
            cambios[campo] = nuevo
    if not cambios:
        return meal
    return {**meal, **cambios}
