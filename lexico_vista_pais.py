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
    los adjetivos de detrás («rojas cocidas»→«rojos cocidos»); si algo más adelante sigue hablando de la palabra en
    femenino («májalas», «hasta que estén blandas») o un vecino no sabe concordar, no sustituye: GLOSA, como el 649;
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
_FIN_DE_CLAUSULA = re.compile(r"[.;:!?\n]")


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


def _clausula_insegura(resto: str, num: str, c: dict) -> bool:
    """¿La cláusula que sigue vuelve sobre la palabra en femenino? Pronombre enclítico («májalas», «añádela»),
    pronombre suelto («ellas») o atributo tras un verbo copulativo («hasta que estén blandas»)."""
    corte = _FIN_DE_CLAUSULA.search(resto)
    clausula = resto[: corte.start()] if corte else resto
    toks = re.findall(_LETRA + "+", clausula)
    sufijo = "las" if num == "pl" else "la"
    encl = set((c.get("encliticos") or {}).get(num, []))
    copulas = set(c.get("copulas") or [])
    adverbios = set(c.get("adverbios") or [])
    for i, tok in enumerate(toks):
        w = tok.lower()
        if w in encl or (len(w) >= 5 and w.endswith(sufijo) and any(a in w for a in _ACENTOS)):
            return True
        if w in copulas:
            for sig in toks[i + 1: i + 4]:
                if sig.lower() in adverbios:
                    continue
                if _femenina(sig, num, c):
                    return True
                break
    return False


def _concordar(texto: str, ini: int, fin: int, num_idx: int, fila: dict, ocupados: list):
    """Las ediciones de concordancia para sustituir [ini, fin) por un nombre de otro género, o None si no es seguro.

    Devuelve (ediciones, adjetivos_concordados, fin_del_sintagma)."""
    c = _concordancia(f"{fila['genero'][0]}>{fila['genero'][1]}")
    if not c:
        return None
    num = "pl" if num_idx else "sg"
    dets = (c.get("determinantes") or {}).get(num, {})
    ediciones = []
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
    if any(_solapa(a, b, ocupados) for a, b, _ in ediciones):
        return None
    if _clausula_insegura(texto[pos:], num, c):
        return ("glosa", adjetivos, pos)
    return (ediciones, adjetivos, pos)


def _glosa_ya_escrita(texto: str, desde: int, destino: str) -> int:
    """Largo de « (destino)» justo en `desde` (0 si no está): el texto ya traía la glosa de lo que se sustituye."""
    g = re.match(r"\s*\(" + re.escape(destino) + r"\)", texto[desde:], re.IGNORECASE)
    return g.end() if g else 0


def _ediciones(texto: str, pais: str, ambito: str = "texto"):
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
            r = _concordar(texto, ini, fin, n, fila, ocupados)
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


def texto_para_leer(texto, pais):
    """El texto de un plato (nombre, descripción, ingrediente o paso) como lo lee un hispanohablante de `pais`:
    el léxico del país y, en la misma pasada, la glosa del 649 para lo que el léxico no cubre."""
    if not isinstance(texto, str) or not texto:
        return texto
    p = _pais(pais)
    ediciones, inserciones, ocupados = _ediciones(texto, p) if activo() else ([], [], [])
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
            nuevo = [texto_para_leer(x, pais) for x in v]
        else:
            continue
        if nuevo != v:
            cambios[campo] = nuevo
    if not cambios:
        return meal
    return {**meal, **cambios}
