# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-47 · 2026-09-14] Lo que se hace con un alimento lo decide el alimento que QUEDA.

Tres pasos de la tercera prueba RD del dueño (plan d8b10b05) que un reemplazo de texto dejó sin sentido:

  · «Pásalos a agua fría para cortar la cocción… (~10-12 min a fuego medio)»: el tiempo por defecto de «El Toque de
    Fuego» cayó en el paso que ENFRÍA los huevos (`paso_frio`).
  · «Enjuaga 30 g de Casabe», «Cocina Casabe en agua hasta que ablanden»: la quinoa pasó a arroz integral por presupuesto
    y el arroz a casabe por la regla del arroz de noche; el nombre cambió dos veces y la técnica se quedó. El casabe es
    una torta de yuca ya hecha: se tuesta; hervirla la deshace (`tecnica_del_sustituto`).
  · «Aparte, revuelve queso blanco en una sartén»: el autofix del tope de huevo cambió el huevo de la arepa por queso y
    dejó el verbo del huevo (`redaccion_queso`).

Texto puro: no toca cantidades ni macros. Knobs `MEALFIT_TIMETEMP_SKIP_COLD_STEP`, `MEALFIT_CARB_SWAP_TECHNIQUE` y
`MEALFIT_SWAP_CHEESE_WORDING` (True). tooltip-anchor: P1-PLAN-LOTE-47-PASOS-SUSTITUCION
"""
from __future__ import annotations

import re
import unicodedata


def _knob(nombre: str) -> bool:
    try:
        from knobs import _env_bool
        return _env_bool(nombre, True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower()).encode("ascii", "ignore").decode("ascii")


# ─────────────── el paso que enfría no lleva tiempo de fuego ───────────────
_FRIO_RE = re.compile(r"\b(?:agua\s+(?:fria|helada|con\s+hielo)|hielo|enfri\w*|refriger\w*|nevera|congel\w*)\b")
_FUEGO_RE = re.compile(
    r"\b(?:cocin\w*|cuec\w*|cocer|hierv\w*|hirv\w*|herv\w*|hornea\w*|horno|sofri\w*|sofre\w*|salte\w*|frie\w*|freir|"
    r"frit\w*|dora\w*|tuest\w*|asa|asar|asad\w*|sancoch\w*|guis\w*|calienta\w*|plancha|parrilla|sarten|caldero|"
    r"microondas)\b")


def paso_frio(paso) -> bool:
    """¿El paso ENFRÍA y no calienta? «Pásalos a agua fría para cortar la cocción» no lleva «~10-12 min a fuego medio».
    Se mira el cuerpo, sin el rótulo del pilar (el rótulo dice «fuego» aunque el paso no lo tenga)."""
    if not _knob("MEALFIT_TIMETEMP_SKIP_COLD_STEP") or not isinstance(paso, str):
        return False
    t = _sa(paso)
    if ":" in t[:32]:
        t = t.split(":", 1)[1]
    return bool(_FRIO_RE.search(t)) and not _FUEGO_RE.search(t)


# ─────────────── la técnica del alimento que queda ───────────────
_HERVIR_RE = re.compile(r"\b(?:hierv\w*|herv\w*|cuec\w*|cocer|sancoch\w*|pon\w*\s+a\s+hervir|cocin\w*\b[^.;:]{0,60}?\ben\s+agua)")
_LAVAR_RE = re.compile(r"\b(?:enjuag\w*|lav[ae]\w*)\b", re.IGNORECASE)
_TOSTAR_CASABE = "Tuesta el casabe en un sartén caliente, un momento por cada lado, hasta que esté crujiente."
_PILAR_RE = re.compile(r"^\s*((?:mise en place|el toque de fuego|montaje)[^:]{0,24}:\s*)", re.IGNORECASE)
_FRASES_RE = re.compile(r"(?<=[.;])\s+")


def _partir(paso: str) -> tuple:
    m = _PILAR_RE.match(paso)
    return (m.group(1), paso[m.end():]) if m else ("", paso)


def _frases(cuerpo: str) -> list:
    return [f for f in _FRASES_RE.split(str(cuerpo).strip()) if f.strip()]


def _juntar(frases: list) -> str:
    out = []
    for f in frases:
        f = f.strip()
        if not f:
            continue
        if not out or out[-1].endswith("."):
            f = f[:1].upper() + f[1:]
        out.append(f)
    return " ".join(out)


def _ten_a_mano(frase: str) -> str:
    return _LAVAR_RE.sub(lambda m: "Ten a mano" if m.group(0)[:1].isupper() else "ten a mano", frase, count=1)


def tecnica_del_sustituto(pasos, sustituto) -> tuple:
    """Tras cambiar un carbohidrato por otro en los pasos, la técnica pasa a ser la del alimento NUEVO. Hoy sólo el casabe
    la necesita (batata, yuca, ñame y auyama se hierven igual que se hervía el arroz): la frase que lo hierve pasa a
    tostarlo, la que lo enjuaga pasa a tenerlo a mano, y sólo se tuesta una vez. `(pasos, cambios)`."""
    if (not _knob("MEALFIT_CARB_SWAP_TECHNIQUE") or not isinstance(pasos, list)
            or _sa(sustituto).strip() != "casabe"):
        return pasos, 0
    hay_hervir = any(isinstance(p, str) and any("casabe" in _sa(f) and _HERVIR_RE.search(_sa(f))
                                                for f in _frases(_partir(p)[1])) for p in pasos)
    out, cambios, tostado = [], 0, False
    for p in pasos:
        if not isinstance(p, str) or "casabe" not in _sa(p):
            out.append(p)
            continue
        pre, cuerpo = _partir(p)
        nuevas = []
        for f in _frases(cuerpo):
            fs = _sa(f)
            if "casabe" in fs and _HERVIR_RE.search(fs):
                cambios += 1
                if not tostado:
                    nuevas.append(_TOSTAR_CASABE)
                    tostado = True
                continue
            if "casabe" in fs and _LAVAR_RE.search(f):
                cambios += 1
                if not hay_hervir and not tostado:
                    nuevas.append(_TOSTAR_CASABE)
                    tostado = True
                else:
                    nuevas.append(_ten_a_mano(f))
                continue
            nuevas.append(f)
        cuerpo2 = re.sub(r"(?<=[a-záéíóúñ,] )Casabe\b", "casabe", _juntar(nuevas))
        out.append(pre + cuerpo2 if cuerpo2 else p)
    return out, cambios


# ─────────────── el desalado es una CLÁUSULA, no la frase ───────────────
# [P1-PLAN-LOTE-48 · 2026-09-14] Plan 358a2cdf: el tope de sodio cambió el arenque del «Locrio de arenque» por pescado
# fresco y el limpiador del desalado borró la FRASE entera «Agrega el arenque (ya desalado y en trozos), remuévelo… y
# añade el arroz blanco». Se fueron el pescado y el ARROZ del locrio; el pescado volvió como «complemento» después de
# servir y el arroz se quedó sin paso. El participio sale; la frase sólo se va si su verbo es desalar o remojar.
_DESALADO_PAREN_RE = re.compile(r"\(\s*(?:ya\s+|bien\s+|previamente\s+)?desalad[oa]s?\s*(?:y\s+|,\s*)?", re.IGNORECASE)
_DESALADO_ADJ_RE = re.compile(r",?\s+(?:ya\s+|bien\s+|previamente\s+)?desalad[oa]s?(?![a-z])", re.IGNORECASE)


def quitar_clausula_desalado(texto) -> str:
    """«el arenque (ya desalado y en trozos)» → «el arenque (en trozos)»; «el bacalao desalado» → «el bacalao». No toca
    los VERBOS («Desala el arenque…»): esa frase entera la quita el limpiador de siempre."""
    if not _knob("MEALFIT_DESALT_CLAUSE_ONLY") or not isinstance(texto, str):
        return texto
    t = _DESALADO_PAREN_RE.sub("(", texto)
    t = re.sub(r"\(\s*\)", "", t)
    t = _DESALADO_ADJ_RE.sub("", t)
    return re.sub(r"\s{2,}", " ", t).replace(" ,", ",").replace("( ", "(")


# ─────────────── [P1-PLAN-LOTE-49 · 2026-09-14] el paso breve y el enlatado que ya no está ───────────────
_BREVE_RE = re.compile(r"\b(?:brevemente|un\s+momento|unos\s+segundos|un\s+par\s+de\s+segundos|rapidamente|al\s+instante)\b")
TIEMPO_BREVE = "1-2 min a fuego medio"      # el contrato pide un tiempo concreto; éste no contradice «brevemente»


def paso_breve(paso) -> bool:
    """¿El paso ya dice que dura poco? «Tuesta el casabe brevemente» no lleva «(~10-12 min a fuego medio)» (plan a059d7bb:
    diez minutos queman un casabe): lleva `TIEMPO_BREVE`. Knob `MEALFIT_TIMETEMP_SKIP_BRIEF_STEP`."""
    if not _knob("MEALFIT_TIMETEMP_SKIP_BRIEF_STEP") or not isinstance(paso, str):
        return False
    return bool(_BREVE_RE.search(_sa(paso)))


def tiene_fuego(paso) -> bool:
    """¿El paso calienta? Se mira el cuerpo, sin el rótulo del pilar."""
    if not isinstance(paso, str):
        return False
    t = _sa(paso)
    if ":" in t[:32]:
        t = t.split(":", 1)[1]
    return bool(_FUEGO_RE.search(t))


_ESCURRIDO_RE = re.compile(r",?\s*(?:ya\s+)?escurrid[oa]s?\s*,?\s*", re.IGNORECASE)
_DESHECHO_RE = re.compile(r"\bdeshech[oa]s?\s+en\s+trozos\b", re.IGNORECASE)
_LIQUIDO_LATA_RE = re.compile(
    r"\s*,?\s*(?:y\s+|con\s+)?(?:un\s+pellizco|un\s+chorrito|un\s+poco|unas\s+gotas|una\s+cucharada)\s+del?\s+"
    r"(?:l[ií]quido|jugo|aceite|agua|caldo)\s+de\s+la\s+lata(?:\s+de\s+[^,.;]+?)?"
    r"(?:\s+si\s+quieres(?:\s+darle)?\s+(?:m[aá]s\s+)?sabor)?(?=[,.;])", re.IGNORECASE)


def quitar_clausula_enlatado(texto) -> str:
    """Tras cambiar un enlatado por uno fresco (sardinas → filete de pescado, por el tope de sodio), el paso no puede seguir
    escurriéndolo ni usando «el líquido de la lata»: «Incorpora filete de pescado blanco, ya escurridas, deshechas en trozos
    grandes» → «Incorpora filete de pescado blanco en trozos grandes»; «… y un pellizco del líquido de la lata de filete de
    pescado blanco si quieres darle sabor, …» se va (plan a059d7bb). Knob `MEALFIT_CANNED_SWAP_CLAUSES`."""
    if not _knob("MEALFIT_CANNED_SWAP_CLAUSES") or not isinstance(texto, str):
        return texto
    t = _LIQUIDO_LATA_RE.sub("", texto)
    t = _ESCURRIDO_RE.sub(" ", t)
    t = _DESHECHO_RE.sub("en trozos", t)
    t = re.sub(r"\s{2,}", " ", t)
    return re.sub(r"\s+([,.;])", r"\1", t).strip()


def limpiar_pasos_enlatado(meal) -> int:
    """`quitar_clausula_enlatado` en todos los pasos del plato. Devuelve cuántos cambió."""
    rec = meal.get("recipe") if isinstance(meal, dict) else None
    if not isinstance(rec, list):
        return 0
    nuevos = [quitar_clausula_enlatado(s) if isinstance(s, str) else s for s in rec]
    n = sum(1 for a, b in zip(rec, nuevos) if a != b)
    if n:
        meal["recipe"] = nuevos
    return n


# ─────────────── lo que se hacía con el huevo no se hace con el queso ───────────────
_QUESOS = ("queso", "mozzarella", "cheddar", "gouda", "ricotta", "cottage", "parmesano", "requeson")
_QUESOS_DE_FREIR = ("queso blanco", "queso de freir", "queso fresco", "queso paisa", "queso de hoja", "halloumi")
_VERBOS_HUEVO = {  # verbo del huevo → (queso que se dora, queso que no)
    "revuelve": ("dora", "incorpora"), "revolver": ("dorar", "incorporar"),
    "bate": ("desmenuza", "mezcla"), "batir": ("desmenuzar", "mezclar"),
    "cuaja": ("dora", "incorpora"), "cuajar": ("dorar", "incorporar"),
    "escalfa": ("dora", "incorpora"), "escalfar": ("dorar", "incorporar"),
}
_ADJ_HUEVO = r"(?:revuelt|batid|escalfad|cuajad)[oa]s?"
_ACENTOS = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}


def es_queso(nombre) -> bool:
    t = _sa(nombre)
    return any(h in t for h in _QUESOS)


def _flex(s: str) -> str:
    return "".join(_ACENTOS.get(c, re.escape(c)) for c in _sa(s))


def redaccion_queso(texto, queso) -> str:
    """«revuelve queso blanco» → «dora el queso blanco»; «queso blanco revuelto» → «queso blanco dorado». Un queso que no
    se dora (cottage, mozzarella…) se incorpora o se mezcla. Sólo toca verbos del huevo pegados al nombre del queso."""
    if not _knob("MEALFIT_SWAP_CHEESE_WORDING") or not isinstance(texto, str) or not es_queso(queso):
        return texto
    q = str(queso or "").strip()
    if not q:
        return texto
    de_freir = any(h in _sa(q) for h in _QUESOS_DE_FREIR)
    qp = _flex(q)

    def _verbo(m):
        v = m.group("v")
        nuevo = _VERBOS_HUEVO[_sa(v)][0 if de_freir else 1]
        if v[:1].isupper():
            nuevo = nuevo[:1].upper() + nuevo[1:]
        return f"{nuevo} {m.group('art') or 'el '}"

    t = re.sub(r"\b(?P<v>" + "|".join(_VERBOS_HUEVO) + r")\s+(?P<art>(?:el|la|los|las)\s+)?(?=" + qp + r"\b)",
               _verbo, texto, flags=re.IGNORECASE)
    t = re.sub(r"(?P<q>" + qp + r")\s+" + _ADJ_HUEVO + r"\b",
               lambda m: m.group("q") + (" dorado" if de_freir else ""), t, flags=re.IGNORECASE)
    return t
