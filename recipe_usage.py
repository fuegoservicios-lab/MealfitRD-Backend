# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-25 · 2026-09-12] C4 del plan de pendientes (H8 de la auditoría del 09-11): asignación paso↔ingrediente
en la receta congelada, y V6 de «puede repartir» a suma exacta.

## El problema, medido antes de escribir

La receta congelada (la biblioteca de RD, `recipe_library_<cc>_v1.json`: 193 recetas, 867 pasos) se escribió A PROPÓSITO sin cantidades de
ingrediente («valen para cualquier porción»). El escáner culinario decide cuánto usa cada paso leyendo NÚMEROS en el texto
(V6, V7a, V7e): sobre estos pasos no ve ninguno, así que sobre la receta congelada nunca dice nada — ni «coherente» ni
«no evaluable»: calla. Medido materializando las 193 como plato y pasándolas por el escáner de hoy: 6 hallazgos en total
(V1 4, V3 1, V5 1) y NINGUNO de cantidad. No porque las recetas repartan bien, sino porque nadie podía mirarlo: «la mitad
del aceite» … «la otra mitad del aceite» … «la otra mitad del aceite» —tres mitades— pasa limpio, y una auyama comprada
que ningún paso toca sólo la ve V3 porque da la casualidad de que tampoco la nombra nadie. La cuarta clase de las notas
humanas del 09-07 («la mitad del ajo queda sin usar») no tenía detector.

## Lo que hace

1. **Asigna** a cada paso los constituyentes de SU plantilla que usa, con la FRACCIÓN de lo comprado:
   `usa[i] = [{"ingredient_id", "fraccion"}]`, paralelo a `pasos[i]` (los pasos siguen siendo texto: viajan tal cual a
   `meal["recipe"]` y a todo lo que lo lee). El vocabulario es el de la plantilla —un mundo cerrado de 3-8 alimentos—,
   no el catálogo: dentro de «Pinchos de pollo…», «el pollo» sólo puede ser la pechuga. Un token que dos constituyentes
   comparten («aceite» con dos aceites) no decide: se declara ambiguo.
2. **Reparte** por las pistas del texto: «la mitad de la sal» → 0,5; «la otra mitad» / «el resto» → lo que queda; «parte
   de», «una pizca de», «un poco de» → una parte SIN cifra, que se estima a partes iguales y queda marcada `estimada`;
   sin pista alguna, toda la cantidad entra en el PRIMER paso que la nombra y las menciones siguientes son referencias
   («el pollo está seguro cuando el termómetro…»). «Resérvala», «queda para untar», «corrige la sal» no consumen.
3. **Cuenta** por constituyente: la suma de fracciones ha de ser 1. Σ = 0 es «comprado y sin usar» (V3); 0 < Σ < 1 es
   «usado a medias» —la cuarta clase que las notas humanas del 09-07 destaparon y el comentario de V7 dejó anotada sin
   implementar—; Σ > 1 es «pide más de lo que hay» (la dirección de V6). Un alimento del catálogo en los pasos que NO es
   constituyente es «fuera de plantilla» (la familia de V5; se inventaría, no se duplica al escáner).
4. **Ata** la asignación al TEXTO con `pasos_hash`. Si alguien edita un paso, la asignación caduca sola y
   `usage_for_template` devuelve `None`: la «firma que caduca» de la auditoría, aplicada mecánicamente y sin ceremonia.

## Dónde engancha

`culinary_coherence.culinary_contract_scan`: en una comida de receta congelada (`_recipe_source == "library"`) con
asignación vigente, V3/V6/V7a/V7e dejan de ADIVINAR por el texto y leen las cuentas (`cuentas_para_comida`); con el knob
`MEALFIT_RECIPE_USAGE_EXACT=False` vuelve la heurística. Medido en la flota al escribirlo: 0 comidas congeladas en 30 días
(el día determinista sigue en canario) — el enganche es para cuando las haya; hoy el valor está en el inventario sobre
la biblioteca (`scripts/asignar_uso_pasos.py`) y en que una receta editada no pueda quedarse con una asignación vieja.

Puro: sin base de datos, sin red. Fail-open en el enganche de runtime (cualquier excepción ⇒ heurística de siempre).
tooltip-anchor: P1-PLAN-LOTE-25-RECIPE-USAGE
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent / "data" / "registry"
SCHEMA_VERSION = 1
#: Σ de fracciones que se da por «1». «⅓ + ⅓ + ⅓» redondeado a 4 decimales no llega a 1,0 exacto.
TOLERANCIA = 0.01
ESTADOS = ("exacta", "estimada", "revisar")
#: Tipos de hallazgo del inventario. Los tres primeros son los que el escáner traduce a violaciones.
HALLAZGOS = ("sin_uso", "a_medias", "sobre_asignado", "ambigua", "fuera_de_plantilla", "sin_uso_condimento")
#: Traducción hallazgo → check del escáner culinario (los demás tipos se inventarían y no se duplican al escáner:
#: `fuera_de_plantilla` ya lo acusa V5 por texto, y `ambigua`/`sin_uso_condimento` son del inventario, no del plato).
CHECK_POR_HALLAZGO = {"sin_uso": "V3", "a_medias": "V7a", "sobre_asignado": "V6"}

_STOP = frozenset({"de", "del", "la", "el", "los", "las", "en", "con", "sin", "y", "a", "al", "o", "u", "para", "por",
                   "un", "una", "unos", "unas", "su", "sus"})


def _norm(s) -> str:
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


_FRONTERA = re.compile(r"[.;:,()!?¡¿]")


def _norm_clausulas(s) -> str:
    """Como `_norm`, pero `.;:,()` quedan como « , »: la pista de reparto no cruza una cláusula («la mitad del huevo.
    Añade sal» no dice nada de la sal). Las formas de los alimentos casan igual: son palabras con frontera."""
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    t = _FRONTERA.sub(" , ", t)
    return " ".join("".join(c if (c.isalnum() or c == ",") else " " for c in t).split())


def pasos_hash(pasos: Iterable) -> str:
    """sha256[:16] del TEXTO de los pasos, en orden. Cambia si cambia una letra: a eso se ata la asignación."""
    return hashlib.sha256(json.dumps([str(p) for p in (pasos or [])], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Vocabulario cerrado de la plantilla
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _tok_pat(tok: str) -> str:
    """Patrón de un token con plural opcional: «huevo» casa «huevos»; «habichuelas» casa «habichuela»."""
    if len(tok) > 3 and tok.endswith("s"):
        return re.escape(tok[:-1]) + r"(?:e?s)?"
    return re.escape(tok) + r"(?:e?s)?"


def _formas(constituents: list) -> tuple[dict, dict]:
    """`{ingredient_id: [regex]}` con el nombre completo y los tokens que SÓLO ese constituyente tiene en la plantilla,
    y `{token: [ingredient_ids]}` con los tokens que dos o más comparten (los ambiguos: no deciden)."""
    nombres: dict = {}
    for c in constituents or []:
        cid = str(c.get("ingredient_id") or "")
        n = _norm(c.get("canonical") or c.get("name") or "")
        if cid and n:
            nombres[cid] = n
    dueños: dict = {}
    for cid, n in nombres.items():
        for t in n.split():
            if t in _STOP or len(t) < 3:
                continue
            dueños.setdefault(_tok_pat(t), set()).add(cid)
    formas: dict = {}
    ambiguos: dict = {}
    for cid, n in nombres.items():
        pats = [r"\b" + r"\s+".join(_tok_pat(t) for t in n.split()) + r"\b"]
        for t in n.split():
            if t in _STOP or len(t) < 3:
                continue
            tp = _tok_pat(t)
            if len(dueños.get(tp, ())) == 1:
                pats.append(r"\b" + tp + r"\b")
            else:
                ambiguos.setdefault(tp, sorted(dueños[tp]))
        formas[cid] = [re.compile(p) for p in pats]
    return formas, ambiguos


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Pistas de reparto: qué dice el texto ANTES (y justo después) de nombrar el alimento
# ─────────────────────────────────────────────────────────────────────────────────────────────

_ART = r"(?:\s+(?:la|el|los|las|del|de|de\s+la|de\s+los|de\s+las))?"
#: Cola que permite la cadena «el resto del morrón y la cebolla»: hasta 3 tokens y una conjunción con su artículo.
_COLA = r"(?:\s+[a-z]+){0,3}?(?:\s+y" + r"(?:\s+(?:la|el|los|las|del))?)?\s*$"
_DE = r"\s+de(?:l|\s+la|\s+los|\s+las)?"
#: (pista, regex sobre la VENTANA previa). El orden importa: «otra mitad» antes que «mitad».
_PISTAS = (
    ("otra_mitad", re.compile(r"\b(?:la\s+)?otra\s+mitad" + _DE + _COLA)),
    ("mitad", re.compile(r"\b(?:la\s+)?(?:mitad|media)" + _DE + _COLA)),
    ("resto", re.compile(r"\b(?:el|lo)\s+(?:resto|restante|sobrante)" + _DE + _COLA)),
    ("resto", re.compile(r"\b(?:el|lo)\s+que\s+(?:queda|quede|sobra|sobre)" + _DE + _COLA)),
    ("resto", re.compile(r"\b(?:el|la|los|las)\s+[a-z]+\s+restantes?\s*$")),
    # [P1-PLAN-LOTE-61 · 2026-09-15] (C4) «el último tercio del aceite» / «el último cuarto» / «el tercer tercio» CIERRAN
    # (toman lo que queda, como «el resto»), y «otro tercio» / «otro cuarto» son fracciones fijas como «un tercio». Las
    # recetas del dueño del 14-sep reparten el aceite así y el parser leía 1/3 y 3/4: `a_medias`, cuando suman 1.
    # Antes que «tercio»/«cuarto» fijos: la frase que cierra contiene la palabra. tooltip-anchor: P1-PLAN-LOTE-61-TERCIOS
    ("resto", re.compile(r"\b(?:el|la)\s+(?:ultim[oa]|tercer[oa]?)\s+(?:tercio|cuarto|cuarta\s+parte)" + _DE + _COLA)),
    ("dos_tercios", re.compile(r"\bdos\s+tercios" + _DE + _COLA)),
    ("tercio", re.compile(r"\b(?:un|otro)\s+tercio" + _DE + _COLA)),
    ("cuarto", re.compile(r"\b(?:(?:un|otro)\s+cuarto|(?:la|otra)\s+cuarta\s+parte)" + _DE + _COLA)),
    ("parte", re.compile(r"\b(?:una\s+)?parte" + _DE + _COLA)),
    ("parte", re.compile(r"\buna?\s+pizca" + _DE + _COLA)),
    ("parte", re.compile(r"\bun\s+poco" + _DE + _COLA)),
    ("parte", re.compile(r"\buna?\s+poquit[oa]" + _DE + _COLA)),
    ("parte", re.compile(r"\buna\s+porcion" + _DE + _COLA)),
    ("parte", re.compile(r"\bun\s+chorrito" + _DE + _COLA)),
    ("parte", re.compile(r"\bun\s+toque" + _DE + _COLA)),
    ("parte", re.compile(r"\bunas?\s+(?:cucharadas?|cucharaditas?|gotas?)" + _DE + _COLA)),
    # no consumen: apartar para después, ajustar el punto, o la ausencia
    ("reserva", re.compile(r"\b(?:reserva|reservar|aparta|apartar|guarda|guardar|separa|separar)" + _ART + r"\s*$")),
    ("referencia", re.compile(r"\b(?:corrige|corregir|rectifica|rectificar|ajusta|ajustar|prueba|probar|comprueba)"
                              r"(?:\s+de|\s+el\s+punto\s+de)?" + _ART + r"\s*$")),
    ("referencia", re.compile(r"\bsin" + _ART + r"\s*$")),
)
_FIJAS = {"mitad": 0.5, "tercio": 1.0 / 3.0, "dos_tercios": 2.0 / 3.0, "cuarto": 0.25}
_CIERRAN = {"otra_mitad", "resto"}
_ABIERTAS = {"parte"}
_NO_CONSUMEN = {"reserva", "referencia"}
#: Lo que sigue al alimento y dice que NO se consume aquí: «la otra mitad del aguacate queda para untar el pan».
_DESPUES_RESERVA = re.compile(
    r"^(?:\s+[a-z]+){0,3}?\s+(?:"
    r"quedan?\s+(?:para|aparte|reservad[oa]s?)|"                       # «queda para untar el pan»
    r"se\s+(?:reserva|guarda|aparta)|"                                  # «se reserva para el final»
    r"(?:la|lo|las|los)\s+(?:dejas|reservas|guardas|apartas)\b|"       # «la dejas batida aparte»
    r"reserval[oa]s?|apartal[oa]s?|guardal[oa]s?"                         # «resérvala tapada»
    r")\b")
#: «el aceite de oliva restante», «la sal que queda», «el ajo sobrante»: la pista va DETRÁS del alimento.
_DESPUES_RESTO = re.compile(r"^(?:\s+[a-z]+){0,2}?\s+(?:restantes?|sobrantes?|que\s+(?:queda|quede|sobra|sobre))\b")
_VENTANA = 48
_CADENA = re.compile(r"\s+y\s+(?:la|el|los|las|del)?\s*$")
_SOLO_LISTA = re.compile(r"\s*(?:(?:el|la|los|las|y|con)\s+|[a-z]+\s+){0,5}")


def _pista(norm_paso: str, ini: int, fin: int) -> tuple:
    """`(pista, encadenada)`: la pista de reparto de ESTA mención, leída en su cláusula. `encadenada` si llegó por una
    conjunción («la mitad de la sal y el orégano»): una fracción FIJA heredada así no cuenta salvo que el alimento tenga
    otra mención con pista — «el resto del morrón y la cebolla» sí reparte a los dos, «la mitad de la sal y el orégano» no
    parte el orégano."""
    despues = norm_paso[fin:fin + 40]
    if "," in despues:
        despues = despues[:despues.index(",")]
    if _DESPUES_RESERVA.match(despues):
        return "reserva", False
    antes = norm_paso[max(0, ini - _VENTANA):ini]
    clausula = antes[antes.rfind(",") + 1:] if "," in antes else antes
    for nombre, rx in _PISTAS:
        if rx.search(clausula):
            return nombre, bool(_CADENA.search(clausula))
    # «el resto del ají, el ajo y el cilantro»: si la cláusula sólo trae artículos, «y» y un par de nombres, la pista
    # está en la cláusula ANTERIOR y llega encadenada (una fija así no cuenta sola; una que cierra sí reparte)
    if "," in antes and _SOLO_LISTA.fullmatch(clausula):
        previa = antes[:antes.rfind(",")]
        previa = previa[previa.rfind(",") + 1:] if "," in previa else previa
        for nombre, rx in _PISTAS:
            if nombre in _NO_CONSUMEN:
                continue
            if rx.search(previa + " "):
                return nombre, True
    if _DESPUES_RESTO.match(despues):
        return "resto", False
    return None, False


def _menciones(pasos: list, formas: dict) -> dict:
    """`{ingredient_id: [(paso_idx, pista)]}` en orden de aparición; una mención por (paso, constituyente, span)."""
    out: dict = {cid: [] for cid in formas}
    for i, paso in enumerate(pasos or []):
        n = _norm_clausulas(paso)
        for cid, rxs in formas.items():
            spans = []
            for rx in rxs:
                for m in rx.finditer(n):
                    if not any(a <= m.start() < b for a, b in spans):
                        spans.append((m.start(), m.end()))
            for a, b in sorted(spans):
                pista, encadenada = _pista(n, a, b)
                out[cid].append((i, pista, encadenada))
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# El reparto
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _repartir(menciones: list) -> tuple[list, bool, Optional[str], Optional[str]]:
    """De las menciones de UN constituyente a `([(paso_idx, fraccion)], estimada, defecto, motivo)`.

    · sin pistas ⇒ toda la cantidad entra en el PRIMER paso que lo nombra; el resto son referencias;
    · fijas («mitad», «tercio») suman lo que dicen; una que CIERRA («otra mitad», «resto») toma lo que queda;
    · abiertas («parte de», «una pizca de») se estiman a partes iguales sobre lo que no está fijado;
    · una mención sin pista ANTES de una que cierra es una parte abierta (se usó algo, después «el resto»);
    · sólo fijas y no llegan a 1 ⇒ `a_medias`; fijas por encima de 1 ⇒ `sobre_asignado`."""
    usos = [(i, p, enc) for i, p, enc in menciones if p not in _NO_CONSUMEN]
    if not usos:
        return [], False, "sin_uso", "comprado y ningún paso lo nombra"
    con_pista = sum(1 for _, p, _e in usos if p is not None)
    # una fracción FIJA heredada por conjunción, sin otra mención con pista, es del alimento de al lado
    usos = [(i, (None if (enc and p in _FIJAS and con_pista < 2) else p)) for i, p, enc in usos]
    if all(p is None for _, p in usos):
        return [(usos[0][0], 1.0)], False, None, None
    cierres = [k for k, (_, p) in enumerate(usos) if p in _CIERRAN]
    idx_cierre = cierres[0] if cierres else None
    fijas = [(k, _FIJAS[p]) for k, (_, p) in enumerate(usos) if p in _FIJAS]
    abiertas = [k for k, (_, p) in enumerate(usos) if p in _ABIERTAS]
    primer_cued = next(k for k, (_, p) in enumerate(usos) if p is not None)
    previas = [k for k, (_, p) in enumerate(usos) if p is None and k < primer_cued]
    if idx_cierre is not None and not fijas and not abiertas and previas:
        # «Licúa … el queso blanco» … «Rellena con el resto del queso blanco»: la primera mención sin pista fue una parte
        abiertas = [previas[0]]
    asignado = sum(f for _, f in fijas)
    fr: dict = {k: f for k, f in fijas}
    estimada = False
    defecto = motivo = None
    if asignado > 1.0 + TOLERANCIA:
        defecto, motivo = "sobre_asignado", f"las fracciones fijas del texto suman {asignado:g} de lo comprado"
    elif len(cierres) >= 2:
        defecto, motivo = "sobre_asignado", ("«el resto» / «la otra mitad» aparece dos veces: la receta reparte más "
                                             "de lo que compra")
    resto = max(0.0, 1.0 - asignado)
    if idx_cierre is None and not abiertas and previas and resto > 0:
        # «Hierve la yuca» … «la mitad de la masa de yuca»: el TODO entró antes, sin pista; las fracciones posteriores
        # son sub-usos del producto intermedio, no otra compra. Lo que las fijas no nombran se queda donde entró.
        fr[previas[0]] = fr.get(previas[0], 0.0) + resto
        estimada = True
        resto = 0.0
    if idx_cierre is not None:
        partes = abiertas + [idx_cierre]
        if abiertas:
            estimada = True
        cuota = resto / len(partes)
        for k in partes:
            fr[k] = fr.get(k, 0.0) + cuota
    elif abiertas:
        # «Añade sal» … «un poquito de la sal»: lo que entró sin pista y las partes sin cifra se reparten a partes iguales
        partes = ([previas[0]] if previas else []) + abiertas
        estimada = True
        cuota = resto / len(partes)
        for k in partes:
            fr[k] = fr.get(k, 0.0) + cuota
    elif resto > TOLERANCIA:
        defecto, motivo = "a_medias", f"las pistas del texto suman {asignado:g} de lo comprado y nada usa el resto"
    out: dict = {}
    for k, f in fr.items():
        if f > 0:
            i = usos[k][0]
            out[i] = out.get(i, 0.0) + f
    return [(i, round(f, 4)) for i, f in sorted(out.items())], estimada, defecto, motivo


def derivar_uso(pasos: list, constituents: list, index: Optional[dict] = None, *, condimentos=None) -> dict:
    """La asignación de UNA receta: `{"pasos_hash", "estado", "usa", "cuentas", "hallazgos"}`.

    `index` (opcional) es el índice culinario del catálogo: con él se detectan alimentos del catálogo que los pasos
    nombran y la plantilla no compra («fuera de plantilla»). `condimentos` es un predicado `nombre → bool` para no
    contar como defecto un condimento que ningún paso nombra (la sal «al gusto» de V3)."""
    pasos = [str(p) for p in (pasos or [])]
    formas, ambiguos = _formas(constituents)
    nombres = {str(c.get("ingredient_id")): (c.get("canonical") or c.get("name") or str(c.get("ingredient_id")))
               for c in (constituents or []) if c.get("ingredient_id")}
    menciones = _menciones(pasos, formas)
    usa: list = [[] for _ in pasos]
    cuentas: dict = {}
    hallazgos: list = []
    estimada_alguna = False
    for cid in formas:
        reparto, estimada, defecto, motivo = _repartir(menciones.get(cid) or [])
        estimada_alguna = estimada_alguna or estimada
        for i, f in reparto:
            item = {"ingredient_id": cid, "fraccion": f}
            if estimada:
                item["estimada"] = True
            usa[i].append(item)
        cuentas[cid] = round(sum(f for _, f in reparto), 4)
        if defecto == "sin_uso":
            es_cond = bool(condimentos and condimentos(nombres.get(cid, "")))
            # ¿lo nombra un token AMBIGUO (dos constituyentes lo comparten)? Entonces no es que falte: es que no se decide.
            amb = [tp for tp, cids in ambiguos.items() if cid in cids
                   and any(re.search(r"\b" + tp + r"\b", _norm(p)) for p in pasos)]
            if amb:
                hallazgos.append({"tipo": "ambigua", "ingredient_id": cid, "alimento": nombres.get(cid),
                                  "detalle": f"el texto dice «{amb[0]}» y en esta plantilla lo comparten "
                                             f"{', '.join(nombres.get(x, x) for x in ambiguos[amb[0]])}"})
            else:
                hallazgos.append({"tipo": "sin_uso_condimento" if es_cond else "sin_uso", "ingredient_id": cid,
                                  "alimento": nombres.get(cid), "detalle": "comprado y ningún paso lo nombra"})
        elif defecto:
            hallazgos.append({"tipo": defecto, "ingredient_id": cid, "alimento": nombres.get(cid), "detalle": motivo})
    if index:
        hallazgos.extend(_fuera_de_plantilla(pasos, constituents, index, condimentos))
    # `fuera_de_plantilla` NO decide el estado: es la familia de V5 y el matcher global confunde técnica con alimento
    # («al sofrito» → Sofrito, «hasta que el agua salga clara» → Clara de huevo: 50 acusaciones medidas, casi todas así).
    graves = {"sin_uso", "a_medias", "sobre_asignado", "ambigua"}
    if any(h["tipo"] in graves for h in hallazgos):
        estado = "revisar"
    elif estimada_alguna:
        estado = "estimada"
    else:
        estado = "exacta"
    return {"pasos_hash": pasos_hash(pasos), "estado": estado, "usa": usa, "cuentas": cuentas, "hallazgos": hallazgos}


def _fuera_de_plantilla(pasos: list, constituents: list, index: dict, condimentos=None) -> list:
    """Alimentos que los pasos nombran y la plantilla no compra: lo que V5 diría del plato materializado.

    Se REUTILIZA V5 (`_v5_paso_usa_lo_que_no_esta`) en vez de pasar el matcher global crudo: medido antes de decidirlo,
    el matcher crudo acusaba 50 veces sobre las 193 recetas y casi todas eran técnica leída como alimento («al sofrito»
    → Sofrito ×18, «hasta que el agua salga clara» → Clara de huevo, «cuajada»); V5, con sus guardas de especificidad
    y de dirección, acusa 1. Un inventario con 49 falsos no es un inventario."""
    try:
        from culinary_coherence import _v5_paso_usa_lo_que_no_esta
    except Exception:
        return []
    ings = [f"{float(c.get('grams') or 0):g} g de {c.get('canonical') or c.get('name')}" for c in (constituents or [])
            if (c.get("canonical") or c.get("name"))]
    meal = {"meal": "Plato", "name": "plantilla", "ingredients": ings, "recipe": list(pasos)}
    out = []
    for v in _v5_paso_usa_lo_que_no_esta(None, meal, index) or []:
        out.append({"tipo": "fuera_de_plantilla", "alimento": v.get("food"),
                    "detalle": f"V5 sobre la plantilla materializada: {v.get('detail')}"})
    return out


def validar_uso(entry: dict, pasos: list, constituents: list) -> list:
    """Problemas ESTRUCTURALES de una asignación guardada frente a la receta y la plantilla de hoy. `[]` = válida.

    No re-deriva: comprueba que el hash ata al texto, que hay una lista por paso, que cada id es constituyente y que
    `cuentas` es la suma de `usa`. En `exacta`/`estimada` además exige Σ = 1 por constituyente; en `revisar` la Σ ≠ 1 ES
    el hallazgo y no un problema del fichero."""
    problemas = []
    if not isinstance(entry, dict):
        return ["la entrada no es un dict"]
    if entry.get("pasos_hash") != pasos_hash(pasos):
        problemas.append("pasos_hash no coincide con el texto actual de la receta (asignación caducada)")
    usa = entry.get("usa")
    if not isinstance(usa, list) or len(usa) != len(pasos or []):
        problemas.append(f"usa tiene {len(usa) if isinstance(usa, list) else 'ningún'} elemento(s) y la receta {len(pasos or [])} pasos")
        return problemas
    ids = {str(c.get("ingredient_id")) for c in (constituents or []) if c.get("ingredient_id")}
    sumas: dict = {}
    for i, items in enumerate(usa):
        for it in items or []:
            cid = str((it or {}).get("ingredient_id"))
            f = (it or {}).get("fraccion")
            if cid not in ids:
                problemas.append(f"paso {i}: `{cid}` no es constituyente de la plantilla")
            if not isinstance(f, (int, float)) or f <= 0 or f > 1.0 + TOLERANCIA:
                problemas.append(f"paso {i}: fracción inválida {f!r} para `{cid}`")
            sumas[cid] = sumas.get(cid, 0.0) + (float(f) if isinstance(f, (int, float)) else 0.0)
    cuentas = entry.get("cuentas") or {}
    for cid in ids:
        if abs(round(sumas.get(cid, 0.0), 4) - float(cuentas.get(cid, 0.0))) > TOLERANCIA:
            problemas.append(f"`{cid}`: cuentas dice {cuentas.get(cid)} y usa suma {sumas.get(cid, 0.0):.4f}")
    if entry.get("estado") not in ESTADOS:
        problemas.append(f"estado desconocido {entry.get('estado')!r}")
    if entry.get("estado") in ("exacta", "estimada"):
        exentos = {str(h.get("ingredient_id")) for h in (entry.get("hallazgos") or []) if h.get("tipo") == "sin_uso_condimento"}
        for cid in ids - exentos:
            if abs(sumas.get(cid, 0.0) - 1.0) > TOLERANCIA:
                problemas.append(f"`{cid}`: Σ = {sumas.get(cid, 0.0):.4f} en una asignación `{entry.get('estado')}`")
    return problemas


# ─────────────────────────────────────────────────────────────────────────────────────────────
# La biblioteca entera
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _condimento_predicado():
    try:
        from culinary_coherence import _CONDIMENT_EXEMPT_RES
        return lambda nombre: any(rx.search(_norm(nombre)) for rx in _CONDIMENT_EXEMPT_RES)
    except Exception:
        return None


def derivar_biblioteca(country: str = "DO", index: Optional[dict] = None, *, indice_huella: Optional[str] = None,
                       generado_el: Optional[str] = None) -> dict:
    """La asignación de TODAS las recetas congeladas del país, lista para escribirse como snapshot."""
    import dish_registry as dr
    lib = dr.library_for_country(country)
    pasos_por_id = dr.recipe_steps_index(lib)
    plantillas = dr.templates_by_id(country)
    cond = _condimento_predicado()
    por_id: dict = {}
    for tid in sorted(pasos_por_id):
        t = plantillas.get(tid)
        if not t:
            continue
        por_id[tid] = derivar_uso(pasos_por_id[tid], t.get("constituents") or [], index, condimentos=cond)
    return {
        "schema_version": SCHEMA_VERSION,
        "country": str(country).upper(),
        "generado_el": generado_el or "",
        "indice_huella": indice_huella,
        "procedencia": {
            "escrito_por": "recipe_usage.derivar_biblioteca (determinista, sin LLM)",
            "que_es": "por receta congelada, qué constituyentes usa cada paso y con qué fracción de lo comprado; "
                      "atado al texto por pasos_hash",
            "veredicto_humano": "NO revisado a mano. `estado=revisar` es lo que la máquina no pudo decidir o "
                                "encontró contradictorio; `estimada` reparte a partes iguales una parte sin cifra.",
        },
        "resumen": resumen(por_id),
        "por_id": por_id,
    }


def resumen(por_id: dict) -> dict:
    est = Counter(e.get("estado") for e in por_id.values())
    tipos = Counter(h.get("tipo") for e in por_id.values() for h in e.get("hallazgos") or [])
    alimentos = Counter((h.get("tipo"), h.get("alimento")) for e in por_id.values() for h in e.get("hallazgos") or [])
    estimadas = sum(1 for e in por_id.values() for items in e.get("usa") or [] for it in items if it.get("estimada"))
    consts = sum(len(e.get("cuentas") or {}) for e in por_id.values())
    cerradas = sum(1 for e in por_id.values() for s in (e.get("cuentas") or {}).values() if abs(s - 1.0) <= TOLERANCIA)
    return {"recetas": len(por_id), "estados": dict(est), "hallazgos": dict(tipos),
            "constituyentes": consts, "constituyentes_con_suma_1": cerradas, "fracciones_estimadas": estimadas,
            "top": [{"tipo": t, "alimento": a, "n": n} for (t, a), n in alimentos.most_common(12)]}


def snapshot_path(country: str = "DO") -> Path:
    return _DIR / f"recipe_usage_{str(country).lower()}_v1.json"


def escribir_snapshot(snap: dict, path: Optional[Path] = None) -> Path:
    p = Path(path) if path else snapshot_path(snap.get("country") or "DO")
    p.write_text(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


@lru_cache(maxsize=4)
def cargar_uso(country: str = "DO") -> dict:
    p = snapshot_path(country)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-25] asignación {country} ilegible: {e!r}")
        return {}


@lru_cache(maxsize=4)
def _pasos_index(country: str = "DO") -> dict:
    """`template_id` → pasos de la biblioteca congelada. Cacheado: `dish_registry.recipe_steps_index` relee el JSON."""
    import dish_registry as dr
    return dr.recipe_steps_index(dr.library_for_country(country))


def clear_caches() -> None:
    cargar_uso.cache_clear()
    _pasos_index.cache_clear()


def usage_for_template(tid: str, country: str = "DO") -> Optional[dict]:
    """La asignación VIGENTE de una plantilla: existe y su `pasos_hash` es el de los pasos de hoy. Si no, `None`."""
    try:
        entry = (cargar_uso(country).get("por_id") or {}).get(str(tid))
        if not isinstance(entry, dict):
            return None
        pasos = _pasos_index(country).get(str(tid))
        if not pasos or entry.get("pasos_hash") != pasos_hash(pasos):
            return None
        return entry
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# El enganche de runtime
# ─────────────────────────────────────────────────────────────────────────────────────────────

def usage_exact_enabled() -> bool:
    """Knob `MEALFIT_RECIPE_USAGE_EXACT` (default True): con asignación vigente, el escáner lee las cuentas en vez de
    adivinar por el texto. Apagarlo devuelve la heurística sin redeploy."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_RECIPE_USAGE_EXACT", True)
    except Exception:
        return True


def cuentas_para_comida(meal: dict, country: str = "DO") -> Optional[dict]:
    """Para una comida de receta CONGELADA con asignación vigente: `{"template_id", "estado", "cuentas", "hallazgos",
    "nombres"}`. `None` si la comida no es de biblioteca, no hay asignación vigente, el número de pasos no cuadra con
    el de la asignación (alguien reescribió la receta en el plato) o el knob está apagado. Jamás lanza."""
    try:
        if not usage_exact_enabled() or not isinstance(meal, dict):
            return None
        if meal.get("_recipe_source") != "library":
            return None
        tid = meal.get("_recipe_template_id") or meal.get("_template_id")
        if not tid:
            return None
        entry = usage_for_template(str(tid), country)
        if not entry:
            return None
        rec = meal.get("recipe")
        if not isinstance(rec, list) or len(rec) != len(entry.get("usa") or []):
            return None
        import dish_registry as dr
        t = dr.templates_by_id(country).get(str(tid)) or {}
        nombres = {str(c.get("ingredient_id")): (c.get("canonical") or c.get("name")) for c in (t.get("constituents") or [])}
        return {"template_id": str(tid), "estado": entry.get("estado"), "cuentas": dict(entry.get("cuentas") or {}),
                "hallazgos": list(entry.get("hallazgos") or []), "nombres": nombres}
    except Exception:
        return None


def hallazgos_para_scanner(cuentas: dict) -> list:
    """`[(check, alimento, detalle)]` que el escáner emite en lugar de V3/V6/V7a por texto. Sólo los tres tipos que
    hablan de la cantidad; lo demás es inventario."""
    out = []
    for h in (cuentas or {}).get("hallazgos") or []:
        check = CHECK_POR_HALLAZGO.get(h.get("tipo"))
        if not check:
            continue
        cid = h.get("ingredient_id")
        nombre = (cuentas.get("nombres") or {}).get(str(cid)) or h.get("alimento") or str(cid)
        suma = (cuentas.get("cuentas") or {}).get(str(cid))
        detalle = f"asignación exacta ({cuentas.get('estado')}): {h.get('detalle')}"
        if suma is not None:
            detalle += f" — Σ = {float(suma):g}"
        out.append((check, nombre, detalle))
    return out
