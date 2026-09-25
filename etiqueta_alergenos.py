# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-270 · 2026-09-25] Nota de etiqueta para los productos comerciales que suelen llevar el alérgeno.

Batería final y verificación (36 corridas): el revisor de IA rechazó planes por «el pan integral familiar tiene alta
probabilidad de contener leche» (alergia a lácteos), «la tortilla de trigo integral no está especificada como libre de
lecitina de soya» (soya) y «el wrap no especifica la envoltura» (gluten). El escáner determinista acierta —el pan no es
un lácteo— pero el usuario compra una marca, y el pan de molde, las galletas, el chocolate, los embutidos, la granola,
las tortillas o el caldo en cubito llevan a menudo leche, huevo, soya, gluten o frutos secos. El plan no lo decía y
cada objeción costaba un reintento.

Ahora cada comida con uno de esos productos, para una alergia DECLARADA que el producto suele esconder, lleva una nota
«⚠️ Alergia declarada: elige … sin …; revisa la etiqueta (y el «puede contener»).». La clase de alergia se lee de la
expansión SSOT (`_expand_allergy_declarations`), así que cuenta también lo tecleado a mano. Es una NOTA (`⚠`: los pases
que leen los pasos la saltan), idempotente y nunca lanza; no toca ingredientes. tooltip-anchor: P1-PLAN-LOTE-270-ETIQUETA
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_PREFIJO = "⚠️ Alergia declarada: elige "
_PAN = r"\bpan(?:es)?\b(?!\s+de\s+(?:maiz|yuca|arroz|platano|casabe))|\bbrioche\b"
_GALLETAS = r"\bgalletas?\b"
_CHOCOLATE = r"\bchocolate\b"
_EMBUTIDOS = r"\b(?:salami|salchich\w*|jamon|embutid\w*|longaniza|chorizo|mortadela|pepperoni)\b"
_GRANOLA = r"\bgranola\b|\bbarras? de cereal"
_TORTILLAS = r"\btortillas? de (?:harina|trigo|maiz)\b|\bwraps?\b"
_PASTA = r"\b(?:pasta|espaguetis?|fideos?|macarrones?|tallarines?|coditos?)\b"
_CUBITO = r"\b(?:cubitos?|sopitas?|caldo en polvo|caldo de sobre)\b"

# clase → [(nombre del producto, patrón)]; la clase se decide por un término CANÓNICO de la expansión SSOT.
_PRODUCTOS = {
    "lácteos": (("pan", _PAN), ("galletas", _GALLETAS), ("chocolate", _CHOCOLATE), ("embutidos", _EMBUTIDOS),
                ("granola", _GRANOLA)),
    "huevo": (("pan", _PAN), ("galletas", _GALLETAS), ("pasta", _PASTA)),
    "soya": (("pan", _PAN), ("tortillas", _TORTILLAS), ("chocolate", _CHOCOLATE), ("galletas", _GALLETAS),
             ("embutidos", _EMBUTIDOS), ("caldo en cubito", _CUBITO)),
    "gluten": (("embutidos", _EMBUTIDOS), ("caldo en cubito", _CUBITO), ("chocolate", _CHOCOLATE)),
    "maní": (("granola", _GRANOLA), ("chocolate", _CHOCOLATE), ("galletas", _GALLETAS)),
    "frutos secos": (("granola", _GRANOLA), ("chocolate", _CHOCOLATE), ("galletas", _GALLETAS)),
    "sésamo": (("pan", _PAN), ("galletas", _GALLETAS)),
}
_CANONICO = {"lácteos": "leche", "huevo": "huevo", "soya": "soya", "gluten": "trigo", "maní": "mani",
             "frutos secos": "almendra", "sésamo": "sesamo"}
_RX = {clase: tuple((n, re.compile(p)) for n, p in prods) for clase, prods in _PRODUCTOS.items()}


def _sin_acentos(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s or "")) if not unicodedata.combining(c)).lower()


def clases_declaradas(form_data) -> list:
    """Las clases de alergia declaradas (chips + texto libre) que este módulo sabe anotar."""
    try:
        import graph_orchestrator as go
        fd = go.profile_with_free_text(form_data if isinstance(form_data, dict) else {})
        alergias = [a for a in (fd.get("allergies") or []) if str(a).strip().lower() not in go._SENTINEL_NONE_VALUES]
        if not alergias:
            return []
        exp = {_sin_acentos(t) for t in go._expand_allergy_declarations(alergias)}
        return [clase for clase, canon in _CANONICO.items() if canon in exp]
    except Exception:
        return []


def nota_para(meal, clases) -> str:
    """La nota de etiqueta de una comida («» si no hace falta)."""
    texto = " | ".join(_sin_acentos(i) for i in (meal.get("ingredients") or []) if isinstance(i, str))
    por_producto = {}
    for clase in clases:
        for nombre, rx in _RX.get(clase, ()):
            if rx.search(texto):
                por_producto.setdefault(nombre, []).append(clase)
    if not por_producto:
        return ""
    partes = []
    for nombre, cls in por_producto.items():
        partes.append(nombre + " sin " + (cls[0] if len(cls) == 1 else ", ".join(cls[:-1]) + " ni " + cls[-1]))
    return _PREFIJO + "; ".join(partes) + ". Revisa la etiqueta, también el «puede contener»."


def anotar_plan(plan, form_data) -> int:
    """Añade la nota a cada comida que la necesite. Devuelve 0 (no es una sustitución: no suma al contador del
    llamador); el recuento va al log. Idempotente, nunca lanza."""
    try:
        clases = clases_declaradas(form_data)
        if not clases or not isinstance(plan, dict):
            return 0
        n = 0
        for d in plan.get("days") or []:
            for m in (d.get("meals") or []) if isinstance(d, dict) else []:
                if not isinstance(m, dict):
                    continue
                rec = m.get("recipe")
                if not isinstance(rec, list) or any(isinstance(p, str) and p.startswith(_PREFIJO) for p in rec):
                    continue
                nota = nota_para(m, clases)
                if nota:
                    rec.append(nota)
                    n += 1
        if n:
            logger.info(f"🏷️ [P1-PLAN-LOTE-270] nota de etiqueta en {n} comida(s) (alergias: {clases}).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-270] nota de etiqueta falló (no bloquea): {type(e).__name__}: {e}")
    return 0
