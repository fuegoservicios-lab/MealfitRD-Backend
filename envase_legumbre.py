# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-282 · 2026-09-25] Una lata de habichuelas no son 425 g de habichuelas SECAS.

El catálogo mide las legumbres en SECO (333-388 kcal por 100 g) y el motor convierte lo cocido a seco antes de
comprar (`shopping_calculator._parse_quantity`, yield 0,35×; `_normalize_cooked_grain_lines` reescribe la línea a
«… secas»): la necesidad que llega al selector de envases está en gramos SECOS. Los envases, en cambio, se medían por
lo que PESAN: la lata 425 g con su líquido, el cartón Tetra de Rica 400 g, el frasco de cocidas 540 g; y la funda del
catálogo que dice «800 g seco» llevaba `grams: 2000` (su equivalente cocido, de cuando la lista contaba en cocido).

Medido en las baterías del 25-sep (208 líneas de legumbres en las listas): 32 compraban 1 lata o 1 cartón para
150-900 g secos —una lata trae ~90 g de legumbre seca, la lista se quedaba hasta ~4,5× corta— y la funda «800 g seco»
cubría 1.814 g secos con UNA funda (2,3× corta). El guard de coherencia no lo veía: compara gramos contra gramos y el
envase decía 425. `P2-LEGUME-NO-LATA-DEFAULT` (jul) sacó las latas del default del súper, pero no el cartón ni el
frasco, ni las latas del propio catálogo (que es a donde cae un alimento cuyo nombre no casa con el súper: «Habichuelas
negras» contra «Habichuela negra»).

Aquí cada envase de una legumbre en base seca se expresa en esa base: lo LISTO (lata, cartón/tetra, frasco, brik,
cocidas, guisadas, al natural…) × `MEALFIT_LEGUMBRE_LISTA_SECO_POR_G` (0,21 — ~57 % escurrido × 1/2,7 de hidratación,
por kcal USDA de enlatado escurrido contra seco); lo SECO que declara su peso en la etiqueta («800 g seco») por esa
etiqueta. La etiqueta que ve el usuario no cambia («1 lata (15 oz)»): cambia CUÁNTAS. Solo filas de legumbre en base
seca (kcal ≥ 250): si una fila pasa a base cocida, esto no la toca. Puro; nunca lanza: ante la duda, el envase tal cual.
tooltip-anchor: P1-PLAN-LOTE-282-ENVASE-EN-SECO"""
from __future__ import annotations

import re
import unicodedata

_LEGUMBRE_RX = re.compile(r"\b(habichuela|frijol|lenteja|garbanzo|guandul|gandul|alubia|judia|poroto)")
#: la fila está en base SECA: las legumbres cocidas o enlatadas rondan 90-160 kcal/100 g; las secas, 330-390
KCAL_BASE_SECA = 250.0
_UNIDAD_LISTA = {"lata", "latas", "tetra", "carton", "cartones", "frasco", "frascos", "brik", "briks", "tarro", "pote"}
_LISTO_RX = re.compile(r"\b(lata|latas|enlatad\w*|tetra|carton|frasco|brik|cocid\w*|guisad\w*|al natural|con coco|"
                       r"con chorizo|con vegetales|precocid\w*|listas? para)\b")
_SECO_RX = re.compile(r"\b(sec[oa]s?|deshidratad\w*)\b")


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def factor_listo_a_seco() -> float:
    """g de legumbre SECA por g de envase listo (lata, cartón, frasco). Knob `MEALFIT_LEGUMBRE_LISTA_SECO_POR_G`."""
    try:
        from knobs import _env_float
        v = float(_env_float("MEALFIT_LEGUMBRE_LISTA_SECO_POR_G", 0.21))
    except Exception:
        v = 0.21
    return min(0.5, max(0.1, v))


def es_legumbre_en_seco(nombre, master_item) -> bool:
    """¿Alimento de la familia legumbre cuya fila de catálogo mide en seco? Sin kcal conocidas, no (fail-open)."""
    try:
        if not _LEGUMBRE_RX.search(_norm(nombre or (master_item or {}).get("name"))):
            return False
        kcal = float((master_item or {}).get("kcal_per_100g") or 0)
        return kcal >= KCAL_BASE_SECA
    except Exception:
        return False


def es_envase_listo(pkg) -> bool:
    """Lata, cartón, frasco, brik… o una etiqueta que dice cocidas/guisadas/al natural — y NO dice «seco»."""
    try:
        etiqueta = _norm(f"{(pkg or {}).get('unit') or ''} {(pkg or {}).get('label') or ''}")
        if _SECO_RX.search(etiqueta):
            return False
        return _norm((pkg or {}).get("unit")) in _UNIDAD_LISTA or bool(_LISTO_RX.search(etiqueta))
    except Exception:
        return False


def _gramos_de_etiqueta_seca(label) -> float | None:
    """«800 g seco» → 800; «1 lb seco» → 453,6; sin «seco» o sin peso → None."""
    try:
        if not _SECO_RX.search(_norm(label)):
            return None
        from shopping_calculator import _parse_presentation_grams
        g = _parse_presentation_grams(label)
        return float(g) if g else None
    except Exception:
        return None


def envase_en_seco(pkg) -> dict:
    """El envase con `grams` en gramos de legumbre SECA. Devuelve una copia si cambia; el mismo dict si no."""
    try:
        g = float((pkg or {}).get("grams") or 0)
        if g <= 0:
            return pkg
        if es_envase_listo(pkg):
            nuevo = dict(pkg)
            nuevo["grams"] = round(g * factor_listo_a_seco(), 1)
            return nuevo
        seco = _gramos_de_etiqueta_seca(pkg.get("label"))
        if seco and abs(seco - g) > 0.15 * seco:
            nuevo = dict(pkg)
            nuevo["grams"] = round(seco, 1)
            return nuevo
        return pkg
    except Exception:
        return pkg


def en_base_del_catalogo(nombre, master_item):
    """El `master_item` con sus `market_packages` en la base del catálogo (copia superficial si algo cambia; jamás
    muta el cache). `container_weight_g` no se toca: lo leen otros caminos (la Nevera, el aviso de envase capado) y las
    filas de legumbre compran siempre por `market_packages`."""
    try:
        if not isinstance(master_item, dict) or not es_legumbre_en_seco(nombre, master_item):
            return master_item
        pkgs = master_item.get("market_packages")
        if not isinstance(pkgs, list) or not pkgs:
            return master_item
        nuevos = [envase_en_seco(p) if isinstance(p, dict) else p for p in pkgs]
        if all(a is b for a, b in zip(nuevos, pkgs)):
            return master_item
        out = dict(master_item)
        out["market_packages"] = nuevos
        return out
    except Exception:
        return master_item


# ── [P1-PLAN-LOTE-285 · 2026-09-25] La receta dice «de lata»: la lista compra la legumbre LISTA ────────────────────────
#
# Con «Nada» de tiempo (lote 283) la receta pide «½ taza de habichuelas negras de lata, escurridas». La lista la contaba
# como legumbre SECA (el 0,35× de lo cocido no miraba «de lata»: 3× de más) y elegía por precio la funda seca, la que
# hay que remojar y hervir una hora. Aquí: si alguna línea de esa legumbre pide la forma lista (de lata / en lata /
# enlatada / escurrida) y ninguna la pide seca, el selector ve solo los envases listos (lata, cartón, frasco) —los del
# súper o, si el súper no trae ninguno, los del catálogo—; si no hay ninguno, decide el precio como siempre. La marca que
# el usuario eligió en «Marcas del súper» manda siempre. tooltip-anchor: P1-PLAN-LOTE-285-LISTA-LISTA
_LINEA_LISTA_RX = re.compile(r"\b(de lata|en lata|enlatad[oa]s?|escurrid[oa]s?)\b")
_LINEA_SECA_RX = re.compile(r"\b(sec[oa]s?|crud[oa]s?|en seco|remoj\w*)\b")


def linea_lista(linea):
    """True = la línea pide la legumbre lista; False = seca; None = no lo dice o no es legumbre."""
    try:
        t = _norm(linea)
        if not _LEGUMBRE_RX.search(t):
            return None
        fuera = re.sub(r"\([^)]*\)", " ", t)
        if _LINEA_SECA_RX.search(fuera):
            return False
        if _LINEA_LISTA_RX.search(fuera):
            return True
        return None
    except Exception:
        return None


def anotar_forma(formas: dict, nombre, linea) -> None:
    """Acumula en `formas[nombre]` las formas (True lista / False seca) que piden las líneas del plan."""
    v = linea_lista(linea)
    if v is not None and nombre:
        formas.setdefault(nombre, set()).add(v)


def prefiere_listo(formas: dict, nombre) -> bool:
    s = (formas or {}).get(nombre) or set()
    return True in s and False not in s


def solo_listos(nombre, master_item, catalogo=None):
    """El master con SOLO sus envases listos: los del `master_item` (súper) o, si no trae ninguno, los del `catalogo`
    (la fila del catálogo sin el overlay). Sin ninguno listo, el master tal cual."""
    try:
        if not isinstance(master_item, dict) or not es_legumbre_en_seco(nombre, master_item):
            return master_item
        for fuente in (master_item, catalogo or {}):
            listos = [p for p in (fuente.get("market_packages") or []) if isinstance(p, dict) and es_envase_listo(p)]
            if listos:
                out = dict(master_item)
                out["market_packages"] = listos
                return out
        return master_item
    except Exception:
        return master_item
