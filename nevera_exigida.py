# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-199 · 2026-09-24] Los cerradores eligen de la Nevera cuando el revisor la va a exigir.

Caso REAL (el único usuario con relleno semanal ligado a su Nevera): sus 6 rellenos de las últimas 72 h se entregaron
FALLANDO la revisión (alerta `review_failed_delivered_rate_high`, 6 de 8 entregas). En 5 de 6 el motivo fue la Nevera
—«Ingredientes COMPLETAMENTE INEXISTENTES en inventario»— y en TODOS los intentos aparecía «edamame cocido» (35-250 g),
que no estaba en su Nevera: lo metía el cerrador de proteína, que elige del catálogo del país
(`_safe_high_density_proteins`), igual que camarones, whey, pechuga y cottage; el cerrador de micronutrientes sembraba
semillas de girasol/linaza. El revisor rechazaba, el reintento volvía a meter lo mismo y, agotado el presupuesto, el plan
salía degradado con «🚨 Compra Urgente: 250 g de edamame cocido, 1 porción de proteína whey…» y un push de «compras
urgentes». Su Nevera tenía 20 huevos, 2 kg de yogur, 5 tilapias, 800 g de habichuelas blancas, soya texturizada y costilla.

Aquí vive:
  1. `lista(fd)`: la MISMA regla con la que `review_plan_node` decide si valida contra la Nevera. Es un espejo (el bloque
     del revisor está anclado por 10 tests de fuente); el test del lote EJECUTA el bloque del revisor y compara.
  2. `admite_linea`: la MISMA coincidencia que aplica el revisor (`validate_ingredients_against_pantry`, modo sonda: sin
     el paso vectorial, que costaría una llamada por candidato, y sin el warning de rechazo). La sonda sólo puede
     rechazar DE MÁS (lo que el revisor aceptaría por vector), nunca aceptar lo que él rechaza.
  3. El formulario de la corrida, fijado por `arun_plan_pipeline` en un ContextVar: el grafo ya propaga el contexto a
     sus hilos (`ctx.run`), así los cerradores no necesitan un parámetro nuevo en sus siete llamadas.

Si la Nevera no tiene NINGUNA proteína densa, el cerrador conserva su conducta previa (un déficit tampoco pasa la
revisión; la compra urgente es la salida honesta). Las semillas, en cambio, simplemente no se siembran: son un extra.
Knob `MEALFIT_CLOSERS_RESPECT_PANTRY` (True) — rollback sin redeploy.
"""
from __future__ import annotations

import contextvars
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_FD: contextvars.ContextVar = contextvars.ContextVar("mealfit_nevera_exigida_form_data", default=None)
#: (nevera, país) → {línea: bool}. La Nevera de una corrida no cambia; el tope evita crecer sin fin en un proceso largo.
_MEMO: dict = {}
_MEMO_MAX_NEVERAS = 64
#: Un log por Nevera y resultado: el builder se llama decenas de veces por corrida.
_AVISADOS: set = set()


def activo() -> bool:
    """Knob. tooltip-anchor: MEALFIT_CLOSERS_RESPECT_PANTRY"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLOSERS_RESPECT_PANTRY", True)
    except Exception:
        return True


def fijar(form_data):
    """Lo llama `arun_plan_pipeline` con su `actual_form_data` (la MISMA referencia que recibe el revisor en el estado,
    así una bandera puesta después —`_is_rotation_reroll`— también cuenta). Devuelve el token para `soltar`."""
    return _FD.set(form_data if isinstance(form_data, dict) else None)


def soltar(token) -> None:
    try:
        _FD.reset(token)
    except (LookupError, ValueError, TypeError, RuntimeError):
        pass


def lista(form_data=None) -> Optional[list]:
    """Espejo de `review_plan_node` (bloque «Validación Estricta de Despensa»): la Nevera contra la que el revisor
    validará la EXISTENCIA de cada ingrediente, o None si no la validará. Sin formulario ⇒ None."""
    fd = _FD.get() if form_data is None else form_data
    if not isinstance(fd, dict):
        return None
    try:
        from horizon import is_renewal_reason
        is_variety_regen = is_renewal_reason(fd.get("update_reason"))
    except Exception:
        return None
    is_rotation = bool(fd.get("_is_rotation_reroll", False))
    is_strict_required = bool(fd.get("_strict_pantry_required", False))
    has_pantry = bool(fd.get("current_pantry_ingredients") or fd.get("current_shopping_list"))
    pantry_advisory_only = bool(fd.get("_pantry_advisory_only", False))
    fridge_checked_empty = ("current_pantry_ingredients" in fd and not fd.get("current_pantry_ingredients"))
    if fridge_checked_empty and has_pantry and not (is_rotation or is_strict_required):
        has_pantry = False
    if pantry_advisory_only or is_variety_regen or not (is_rotation or is_strict_required or has_pantry):
        return None
    current = fd.get("current_pantry_ingredients") or fd.get("current_shopping_list", [])
    if not current or not isinstance(current, list):
        return None
    clean = [item.strip() for item in current if item and isinstance(item, str) and len(item) > 2]
    return clean or None


_ANOTACION = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _tokens(texto) -> list:
    """Tokens canónicos del NOMBRE, leído como lo lee el validador del revisor (`_parse_quantity`: «1960 g de Yogurt»,
    «0.5 lbs de …», «20 Huevo»), sin las anotaciones del inventario («[⚠️ URGENTE: …]», «(4 uds.)»)."""
    from constants import canonical_pantry_key
    limpio = _ANOTACION.sub(" ", str(texto or ""))
    try:
        from shopping_calculator import _parse_quantity
        nombre = _parse_quantity(limpio)[2] or limpio
    except Exception:
        nombre = limpio
    return canonical_pantry_key(_ANOTACION.sub(" ", str(nombre))).split()


def _es_de_la_nevera(linea: str, nevera: list) -> bool:
    """El candidato ES (una variante de) algo que la Nevera tiene: TODOS los tokens de ese alimento están, como token
    completo (singular/plural), en el candidato. «pollo» → «pechuga de pollo» sí; «sal» → «salmón» NO; «queso blanco» →
    «queso cottage» NO. El revisor casa por SUBCADENA y acepta «salmón» con sal en la Nevera: el filtro no hereda eso."""
    from constants import _pantry_token_variants
    cand = [_pantry_token_variants(t) for t in _tokens(linea)]
    if not cand:
        return False
    for item in nevera:
        propios = _tokens(item)
        if propios and all(any(_pantry_token_variants(t) & v for v in cand) for t in propios):
            return True
    return False


def admite_linea(linea, form_data=None) -> bool:
    """¿El revisor aceptaría esta línea contra la Nevera exigida, y es de verdad algo que la Nevera tiene? Sin Nevera
    exigida, knob apagado o error ⇒ True (conducta previa)."""
    fd = _FD.get() if form_data is None else form_data
    nevera = lista(fd)
    if not nevera or not activo():
        return True
    texto = str(linea or "").strip()
    if not texto:
        return True
    try:
        from constants import country_for_form_data, validate_ingredients_against_pantry
        pais = country_for_form_data(fd)
    except Exception:
        return True
    clave = (tuple(nevera), pais)
    memo = _MEMO.get(clave)
    if memo is None:
        if len(_MEMO) >= _MEMO_MAX_NEVERAS:
            _MEMO.clear()
        memo = _MEMO[clave] = {}
    if texto not in memo:
        try:
            memo[texto] = (_es_de_la_nevera(texto, nevera) and validate_ingredients_against_pantry(
                [texto], nevera, strict_quantities=False, country=pais, probe_only=True) is True)
        except Exception:
            memo[texto] = True
    return memo[texto]


def admite(nombre, form_data=None) -> bool:
    """Un NOMBRE de alimento, con la forma de línea que escribe el cerrador («N g de <nombre>»)."""
    return admite_linea(f"100 g de {str(nombre or '').strip().lower()}", form_data)


def preferir(opciones, indice: int):
    """La rotación determinista de un autofix (arroz de noche → batata/yuca/casabe…), primero entre las opciones que la
    Nevera exigida tiene. Sin Nevera exigida, o si no tiene ninguna, la rotación de siempre (byte-idéntica)."""
    ops = list(opciones)
    dentro = [o for o in ops if admite(o)]
    base = dentro or ops
    return base[int(indice) % len(base)]


def filtrar_proteinas(cands: list) -> list:
    """Salida de `_safe_high_density_proteins` — `[(densidad, nombre, info)]` — reducida a lo que la Nevera tiene.
    Se evalúa `info.name` (lo que el cerrador ESCRIBE en la línea) y, si no hay, el nombre del pool. Si nada queda, la
    lista entera (conducta previa)."""
    if not cands:
        return cands
    fd = _FD.get()
    nevera = lista(fd)
    if nevera is None or not activo():
        return cands
    try:
        dentro = [c for c in cands if admite(getattr(c[2], "name", None) or c[1], fd)]
    except Exception:
        return cands
    aviso = (tuple(nevera), len(dentro), len(cands))
    if aviso not in _AVISADOS and len(_AVISADOS) < 1024:
        _AVISADOS.add(aviso)
        if not dentro:
            logger.info("🧊 [P1-PLAN-LOTE-199] la Nevera no tiene ninguna proteína densa del pool: el cerrador conserva "
                        "el catálogo (compra urgente honesta)")
        elif len(dentro) < len(cands):
            logger.info(f"🧊 [P1-PLAN-LOTE-199] cerrador de proteína limitado a la Nevera: {len(dentro)}/{len(cands)} "
                        f"candidatos ({', '.join(str(getattr(c[2], 'name', None) or c[1]) for c in dentro[:6])})")
    return dentro or cands
