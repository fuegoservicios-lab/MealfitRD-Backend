# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-813 · 2026-09-29] Los mutadores de contenido de la cola de assemble, ANTES de la cadena de calidad.

El defecto (refutación del informe «finalize», 28-sep): la cola de `assemble_plan_node` decía correr la cadena de
calidad/banda (`db_plans.apply_plan_quality_finalize_chain`, surface `assemble-tail`) «tras la ÚLTIMA mutación de
assemble». No era así: detrás seguían cinco pases que AÑADEN o REESCRIBEN líneas —el ingrediente que los pasos declaran
y la lista no tiene, el «queso» genérico con nombre, el lácteo que promete el nombre, cocido→seco y la fusión de
duplicados—. En el journal del VPS al menos uno actuó entre el fin de la cadena y el revisor en 71 de 81 ventanas (sin
contar el AUTO-PATCH), metiendo líneas como «1 taza de Queso blanco». Esas líneas no pasaban por los topes, el cerrador
ni el contrato hasta el escudo pre-INSERT, que entonces recortaba y sacaba días de banda (13 bajadas reales de 81).

Aquí viven:
  · `aplicar(result)`: los cinco pases, en el orden y con los knobs y la telemetría de siempre (el código salió tal cual
    de `graph_orchestrator.py`; las funciones siguen allí).
  · `en_posicion(result, "antes"|"despues")`: assemble la llama en los DOS sitios y sólo corre en el que manda el knob
    `MEALFIT_ASSEMBLE_MUTATORS_BEFORE_CHAIN` (default True = antes de la cadena; False = el sitio viejo, detrás).
    Se quedan detrás a propósito: los re-autofix tardíos (detectan lo que la cadena reintroduce), el reconciliador
    display↔raw (la cadena ya corre el suyo), el AUTO-PATCH del revisor (posterior al revisor por diseño) y, además de
    delante, la fusión de duplicados (ver abajo).
  · El instrumento de la fila `clinical_band_final` (ver `metadata_banda_final`).

Dependencias verificadas (replay sin IA del lote sobre 19 pipeline_result del VPS; sonda = los cinco sobre la salida de
la cadena en el orden nuevo):
  · fantasma de los pasos: lee pasos declarados con cantidad; la cadena no escribe pasos «N g de X» sin su línea (sus
    notas 💪/⚠ se saltan) y su contrato final re-sincroniza los pasos con la lista. Delante de la cadena gana además
    que ya no puede RE-insertar una línea que la cadena retiró (topes, restricciones) y cuyo paso quedó.
  · «queso» genérico: la cadena ya lo corre dentro de `finalize_plan_data_coherence`.
  · lácteo del nombre: si la cadena quitara el lácteo, `identidad_plato.restaurar_identidad` lo devuelve. La sonda lo
    ve «volver a actuar» tras la cadena, pero es su piso de 30 g peleando con el cerrador, que lo baja: en el orden viejo
    el pre-INSERT lo bajaba igual (mismo gramaje final o ±5 g).
  · cocido→seco: el único escritor de la cadena de «N g de arroz blanco cocido» es el relleno de ganancia muscular, que
    ya lo corre detrás de sí (P1-FINALIZE-TAIL-PARITY). Sonda del replay: 0 de 19 planes lo piden tras la cadena.
  · duplicados: SÍ depende de la cadena, que crea duplicados (la identidad del plato, los cerradores) y sólo los funde
    tras el relleno de ganancia muscular. Por eso corre en LOS DOS sitios: antes (los topes ven el total real de una
    línea fantasma fundida con la suya) y detrás (lo que la cadena duplicó). Ver `en_posicion`.
tooltip-anchor: P1-PLAN-LOTE-813-MUTADORES-ANTES
"""
from __future__ import annotations

import hashlib
import json
import logging

from knobs import _env_bool

logger = logging.getLogger(__name__)

ASSEMBLE_MUTATORS_BEFORE_CHAIN = _env_bool("MEALFIT_ASSEMBLE_MUTATORS_BEFORE_CHAIN", True)


def antes_de_cadena() -> bool:
    return bool(ASSEMBLE_MUTATORS_BEFORE_CHAIN)


def en_posicion(result, posicion: str, ck=None) -> bool:
    """Assemble la llama en los dos sitios ('antes' y 'despues' de la cadena). Knob encendido: 'antes' corre los cinco
    pases y 'despues' SÓLO la fusión de duplicados, porque la cadena CREA duplicados que nadie más funde (replay del lote:
    la identidad del plato devuelve «2 rebanadas de pan integral» junto a «Pan integral familiar» en 3 de 19 planes, y
    sin fundir el pre-INSERT bajó uno de 1,00 a 0,917). Knob apagado: 'despues' corre los cinco, el orden viejo.
    Devuelve si corrió algo."""
    if posicion == "antes":
        if not antes_de_cadena():
            return False
        aplicar(result, ck=ck)
        if ck is not None:
            ck("pre_assemble_tail_chain")   # el mapa de tramos (P1-ASSEMBLE-PASS-MAP) no carga la cadena a los mutadores
        return True
    if antes_de_cadena():
        fundir_duplicados(result)
    else:
        aplicar(result, ck=ck)
    return True


def aplicar(result, ck=None) -> None:
    """Los cinco mutadores de contenido de la cola de assemble. Cada uno es fail-safe y nunca bloquea."""
    import graph_orchestrator as go
    if not isinstance(result, dict):
        return
    log = go.logger
    # [P1-PLAN-LOTE-818 · 2026-09-29] La telemetría de duplicados se ACUMULA dentro de una corrida (antes y detrás de la
    # cadena), no entre corridas: una re-entrada sobre el mismo dict heredaba las fusiones de la anterior.
    result.pop("_duplicate_food_lines_merged", None)

    # [P1-PHANTOM-INGREDIENT · 2026-07-24] Dirección INVERSA del validador de coherencia.
    # DEBE correr antes de construir la lista de compras (la línea insertada tiene que llegar a la lista) y antes del
    # truth-up (que recalcula macros desde strings). [P1-PLAN-LOTE-813] y, con el knob encendido, antes de la cadena:
    # así la línea insertada pasa por topes, cerrador y contrato.
    if go.PHANTOM_INGREDIENT_REPAIR:
        try:
            if ck is not None:
                ck("pre_phantom_repair")
            _ph_fixed = go._repair_declared_but_unlisted_ingredients(result.get("days") or [])
            if _ph_fixed:
                result["_phantom_ingredients_repaired"] = _ph_fixed
                log.info(f"👻 [P1-PHANTOM-INGREDIENT] {len(_ph_fixed)} ingrediente(s) fantasma "
                         f"reinsertado(s): " + "; ".join(
                             f"D{f['day']} {f['meal'][:24]!r} → {f['line']!r}" for f in _ph_fixed[:6]))
        except Exception as _ph_e:
            log.warning(f"[P1-PHANTOM-INGREDIENT] falló (no bloquea): {type(_ph_e).__name__}: {_ph_e}")

    # [P1-PLAN-LOTE-48 · 2026-09-14] El «queso» genérico de un cerrador toma el nombre del queso del plato ANTES de que
    # el lácteo del nombre se inserte aparte y de la lista de compras (plan 358a2cdf: la lista compró queso blanco para
    # dos platos «con queso cottage»). tooltip-anchor: P1-PLAN-LOTE-48-QUESO-NOMBRADO
    go._ccr.nombrar_quesos_genericos(result.get("days") or [])

    # [P1-NAME-PHANTOM-DAIRY · 2026-07-25] El lácteo que el NOMBRE promete y el plato no lleva.
    # Va DESPUÉS del repair por cantidad declarada (si los pasos ya la traían, ese lo resolvió) y antes de la lista de
    # compras. Los caps corren después como última palabra.
    if go.NAME_PHANTOM_DAIRY_REPAIR:
        try:
            _npd = go._repair_name_phantom_dairy(result.get("days") or [])
            if _npd:
                result["_name_phantom_dairy_repaired"] = _npd
                log.info(f"🧀 [P1-NAME-PHANTOM-DAIRY] {len(_npd)} lácteo(s) del NOMBRE añadido(s) al "
                         f"plato: " + "; ".join(f"D{d['day']} {d['meal'][:26]!r} → {d['line']!r}"
                                                for d in _npd[:5]))
        except Exception as _npd_e:
            log.warning(f"[P1-NAME-PHANTOM-DAIRY] falló (no bloquea): {type(_npd_e).__name__}: {_npd_e}")

    # [P1-COOKED-GRAIN-DRY · 2026-07-24] Gramos COCIDOS → gramos SECOS (unidades del catálogo).
    # Mismo requisito de orden: antes de la lista de compras y del truth-up pre-INSERT.
    # Va DESPUÉS del kcal-floor de gain_muscle de assemble, que es uno de los escritores de
    # "{g}g de arroz blanco cocido" — así su línea también queda normalizada.
    if go.COOKED_GRAIN_DRY_REWRITE:
        try:
            _cg = go._normalize_cooked_grain_lines(result.get("days") or [])
            if _cg:
                result["_cooked_grain_dry_rewrites"] = _cg
                log.info(f"🍚 [P1-COOKED-GRAIN-DRY] {len(_cg)} línea(s) cocido→crudo: " + "; ".join(
                    f"D{c['day']} {c['before']!r}→{c['after'].strip()!r}" for c in _cg[:6]))
        except Exception as _cg_e:
            log.warning(f"[P1-COOKED-GRAIN-DRY] falló (no bloquea): {type(_cg_e).__name__}: {_cg_e}")

    # [P1-DUP-FOOD-LINE-MERGE · 2026-07-24] El mismo alimento en dos líneas de la misma comida.
    # Va DESPUÉS del repair de fantasmas (si la línea reinsertada coincide con una existente, aquí se funden) y antes
    # de la lista. [P1-PLAN-LOTE-813] Con el knob encendido corre además detrás de la cadena (ver `en_posicion`).
    fundir_duplicados(result)


def fundir_duplicados(result) -> None:
    """[P1-DUP-FOOD-LINE-MERGE] Funde el mismo alimento repetido en una comida. La telemetría se ACUMULA: con el knob
    encendido corre dos veces (antes de la cadena y detrás). Fail-safe."""
    import graph_orchestrator as go
    if not isinstance(result, dict):
        return
    log = go.logger
    if go.MERGE_DUPLICATE_FOOD_LINES:
        try:
            _dm = go._merge_duplicate_food_lines(result.get("days") or [])
            if _dm:
                result["_duplicate_food_lines_merged"] = list(result.get("_duplicate_food_lines_merged") or []) + _dm
                log.info(f"🔗 [P1-DUP-FOOD-LINE-MERGE] {len(_dm)} grupo(s) fundido(s): " + "; ".join(
                    f"D{d['day']} {d['food']} → {d['into']!r}" for d in _dm[:6]))
        except Exception as _dm_e:
            log.warning(f"[P1-DUP-FOOD-LINE-MERGE] falló (no bloquea): {type(_dm_e).__name__}: {_dm_e}")


# ─────────────── el instrumento: ¿el pre-INSERT recibe la salida de la cadena? ───────────────
# Superficies de GENERACIÓN cuya salida alimenta al pre-INSERT del Bloque 1. Las del chunk (vista descartable) y la del
# seam T2 no dejan huella: sus `plan_data` se escriben por otros caminos y la clave no debe llegar a la base.
_SUPERFICIES_DE_GENERACION = frozenset({"assemble-tail", "assemble-budget-convergence", "review-band-gate",
                                        "post-review-patch"})
_CLAVE_SALIDA = "_cadena_salida_813"   # {"h", "surface"} de la última cadena de generación; el pre-INSERT la RETIRA
_CLAVE_CTX = "_cadena_ctx_813"         # contexto de la corrida en curso; `salida_cadena` lo retira siempre
# [P1-PLAN-LOTE-818 · 2026-09-29] Las claves de ESTE instrumento no salen de la generación. El pre-INSERT trabaja sobre
# una COPIA, así que el `result` original las llevaba al cliente (SSE/sync), a la KV del invitado y, de vuelta, a la fila
# por `restore-local` (la fila es lo que lee la caché semántica). Lista y helper únicos; los puntos de salida los llaman
# DESPUÉS de persistir: antes, el pre-INSERT no vería la huella y `input_equals_chain_out` quedaría en None.
# tooltip-anchor: P1-PLAN-LOTE-818-HUELLA-FUERA
CLAVES_PRIVADAS = (_CLAVE_SALIDA, _CLAVE_CTX)
# Lo que la cadena puede cambiar de una comida. Fuera a propósito: `date`/`day_name` y `_display`, que se estampan
# entre la generación y el guardado y harían «distinto» un plato idéntico.
_CAMPOS = ("meal", "name", "ingredients", "ingredients_raw", "recipe", "cals", "protein", "carbs", "fats")


def retirar_claves_privadas(plan_data) -> int:
    """[P1-PLAN-LOTE-818] Quita de `plan_data`, en su sitio, las claves de `CLAVES_PRIVADAS`. Devuelve cuántas quitó
    (0 si no es dict). Fail-safe."""
    if not isinstance(plan_data, dict):
        return 0
    quitadas = [k for k in CLAVES_PRIVADAS if k in plan_data]
    for k in quitadas:
        plan_data.pop(k, None)
    return len(quitadas)


def huella_days(days):
    """sha256 (16 hex) del CONTENIDO de `days`: por comida, los campos de `_CAMPOS`. None si no se puede."""
    try:
        proy = [[{k: m.get(k) for k in _CAMPOS if k in m} for m in (d.get("meals") or []) if isinstance(m, dict)]
                for d in (days or []) if isinstance(d, dict)]
        return hashlib.sha256(json.dumps(proy, sort_keys=True, ensure_ascii=False, default=str)
                              .encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def entrada_cadena(plan_data, data, surface) -> None:
    """Al entrar al escudo: compara la huella de `days` con la que dejó la última cadena de generación y deja el contexto
    que lee `metadata_banda_final`. En el pre-INSERT retira la huella guardada (no se persiste). Fail-safe."""
    if not isinstance(plan_data, dict):
        return
    try:
        _prev = plan_data.pop(_CLAVE_SALIDA, None) if surface == "pre-INSERT" else plan_data.get(_CLAVE_SALIDA)
        _h = huella_days(plan_data.get("days"))
        _eq = (_prev.get("h") == _h) if (isinstance(_prev, dict) and _prev.get("h") and _h) else None
        _data = data if isinstance(data, dict) else {}
        _fd = _data.get("form_data") if isinstance(_data.get("form_data"), dict) else {}
        _fence = plan_data.get("_chunk_fence") if isinstance(plan_data.get("_chunk_fence"), dict) else {}
        plan_data[_CLAVE_CTX] = {
            "plan_id": _data.get("plan_id") or plan_data.get("plan_id"),
            "run_id": plan_data.get("_run_id") or _fd.get("_run_id"),
            "chunk_task_id": _fence.get("task_id"),
            "input_equals_chain_out": _eq,
            "chain_out_surface": _prev.get("surface") if isinstance(_prev, dict) else None,
        }
    except Exception as _e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-813] entrada de la cadena no-op: {type(_e).__name__}: {_e}")


def salida_cadena(plan_data, surface) -> None:
    """Al salir del escudo (también si falló): retira el contexto y, si la superficie es de generación, guarda la huella
    de salida para que el pre-INSERT la compare. Fail-safe."""
    if not isinstance(plan_data, dict):
        return
    try:
        plan_data.pop(_CLAVE_CTX, None)
        if surface in _SUPERFICIES_DE_GENERACION:
            _h = huella_days(plan_data.get("days"))
            if _h:
                plan_data[_CLAVE_SALIDA] = {"h": _h, "surface": surface}
    except Exception as _e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-813] salida de la cadena no-op: {type(_e).__name__}: {_e}")


def metadata_banda_final(plan_data, score) -> dict:
    """Claves nuevas de la fila `clinical_band_final`. `final_score` va como double en el JSON: `confidence` es float4
    y 0,917 se guardaba como 0,91699 < 0,917, así que 10 de las 23 «bajadas» medidas eran empates."""
    out = {"final_score": float(score) if score is not None else None}
    ctx = plan_data.get(_CLAVE_CTX) if isinstance(plan_data, dict) else None
    if isinstance(ctx, dict):
        for k in ("plan_id", "run_id", "chunk_task_id", "input_equals_chain_out", "chain_out_surface"):
            out[k] = ctx.get(k)
    return out
