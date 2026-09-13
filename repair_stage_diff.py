# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-27 · 2026-09-12] C5 (segunda parte) · CUL-P1-04: la cadena de reparación se MIDE por etapas.

«Ajustar nutrición sin desarmar la receta» tiene una parte que ya estaba (los reparadores re-escalan porciones EXISTENTES,
la sustitución de huevo reescribe sus pasos, C2/C3 sincronizan cantidades y formas en la cola) y una que faltaba: nadie
comparaba el plato ANTES y DESPUÉS de la cadena. Un reparador que arregla la banda y deja un paso hablando del alimento
anterior produce un defecto NUEVO, y un defecto nuevo que nadie mide es un defecto que nadie ve.

Este módulo corre el escáner culinario (capa 1, determinista) en tres puntos del persist boundary
(`db_plans._finalize_plan_data_for_insert`): a la ENTRADA, tras los CAPS de realismo (el pase que más mueve cantidades) y
a la SALIDA (tras el contrato final). Escribe en `plan_data["_repair_stage_diff"]`:

    {"etapas": {"entrada": {check: n}, "tras_caps": {...}, "salida": {...}},
     "nuevos": [{"etapa", "check", "meal_index", "day", "food"}],     # hallazgos que NO estaban en la entrada
     "resueltos": n, "comidas": n, "reglas_huella": ..., "surface": ...}   # determinista: sin tiempos

y lo LOGUEA cuando hay nuevos. No muta el plan, no bloquea, no repara: informa. Knob `MEALFIT_REPAIR_STAGE_DIFF` (default
`True`); coste medido: un scan ≈ 0,3 s por 64 comidas (tres por plan de 30 días ≈ 1,5 s sobre una cadena de 10-25 s).
Sin catálogo (fuera de FastAPI, sin pool) no mide y lo dice (`estado = sin_catalogo`).

Puro salvo la lectura del catálogo (`shopping_calculator.get_master_ingredients`, cacheado); nunca lanza.
tooltip-anchor: P1-PLAN-LOTE-27-REPAIR-STAGE-DIFF
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

ETAPAS = ("entrada", "tras_caps", "salida")
MAX_COMIDAS = 200          # por encima, el plan no se mide (coste): se dice


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REPAIR_STAGE_DIFF", True)
    except Exception:
        return True


def _catalogo(catalog=None):
    if catalog is not None:
        return catalog
    try:
        from shopping_calculator import get_master_ingredients
        return get_master_ingredients() or None
    except Exception:
        return None


def _firma(v: dict) -> tuple:
    return (str(v.get("check")), v.get("day"), v.get("meal_index"), str(v.get("food")))


def _scan(plan_data: dict, catalog, form_data=None) -> tuple:
    """`(firmas, conteo_por_check, estado)`; fail-open."""
    try:
        from culinary_coherence import culinary_contract_scan_status
        viol, est = culinary_contract_scan_status(plan_data, catalog, form_data=form_data)
        return {_firma(v) for v in viol}, dict(Counter(str(v.get("check")) for v in viol)), est
    except Exception as e:                                                     # noqa: BLE001
        return set(), {}, {"status": "error", "error": f"{type(e).__name__}: {e}"[:200]}


def start(plan_data: dict, *, surface: str = "", catalog=None, form_data=None) -> Optional[dict]:
    """La foto de ENTRADA. `None` si no procede (knob, sin catálogo, plan enorme o sin días): entonces nada más se mide."""
    try:
        if not enabled() or not isinstance(plan_data, dict):
            return None
        n = sum(len(d.get("meals") or []) for d in (plan_data.get("days") or []) if isinstance(d, dict))
        if not n or n > MAX_COMIDAS:
            return None
        cat = _catalogo(catalog)
        if not cat:
            plan_data["_repair_stage_diff"] = {"estado": "sin_catalogo", "surface": str(surface or "")[:40]}
            return None
        t0 = time.monotonic()
        firmas, conteo, est = _scan(plan_data, cat, form_data)
        if est.get("status") not in ("scanned",):
            plan_data["_repair_stage_diff"] = {"estado": f"no_medible:{est.get('status')}", "surface": str(surface or "")[:40]}
            return None
        return {"surface": str(surface or "")[:40], "catalog": cat, "form_data": form_data, "comidas": n,
                "reglas_huella": est.get("reglas_huella"), "t0": t0,
                "firmas": {"entrada": firmas}, "etapas": {"entrada": conteo}}
    except Exception:
        return None


def mark(ctx: Optional[dict], etapa: str, plan_data: dict) -> None:
    """Una foto intermedia (`tras_caps`). No-op sin contexto."""
    if not ctx or etapa not in ETAPAS:
        return
    try:
        firmas, conteo, _est = _scan(plan_data, ctx["catalog"], ctx.get("form_data"))
        ctx["firmas"][etapa] = firmas
        ctx["etapas"][etapa] = conteo
    except Exception:
        return


def finish(ctx: Optional[dict], plan_data: dict) -> Optional[dict]:
    """La foto de SALIDA y el informe: lo que apareció en cada etapa y no estaba en la entrada. Escribe
    `plan_data["_repair_stage_diff"]` y devuelve el informe. No-op sin contexto."""
    if not ctx or not isinstance(plan_data, dict):
        return None
    try:
        firmas, conteo, _est = _scan(plan_data, ctx["catalog"], ctx.get("form_data"))
        ctx["firmas"]["salida"] = firmas
        ctx["etapas"]["salida"] = conteo
        base = ctx["firmas"].get("entrada", set())
        nuevos, vistos = [], set()
        previo = base
        for etapa in ("tras_caps", "salida"):
            if etapa not in ctx["firmas"]:
                continue
            for f in sorted(ctx["firmas"][etapa] - base - vistos, key=str):
                if f in previo and etapa != "tras_caps":
                    continue
                nuevos.append({"etapa": etapa, "check": f[0], "day": f[1], "meal_index": f[2], "food": f[3]})
                vistos.add(f)
            previo = ctx["firmas"][etapa]
        # DETERMINISTA a propósito: dos corridas iguales dan el mismo dict (los tests del seam T2 comparan planes byte a
        # byte); el tiempo va al log, no al plan.
        ms = int((time.monotonic() - ctx["t0"]) * 1000)
        informe = {
            "estado": "medido", "surface": ctx["surface"], "comidas": ctx["comidas"],
            "etapas": ctx["etapas"], "nuevos": nuevos[:40], "n_nuevos": len(nuevos),
            "resueltos": len(base - firmas), "reglas_huella": ctx.get("reglas_huella"),
        }
        plan_data["_repair_stage_diff"] = informe
        if nuevos:
            por = Counter((x["etapa"], x["check"]) for x in nuevos)
            logger.warning(f"🧪 [P1-PLAN-LOTE-27] {ctx['surface']}: la cadena de reparación introdujo {len(nuevos)} hallazgo(s) "
                           f"culinario(s) nuevo(s) ({dict(por)}); resueltos {informe['resueltos']}.")
        else:
            logger.info(f"🧪 [P1-PLAN-LOTE-27] {ctx['surface']}: cadena de reparación sin hallazgos nuevos "
                        f"(entrada {sum(ctx['etapas'].get('entrada', {}).values())} → salida {sum(conteo.values())}, "
                        f"resueltos {informe['resueltos']}, {ms} ms).")
        return informe
    except Exception:
        return None


def medir_cadena(plan_data: dict, cadena, *, surface: str = "bench", catalog=None, form_data=None) -> Optional[dict]:
    """Para el benchmark de superficies: `cadena(plan_data)` corre entre la entrada y la salida. Devuelve el informe."""
    ctx = start(plan_data, surface=surface, catalog=catalog, form_data=form_data)
    try:
        cadena(plan_data)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-27] cadena {surface} lanzó {type(e).__name__}: {e}")
    return finish(ctx, plan_data)
