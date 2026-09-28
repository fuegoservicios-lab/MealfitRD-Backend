# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-661 · 2026-09-28] Una proteína asignada que el día no usa es un AVISO si el día cumple su proteína.

Decisión del dueño (28-sep). Producción 14-27 sep: 6 de los 36 rechazos del revisor fueron «Día N omitió múltiples
proteínas clave asignadas: ['tilapia', 'queso blanco fresco']» —el planificador reparte proteínas de la Nevera por día y
el generador de días elige otras— y cada uno regeneraba el plan entero aunque el día llegara a su proteína. El lote
P1-FIDELITY-FINAL-ADVISORY ya lo degradaba a aviso en el ÚLTIMO intento; los anteriores seguían quemándose.

Aquí, en el revisor: el error de fidelidad de un día que NO está bajo el piso de proteína (el MISMO cálculo y la misma
tolerancia que el gate de proteína, `_protein_floor_shortfall`) pasa a `plan["_skeleton_fidelity_advisory"]` y no
rechaza. El día que además se queda corto de proteína conserva el error (y el gate de proteína lo acusa por su lado).
Sin gate de proteína o sin objetivo de proteína en el plan no se puede medir: nada cambia. Knob
`MEALFIT_FIDELITY_ADVISORY_IF_PROTEIN_OK`. tooltip-anchor: P1-PLAN-LOTE-661
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DIA = re.compile(r"^\s*D[ií]a\s+(\S+)\s+omiti")


def _activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_FIDELITY_ADVISORY_IF_PROTEIN_OK", True)
    except Exception:
        return True


def filtrar(plan, errores, form_data=None) -> list:
    """Los errores de fidelidad que siguen rechazando; los de días que cumplen su proteína quedan como aviso."""
    errores = list(errores or [])
    try:
        if not errores or not isinstance(plan, dict) or not _activo():
            return errores
        import graph_orchestrator as go
        if not go.PROTEIN_FLOOR_HARD_GATE:
            return errores
        if not re.search(r"\d", str((plan.get("macros") or {}).get("protein", ""))):
            return errores                             # sin objetivo de proteína no se puede decir que el día cumpla
        renal = bool((plan.get("renal_protein_cap") or {}).get("applied"))
        cortos = {str(d) for d, _p, _t in go._protein_floor_shortfall(
            plan, renal_capped=renal, form_data=form_data, tolerance_pct=go.PROTEIN_FLOOR_RETRY_TOLERANCE_PCT)}
        quedan, avisos = [], []
        for e in errores:
            m = _DIA.match(str(e))
            (avisos if m and m.group(1) not in cortos else quedan).append(e)
        if avisos:
            plan.setdefault("_skeleton_fidelity_advisory", [])
            plan["_skeleton_fidelity_advisory"] += [a for a in avisos if a not in plan["_skeleton_fidelity_advisory"]]
            logger.info(f"🧬 [P1-PLAN-LOTE-661] {len(avisos)} día(s) con proteínas asignadas sin usar pero con su "
                        f"proteína cumplida → aviso, no rechazo: {avisos[0][:90]}")
        return quedan
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-661] no-op: {type(e).__name__}: {e}")
        return errores


__all__ = ["filtrar"]
