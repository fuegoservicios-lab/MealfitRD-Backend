# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-817 · 2026-09-29] Las anclas de una política CONGELADA frente al perfil VIGENTE.

La renovación (`rolling_refill`, lote 811) lleva la política efectiva con la que NACIÓ el plan
(`plan_data._plan_policy.effective`): no se recompila, porque el formulario de la renovación no es el del wizard. Pero
el perfil sí cambia: una alergia nueva, un «no me gusta» nuevo o un cambio de dieta después de crear el plan. El worker
fusiona el perfil vivo en `form_data` (`cron_tasks._merge_chunk_live_profile`), así que los guards de la generación ven
la restricción nueva; la política no. Un ancla que el guard rechaza (Camarones con una alergia nueva a mariscos) es una
regla INSATISFACIBLE: el bloque 📐 la exige, el validador de fidelidad la echa en falta, el seeder la inyecta, y el
guard rechaza el plato en cada reintento — gasta reintentos y empeora el plato.

`anclas_vigentes(effective, perfil)` retira esas anclas con la MISMA puerta que usa el guard (no otra tabla; la lección
de P1-DIET-CANON-SSOT):
  · alergia → `graph_orchestrator._allergen_pool_item_banned` (el escáner `_scan_allergen_violations` del guard, con
    sus sinónimos y excusas; alergias con la unión del texto libre `profile_with_free_text`, centinela «Ninguna» fuera
    por `_has_real_medical_flags`, como el guard de la promoción). Rango 1, `anchor_conflicts_allergy` (el de F2).
  · dieta → `graph_orchestrator._diet_pool_item_banned` con la dieta del perfil. Rango 2, `anchor_conflicts_diet`.
  · rechazo («no me gusta», texto libre y restricción religiosa) → `rechazos._scan_dislike_violations`. Rango 4 (las
    exclusiones son restricción dura en §6.3), `anchor_conflicts_exclusion`: código sólo del snapshot, sin copy en el
    panel (no llega al frontend; `_REASON_COPY` exige paridad con `planPolicy.js`).
Los tres se comprueban aunque su knob de guard esté apagado: con el guard apagado no hay bucle, pero exigir un alérgeno
o algo que el usuario dijo que no quiere sigue siendo incorrecto.

Qué devuelve: con alguna retirada, una COPIA de la política sin esas anclas, con `renewal_relaxations` (entradas con la
forma de F2: `plan_policy._relax`), `renewal_parent_policy_hash` y `policy_hash` recalculado (la política que obedece el
bloque ya no es la del plan, y la métrica de fidelidad tiene que poder decirlo). Sin retirada, el MISMO objeto: el
snapshot queda byte a byte como el del lote 811. Nunca toca `plan_data`. Fail-open: si la puerta no se puede cargar,
la política va tal cual (conducta del 811) y queda un WARNING; el guard de la generación sigue en pie.

Alcance: sólo las anclas (`food_anchors`), que es lo único que la política EXIGE. `diet.allergies`/`exclusions`/`type`
del snapshot siguen siendo los de la creación (los leen el selector del registry y la proyección de compra única como
filtro, no como exigencia). Se revisa al ARMAR el snapshot; un bloque ya encolado cuyo perfil cambia antes de ejecutarse
no pasa por aquí.

Knob `MEALFIT_REFILL_POLICY_ANCHORS_RECHECK` (True). False ⇒ el snapshot del lote 811. Doc:
backend/docs/plan_policy_f3.md (fila «Relleno rolling»). Test: tests/test_p1_plan_lote_817.py.
tooltip-anchor: P1-PLAN-LOTE-817
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def recheck_activo() -> bool:
    """tooltip-anchor: MEALFIT_REFILL_POLICY_ANCHORS_RECHECK"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REFILL_POLICY_ANCHORS_RECHECK", True)
    except Exception:
        return True


def _mini(nombre: str) -> dict:
    return {"days": [{"meals": [{"name": "_ancla", "ingredients": [nombre]}]}]}


def _conflicto(nombre: str, go, rechazos, alergias: list, dieta, perfil: dict) -> Optional[tuple]:
    """(reason_code, rank, evidence) del primer choque del ancla con el perfil vigente, en el orden de §6.3; o None."""
    if alergias and go._allergen_pool_item_banned(nombre, alergias):
        culpable = next((a for a in alergias if go._has_real_medical_flags([a])
                         and go._allergen_pool_item_banned(nombre, [a])), None)
        return "anchor_conflicts_allergy", 1, {"allergy": str(culpable) if culpable else ", ".join(map(str, alergias))}
    if dieta and go._diet_pool_item_banned(nombre, dieta):
        try:
            from constants import canonicalize_diet_type
            dieta_c = canonicalize_diet_type(dieta)
        except Exception:
            dieta_c = str(dieta)
        return "anchor_conflicts_diet", 2, {"diet": dieta_c}
    viol = rechazos._scan_dislike_violations(_mini(nombre), perfil)
    if viol:
        return "anchor_conflicts_exclusion", 4, {"exclusion": str(viol[0][2])}
    return None


def anclas_vigentes(effective: Optional[dict], perfil: Optional[dict], *, user_id: Optional[str] = None
                    ) -> tuple[Optional[dict], list]:
    """(política, retiradas). Sin retiradas devuelve el MISMO `effective`. Nunca lanza."""
    if not isinstance(effective, dict) or not effective.get("food_anchors") or not isinstance(perfil, dict):
        return effective, []
    try:
        import graph_orchestrator as go
        import rechazos
        from plan_policy import _relax, policy_hash
        unido = go.profile_with_free_text(perfil)
        crudas = unido.get("allergies")
        alergias = [a for a in ([crudas] if isinstance(crudas, str) else list(crudas or [])) if str(a).strip()]
        if not go._has_real_medical_flags(alergias):
            alergias = []
        dieta = perfil.get("dietType") or ((perfil.get("dietTypes") or [None])[0])
        quedan, rels = [], []
        for a in effective.get("food_anchors") or []:
            nombre = str((a or {}).get("name") or (a or {}).get("ingredient_id") or "") if isinstance(a, dict) else ""
            choque = _conflicto(nombre, go, rechazos, alergias, dieta, perfil) if nombre else None
            if choque:
                code, rank, ev = choque
                _relax(rels, field="food_anchors", requested=nombre, applied=None, reason=code, rank=rank,
                       evidence={**ev, "source": "renewal_recheck"})
            else:
                quedan.append(a)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-817] revisión de anclas contra el perfil vigente no-op (fail-open, política "
                       f"del plan tal cual): {type(e).__name__}: {e}")
        return effective, []
    if not rels:
        return effective, []
    nueva = copy.deepcopy(effective)
    nueva["food_anchors"] = copy.deepcopy(quedan)
    nueva["renewal_relaxations"] = rels
    nueva["renewal_parent_policy_hash"] = effective.get("policy_hash")
    try:
        nueva["policy_hash"] = policy_hash(nueva)
    except Exception:
        pass
    logger.warning(
        f"[P1-PLAN-LOTE-817] renovación (user {str(user_id or '')[:8]}): {len(rels)} ancla(s) de la política chocan con "
        f"el perfil vigente y no viajan: "
        + "; ".join(f"{r['requested']} ({r['reason_code']}: {next(iter(r['evidence'].values()))})" for r in rels))
    return nueva, rels
