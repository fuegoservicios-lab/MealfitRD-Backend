# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-815 · 2026-09-29] La fidelidad del tiempo de cocina ya puede fallar en el camino del modelo.

La dimensión `prep_time` del informe de fidelidad (`horizon._prep_time_issues`) lee el `prep_time` DECLARADO. En el día
determinista ese número es el de la receta (`_prep_time_source` = `receta`/`tecnica`); en el camino del modelo lo escribe
el propio modelo y ninguna marca lo decía: el instrumento juzgaba a uno con su receta y al otro con su palabra.
Refutación del informe de fidelidad (29-sep), plan 6594aae1 («Nada» = 10 min): 20 de 20 comidas «cumplían» por lo
declarado y 5 de 20 pasaban de 12,5 min por los minutos de sus pasos (la cena del día 2 declara 10 y sus pasos suman 21).
Y el `score` (`1 − issues/n_checks`, un issue POR COMIDA) ponía a 0 un plan con 6 de 12 comidas fuera.

Tres piezas, todas instrumento: no cambian qué se genera, ni el `score`, ni `issues` (el panel «solicitaste / aplicamos»
y el coach leen `issues`).

  · `sellar_llm` — en `assemble_plan_node`, lo que el modelo declaró queda `_prep_time_source="llm"`. Lo vacío sigue
    yendo a `recipe_library.fill_prep_time` (registry/unknown) y lo que ya trae fuente (receta/técnica) no se toca.
    `horizon._prep_minutes` mide `llm` igual que antes medía la ausencia de marca. El día de contingencia
    (`_day_fallback`, `_build_fallback_day` en `generate_days_parallel`) NO se sella: su «15 min» es un relleno de
    plantilla (P1-AUDITORIA-ARQ-VERIFICADA), no la palabra del modelo; tampoco se contrasta (`fallback_meals` lo cuenta).
  · `contraste_por_pasos` — los minutos EXPLÍCITOS de los pasos (`tiempo_pasos.minutos_de_fuego`, el SSOT del lote 323:
    extremo BAJO de los rangos, «por lado» ×2, lo que va «mientras / a la vez / aparte» en paralelo, sin tiempos pasivos
    ni mise en place) contra el mismo tope del formulario y la misma tolerancia (×1,25) que la auditoría ⇒
    `prep_time_measured` / `prep_time_over` por comida del modelo, en `_fidelity_report.prep_time_steps` y aplanado en
    `pipeline_metrics`. FUERA del score. Es una cota conservadora: la mise en place no cuenta y una cláusula paralela
    cuenta su máximo, así que infra-cuenta más de lo que sobre-cuenta (calibración en el informe del lote).
  · `score_v2` — por DIMENSIÓN, junto al `score` de siempre (que sigue siendo `1 − issues/n_checks`: el criterio
    «score ≥ 0.9 sostenido» de plan_policy_f3.md se lee con él y el ancla `n_checks == len(checks_run)` no cambia). Cada
    dimensión pesa lo mismo; la del tiempo vale `1 − fuera/medidas` (lo DECLARADO, sin tope de 10), las anclas
    `1 − issues/anclas`, las raciones de anclas `1 − anclas con issue/anclas medidas`, el resto 1 ó 0.

Knob `MEALFIT_FIDELITY_PREP_TIME_MEASURED` (True): False ⇒ ni sello ni campos nuevos, lo de antes byte a byte.

`by_source.sin_sello` solo aparece al medir fuera del pipeline (replay de `plan_data` guardado antes del lote, o de
un plan que perdió sus marcas después): dentro del pipeline toda comida pasa por `assemble_plan_node` antes de
`review_fidelity_gate`, y assemble sella `llm` todo lo que trae minutos sin fuente. Por eso un camino que pierde la
marca ANTES de assemble (p. ej. surgical regen → assemble) sale `llm`, no `sin_sello`: si esa comida era una receta
que el corrector no restauró (`reeleccion_dia.restaurar_procedencia` devuelve la marca a las que dejó iguales), queda
mal etiquetada. Fuera de alcance: el camino que BORRA las marcas del plan guardado (`_day_source`, `_prep_time_source`,
`_candidate_source`; plan d8b10b05, apuntado a P6-SURGICAL-PROMOTE sin aislar); en un replay sus comidas son `sin_sello`.
tooltip-anchor: P1-PLAN-LOTE-815
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

FUENTE_LLM = "llm"
#: `_prep_time_source` → etiqueta del contraste. Solo la palabra del modelo se contrasta con sus pasos.
_CONTRASTADAS = {FUENTE_LLM: "llm", "": "sin_sello"}
_TOLERANCIA = 1.25            # la de `horizon._prep_time_issues`
_TOPE_ITEMS = 10              # el detalle por comida; los conteos no tienen tope

_ANCLAS = ("anchor_missing_day", "anchor_slot_mismatch", "anchor_under_scheduled", "recurrence_above_band",
           "recurrence_below_band")
_COMPRA_UNICA = ("fresh_beyond_horizon", "protein_beyond_freeze_window", "frozen_needs_freezer")
_DIMENSION = {
    "exact_repeat_exceeded": "exact_repeat", "ingredient_days_exceeded": "ingredient_days",
    "culture_share_below": "culture_share", "culture_share_above": "culture_share",
    "culture_unavailable": "culture_share", "prep_time_over_budget": "prep_time",
    "anchor_portion_below": "anchor_portion", "anchor_portion_above": "anchor_portion",
    "equipment_unavailable": "equipment",
    **{c: "anchors" for c in _ANCLAS}, **{c: "single_trip" for c in _COMPRA_UNICA},
}


def activo() -> bool:
    try:
        from knobs import _env_bool
        return bool(_env_bool("MEALFIT_FIDELITY_PREP_TIME_MEASURED", True))
    except Exception:                                                          # noqa: BLE001
        return True


def es_contingencia(day) -> bool:
    """El día de plantilla matemática (`_day_fallback`): su «15 min» lo puso `_build_fallback_day`, no el modelo."""
    return isinstance(day, dict) and bool(day.get("_day_fallback"))


def sellar_llm(meal, day=None) -> None:
    """`_prep_time_source="llm"` en la comida cuyo `prep_time` declaró el modelo (tiene minutos y ninguna fuente).
    Con el `day` al lado, la contingencia (`_day_fallback`) no se sella."""
    try:
        if es_contingencia(day):
            return
        if isinstance(meal, dict) and meal.get("prep_time") and not meal.get("_prep_time_source") and activo():
            meal["_prep_time_source"] = FUENTE_LLM
    except Exception:                                                          # noqa: BLE001
        pass


def _presupuesto(form_data) -> Optional[int]:
    from horizon import _COOKING_TIME_BUDGET_MIN
    return _COOKING_TIME_BUDGET_MIN.get(str((form_data or {}).get("cookingTime") or "").strip().lower())


def _minutos(x: float):
    return int(x) if float(x).is_integer() else round(float(x), 1)


def contraste_por_pasos(days, form_data) -> dict:
    """Las comidas del modelo, su minuto declarado y el que dicen sus pasos. Puro; nunca lanza."""
    from horizon import _prep_minutes, _tanda_581
    from tiempo_pasos import minutos_de_fuego
    budget = _presupuesto(form_data)
    techo = budget * _TOLERANCIA if budget else None
    por_fuente = {"llm": 0, "sin_sello": 0}
    comidas = medidas = fuera = fuera_declarado = subdeclaradas = contingencia = 0
    items: list = []
    for i, d in enumerate(days or []):
        for m in ((d.get("meals") or []) if isinstance(d, dict) else []):
            if not isinstance(m, dict):
                continue
            if es_contingencia(d):
                contingencia += 1 if m.get("prep_time") else 0
                continue                      # plantilla, no palabra del modelo
            fuente = _CONTRASTADAS.get(str(m.get("_prep_time_source") or ""))
            declarado = _prep_minutes(m) if fuente else None
            if declarado is None:
                continue                      # receta/técnica/registry/unknown, o el modelo no dio minutos
            comidas += 1
            por_fuente[fuente] += 1
            if techo is not None and declarado > techo:
                fuera_declarado += 1
            pasos = float(minutos_de_fuego(m.get("recipe")) or 0)
            if pasos <= 0:
                continue                      # pasos sin minutos explícitos: no se puede decir
            medidas += 1
            over = (pasos > techo) if techo is not None else None
            fuera += 1 if over else 0
            subdeclaradas += 1 if pasos > declarado else 0
            if (over or pasos > declarado) and len(items) < _TOPE_ITEMS:
                items.append({"day": i + 1, "meal": str(m.get("meal") or ""), "name": str(m.get("name") or "")[:60],
                              "source": fuente, "declared": declarado, "prep_time_measured": _minutos(pasos),
                              "prep_time_over": over})
    return {"version": 1, "budget": budget, "tolerance": _TOLERANCIA, "tanda_581": bool(_tanda_581(form_data)),
            "meals": comidas, "by_source": por_fuente, "prep_time_measured": medidas, "prep_time_over": fuera,
            "over_by_declared": fuera_declarado, "understated": subdeclaradas, "fallback_meals": contingencia,
            "items": items}


def score_v2(report: dict, days, form_data, effective, sl=None) -> tuple[Optional[float], dict]:
    """(score por dimensión, {dimensión: valor}). Mismas mediciones que el `score` de siempre, otro agregado."""
    from horizon import FRESH_HORIZON_DAYS, _prep_minutes, single_trip_policy
    issues = [i for i in (report.get("issues") or []) if isinstance(i, dict)]
    checks = list(report.get("checks_run") or [])
    por_dim: dict = {}
    for it in issues:
        por_dim.setdefault(_DIMENSION.get(str(it.get("code") or ""), "otros"), []).append(it)
    dims: dict = {}
    n_anclas = int(report.get("n_checks") or 0) - len(checks)
    if n_anclas > 0 or por_dim.get("anchors"):
        dims["anchors"] = round(1.0 - min(1.0, len(por_dim.get("anchors") or []) / float(max(1, n_anclas))), 3)
    raciones = [c for c in checks if str(c).startswith("anchor_portion:")]
    for c in checks:
        if c in dims or str(c).startswith("anchor_portion:"):
            continue
        if c == "prep_time":
            budget = _presupuesto(form_data)
            mins = [x for x in (_prep_minutes(m) for d in (days or []) if isinstance(d, dict)
                                for m in (d.get("meals") or []) if isinstance(m, dict)) if x is not None]
            if budget and mins:
                dims[c] = round(1.0 - sum(1 for x in mins if x > budget * _TOLERANCIA) / float(len(mins)), 3)
                continue
        dims[c] = 0.0 if por_dim.get(c) else 1.0
    if raciones or por_dim.get("anchor_portion"):
        con_issue = {str(i.get("anchor") or "") for i in (por_dim.get("anchor_portion") or [])}
        dims["anchor_portion"] = round(1.0 - min(1.0, len(con_issue) / float(max(1, len(raciones)))), 3)
    # la compra única solo se mide en los días del ciclo más allá del horizonte de frescos
    off = int((sl or {}).get("days_offset") or 0) if isinstance(sl, dict) else 0
    n_dias = len([d for d in (days or []) if isinstance(d, dict)])
    if (single_trip_policy(effective) and off + n_dias > FRESH_HORIZON_DAYS) or por_dim.get("single_trip"):
        dims["single_trip"] = 0.0 if por_dim.get("single_trip") else 1.0
    for dim in por_dim:
        dims.setdefault(dim, 0.0)             # una dimensión con issues que no se declaró medida también cuenta
    if not dims:
        return None, {}
    return round(sum(dims.values()) / len(dims), 3), dims


def telemetria(days, form_data, report: dict, sl=None, effective=None) -> dict:
    """Lo que `horizon.fidelity_report` añade a su informe. {} con el knob apagado o ante error (fail-open)."""
    if not activo():
        return {}
    out: dict = {}
    try:
        out["prep_time_steps"] = contraste_por_pasos(days, form_data)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-815] contraste por pasos falló (fail-open): {e!r}")
    try:
        s, dims = score_v2(report, days, form_data, effective, sl)
        out["score_v2"], out["score_v2_dims"] = s, dims
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-815] score_v2 falló (fail-open): {e!r}")
    return out


def metadata_plana(report: dict) -> dict:
    """Aplanado para `pipeline_metrics.metadata` (el cron agrega por columnas jsonb, no por sub-objeto)."""
    try:
        if not isinstance(report, dict) or ("score_v2" not in report and "prep_time_steps" not in report):
            return {}
        pt = report.get("prep_time_steps") if isinstance(report.get("prep_time_steps"), dict) else {}
        return {"score_v2": report.get("score_v2"), "prep_time_llm_meals": pt.get("meals"),
                "prep_time_measured": pt.get("prep_time_measured"), "prep_time_over": pt.get("prep_time_over"),
                "prep_time_over_declared": pt.get("over_by_declared"), "prep_time_understated": pt.get("understated")}
    except Exception:                                                          # noqa: BLE001
        return {}


__all__ = ["FUENTE_LLM", "activo", "es_contingencia", "sellar_llm", "contraste_por_pasos", "score_v2", "telemetria", "metadata_plana"]
