# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-288 · 2026-09-25] La meta del bariátrico que BAJA de peso respeta su propia regla clínica.

El sistema se contradecía a sí mismo. La regla bariátrica que recibe la IA dice «1.400-1.700 kcal al día, aceite ≤ 1
cdta por comida»; la meta calculada para la bariátrica de la batería (95 kg, SOP, perder grasa a ritmo gradual) era
2.000 kcal con 93 g de grasa (42 %): el techo bariátrico general es 2.000 (decisión clínica para el bariátrico ESTABLE en
mantenimiento: no inducir déficit) y el tope de proteína del pouch (80 g) mandaba TODO lo que liberaba a grasa. Los pases
de ajuste escalaban el aceite hasta 20 g por comida para llegar, el revisor lo rechazaba («precisión de macros baja,
sobre todo grasas») y el plan salía con aviso tras ~11-13 minutos. Medido: bajar solo el techo a 1.700 dejaba la grasa en
44 %.

Dos cambios, SOLO para quien pidió perder grasa (el mantenimiento conserva su 2.000 y su reparto):
  · techo de la meta = `MEALFIT_BARIATRIC_LOSS_KCAL_CEILING_KCAL` (1.700, el tope de la propia regla);
  · lo que libera el tope de proteína va a grasa solo hasta `MEALFIT_BARIATRIC_FAT_MAX_PCT` (35 %) de la meta; el resto
    no se reasigna (el pouch no tiene volumen para él) y la meta baja con él — sin piso nuevo: el piso calórico general
    del cálculo ya corrió antes.
Knob maestro `MEALFIT_BARIATRIC_LOSS_TARGET` (on). Decisión tomada con la delegación del dueño (25-sep); revertible sin
redeploy. tooltip-anchor: P1-PLAN-LOTE-288-BARIATRICO-PIERDE"""
from __future__ import annotations


def activo() -> bool:
    try:
        from knobs import _env_bool
        return bool(_env_bool("MEALFIT_BARIATRIC_LOSS_TARGET", True))
    except Exception:
        return True


def techo_perdida() -> int:
    try:
        from knobs import _env_int
        return int(_env_int("MEALFIT_BARIATRIC_LOSS_KCAL_CEILING_KCAL", 1700, validator=lambda v: 1400 <= v <= 2200))
    except Exception:
        return 1700


def grasa_max_pct() -> float:
    try:
        from knobs import _env_float
        return float(_env_float("MEALFIT_BARIATRIC_FAT_MAX_PCT", 0.35, validator=lambda v: 0.25 <= v <= 0.45))
    except Exception:
        return 0.35


def techo(goal, techo_general) -> int:
    """El techo calórico bariátrico que aplica a este objetivo."""
    try:
        if activo() and str(goal or "") == "lose_fat":
            return min(int(techo_general), techo_perdida())
    except Exception:
        pass
    return int(techo_general)


def capar_proteina(mac: dict, tope_g: float, meta_kcal, goal) -> float:
    """Capa la proteína de `mac` a `tope_g` y pasa lo liberado a grasa (hasta el % máximo si pierde grasa). Devuelve las
    kcal que NO se reasignaron (0 fuera de pérdida). Muta `mac`; nunca lanza."""
    try:
        p = float(mac.get("protein_g") or 0)
        if p <= tope_g:
            return 0.0
        liberado = (p - tope_g) * 4.0
        mac["protein_g"] = round(tope_g)
        mac["protein_str"] = f"{round(tope_g)}g"
        grasa = float(mac.get("fats_g") or 0)
        if activo() and str(goal or "") == "lose_fat" and float(meta_kcal or 0) > 0:
            # x kcal a grasa con la meta FINAL = meta − (liberado − x): 9·grasa + x ≤ pct·(meta − liberado + x)
            pct = grasa_max_pct()
            x_max = max(0.0, (pct * (float(meta_kcal) - liberado) - 9.0 * grasa) / (1.0 - pct))
            sube = min(liberado, x_max) / 9.0
            sobra = liberado - sube * 9.0
        else:
            sube, sobra = liberado / 9.0, 0.0
        nueva = round(grasa + sube)
        mac["fats_g"] = nueva
        mac["fats_str"] = f"{nueva}g"
        return max(0.0, sobra)
    except Exception:
        return 0.0
