"""[P2-MDDA-PORTION-SOLVER · 2026-06-13] Solver determinista de porciones —
el "lado matemático" del cerebro dividido (MDDA).

Problema: el LLM elige los alimentos de cada comida y porciona "a ojo", lo que
produce drift de macros (sub-entrega de proteína sistemática — ver
`project_plan_quality_degraded_finding_2026_06_13`). Este solver toma los
ingredientes que el LLM eligió + el target de macros del slot, computa los macros
REALES por ingrediente (vía `nutrition_db`), y RE-ESCALA las porciones para clavar
el target — sin tocar QUÉ alimentos eligió el LLM (preserva la creatividad).

Algoritmo v1 — escalado proporcional por grupo de macro dominante (determinista,
sin scipy):
  1. Por ingrediente resoluble, clasificar por su macro DOMINANTE (mayor aporte
     calórico: 4·P vs 4·C vs 9·F).
  2. Por cada macro {protein, carbs, fats}: factor = target_macro / Σ(macro en su
     grupo), clamp a [min_scale, max_scale]. Escalar la cantidad de cada
     ingrediente del grupo por ese factor (los macros escalan lineal con la
     cantidad, sea cual sea la unidad → escalamos `quantity` directo).
  3. Ingredientes no-resolubles (sin macros / sin gramos) se dejan intactos.

Por qué proporcional y no LP/scipy: el 80% del valor (cerrar el déficit de
proteína 110g→154g) se logra con escalado por grupo, sin dependencia pesada ni
soluciones no-deterministas. Si la telemetría muestra que grupos acoplados
(un ingrediente que es P y C a la vez) necesitan optimización conjunta, se añade
LP entonces — no antes (convención del repo: no diseñar para requisitos hipotéticos).
"""
from __future__ import annotations

import logging
import os
import re as _re_ps
from typing import Optional


_log = logging.getLogger(__name__)

# Aportes calóricos Atwater (kcal/g) por macro — para decidir el macro dominante.
_KCAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fats": 9.0}


# [P2-SOLVER-KNOBS-REGISTRY · 2026-06-18] (audit fresco P2) Delegamos a los helpers de knobs.py para que
# los 6 knobs MEALFIT_SOLVER_* se auto-registren en _KNOBS_REGISTRY → visibles en /health/version. Antes
# leían os.environ crudo y eludían el registry: un override de los pesos del solver (núcleo de precisión)
# era invisible al operador. Fail-safe: si knobs no importa, helpers locales equivalentes (raw os.environ).
try:
    from knobs import (_env_float as _envf, _env_bool as _envb,
                       _env_int as _envi)  # auto-registran en _KNOBS_REGISTRY
except Exception:  # pragma: no cover - knobs siempre disponible en prod
    def _envi(name: str, default: int, validator=None) -> int:
        # [P1-SOLVER-LSQ-ITERS] espejo offline de knobs._env_int (mismo contrato que _envf).
        try:
            v = int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
        if validator is not None:
            try:
                if not validator(v):
                    return default
            except Exception:
                return default
        return v

    def _envf(name: str, default: float, validator=None) -> float:
        # [S-P3-a] acepta `validator` para paridad con knobs._env_float (fallback offline).
        try:
            v = float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
        if validator is not None:
            try:
                if not validator(v):
                    return default
            except Exception:
                return default
        return v

    def _envb(name: str, default: bool) -> bool:
        return str(os.environ.get(name, str(default))).strip().lower() in ("1", "true", "yes", "on")


# [M2-SOLVER-NNLS · 2026-06-14] Solver multi-restricción: reemplaza el escalado GREEDY por-grupo
# (que con ingredientes acoplados —pollo=P+grasa, arroz=C+P— no clava los 4 macros a la vez) por
# mínimos cuadrados ACOTADOS con regularización hacia x≈1, resuelto por descenso por coordenadas
# (box-QP convexo, exacto por coordenada, determinista, SIN dependencias — scipy no está instalado).
# El benchmark M2 midió la fuga: proteína 16% MAPE / solo 48% en ±10%. Este es el fix. Fallback al
# greedy si falla o se desactiva. Pesos: kcal+proteína priorizados (lo clínicamente crítico).
SOLVER_LSQ = _envb("MEALFIT_SOLVER_LSQ", True)
# Pesos TUNED por el benchmark M2 (A/B 2026-06-14): proteína 2.0 sobre-priorizaba y regresaba la
# grasa (12.2→15.8% MAPE). Rebalanceados → all-4-en-±10% subió de 24%→50% y la grasa volvió a 12.1%.
# [S-P3-a · 2026-07-07] (audit solver+seeder v3) validators de rango — espejo de los knobs hermanos
# SOLVER_MIN_COVERAGE/SOLVER_PARTIAL_MAX_SCALE (graph_orchestrator.py), que SÍ los usan. Un override
# swapeado/negativo era aceptado en silencio y re-escalaba cada ingrediente sin WARNING. Fuera de rango →
# WARNING + fallback al default (knobs._env_float).
# [P1-SOLVER-KCAL-ROW-REDUNDANT · 2026-07-29] (audit solver+seeder v4) Default 1.2 → 0.1.
#
# La fila kcal del LSQ es una restricción REDUNDANTE, medida: sobre las 201 filas de
# `master_ingredients` con los 4 macros no nulos, |kcal_catálogo − (4·(P+C) + 9·F)| / kcal tiene
# p50 = 0.001 (0.1%) y 63% de las filas bajo el 2% — o sea, para todo lo que pesa en un plato la
# fila kcal ES `4·fila_P + 4·fila_C + 9·fila_F`. No añade información nueva, pero como las filas
# se arman en unidades ABSOLUTAS (`A_rows`/`brow` abajo, ~500-720 kcal vs ~16-72 g de macro) el
# peso EFECTIVO de cada ecuación es `w·b²` → con w=1.2 la fila kcal se llevaba el **98.2% del
# objetivo**: los otros tres knobs `SOLVER_W_*` movían décimas de punto porcentual, y el problema
# quedaba mal condicionado para el descenso por coordenadas (ver `iters` en `_box_lsq`).
#
# ⚠️ Lo que este cambio NO hace: con 0.1 la fila kcal SIGUE dominando (share 81.7-84.0% según el
# slot) — solo domina MENOS. Los pesos declarados siguen sin ser los efectivos. Devolverles su
# significado exige re-tunear los CUATRO simultáneamente contra un harness de comidas vivas; esto
# compra los +10.1 pp medidos y nada más. No leer este knob como "el objetivo ya está bien puesto".
#
# Medido re-solviendo 416 comidas VIVAS (30 planes) con el mismo `_box_lsq`, variando SOLO este peso:
#   w=1.2 (previo) → 67.5% de comidas con P/C/F en ±10%   ·  w=0.1 → 77.6% (+10.1 pp)
# NO bajarlo a 0 (quitar la fila): P/C/F sube a 80.3% pero el MAPE de kcal salta a 2.3% y la banda
# kcal es la ESTRECHA [0.95, 1.05] → el all-4 empeora. Se conserva como regularizador suave del
# agregado. Rollback sin redeploy: `MEALFIT_SOLVER_W_KCAL=1.2`.
#
# ⚠️ NO "arreglar" esto normalizando las filas (`A/b`, `b=1`) para que los pesos declarados sean los
# efectivos: MEDIDO sobre las mismas 416 comidas, regresa **−27.1 pp** de convergencia (78.1% → 51.0%,
# MAPE proteína 4.7% → 9.9%). Con filas absolutas `w·b²` da a la proteína (b≈40 g) ~6.7× el peso de la
# grasa (b≈16 g), que es clínicamente lo que se quiere; normalizar las iguala y de-prioriza la proteína.
# Lo que sobra es la FILA KCAL, no la escala. tooltip-anchor: P1-SOLVER-KCAL-ROW-REDUNDANT
SOLVER_W_KCAL = _envf("MEALFIT_SOLVER_W_KCAL", 0.1, lambda v: 0.0 < v <= 10.0)
SOLVER_W_PROTEIN = _envf("MEALFIT_SOLVER_W_PROTEIN", 1.5, lambda v: 0.0 < v <= 10.0)
SOLVER_W_CARBS = _envf("MEALFIT_SOLVER_W_CARBS", 1.1, lambda v: 0.0 < v <= 10.0)
SOLVER_W_FATS = _envf("MEALFIT_SOLVER_W_FATS", 1.4, lambda v: 0.0 < v <= 10.0)
# Regularización hacia el porcionado original del LLM (x=1): evita porciones absurdas (un
# ingrediente a min_scale y otro a max_scale solo para clavar macros). Más alto = más fiel al LLM.
SOLVER_LSQ_REG = _envf("MEALFIT_SOLVER_LSQ_REG", 0.10, lambda v: 0.0 <= v <= 5.0)
# [P1-SOLVER-LSQ-ITERS · 2026-07-29] (audit solver+seeder v4) El tope de barridos de `_box_lsq` era un
# default de PARÁMETRO plano (`iters: int = 150`) que ningún callsite pasaba → ni knob, ni rollback, ni
# A/B sin redeploy (mismo defecto que S-P2-a cerró para MIN_SCALE/MAX_SCALE).
#
# Peor: el criterio de parada NO DISPARA en producción. Instrumentando `_box_lsq` con el criterio real
# (`max_delta < 1e-7`) y tope 100.000 sobre 416 comidas VIVAS: barridos necesarios p50 = 82.560, y el
# **99.0% de las comidas necesitan más de 150** (95.4% más de 400). O sea que hoy el solver SIEMPRE
# retorna por agotamiento, en un punto que no es su propio óptimo — el docstring que promete "converge
# al óptimo global" describía la teoría, no la corrida.
#
# Medido re-solviendo las mismas 416 comidas variando SOLO el tope (objetivo de producción intacto):
#   150 (previo) → 67.5% con P/C/F en ±10%  ·  400 → 74.8% (+7.3 pp)  ·  5.000 → 78.1%
#   MAPE grasa 11.0% → 9.8% → 9.0%   ·   MAPE proteína 6.1% → 5.1% → 4.7%
# Default 400 y no 5.000: los ~3 pp extra se solapan con lo que P1-SOLVER-KCAL-ROW-REDUNDANT consigue
# más barato (bajar el peso de la fila kcal mejora el condicionamiento, que es la causa del mal
# condicionamiento que exige tantos barridos). El coste CPU de 400 es indistinguible (≤15 vars, ≤4 filas,
# pure-python). Rollback sin redeploy: `MEALFIT_SOLVER_LSQ_ITERS=150`. tooltip-anchor: P1-SOLVER-LSQ-ITERS
SOLVER_LSQ_ITERS = _envi("MEALFIT_SOLVER_LSQ_ITERS", 400, lambda v: 50 <= v <= 20000)

# [P3-SOLVER-CONVERGED-BAND · 2026-07-29] (audit solver+seeder v4) `converged` no era comparable con
# la banda clínica que alimenta, por tres desalineaciones acumuladas:
#   (a) el bucle solo mira ("protein","carbs","fats") — kcal queda FUERA del criterio, justo la fila
#       que domina el objetivo;
#   (b) `|achieved−t|/t > tol` es SIMÉTRICO ±10% mientras la banda real es ASIMÉTRICA [0.90, 1.12];
#   (c) `tolerance_pct` era un default de parámetro, no un knob.
# Efecto: una comida a 1.11× de proteína se marcaba NO convergida (está DENTRO de la banda) y una a
# 1.30× de kcal se reportaba convergida (la banda kcal es la MÁS estrecha, [0.95, 1.05]). Con eso el
# 57.2% medido no se puede mapear a resultados de banda: un operador que lo vea mejorar no sabe si
# mejoró el plan.
# `..._USES_BAND` nace OFF: cambia la semántica de una serie que ya tiene línea base medida hoy
# (57.2%) — encenderlo sin avisar rompería la comparabilidad histórica, que es justo lo que este fix
# quiere ganar. Encender junto con un corte de serie. tooltip-anchor: P3-SOLVER-CONVERGED-BAND
SOLVER_TOLERANCE_PCT = _envf("MEALFIT_SOLVER_TOLERANCE_PCT", 0.10, lambda v: 0.01 <= v <= 0.50)
SOLVER_CONVERGED_USES_BAND = _envb("MEALFIT_SOLVER_CONVERGED_USES_BAND", False)
SOLVER_BAND_LOWER = _envf("MEALFIT_SOLVER_BAND_LOWER", 0.90, lambda v: 0.5 <= v < 1.0)
SOLVER_BAND_UPPER = _envf("MEALFIT_SOLVER_BAND_UPPER", 1.12, lambda v: 1.0 < v <= 2.0)
SOLVER_BAND_KCAL_LOWER = _envf("MEALFIT_SOLVER_BAND_KCAL_LOWER", 0.95, lambda v: 0.5 <= v < 1.0)
SOLVER_BAND_KCAL_UPPER = _envf("MEALFIT_SOLVER_BAND_KCAL_UPPER", 1.05, lambda v: 1.0 < v <= 2.0)
# [P3-SOLVER-FEASIBILITY · 2026-07-29] (audit solver+seeder v4) Telemetría PURA: el solver sabe si el
# target es físicamente inalcanzable con los alimentos elegidos (cotas exactas de la caja, O(n)) y no
# lo decía. Caso medido: merienda 'Yogurt Griego con Guineo y Avena Tostada' con slot-target de grasa
# 8.5 g cuando el MÁXIMO escalando todo al tope es 6.0 g — no falta escalado, falta un PORTADOR de
# grasa. La reparación aguas abajo no puede distinguir un caso del otro sin esta señal.
SOLVER_FEASIBILITY_SIGNAL = _envb("MEALFIT_SOLVER_FEASIBILITY_SIGNAL", True)

# [S-P2-a / P2-SOLVER-SCALE-KNOBS · 2026-07-07] (audit solver+seeder v2) El clamp de escala del solver eran
# params default PLANOS (0.3/3.5), NO knobs → invisibles en /health/version y sin rollback/A-B sin redeploy
# (viola la convención del repo: "cambios de comportamiento reversibles van como knob"). Promovidos a knobs.
# Además `max_scale` para líneas PROTEÍNA-dominantes sube a 5.0 (la "opción b" del audit): la telemetría
# `solver_clamp` mostró ~74% de meals saturando el clamp, mayormente ARRIBA (sub-entrega de proteína). Con un
# techo mayor SOLO para la proteína, el solver clava la proteína ESCALANDO la línea existente antes de que el
# closer AÑADA una línea nueva (mejor coherencia/variedad). Acotado aguas abajo por los realism-caps
# (PORTION_CAP_PROTEIN_G / _cap_unrealistic_portions) y el protein-ceiling-trim (g/kg goal-aware). Rollback:
# MEALFIT_SOLVER_MAX_SCALE_PROTEIN=3.5 (iguala al general → comportamiento previo).
SOLVER_MIN_SCALE = _envf("MEALFIT_SOLVER_MIN_SCALE", 0.3, lambda v: 0.05 <= v <= 1.0)
SOLVER_MAX_SCALE = _envf("MEALFIT_SOLVER_MAX_SCALE", 3.5, lambda v: 1.0 <= v <= 8.0)
SOLVER_MAX_SCALE_PROTEIN = _envf("MEALFIT_SOLVER_MAX_SCALE_PROTEIN", 5.0, lambda v: 1.0 <= v <= 8.0)
# [S-P3-a · 2026-07-07] Guard de INVERSIÓN post-validators: aun con cada knob en su rango, un swap
# (MIN=1.0/MAX=1.0 imposible, pero MIN cerca de MAX, o proteína < general) degeneraría el clamp
# por-coordenada de `_box_lsq` (todo forzado a un bound). Fail-safe a defaults + WARNING. tooltip-anchor: S-P3-a
if not (SOLVER_MIN_SCALE < SOLVER_MAX_SCALE):
    logging.getLogger(__name__).warning(
        f"[S-P3-a] MEALFIT_SOLVER_MIN_SCALE ({SOLVER_MIN_SCALE}) >= MAX_SCALE ({SOLVER_MAX_SCALE}) — "
        f"clamp degenerado. Fallback a defaults 0.3/3.5.")
    SOLVER_MIN_SCALE, SOLVER_MAX_SCALE = 0.3, 3.5
if SOLVER_MAX_SCALE_PROTEIN < SOLVER_MAX_SCALE:
    logging.getLogger(__name__).warning(
        f"[S-P3-a] MEALFIT_SOLVER_MAX_SCALE_PROTEIN ({SOLVER_MAX_SCALE_PROTEIN}) < MAX_SCALE "
        f"({SOLVER_MAX_SCALE}) — la proteína no debe escalar MENOS que el general. Igualado al general.")
    SOLVER_MAX_SCALE_PROTEIN = SOLVER_MAX_SCALE


def _box_lsq(A_rows: list, b: list, weights: list, lo: float, hi: float,
             reg: float, iters: int = None) -> list:
    """Mínimos cuadrados ACOTADOS con regularización hacia x=1, por descenso por coordenadas.
    Minimiza  Σ_r w_r (Σ_j A[r][j]·x_j − b[r])²  +  reg·Σ_j (x_j − 1)²  s.a. x_j ∈ [lo, hi].
    Problema convexo pequeño (≤~15 vars, ≤4 filas) → CD con minimización 1D exacta por coordenada
    converge al óptimo global. Determinista, pure-python (sin numpy/scipy). Retorna x (factores).

    [P1-SOLVER-LSQ-ITERS · 2026-07-29] `iters=None` → `SOLVER_LSQ_ITERS` (knob). OJO al leer esto:
    la convergencia al óptimo global es la propiedad del ALGORITMO, no de la corrida — medido en
    prod, el 99% de las comidas AGOTA el tope antes de que `max_delta < 1e-7` dispare, así que el
    retorno es un punto sub-óptimo cuya calidad depende directamente de `iters`."""
    if iters is None:
        iters = SOLVER_LSQ_ITERS
    nrows = len(A_rows)
    n = len(A_rows[0]) if nrows else 0
    x = [1.0] * n
    if n == 0:
        return x
    # [S-P2-a] lo/hi por-COORDENADA (protein-dominante → hi mayor) o escalar (retro-compat).
    _lo = list(lo) if isinstance(lo, (list, tuple)) else [lo] * n
    _hi = list(hi) if isinstance(hi, (list, tuple)) else [hi] * n
    denom = [reg + sum(weights[r] * A_rows[r][i] ** 2 for r in range(nrows)) for i in range(n)]
    res = [sum(A_rows[r][i] * x[i] for i in range(n)) - b[r] for r in range(nrows)]  # Σ A·x − b
    for _ in range(iters):
        max_delta = 0.0
        for i in range(n):
            if denom[i] <= 0:
                continue
            num = reg  # = reg·1 (target del prior)
            for r in range(nrows):
                a = A_rows[r][i]
                if a != 0.0:
                    num -= weights[r] * a * (res[r] - a * x[i])  # c_r = res_r − a·x_i
            xi = num / denom[i]
            xi = _lo[i] if xi < _lo[i] else (_hi[i] if xi > _hi[i] else xi)
            d = xi - x[i]
            if d != 0.0:
                for r in range(nrows):
                    res[r] += A_rows[r][i] * d
                x[i] = xi
                if abs(d) > max_delta:
                    max_delta = abs(d)
        if max_delta < 1e-7:
            break
    return x


def _compute_scale_factors(entries: list, tgt: dict, min_scale: float, max_scale: float,
                           max_scale_protein: float = None) -> tuple:
    """Factor de escalado POR-INGREDIENTE (alineado con `entries`). Usa el solver LSQ multi-macro
    si está habilitado; si no (o si falla), cae al greedy por-grupo. `entries[i]` debe tener
    `macros` ({kcal,protein,carbs,fats}|None) y `group` (macro dominante|None).
    [S-P2-a] `hi` por-COORDENADA: las líneas PROTEÍNA-dominantes usan `max_scale_protein` (≥ max_scale);
    el resto `max_scale`. Retorna (factors, method, saturated_hi, saturated_lo) — `saturated_*` cuenta los
    factores clavados en su bound per-línea (telemetría exacta, no un umbral fijo)."""
    factors = [1.0] * len(entries)
    sc = [i for i, e in enumerate(entries) if e.get("macros") and e.get("group")]
    if not sc:
        return factors, "none", 0, 0
    _mxp = max_scale if max_scale_protein is None else max_scale_protein
    _hi_sc = [(_mxp if entries[i]["group"] == "protein" else max_scale) for i in sc]  # hi por-coordenada
    if SOLVER_LSQ:
        try:
            _w = {"kcal": SOLVER_W_KCAL, "protein": SOLVER_W_PROTEIN,
                  "carbs": SOLVER_W_CARBS, "fats": SOLVER_W_FATS}
            A_rows, brow, wrow = [], [], []
            for m in ("kcal", "protein", "carbs", "fats"):
                if tgt.get(m, 0) > 0:  # solo ecuaciones con target real (evita forzar macro→0)
                    A_rows.append([entries[i]["macros"][m] for i in sc])
                    brow.append(float(tgt[m]))
                    wrow.append(_w[m])
            if A_rows:
                xs = _box_lsq(A_rows, brow, wrow, min_scale, _hi_sc, SOLVER_LSQ_REG)
                sat_hi = sat_lo = 0
                for j, i in enumerate(sc):
                    factors[i] = xs[j]
                    if xs[j] >= _hi_sc[j] * 0.999:
                        sat_hi += 1
                    elif xs[j] <= min_scale * 1.001:
                        sat_lo += 1
                return factors, "lsq", sat_hi, sat_lo
        except Exception:
            pass
    # Fallback GREEDY por grupo de macro dominante (algoritmo v1).
    gf = {}
    for macro in _KCAL_PER_G:
        current = sum(entries[i]["macros"][macro] for i in sc if entries[i]["group"] == macro)
        tv = tgt.get(macro, 0)
        _hi_m = _mxp if macro == "protein" else max_scale
        gf[macro] = max(min_scale, min(_hi_m, tv / current)) if (current > 0 and tv > 0) else 1.0
    sat_hi = sat_lo = 0
    for i in sc:
        factors[i] = gf[entries[i]["group"]]
        _hi_i = _mxp if entries[i]["group"] == "protein" else max_scale
        if factors[i] >= _hi_i * 0.999:
            sat_hi += 1
        elif factors[i] <= min_scale * 1.001:
            sat_lo += 1
    return factors, "greedy", sat_hi, sat_lo


def _converged_report(achieved: dict, tgt: dict, tolerance_pct: float) -> tuple:
    """[P3-SOLVER-CONVERGED-BAND · 2026-07-29] `(converged: bool, per_macro: dict)`.

    `per_macro` existe porque el criterio viejo hacía `break` al primer macro fuera → se PERDÍA cuál
    falló, que es justo el dato que vuelve accionable la métrica de no-convergencia.

    Con `SOLVER_CONVERGED_USES_BAND` OFF (default) el bool es idéntico al de siempre (±tol simétrico
    sobre P/C/F); ON evalúa el ratio contra la banda REAL y añade kcal con su banda estrecha."""
    per: dict = {}
    _macros = ("protein", "carbs", "fats", "kcal") if SOLVER_CONVERGED_USES_BAND \
        else ("protein", "carbs", "fats")
    for m in _macros:
        t = float(tgt.get(m) or 0.0)
        if t <= 0:
            continue
        ratio = float(achieved.get(m) or 0.0) / t
        if SOLVER_CONVERGED_USES_BAND:
            lo, hi = ((SOLVER_BAND_KCAL_LOWER, SOLVER_BAND_KCAL_UPPER) if m == "kcal"
                      else (SOLVER_BAND_LOWER, SOLVER_BAND_UPPER))
            per[m] = bool(lo <= ratio <= hi)
        else:
            per[m] = bool(abs(ratio - 1.0) <= tolerance_pct)
    return (all(per.values()) if per else True), per


def _feasibility_report(entries: list, tgt: dict, min_scale: float,
                        hi_by_entry: list) -> "dict | None":
    """[P3-SOLVER-FEASIBILITY · 2026-07-29] ¿El target es alcanzable con ESTOS alimentos dentro del
    clamp? Cota exacta por coordenada: `m_max = Σ hi_i·a_mi`, `m_min = Σ lo_i·a_mi` (condición
    NECESARIA, O(n), determinista, sin dependencias).

    Devuelve `{macro: 'high'|'low'}` para los macros infactibles, o `{}` si todos son alcanzables.
    'high' = ni escalando todo al techo se llega (falta un PORTADOR, no escalado); 'low' = ni
    bajando todo al piso se baja lo suficiente. `None` si no hay nada que evaluar."""
    if not SOLVER_FEASIBILITY_SIGNAL or not entries:
        return None
    try:
        out: dict = {}
        for m in ("kcal", "protein", "carbs", "fats"):
            t = float(tgt.get(m) or 0.0)
            if t <= 0:
                continue
            _mx = _mn = 0.0
            for j, e in enumerate(entries):
                _a = float((e.get("macros") or {}).get(m) or 0.0)
                if _a <= 0:
                    continue
                _hi = hi_by_entry[j] if j < len(hi_by_entry) else 1.0
                _mx += _a * _hi
                _mn += _a * min_scale
            if _mx > 0 and t > _mx:
                out[m] = "high"
            elif t < _mn:
                out[m] = "low"
        return out
    except Exception:
        return None


def _get(d: dict, *keys, default=0.0):
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _coerce_line(ing) -> tuple:
    """ing dict o string → (quantity, unit, name). Tolerante a aliases de key."""
    if isinstance(ing, dict):
        qty = ing.get("quantity", ing.get("qty", ing.get("amount")))
        unit = ing.get("unit", "unidad")
        name = ing.get("name") or ing.get("ingredient_name") or ing.get("item_name") or ""
        return qty, unit, name
    # string "150 g pechuga de pollo" → delega el parseo al shopping_calculator
    try:
        from shopping_calculator import _parse_quantity
        q, u, n = _parse_quantity(ing, apply_yield_multiplier=False)
        return q, u, n
    except Exception:
        return None, "unidad", str(ing)


def _target_macros(target: dict) -> dict:
    return {
        "kcal": _get(target, "kcal", "cals", "calories", "target_calories"),
        "protein": _get(target, "protein", "protein_g", "proteina"),
        "carbs": _get(target, "carbs", "carbs_g", "carbohidratos"),
        "fats": _get(target, "fats", "fat", "fats_g", "grasas"),
    }


def solve_portion_macros(
    ingredients: list,
    target: dict,
    db=None,
    *,
    min_scale: float = None,
    max_scale: float = None,
    max_scale_protein: float = None,
    tolerance_pct: float = 0.10,
) -> dict:
    """Re-escala porciones para clavar el target de macros del slot.

    Args:
        ingredients: lista de dicts {name, quantity, unit} (o strings parseables).
        target: macros objetivo del slot {kcal, protein, carbs, fats} (acepta aliases).
        db: IngredientNutritionDB; si None se instancia uno (carga master_ingredients).
        min_scale/max_scale: clamp del factor por grupo (evita porciones absurdas).
        tolerance_pct: para reportar `converged` (|achieved-target|/target ≤ tol).

    Returns:
        dict con:
          - ingredients: lista re-escalada (mismas keys de entrada, `quantity` ajustada).
          - achieved: {kcal,protein,carbs,fats} reales tras el escalado (solo resolubles).
          - target: macros objetivo normalizados.
          - report: por macro {current, target, factor, applied}.
          - resolved_count / unresolved: cuántos ingredientes se pudieron computar.
          - converged: bool (todos los macros con target>0 dentro de tolerancia).
    """
    if db is None:
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
    tgt = _target_macros(target)
    # [S-P2-a] defaults desde knobs (None → knob).
    min_scale = SOLVER_MIN_SCALE if min_scale is None else min_scale
    max_scale = SOLVER_MAX_SCALE if max_scale is None else max_scale
    max_scale_protein = SOLVER_MAX_SCALE_PROTEIN if max_scale_protein is None else max_scale_protein

    # 1) computar macros por ingrediente + clasificar por macro dominante.
    entries = []  # cada uno: {idx, qty, unit, name, macros|None, group|None}
    for idx, ing in enumerate(ingredients):
        qty, unit, name = _coerce_line(ing)
        macros = db.macros_for_line(qty, unit, name) if name else None
        group = None
        if macros:
            contrib = {m: macros[m] * _KCAL_PER_G[m] for m in _KCAL_PER_G}
            if any(contrib.values()):
                group = max(contrib, key=contrib.get)
        entries.append({"idx": idx, "qty": _get({"q": qty}, "q") if qty is not None else qty,
                        "raw_qty": qty, "unit": unit, "name": name,
                        "macros": macros, "group": group})

    # 2) factor de escalado POR-INGREDIENTE (LSQ multi-macro; greedy fallback). El `report`
    #    greedy por-macro se conserva como telemetría.
    ing_factors, method, _sat_hi, _sat_lo = _compute_scale_factors(
        entries, tgt, min_scale, max_scale, max_scale_protein)
    report = {}
    for macro in _KCAL_PER_G:  # protein, carbs, fats
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        gfactor = max(min_scale, min(max_scale, target_v / current)) if (current > 0 and target_v > 0) else 1.0
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(gfactor, 4), "applied": abs(gfactor - 1.0) > 1e-9}

    # 3) aplicar el factor por-ingrediente a la cantidad.
    out_ingredients = []
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for idx, (e, ing) in enumerate(zip(entries, ingredients)):
        new_ing = dict(ing) if isinstance(ing, dict) else {"name": e["name"],
                                                            "quantity": e["raw_qty"], "unit": e["unit"]}
        if e["macros"] and e["group"]:
            f = ing_factors[idx]
            base_q = e["raw_qty"]
            try:
                new_ing["quantity"] = round(float(base_q) * f, 2)
            except (TypeError, ValueError):
                pass
            for m in achieved:
                achieved[m] += e["macros"][m] * f
            resolved += 1
        elif e["macros"]:  # resoluble pero sin grupo (macros todos 0, e.g. agua)
            for m in achieved:
                achieved[m] += e["macros"][m]
            resolved += 1
        out_ingredients.append(new_ing)

    achieved = {m: round(v, 1) for m, v in achieved.items()}
    # [P3-SOLVER-FEASIBILITY · 2026-07-29] cotas exactas de la caja con los MISMOS bounds que usó
    # el solver (proteína-dominante lleva su techo propio). Telemetría: no cambia ni un gramo.
    _hi_all = [(max_scale_protein if e.get("group") == "protein" else max_scale) for e in entries]
    _infeasible = _feasibility_report(entries, tgt, min_scale, _hi_all)

    # [P3-SOLVER-CONVERGED-BAND · 2026-07-29] criterio compartido + desglose per-macro (el `break`
    # previo perdía CUÁL macro falló, que es el dato que vuelve accionable la métrica).
    converged, converged_per_macro = _converged_report(achieved, tgt, tolerance_pct)

    return {
        "ingredients": out_ingredients,
        "achieved": achieved,
        "target": tgt,
        "report": report,
        # [P3-6 · 2026-07-07] `report` es una referencia GREEDY per-macro (target/current por grupo), NO los
        # factores LSQ realmente aplicados (esos están en factors_applied/method/saturated_*). Etiquetado.
        "report_basis": "greedy-reference",
        "method": method,
        "resolved_count": resolved,
        "unresolved": len(ingredients) - resolved,
        "converged": converged,
        # [P3-SOLVER-CONVERGED-BAND] qué macro falló, no solo que falló alguno.
        "converged_per_macro": converged_per_macro,
        # [P3-SOLVER-FEASIBILITY] {macro: 'high'|'low'} si el target es inalcanzable con estos
        # alimentos dentro del clamp: distingue "falta escalado" de "falta un PORTADOR".
        "infeasible": _infeasible,
        "residuals": {m: (round(achieved[m] / tgt[m], 3) if tgt.get(m) else None)
                      for m in ("kcal", "protein", "carbs", "fats")},
        "saturated_hi": _sat_hi,
        "saturated_lo": _sat_lo,
    }


def solve_meal_macros(
    ingredient_strings: list,
    target: dict,
    db=None,
    *,
    min_scale: float = None,
    max_scale: float = None,
    max_scale_protein: float = None,
    tolerance_pct: float = 0.10,
) -> dict:
    """Variante para los ingredientes-STRING de un meal del plan ("0.5 taza de avena
    (50g)"). Mismo algoritmo que `solve_portion_macros` pero re-escribe los strings
    (cantidad líder + hint de gramos) en vez de un campo `quantity`, preservando el
    formato que consumen el coherence guard + shopping aggregator + frontend.

    Returns dict con `ingredients` (lista de strings re-escalados), `achieved`,
    `target`, `report`, `resolved_count`, `unresolved`, `converged`.
    """
    if db is None:
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
    from nutrition_db import rescale_ingredient_string
    tgt = _target_macros(target)
    # [S-P2-a] defaults desde knobs (None → knob; el caller override, e.g. cobertura parcial pasa 2.0).
    min_scale = SOLVER_MIN_SCALE if min_scale is None else min_scale
    max_scale = SOLVER_MAX_SCALE if max_scale is None else max_scale
    max_scale_protein = SOLVER_MAX_SCALE_PROTEIN if max_scale_protein is None else max_scale_protein

    entries = []
    for s in ingredient_strings:
        macros = db.macros_from_ingredient_string(s)
        group = None
        if macros:
            contrib = {m: macros[m] * _KCAL_PER_G[m] for m in _KCAL_PER_G}
            if any(contrib.values()):
                group = max(contrib, key=contrib.get)
        entries.append({"s": s, "macros": macros, "group": group})

    # [M2-SOLVER-NNLS] Factor POR-INGREDIENTE (LSQ multi-macro; greedy fallback). Reemplaza el
    # factor único por-grupo. El `report` greedy se conserva como telemetría por-macro.
    ing_factors, method, _sat_hi, _sat_lo = _compute_scale_factors(
        entries, tgt, min_scale, max_scale, max_scale_protein)
    report = {}
    for macro in _KCAL_PER_G:
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        gfactor = max(min_scale, min(max_scale, target_v / current)) if (current > 0 and target_v > 0) else 1.0
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(gfactor, 4), "applied": abs(gfactor - 1.0) > 1e-9}

    out_strings = []
    factors_applied = []  # factor por-ingrediente (1.0 = intacto), alineado con input
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for idx, e in enumerate(entries):
        if e["macros"] and e["group"]:
            f = ing_factors[idx]
            out_strings.append(rescale_ingredient_string(e["s"], f))
            factors_applied.append(f)
            for m in achieved:
                achieved[m] += e["macros"][m] * f
            resolved += 1
        elif e["macros"]:
            out_strings.append(e["s"])
            factors_applied.append(1.0)
            for m in achieved:
                achieved[m] += e["macros"][m]
            resolved += 1
        else:
            out_strings.append(e["s"])
            factors_applied.append(1.0)

    achieved = {m: round(v, 1) for m, v in achieved.items()}
    # [P3-SOLVER-FEASIBILITY · 2026-07-29] cotas exactas de la caja con los MISMOS bounds que usó
    # el solver (proteína-dominante lleva su techo propio). Telemetría: no cambia ni un gramo.
    _hi_all = [(max_scale_protein if e.get("group") == "protein" else max_scale) for e in entries]
    _infeasible = _feasibility_report(entries, tgt, min_scale, _hi_all)
    # [P3-SOLVER-CONVERGED-BAND · 2026-07-29] criterio compartido + desglose per-macro (el `break`
    # previo perdía CUÁL macro falló, que es el dato que vuelve accionable la métrica).
    converged, converged_per_macro = _converged_report(achieved, tgt, tolerance_pct)

    return {
        "ingredients": out_strings,
        "factors_applied": factors_applied,
        "achieved": achieved,
        "target": tgt,
        "report": report,
        # [P3-6 · 2026-07-07] `report` = referencia GREEDY per-macro, NO los factores LSQ aplicados
        # (esos en factors_applied/method/saturated_*). Etiquetado para no confundir al lector.
        "report_basis": "greedy-reference",
        "method": method,
        "resolved_count": resolved,
        "unresolved": len(ingredient_strings) - resolved,
        "converged": converged,
        # [P3-SOLVER-CONVERGED-BAND] qué macro falló, no solo que falló alguno.
        "converged_per_macro": converged_per_macro,
        # [P3-SOLVER-FEASIBILITY] {macro: 'high'|'low'} si el target es inalcanzable con estos
        # alimentos dentro del clamp: distingue "falta escalado" de "falta un PORTADOR".
        "infeasible": _infeasible,
        "residuals": {m: (round(achieved[m] / tgt[m], 3) if tgt.get(m) else None)
                      for m in ("kcal", "protein", "carbs", "fats")},
        "saturated_hi": _sat_hi,
        "saturated_lo": _sat_lo,
    }


# ============================================================
# [P1-NEXT-LEVEL-BATCH · 2026-07-02] Refinador GLOBAL entero del día.
# ------------------------------------------------------------
# La precisión era una cadena SECUENCIAL (solver per-meal → closers → caps →
# quantize → recheck) donde cada pasada des-hace un poco a la anterior, y el
# rebalance del recheck es CONTINUO + re-quantize (el redondeo re-abre drift).
# Este refinador opera DIRECTO sobre el estado post-quantize en PASOS ENTEROS
# de 5g (las porciones siguen humanas — cero re-quantize) optimizando el DÍA
# COMPLETO de forma conjunta: local search greedy que en cada iteración aplica
# el movimiento ±step de UNA línea que más reduce el error ponderado de banda
# (kcal+P+C+F simultáneos). Determinista, sin dependencias, ~O(iters × líneas).
#
# Respeta el plato: bounds por línea [max(floor_g, 0.5×orig), min(cap_g, 2×orig)]
# — jamás convierte una guarnición en plato ni un plato en migaja. Las líneas
# exentas (condimentos/aceites vía exempt_tokens del caller) no se tocan.
# tooltip-anchor: P1-NEXT-LEVEL-SOLVER. Test: test_p1_next_level_batch.py.
# ============================================================

# [S-P2-c / P2-REFINE-HOUSEHOLD · 2026-07-07] (audit solver+seeder v2) El refinador SOLO movía líneas
# gram-led ("150 g de X"); las líneas en unidad CASERA ("1 taza de arroz (150g)") — la mayoría de
# carbos/grasas — quedaban fuera, JUSTO cuando el rebalance unit-agnóstico ya saturó. Con el knob, el
# refinador también las mueve (grams desde el hint via grams_from_ingredient_string) pero re-renderiza
# vía quantize (el lead casero se mantiene HUMANO). Nace OFF: el quantize re-introduce un snap que puede
# desviar el delta predicho por el greedy (aproximación acotada por los bounds [0.5×,2×] + el truth-up
# del caller) → requiere A/B antes de flipear. Rollback/estado actual: MEALFIT_REFINE_HOUSEHOLD_LINES=false.
REFINE_HOUSEHOLD_LINES = _envb("MEALFIT_REFINE_HOUSEHOLD_LINES", False)

# [P1-REFINE-RAW-BY-FOOD · 2026-07-29] (audit solver+seeder v4) El sync a `ingredients_raw` del
# refinador escribía `raw[idx]` con el ÍNDICE de `ingredients` y el único guard `idx < len(raw)` —
# un guard MÁS FLOJO que el de sus dos pases hermanos (`P1-SOLVER-RAW-BY-FOOD` en el solver per-meal
# y `P1-CAP-RAW-BY-FOOD` en los caps), que exigen `len(raw) == len(ings)` antes de confiar en el índice.
# Y el repo YA MIDIÓ que las dos listas no son paralelas (tracer P1-MISALIGN-DEEP-TRACE: el desajuste
# nace en `pre_engine`, o sea antes de todo este bloque). Consecuencia: el factor de la línea de display
# `idx` se aplicaba a lo que ocupara esa posición en raw → la LISTA DE COMPRAS y el PANEL DE MICROS
# (ambos leen raw) escalaban el alimento equivocado, en silencio.
# Agravante de ubicación: en el shield pre-INSERT el refinador corre DESPUÉS del último reconciliador
# display↔raw, así que el desalineado que introduce llega tal cual a la DB.
# Fix = el mismo contrato que los hermanos: largos iguales → índice (exacto y barato); largos distintos
# → mapeo por ALIMENTO. Un alimento con factores distintos en varias líneas se deja INTACTO (preferimos
# no escalar a escalar con el factor equivocado). Rollback: MEALFIT_REFINE_RAW_BY_FOOD=false → el sync
# se SALTA cuando los largos difieren (nunca vuelve al índice ciego: ese era el bug).
# tooltip-anchor: P1-REFINE-RAW-BY-FOOD
REFINE_RAW_BY_FOOD = _envb("MEALFIT_REFINE_RAW_BY_FOOD", True)

# [P2-REFINE-COVERAGE-GATE · 2026-07-29] (audit solver+seeder v4) El refinador acumula `delivered`
# SOLO con las líneas que el catálogo resuelve (`if mc:`), pero su `target` es el del DÍA COMPLETO.
# La masa que el catálogo no ve es invisible a la izquierda y contable a la derecha ⇒ el greedy
# empuja cada línea movible hacia su techo 2× para "cubrir" macros que YA están en el plato.
#
# Es el mismo modo de fallo que `P1-SOLVER-COVERAGE-GATE` cerró para el solver per-meal, y el
# refinador —que agrega el día ENTERO— no tenía rama de abstención. Escenario: día de 4 comidas
# cuyo almuerzo es un sancocho que no resuelve (~500 g = 40P/60C/15F reales). El día está en target,
# pero `delivered` ve todo ~27% bajo y el greedy sube arroz/pollo/aguacate de las OTRAS 3 comidas
# hasta un día entregado de ~2.540 kcal contra un target de 2.003 (+27%).
#
# El gate mide cobertura por DÍA (que es lo que el refinador agrega), excluyendo del denominador las
# líneas benignas que no son masa oculta (agua/hielo/hierbas) — mismo criterio que
# P1-SOLVER-COVERAGE-BENIGN. Rollback sin redeploy: MEALFIT_REFINE_MIN_COVERAGE=0.0 (nunca dispara).
# tooltip-anchor: P2-REFINE-COVERAGE-GATE
REFINE_MIN_COVERAGE = _envf("MEALFIT_REFINE_MIN_COVERAGE", 0.6, lambda v: 0.0 <= v <= 1.0)

# [P3-REFINE-WEIGHTS-KNOBS · 2026-07-29] (audit solver+seeder v4) Eran los ÚNICOS pesos del motor de
# precisión hardcodeados: sus gemelos `SOLVER_W_*` son knobs con validador desde
# P2-SOLVER-KNOBS-REGISTRY, precisamente porque tunearlos movió el all-4-en-banda. El refinador es el
# ÚLTIMO optimizador de la cadena y decide qué línea mover en cada una de sus hasta 250 iteraciones,
# así que el operador no podía A/B-earlo ni verlo en /health/version.
# Defaults IDÉNTICOS a los literales previos ⇒ cero cambio de comportamiento en el merge; lo que se
# gana es reversibilidad y visibilidad.
# ⚠️ Nota para quien los tunee: difieren de los del LSQ (kcal 1.0 vs 0.1 · carbos 1.0 vs 1.1 ·
# grasa 1.2 vs 1.4) y nadie documentó por qué. Además `P1-FATS-RELEVEL-UNIVERSAL` corre JUSTO después
# del refinador y recorta grasa de los días sobre banda: con `fats` alto el refinador empuja grasa y
# el relevel la recorta — el gancho "dos guardas sobre la misma condición → OSCILA". Si vas a
# tocarlos, mide contra `all4_ratio`, no a ojo. tooltip-anchor: P3-REFINE-WEIGHTS-KNOBS
_REFINE_WEIGHTS = {
    "kcal": _envf("MEALFIT_REFINE_W_KCAL", 1.0, lambda v: 0.0 < v <= 10.0),
    "protein": _envf("MEALFIT_REFINE_W_PROTEIN", 1.5, lambda v: 0.0 < v <= 10.0),
    "carbs": _envf("MEALFIT_REFINE_W_CARBS", 1.0, lambda v: 0.0 < v <= 10.0),
    "fats": _envf("MEALFIT_REFINE_W_FATS", 1.2, lambda v: 0.0 < v <= 10.0),
}

# [P3-REFINE-EXEMPT-BOUNDARY · 2026-07-29] (audit solver+seeder v4) La exención del refinador era un
# `in` de substring sobre la línea sin acentos, y fallaba en las DOS direcciones a la vez:
#   (a) `"sal"` ⊂ salmón / ensalada / salchicha / salami / bacalao salado ⇒ el refinador PERDÍA su
#       palanca sobre líneas de proteína y grasa mayores (cena de salmón: el día se entrega fuera de
#       banda con banner `low_band_macro:protein` porque no había de dónde sacar proteína);
#   (b) los portadores que el micro-closer acaba de SEMBRAR (linaza/girasol/maní/zanahoria/auyama/
#       espinaca) NO estaban exentos ⇒ el refinador podía recortarlos a 0.5× y deshacer el cierre.
# Es la 13ª mordida de la clase 'res'⊂'fresco'. Rollback independiente por knob.
REFINE_EXEMPT_WORD_BOUNDARY = _envb("MEALFIT_REFINE_EXEMPT_WORD_BOUNDARY", True)
REFINE_PROTECT_MICRO_CARRIERS = _envb("MEALFIT_REFINE_PROTECT_MICRO_CARRIERS", True)
# Portadores que el micro-closer siembra/escala. SSOT propio del refinador (no puede importar
# graph_orchestrator a module-init sin ciclo); el test de paridad lo ancla contra `_SEED_NUT_TOKENS`.
_MICRO_CARRIER_TOKENS = ("linaza", "chia", "girasol", "mani", "almendra", "almendras", "nuez",
                         "nueces", "pistacho", "merey", "maranon", "ajonjoli", "semilla", "semillas",
                         "zanahoria", "auyama", "espinaca", "espinacas")

_EXEMPT_RE_CACHE: dict = {}


def _exempt_matcher(exempt_tokens: tuple):
    """[P3-REFINE-EXEMPT-BOUNDARY] Devuelve `fn(linea_normalizada) -> bool` con frontera de palabra.
    Cacheado por tupla de tokens: recompilar por línea sería O(líneas × tokens) en el hot path."""
    _key = tuple(exempt_tokens or ())
    _hit = _EXEMPT_RE_CACHE.get(_key)
    if _hit is not None:
        return _hit
    if not REFINE_EXEMPT_WORD_BOUNDARY:
        def _fn(il, _toks=_key):
            return any(t and t in il for t in _toks)
    else:
        _toks = [t for t in _key if t]
        if REFINE_PROTECT_MICRO_CARRIERS:
            _toks = list(_toks) + [t for t in _MICRO_CARRIER_TOKENS if t not in _toks]
        if not _toks:
            def _fn(il):
                return False
        else:
            _rx = _re_ps.compile(r"\b(?:" + "|".join(_re_ps.escape(t) for t in _toks) + r")\b")

            def _fn(il, _r=_rx):
                return bool(_r.search(il))
    _EXEMPT_RE_CACHE[_key] = _fn
    return _fn


def _refine_error(delivered: dict, targets: dict) -> float:
    err = 0.0
    for k, w in _REFINE_WEIGHTS.items():
        t = float(targets.get(k) or 0.0)
        if t <= 0:
            continue
        err += w * ((float(delivered.get(k) or 0.0) - t) / t) ** 2
    return err


def refine_day_portions_integer(
    meals: list,
    targets: dict,
    db,
    step_g: float = 5.0,
    floor_g: float = 15.0,
    cap_g: float = 300.0,
    exempt_tokens: tuple = (),
    max_iters: int = 250,
) -> int:
    """Refina las porciones del DÍA en pasos enteros de `step_g` para clavar la banda
    all-4 (kcal/P/C/F conjuntos). Muta `ingredients` (+`ingredients_raw` lockstep) y
    NO toca los macros del meal (el caller hace truth-up por meal tocado — mismo
    contrato que _cap_unrealistic_portions). Devuelve nº de movimientos aplicados.

    `targets`: {"kcal","protein","carbs","fats"} en unidades absolutas del día.
    Fail-safe: cualquier error → 0 movimientos (día intacto)."""
    import re as _re
    try:
        from nutrition_db import rescale_ingredient_string as _resc, quantize_ingredient_string as _quant
        try:
            from constants import strip_accents as _sa
        except Exception:
            def _sa(s):
                return s

        # [P2-REFINE-COVERAGE-GATE · 2026-07-29] benignos fuera del denominador de cobertura: no son
        # masa oculta, así que contarlos deprimiría la cobertura sin motivo (mismo criterio y misma
        # forma word-boundary que `_SOLVER_COV_BENIGN_RE`, para no re-abrir 'agua'⊂'aguacate').
        _BENIGN_RE = _re.compile(
            r"\b(?:agua|hielo|perejil|cilantro|cilantrico|culantro|albahaca|hierbabuena|cebollin|cebollino)\b")
        _n_quant = _n_res = 0
        _is_exempt = _exempt_matcher(tuple(exempt_tokens or ()))

        # 1) Censo de líneas móviles: gram-based, resolubles, no exentas.
        lines = []  # dicts: meal, idx, grams, per_g {kcal,p,c,f}, orig_grams
        delivered = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        for meal in meals or []:
            if not isinstance(meal, dict):
                continue
            ings = meal.get("ingredients")
            if not isinstance(ings, list):
                continue
            for idx, ing in enumerate(ings):
                s = str(ing)
                mc = None
                try:
                    mc = db.macros_from_ingredient_string(s)
                except Exception:
                    mc = None
                # cobertura del DÍA (el refinador agrega el día, así que el gate va al mismo nivel)
                _s_low = _sa(s.lower())
                if not ("al gusto" in _s_low or "opcional" in _s_low or _BENIGN_RE.search(_s_low)):
                    _n_quant += 1
                    if mc:
                        _n_res += 1
                if mc:
                    delivered["kcal"] += float(mc.get("kcal") or 0.0)
                    delivered["protein"] += float(mc.get("protein") or 0.0)
                    delivered["carbs"] += float(mc.get("carbs") or 0.0)
                    delivered["fats"] += float(mc.get("fats") or 0.0)
                il = _sa(s.lower())
                if "al gusto" in il or "opcional" in il:
                    continue
                # [P3-REFINE-EXEMPT-BOUNDARY] frontera de palabra + protección de portadores de micros
                if _is_exempt(il):
                    continue
                m_g = _re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", il)
                _gram_led = bool(m_g)
                # [S-P2-c] gram-led → mueve directo (queda humano). Unidad casera con hint de gramos
                # ("1 taza de arroz (150g)") → movible SOLO con el knob (re-render vía quantize abajo).
                if not mc or (not _gram_led and not REFINE_HOUSEHOLD_LINES):
                    continue
                if _gram_led:
                    grams = float(m_g.group(1).replace(",", "."))
                else:
                    try:
                        grams = float(db.grams_from_ingredient_string(s) or 0.0)
                    except Exception:
                        grams = 0.0
                if grams <= 0:
                    continue
                per_g = {k: float(mc.get(k2) or 0.0) / grams
                         for k, k2 in (("kcal", "kcal"), ("protein", "protein"),
                                       ("carbs", "carbs"), ("fats", "fats"))}
                if all(abs(v) < 1e-9 for v in per_g.values()):
                    continue
                lines.append({"meal": meal, "idx": idx, "grams": grams,
                              "orig": grams, "per_g": per_g, "gram_led": _gram_led})
        if not lines:
            return 0
        # [P2-REFINE-COVERAGE-GATE · 2026-07-29] Abstención: con masa no-resuelta significativa el
        # lado izquierdo (`delivered`) subestima al derecho (`targets`, del día completo) y el greedy
        # infla hasta 2× las líneas que SÍ resuelve para cubrir macros que ya están en el plato.
        # Abstenerse deja el día intacto: los closers/rebalance aguas abajo siguen dimensionando.
        _cov = (_n_res / _n_quant) if _n_quant else 1.0
        if _cov < REFINE_MIN_COVERAGE:
            _log.warning(
                f"🔎 [P2-REFINE-COVERAGE-GATE] cobertura del día {_cov:.2f} < {REFINE_MIN_COVERAGE} "
                f"({_n_res}/{_n_quant} líneas resueltas) — el refinador se abstiene (con masa "
                f"invisible al catálogo inflaría las líneas resolubles de las OTRAS comidas).")
            return 0
        tg = {k: float(targets.get(k) or 0.0) for k in ("kcal", "protein", "carbs", "fats")}
        if not any(v > 0 for v in tg.values()):
            return 0

        # 2) Greedy: el mejor movimiento ±step por iteración hasta converger.
        moves = 0
        err = _refine_error(delivered, tg)
        _err0 = err
        _exhausted = True   # [P3-REFINE-OBSERVABILITY] se pone False si sale por convergencia
        for _ in range(int(max_iters)):
            best = None  # (new_err, line, direction)
            for ln in lines:
                lo = max(float(floor_g), 0.5 * ln["orig"])
                hi = min(float(cap_g), 2.0 * ln["orig"])
                for direction in (+1.0, -1.0):
                    ng = ln["grams"] + direction * float(step_g)
                    if ng < lo - 1e-9 or ng > hi + 1e-9:
                        continue
                    cand = {k: delivered[k] + direction * float(step_g) * ln["per_g"][k]
                            for k in delivered}
                    ne = _refine_error(cand, tg)
                    if ne < err - 1e-9 and (best is None or ne < best[0]):
                        best = (ne, ln, direction)
            if best is None:
                _exhausted = False   # no hay movimiento que mejore: convergió de verdad
                break
            ne, ln, direction = best
            ln["grams"] += direction * float(step_g)
            for k in delivered:
                delivered[k] += direction * float(step_g) * ln["per_g"][k]
            err = ne
            moves += 1

        # [P3-REFINE-OBSERVABILITY · 2026-07-29] (audit solver+seeder v4) El refinador era MUDO: ni
        # una línea de log, ni siquiera al agotar `max_iters` (que es cuando su resultado depende del
        # tope y no de la convergencia — el mismo modo de fallo que P1-SOLVER-LSQ-ITERS destapó en
        # `_box_lsq`, donde el 99% de las comidas agotaba el tope sin que nadie lo supiera).
        if _exhausted and moves:
            _log.info(f"🎯 [P3-REFINE-OBSERVABILITY] refinador AGOTÓ {max_iters} iteraciones con "
                      f"{moves} movimiento(s) (err {_err0:.4f} → {err:.4f}) — el resultado depende "
                      f"del tope, no de la convergencia; subir max_iters podría mejorar el día.")
        if not moves:
            return 0

        # 3) Aplicar los cambios a los strings (lockstep raw) por línea tocada.
        # [P1-REFINE-RAW-BY-FOOD · 2026-07-29] El display se muta línea a línea; el raw se sincroniza
        # DESPUÉS, por meal, para poder decidir entre índice (largos iguales) y alimento (distintos).
        # `_pending[id(meal)] = (meal, [display_original...], {idx: (factor, es_casera)})`.
        touched_meals = set()
        _pending: dict = {}
        for ln in lines:
            if abs(ln["grams"] - ln["orig"]) < 1e-9:
                continue
            meal = ln["meal"]
            idx = ln["idx"]
            factor = ln["grams"] / ln["orig"]
            _hh = not ln.get("gram_led", True)  # línea en unidad casera → re-render vía quantize
            try:
                s = str(meal["ingredients"][idx])
                # [S-P2-c] gram-led → rescale directo (humano). Unidad casera → rescale + quantize
                # (el lead casero se re-renderiza a porción humana; el caller hace truth-up de macros).
                new_s = _quant(_resc(s, factor))[0] if _hh else _resc(s, factor)
                if new_s and new_s != s:
                    _slot = _pending.get(id(meal))
                    if _slot is None:
                        # snapshot del display ANTES de mutarlo (el mapeo por alimento lo necesita)
                        _slot = (meal, [str(x) for x in meal["ingredients"]], {})
                        _pending[id(meal)] = _slot
                    meal["ingredients"][idx] = new_s
                    _slot[2][idx] = (factor, _hh)
                    touched_meals.add(id(meal))
                    meal["_global_refine_applied"] = True
            except Exception:
                continue

        # 3b) Sync de `ingredients_raw`, por meal, con el MISMO contrato que los pases hermanos
        #     (P1-SOLVER-RAW-BY-FOOD / P1-CAP-RAW-BY-FOOD): el índice solo se usa con paralelismo
        #     VERIFICADO por alimento (P2-RAW-PAIR-BY-FOOD); si no, mapeo por alimento.
        for _meal, _disp_orig, _fmap in _pending.values():
            raw = _meal.get("ingredients_raw")
            if not (isinstance(raw, list) and raw and _fmap):
                continue
            try:
                # [P2-RAW-PAIR-BY-FOOD · 2026-07-29] P1-REFINE-RAW-BY-FOOD (arriba) todavía se fiaba
                # del LARGO para tomar el camino por índice. Medido: el 93.5% de las comidas tiene
                # largos iguales y solo el 48.1% de ESAS son paralelas de verdad — el reconciliador
                # display↔raw preserva el largo y cambia el ORDEN. Ahora hay que demostrar el
                # paralelismo por alimento. Repro del caso permutado: display queda
                # `pollo 240 / arroz 75 / aceite 15` y raw `arroz 300 / aceite 5 / pollo 180`.
                _parallel = len(raw) == len(_disp_orig)
                if _parallel and REFINE_RAW_BY_FOOD:
                    try:
                        from graph_orchestrator import (RAW_PAIR_BY_FOOD as _RPBF,
                                                        _raw_display_parallel_by_food as _par_ok)
                        if _RPBF:
                            _parallel = _par_ok(_disp_orig, raw)
                    except Exception:
                        pass  # sin el verificador, el largo sigue siendo el criterio (estado previo)
                if _parallel:
                    for idx, (factor, _hh) in _fmap.items():
                        if idx >= len(raw):
                            continue
                        try:
                            _rw = _resc(str(raw[idx]), factor)
                            raw[idx] = _quant(_rw)[0] if _hh else _rw
                        except Exception:
                            pass
                    continue
                if not REFINE_RAW_BY_FOOD:
                    # Rollback explícito: se SALTA el sync (no se vuelve al índice ciego — ese era el bug).
                    _log.warning(
                        f"[P1-REFINE-RAW-BY-FOOD] display={len(_disp_orig)} vs raw={len(raw)} en meal "
                        f"{str(_meal.get('name'))[:32]!r} y el knob está OFF — raw sin sincronizar.")
                    continue
                _factors = [_fmap.get(i, (1.0, False))[0] for i in range(len(_disp_orig))]
                from graph_orchestrator import _rescale_raw_by_food as _rrbf
                _new_raw, _n = _rrbf(raw, _disp_orig, _factors)
                if _n:
                    _meal["ingredients_raw"] = _new_raw
                    _meal["_refine_raw_by_food"] = _n
                    _log.info(
                        f"⚖️ [P1-REFINE-RAW-BY-FOOD] {str(_meal.get('name'))[:32]!r}: {_n} línea(s) de "
                        f"raw escaladas por alimento (display={len(_disp_orig)} vs raw={len(raw)} — el "
                        f"guard por índice habría escalado la línea EQUIVOCADA).")
            except Exception as _rs_e:
                # Fail-safe: raw sin tocar es un estado consistente (el truth-up del caller lo detecta);
                # escalar la línea equivocada NO lo es.
                _log.warning(f"[P1-REFINE-RAW-BY-FOOD] sync no-op en meal "
                             f"{str(_meal.get('name'))[:32]!r}: {type(_rs_e).__name__}: {_rs_e}")
        return moves if touched_meals else 0
    except Exception:
        return 0
