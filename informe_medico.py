# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1032 · 2026-11-17] Informe para el médico: los DATOS (sin IA).

El dueño: «quiero que el coach pueda dar un diagnóstico para dárselo a tu médico y así hacerles la vida más fácil».
La app NO diagnostica (plan D-A): resume lo que la persona registró para que su médico lo lea en dos minutos. Todas las
cifras salen de aquí, deterministas (D-B); el redactor (lote 1033) solo escribe sobre este JSON y no puede añadir cifras.

Reglas: cada sección se OMITE si no hay datos suficientes (nunca ceros inventados) y dice su cobertura; un día sin
registro no cuenta como cero en las medias; se redondea aquí (el cliente no recalcula); del texto libre del usuario solo
entran las listas que él mismo declaró en su perfil. Nada de este módulo va a los logs (datos de salud).

Arriba, funciones puras (`armar_datos` recibe todo ya leído y es lo que prueban los tests); abajo, la lectura en Neon.
"""
from __future__ import annotations

import logging
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

VERSION = 1
PERIODOS = (14, 30, 90)
PERIODO_POR_DEFECTO = 30
MIN_DIAS_PARA_MEDIAS = 3          # con menos días registrados, las medias no dicen nada: se omiten
MIN_DIAS_MICRO = 3                # días con el nutriente completo para dar su media
UMBRAL_BAJO_PCT = 70              # media < 70 % de la referencia (RDA/AI) ⇒ «bajo la meta»
TOP_ALIMENTOS = 8
_TIPOS_NO_COMIDA = ("suplemento", "complemento", "extra")

# ─────────────────────────────────────────────────────────────── la banda de «día en meta» (paridad con el cliente)
# Porte de `frontend/src/utils/calorieGoalStatus.js` (`getCalorieGoalBand` / `getCalorieGoalState`). Un test compara
# los seis números y las subcadenas con el .js: si alguien cambia un lado, cae.
BANDA_MUSCULO = (0.90, 1.15)
BANDA_PERDER_GRASA = (0.80, 1.05)
BANDA_OTRA = (0.90, 1.10)
SUBCADENAS_MUSCULO = ("muscle", "músculo", "musculo", "bulk")
SUBCADENAS_PERDER_GRASA = ("lose", "fat", "weight_loss", "perder", "pérdida", "perdida", "grasa", "deficit", "déficit")


def banda_de_la_meta(objetivo: Any) -> tuple[float, float]:
    raw = str(objetivo or "").strip().lower()
    if any(s in raw for s in SUBCADENAS_MUSCULO):
        return BANDA_MUSCULO
    if any(s in raw for s in SUBCADENAS_PERDER_GRASA):
        return BANDA_PERDER_GRASA
    return BANDA_OTRA


def estado_del_dia(kcal: float, meta: float, objetivo: Any) -> str:
    """'en_meta' | 'bajo' | 'sobre' — misma regla que la tarjeta de Progreso (`green` / `under` / `over`)."""
    meta = float(meta or 0)
    if meta <= 0:
        return "bajo"
    ratio = float(kcal or 0) / meta
    lo, hi = banda_de_la_meta(objetivo)
    if ratio > hi:
        return "sobre"
    if ratio >= lo:
        return "en_meta"
    return "bajo"


# ─────────────────────────────────────────────────────────────── utilidades puras
def hora_fiable(consumed_at: Any, created_at: Any, tolerancia_min: float = 20.0) -> bool:
    """[P1-PLAN-LOTE-1075] ¿La hora de la comida es REAL? Lo es si se anotó al comer (`consumed_at` ≈ `created_at`);
    una anotada tarde lleva la hora estimada de su franja y no sirve para los horarios. Sin `created_at`: fiable (filas
    antiguas). Misma regla que `horaFiable` del diario en la app."""
    if not isinstance(consumed_at, datetime) or not isinstance(created_at, datetime):
        return True
    a = consumed_at if consumed_at.tzinfo else consumed_at.replace(tzinfo=timezone.utc)
    b = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return abs((a - b).total_seconds()) <= tolerancia_min * 60


def _r(x: float, nd: int = 0):
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


def _norm(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "").lower()) if not unicodedata.combining(c)).strip()


_NINGUNO = {"ninguna", "ninguno", "ningun", "none", "no", "nada", "n/a", "na", "-", "no tengo", "sin", "ninguna condicion"}


def lista_declarada(*valores: Any) -> list[str]:
    """Une lo que el paciente declaró (listas o texto «a, b»), sin «Ninguna/none», sin duplicados, recortado."""
    out: list[str] = []
    vistos: set[str] = set()
    for v in valores:
        items = v if isinstance(v, (list, tuple)) else re.split(r"[,;\n]", str(v or ""))
        for it in items:
            if isinstance(it, dict):
                it = it.get("name") or it.get("nombre") or it.get("label") or ""
            t = str(it or "").strip()[:80]
            if not t or _norm(t) in _NINGUNO:
                continue
            k = _norm(t)
            if k not in vistos:
                vistos.add(k)
                out.append(t)
    return out[:20]


def _num(v: Any) -> Optional[float]:
    try:
        n = float(v)
        return n if math.isfinite(n) and n > 0 else None
    except (TypeError, ValueError):
        return None


def peso_en_kg(valor: Any, unidad: Any) -> Optional[float]:
    n = _num(valor)
    if n is None:
        return None
    u = _norm(unidad)
    if u in ("lb", "lbs", "libras", "libra", "pounds", "pound"):
        n = n * 0.45359237
    return round(n, 1) if 20 <= n <= 400 else None


def _texto_corto(v: Any) -> Optional[str]:
    t = str(v or "").strip()
    return t[:60] if t else None


_RE_PAREN = re.compile(r"\([^)]*\)")
_RE_CANT = re.compile(r"^\s*(?:\d+[.,/]?\d*|un|una|uno|dos|tres|cuatro|media|medio|1/2)\s*(?:x\s*)?"
                      r"(?:unidad(?:es)?|tazas?|cucharadas?|cdas?|cditas?|lascas?|rebanadas?|rodajas?|porci[oó]n(?:es)?|"
                      r"vasos?|g|gr|gramos?|ml|piezas?|latas?|envases?|scoops?)?\s*(?:de\s+)?", re.I)


def nombre_de_alimento(ingrediente: Any) -> str:
    """«arroz blanco cocido con trocitos de vegetales (≈180 g)» → «arroz blanco cocido». Solo para contar repetidos."""
    t = _RE_PAREN.sub("", str(ingrediente or "")).strip()
    t = _RE_CANT.sub("", t, count=1)
    t = re.split(r"\s+(?:con|para|en|al|a la|y)\s+|[,;:]", t, maxsplit=1)[0]
    t = re.sub(r"\s+", " ", t).strip(" .-").lower()
    return t[:40] if len(t) >= 3 else ""


# ─────────────────────────────────────────────────────────────── el armado (puro)
def _seccion_persona(perfil: dict) -> Optional[dict]:
    p: dict[str, Any] = {}
    edad = _num(perfil.get("age"))
    if edad and 10 <= edad <= 120:
        p["edad"] = int(edad)
    sexo = _texto_corto(perfil.get("gender"))
    if sexo:
        p["sexo"] = sexo
    talla = _num(perfil.get("height"))
    if talla and 100 <= talla <= 250:
        p["talla_cm"] = _r(talla)
    peso = peso_en_kg(perfil.get("weight"), perfil.get("weightUnit"))
    if peso:
        p["peso_declarado_kg"] = peso
    for k_out, k_in in (("objetivo", "mainGoal"), ("actividad", "activityLevel")):
        v = _texto_corto(perfil.get(k_in))
        if v:
            p[k_out] = v
    return p or None


def _seccion_declarado(perfil: dict) -> dict:
    habitos = {}
    for k_out, k_in in (("alcohol", "habitAlcohol"), ("tabaco", "habitSmoking"), ("cafeina", "habitCaffeine")):
        v = _texto_corto(perfil.get(k_in))
        if v:
            habitos[k_out] = v
    sueno = _num(perfil.get("sleepHours"))
    if sueno and sueno <= 24:
        habitos["sueno_h"] = _r(sueno, 1) if sueno % 1 else int(sueno)
    out = {
        "condiciones": lista_declarada(perfil.get("medicalConditions"), perfil.get("otherConditions")),
        "alergias": lista_declarada(perfil.get("allergies"), perfil.get("otherAllergies")),
        "medicamentos": lista_declarada(perfil.get("medications"), perfil.get("otherMedications")),
        "suplementos": lista_declarada(perfil.get("selectedSupplements"), perfil.get("currentSupplements")),
    }
    if habitos:
        out["habitos"] = habitos
    return out


def armar_datos(
    *,
    periodo_dias: int,
    hasta: date,
    perfil: dict,
    dias: dict[str, dict],
    comidas: list[dict],
    metas_por_dia: dict[str, dict],
    meta_de_hoy: Optional[dict],
    week1_start: Optional[date] = None,
    agua: Optional[dict[str, float]] = None,
    pesos: Optional[list[tuple[date, float]]] = None,
    micros_por_dia: Optional[dict[str, dict]] = None,
    referencias_micros: Optional[dict] = None,
) -> dict:
    """Arma el JSON del informe a partir de datos YA leídos.

    `dias`: {fecha_iso: {calories, protein, carbs, healthy_fats, meals_count}} solo de días con registro.
    `comidas`: [{dia (iso, día biológico), hora_local (0-24 float), dow (0=lun), meal_type, ingredients, meal_name}].
    `metas_por_dia`: {fecha_iso: {calories, main_goal?}}; un día sin meta usa `meta_de_hoy`.
    `micros_por_dia`: {fecha_iso: {nutriente: valor o None}} solo con los nutrientes COMPLETOS ese día.
    """
    periodo_dias = periodo_dias if periodo_dias in PERIODOS else PERIODO_POR_DEFECTO
    desde = hasta - timedelta(days=periodo_dias - 1)
    datos: dict[str, Any] = {
        "version": VERSION,
        "periodo": {"desde": desde.isoformat(), "hasta": hasta.isoformat(), "dias": periodo_dias},
    }
    persona = _seccion_persona(perfil or {})
    if persona:
        datos["persona"] = persona
    datos["declarado_por_el_paciente"] = _seccion_declarado(perfil or {})

    en_periodo = {d: v for d, v in (dias or {}).items() if desde.isoformat() <= d <= hasta.isoformat()
                  and float(v.get("calories") or 0) > 0}
    fechas = sorted(en_periodo)
    limitaciones = ["datos_autorreportados", "porciones_estimadas"]
    sin_registro = periodo_dias - len(fechas)
    if sin_registro > 0:
        limitaciones.append(f"dias_sin_registro: {sin_registro}")

    # adherencia + energía
    estados: dict[str, str] = {}
    metas_usadas: dict[str, float] = {}
    for d in fechas:
        m = metas_por_dia.get(d) or meta_de_hoy or {}
        meta = _num(m.get("calories"))
        if not meta:
            continue
        metas_usadas[d] = meta
        objetivo = m.get("main_goal") or (perfil or {}).get("mainGoal")
        estados[d] = estado_del_dia(float(en_periodo[d]["calories"]), meta, objetivo)
    if fechas:
        adh: dict[str, Any] = {"dias_con_registro": len(fechas), "dias_del_periodo": periodo_dias}
        if estados:
            c = Counter(estados.values())
            adh.update(dias_en_meta=c.get("en_meta", 0), dias_bajo=c.get("bajo", 0), dias_sobre=c.get("sobre", 0))
            if week1_start:
                por_semana: dict[int, Counter] = defaultdict(Counter)
                for d, e in estados.items():
                    n = (date.fromisoformat(d) - week1_start).days // 7 + 1
                    if n >= 1:
                        por_semana[n][e] += 1
                if por_semana:
                    adh["semanas"] = [{"numero": n, "dias_en_meta": por_semana[n].get("en_meta", 0),
                                       "dias_con_registro": sum(por_semana[n].values())}
                                      for n in sorted(por_semana)]
        datos["adherencia"] = adh

    if len(fechas) >= MIN_DIAS_PARA_MEDIAS:
        kcal = [float(en_periodo[d]["calories"]) for d in fechas]
        ene: dict[str, Any] = {"kcal_media": _r(statistics.fmean(kcal)), "kcal_min": _r(min(kcal)),
                               "kcal_max": _r(max(kcal)), "desviacion": _r(statistics.pstdev(kcal))}
        if metas_usadas:
            meta_media = statistics.fmean(metas_usadas.values())
            kcal_con_meta = statistics.fmean(float(en_periodo[d]["calories"]) for d in metas_usadas)
            ene["meta_media"] = _r(meta_media)
            ene["pct_de_la_meta"] = _r(100 * kcal_con_meta / meta_media)
        if week1_start:
            sem: dict[int, list[float]] = defaultdict(list)
            for d in fechas:
                n = (date.fromisoformat(d) - week1_start).days // 7 + 1
                if n >= 1:
                    sem[n].append(float(en_periodo[d]["calories"]))
            if sem:
                ene["por_semana"] = [{"numero": n, "kcal_media": _r(statistics.fmean(v)), "dias": len(v)}
                                     for n, v in sorted(sem.items())]
        datos["energia"] = ene

        pro = statistics.fmean(float(en_periodo[d].get("protein") or 0) for d in fechas)
        car = statistics.fmean(float(en_periodo[d].get("carbs") or 0) for d in fechas)
        gra = statistics.fmean(float(en_periodo[d].get("healthy_fats") or 0) for d in fechas)
        kcal_macros = pro * 4 + car * 4 + gra * 9
        mac: dict[str, Any] = {"proteina_g": _r(pro), "carbos_g": _r(car), "grasas_g": _r(gra)}
        if kcal_macros > 0:
            mac["pct_kcal"] = {"proteina": _r(100 * pro * 4 / kcal_macros), "carbos": _r(100 * car * 4 / kcal_macros),
                               "grasas": _r(100 * gra * 9 / kcal_macros)}
        peso_ref = None
        if pesos:
            peso_ref = sorted(pesos)[-1][1]
        peso_ref = peso_ref or (persona or {}).get("peso_declarado_kg")
        if peso_ref:
            mac["proteina_g_por_kg"] = _r(pro / peso_ref, 1)
        datos["macros"] = mac

    # micronutrientes: media de los días en que el nutriente estaba COMPLETO, frente a su referencia
    if micros_por_dia and referencias_micros:
        bajo, sobre = [], []
        dias_micro = [d for d in fechas if d in micros_por_dia]
        for clave, ref in referencias_micros.items():
            vals = [micros_por_dia[d].get(clave) for d in dias_micro]
            vals = [float(v) for v in vals if v is not None]
            objetivo = _num(ref.get("target"))
            if len(vals) < MIN_DIAS_MICRO or not objetivo:
                continue
            pct = 100 * statistics.fmean(vals) / objetivo
            nombre = clave.rsplit("_", 1)[0] if clave.endswith(("_mg", "_mcg", "_g")) else clave
            if ref.get("kind") == "ceiling":
                if pct > 100:
                    sobre.append({"nutriente": nombre, "media_pct_del_limite": _r(pct), "dias_con_dato": len(vals)})
            elif pct < UMBRAL_BAJO_PCT:
                bajo.append({"nutriente": nombre, "media_pct_de_la_meta": _r(pct), "dias_con_dato": len(vals)})
        if bajo or sobre:
            datos["micronutrientes"] = {
                "dias_con_datos": len(dias_micro), "dias_con_registro": len(fechas),
                "bajo_la_meta": sorted(bajo, key=lambda x: x["media_pct_de_la_meta"]),
                "sobre_el_limite": sorted(sobre, key=lambda x: -x["media_pct_del_limite"]),
            }
            limitaciones.append("micronutrientes_parciales")

    # hidratación
    agua_p = {d: g for d, g in (agua or {}).items() if desde.isoformat() <= d <= hasta.isoformat() and g and g > 0}
    if len(agua_p) >= MIN_DIAS_PARA_MEDIAS:
        datos["hidratacion"] = {"vasos_media": _r(statistics.fmean(agua_p.values()), 1), "dias_con_dato": len(agua_p)}

    # peso (≥ 2 mediciones dentro del periodo)
    pesos_p = sorted((d, kg) for d, kg in (pesos or []) if desde <= d <= hasta)
    if len(pesos_p) >= 2:
        datos["peso"] = {"mediciones": len(pesos_p), "primera_kg": pesos_p[0][1], "ultima_kg": pesos_p[-1][1],
                         "cambio_kg": round(pesos_p[-1][1] - pesos_p[0][1], 1),
                         "primera_fecha": pesos_p[0][0].isoformat(), "ultima_fecha": pesos_p[-1][0].isoformat()}

    # patrones y alimentos (solo comidas, no suplementos)
    reales = [c for c in (comidas or []) if c.get("dia") in en_periodo
              and str(c.get("meal_type") or "").lower() not in _TIPOS_NO_COMIDA]
    if len(fechas) >= MIN_DIAS_PARA_MEDIAS and reales:
        por_dia: dict[str, list[dict]] = defaultdict(list)
        for c in reales:
            por_dia[c["dia"]].append(c)
        # [P1-PLAN-LOTE-1075] Los HORARIOS solo con horas fiables: una comida anotada tarde guarda una hora estimada de
        # su franja (`hora_de_la_franja`), y «primera comida a las 22:23» sería falso. Sin horas fiables, sin horarios.
        fiables: dict[str, list[dict]] = defaultdict(list)
        for c in reales:
            if c.get("hora_fiable", True):
                fiables[c["dia"]].append(c)
        primeras = [min(x["hora_local"] for x in v) for v in fiables.values()]
        ultimas = [max(x["hora_local"] for x in v) for v in fiables.values()]
        cenas = [c for c in reales if str(c.get("meal_type") or "").lower() == "cena" and c.get("hora_fiable", True)]
        pat: dict[str, Any] = {
            "comidas_por_dia": _r(len(reales) / len(por_dia), 1),
            "pct_dias_con_desayuno": _r(100 * sum(1 for v in por_dia.values()
                                                  if any(str(x.get("meal_type")).lower() == "desayuno" for x in v))
                                        / len(por_dia)),
        }
        if primeras:
            pat["hora_primera_comida"] = _hora(statistics.median(primeras))
            pat["hora_ultima_comida"] = _hora(statistics.median(ultimas))
        if cenas:
            pat["pct_cenas_despues_de_las_22"] = _r(100 * sum(1 for c in cenas if 22 <= c["hora_local"] or c["hora_local"] < 4)
                                                     / len(cenas))
        semana = [float(en_periodo[d]["calories"]) for d in fechas if date.fromisoformat(d).weekday() < 5]
        finde = [float(en_periodo[d]["calories"]) for d in fechas if date.fromisoformat(d).weekday() >= 5]
        if len(semana) >= 2 and len(finde) >= 2:
            pat["entre_semana_kcal"] = _r(statistics.fmean(semana))
            pat["fin_de_semana_kcal"] = _r(statistics.fmean(finde))
        datos["patrones"] = pat

        cuenta: Counter = Counter()
        for c in reales:
            vistos = set()
            for ing in (c.get("ingredients") or []):
                n = nombre_de_alimento(ing)
                if n and n not in vistos:
                    vistos.add(n)
                    cuenta[n] += 1
        frecuentes = [{"nombre": n, "veces": v} for n, v in cuenta.most_common(TOP_ALIMENTOS) if v >= 2]
        if frecuentes:
            datos["alimentos_frecuentes"] = frecuentes

    datos["limitaciones"] = limitaciones
    return datos


def _hora(h: float) -> str:
    h = float(h) % 24
    mins = int(round(h * 60)) % (24 * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


# ─────────────────────────────────────────────────────────────── la lectura (Neon)
def _periodo_valido(dias: Any) -> int:
    try:
        n = int(dias)
    except (TypeError, ValueError):
        return PERIODO_POR_DEFECTO
    return n if n in PERIODOS else PERIODO_POR_DEFECTO


def construir_datos(user_id: str, dias: Any = PERIODO_POR_DEFECTO, tz_offset: Optional[int] = None,
                    locale: str = "es") -> dict:
    """Lee el diario, el perfil, el agua y el peso de `user_id` y arma el informe. Toda sentencia filtra `user_id`."""
    from db_core import execute_sql_query
    from diary_targets import resolve_targets_by_day, resolve_user_effective_off_min

    n = _periodo_valido(dias)
    eff_off = resolve_user_effective_off_min(user_id, tz_offset)
    tz_off = eff_off - 4 * 60
    try:
        from db_facts import _diary_rollover_hour
        tz_off = eff_off - int(_diary_rollover_hour()) * 60
    except Exception:
        pass
    hoy = (datetime.now(timezone.utc) - timedelta(minutes=eff_off)).date()
    desde = hoy - timedelta(days=n - 1)

    fila = execute_sql_query(
        "SELECT health_profile, water_tracker_enabled FROM user_profiles WHERE id = %s",
        (user_id,), fetch_one=True,
    ) or {}
    perfil = fila.get("health_profile") or {}
    if isinstance(perfil, str):
        import json
        try:
            perfil = json.loads(perfil)
        except Exception:
            perfil = {}

    filas = execute_sql_query(
        """
        SELECT (consumed_at - make_interval(mins => %s))::date AS dia, consumed_at, created_at, meal_type, meal_name,
               calories, protein, carbs, healthy_fats, ingredients, nutrition
        FROM consumed_meals
        WHERE user_id = %s
          AND consumed_at >= now() - make_interval(days => %s)
        ORDER BY consumed_at
        """,
        (eff_off, user_id, n + 1),
        fetch_all=True,
    ) or []

    dias_tot: dict[str, dict] = {}
    comidas: list[dict] = []
    por_dia_meals: dict[str, list[dict]] = defaultdict(list)
    for r in filas:
        d = str(r.get("dia"))
        if not (desde.isoformat() <= d <= hoy.isoformat()):
            continue
        t = dias_tot.setdefault(d, {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "healthy_fats": 0.0, "meals_count": 0})
        for k in ("calories", "protein", "carbs", "healthy_fats"):
            t[k] += float(r.get(k) or 0)
        t["meals_count"] += 1
        ca = r.get("consumed_at")
        if isinstance(ca, datetime):
            local = (ca.astimezone(timezone.utc) if ca.tzinfo else ca.replace(tzinfo=timezone.utc)) - timedelta(minutes=tz_off)
            hora = local.hour + local.minute / 60
        else:
            hora = 12.0
        ings = r.get("ingredients") if isinstance(r.get("ingredients"), list) else []
        comidas.append({"dia": d, "hora_local": hora, "meal_type": r.get("meal_type"), "ingredients": ings,
                        "meal_name": r.get("meal_name"), "hora_fiable": hora_fiable(ca, r.get("created_at"))})
        por_dia_meals[d].append({"calories": r.get("calories"), "ingredients": ings, "nutrition": r.get("nutrition")})

    metas = {}
    meta_hoy = None
    try:
        metas = resolve_targets_by_day(user_id, list(dias_tot) + [hoy.isoformat()], eff_off) or {}
        meta_hoy = metas.get(hoy.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1032] metas por día no resueltas: {type(exc).__name__}")

    week1 = None
    try:
        j = execute_sql_query("SELECT week1_start FROM user_journey WHERE user_id = %s", (user_id,), fetch_one=True)
        if j and j.get("week1_start"):
            week1 = j["week1_start"] if isinstance(j["week1_start"], date) else date.fromisoformat(str(j["week1_start"]))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1032] día 1 no leído: {type(exc).__name__}")

    agua: dict[str, float] = {}
    if fila.get("water_tracker_enabled"):
        try:
            for w in execute_sql_query(
                "SELECT log_date, glasses FROM water_intake_log WHERE user_id = %s AND log_date >= %s AND log_date <= %s",
                (user_id, desde, hoy), fetch_all=True,
            ) or []:
                agua[str(w.get("log_date"))[:10]] = float(w.get("glasses") or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[LOTE-1032] agua no leída: {type(exc).__name__}")

    pesos: list[tuple[date, float]] = []
    try:
        for w in execute_sql_query(
            "SELECT weight, unit, (created_at - make_interval(mins => %s))::date AS dia FROM weight_log "
            "WHERE user_id = %s AND created_at >= now() - make_interval(days => %s) ORDER BY created_at",
            (eff_off, user_id, n + 1), fetch_all=True,
        ) or []:
            kg = peso_en_kg(w.get("weight"), w.get("unit"))
            d = w.get("dia")
            if kg and d:
                pesos.append((d if isinstance(d, date) else date.fromisoformat(str(d)[:10]), kg))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1032] peso no leído: {type(exc).__name__}")

    micros_dia, refs = _micros_por_dia(por_dia_meals, perfil)

    return armar_datos(
        periodo_dias=n, hasta=hoy, perfil=perfil, dias=dias_tot, comidas=comidas, metas_por_dia=metas,
        meta_de_hoy=meta_hoy, week1_start=week1, agua=agua, pesos=pesos, micros_por_dia=micros_dia,
        referencias_micros=refs,
    )


def _micros_por_dia(por_dia_meals: dict[str, list[dict]], perfil: dict) -> tuple[dict, dict]:
    """Por día, los nutrientes con dato COMPLETO (todas las comidas lo conocen); y las referencias de la persona."""
    try:
        from diary_micros import micros_de_ingredientes, micros_plausibles, resumen_micros, metas_micros, CLAVES
        from diary_nutrients import life_stage
        from food_label_nutrients import micros_from_snapshot
        from nutrition_db import IngredientNutritionDB
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1032] micros no disponibles: {type(exc).__name__}")
        return {}, {}
    try:
        embarazo, lactancia = life_stage(perfil or {})
        refs = metas_micros((perfil or {}).get("gender"), (perfil or {}).get("age"), embarazo, lactancia) or {}
        if not refs:
            return {}, {}
        ndb = IngredientNutritionDB()
        out: dict[str, dict] = {}
        for d, meals in por_dia_meals.items():
            ms = []
            for m in meals:
                mi = micros_from_snapshot(m.get("nutrition")) or __import__("micros_memo").micros_de_ingredientes_memo(m.get("ingredients"), ndb)  # [P1-PLAN-LOTE-1085]
                ms.append({"micros": micros_plausibles(mi, m.get("calories"))})
            res = resumen_micros(ms)
            by = (res.get("micros_coverage") or {}).get("by_nutrient") or {}
            vals = res.get("micros") or {}
            out[d] = {k: vals.get(k) for k in CLAVES if (by.get(k) or {}).get("status") == "complete"}
        return out, refs
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1032] micros no calculados: {type(exc).__name__}")
        return {}, {}


def rutas_de(datos: Any, prefijo: str = "") -> Iterable[str]:
    """Todas las rutas de `datos` («energia.kcal_media», «micronutrientes.bajo_la_meta[0]»…): las usa el verificador."""
    if isinstance(datos, dict):
        for k, v in datos.items():
            p = f"{prefijo}.{k}" if prefijo else str(k)
            yield p
            yield from rutas_de(v, p)
    elif isinstance(datos, list):
        for i, v in enumerate(datos):
            p = f"{prefijo}[{i}]"
            yield p
            yield from rutas_de(v, p)
