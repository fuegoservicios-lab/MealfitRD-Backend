# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-132 · 2026-09-20] Lo que al usuario le FALTA hoy, y una comida A MEDIDA para cerrarlo.

El encargo del dueño: «imagínate que al usuario le falta proteína y calorías en el día y son las 9 de la noche… que
pueda decirle "dame una receta para comer hoy" o "dame una comida para el desayuno" y el agente resuelva de manera 100 %
eficiente de acuerdo a lo que el usuario necesita».

Lo que el coach tenía: las kcal consumidas y las macros acumuladas «(meta N g)» — la RESTA la hacía el modelo, y sin
diario registrado no recibía meta ninguna. Y para proponer una comida escribía de memoria: macros inventadas, sin mirar
la Nevera, sin pasar por el backstop clínico.

Dos piezas, las dos deterministas y sin LLM:

  1. `build_day_gap_context` — la resta hecha: kcal y gramos que faltan (o sobran), la hora, y cómo se cierra un día
     A ESA HORA. De noche la guía está calibrada contra la evidencia, no contra el mito: una batida de proteína a las
     9 pm no es una «deshora» (proteína antes de dormir no perjudica y ayuda a llegar a la meta); lo que sí se
     desaconseja cerca de acostarse es lo pesado/frito, el picante con reflujo, la cafeína de un pre-entreno y beber
     mucho justo antes de dormir. Y un diario casi vacío a las 9 pm casi siempre es «no registró», no «no comió».

  2. `proponer_comidas` — «antes de construir la maquinaria, busca la maquinaria»: el día determinista ya sabe elegir
     una plantilla del registro por franja, escalarla a un objetivo de macros con la tabla de alimentos, y pasarla por
     el escáner culinario y el backstop clínico. Aquí se reutiliza TAL CUAL (`_candidatos`, `elegir_plantillas`,
     `_macros`, `_inclinar`, `verifica_comida`) para UNA comida, con la receta escrita de la biblioteca y la cobertura
     contra la Nevera real. Los pasos se leen de la biblioteca directamente: `recipe_for_dish_name` va detrás del knob
     de GENERACIÓN (`MEALFIT_RECIPE_LIBRARY_SELECT`) y esta superficie tiene el suyo.

Knobs: `MEALFIT_CHAT_DAY_GAP_BLOCK` (el bloque) y `MEALFIT_CHAT_MEAL_PROPOSAL_TOOL` (la herramienta, en `tools.py`).
"""
from __future__ import annotations

import logging
from typing import Optional
import unicodedata

logger = logging.getLogger(__name__)

FRANJAS = ("desayuno", "almuerzo", "merienda", "cena")
# Cuánto «cuesta», en puntos del ajuste de macros, cada ingrediente que no está en su Nevera (un buen ajuste ronda
# 0,1-0,5): con Nevera registrada gana el plato que ya puede cocinar, salvo que otro encaje mucho mejor.
PENA_POR_INGREDIENTE_QUE_FALTA = 0.08
# Condición declarada → atributos de `intrinsic_risk_attributes` del registro que la propuesta debe EVITAR si puede.
# Prefiere, no descarta (la doctrina del día determinista): el plato que los lleva queda detrás, no fuera. Medido en la
# batería del lote: a un diabético se le proponía «Arroz con lentejas y queso gouda», 145 g de carbohidrato.
_RIESGOS_POR_CONDICION = (
    (("diabet", "prediabet", "insulin", "glucosa", "glicemi"), ("glycemic_load_high", "sugar_high")),
    (("hipertens", "presion alta", "presión alta"), ("sodium_high", "processed_meat")),
    (("colesterol", "dislipid", "triglicerid", "cardio"), ("sat_fat_high", "processed_meat")),
    (("renal", "riñon", "rinon"), ("phosphorus_high", "potassium_high", "sodium_high")),
)
PENA_POR_RIESGO_CLINICO = 0.6


def _con_texto_libre(d) -> dict:
    """[P1-PLAN-LOTE-166 · 2026-09-22] El perfil con lo tecleado en «Otra…» sumado a sus listas (la unión del
    generador, `profile_with_free_text`). El perfil GUARDADO no los une: sin esto, «Maní» escrito a mano no existía
    para las propuestas de comida del coach, que en el modo contador son su recomendación principal."""
    try:
        from graph_orchestrator import profile_with_free_text
    except Exception as e:  # pragma: no cover — sin el grafo no arranca ni la app
        logger.error(f"[P1-PLAN-LOTE-166] propuestas sin el texto libre del perfil: {e!r}")
        return d if isinstance(d, dict) else {}
    return profile_with_free_text(d)


def restricciones_del_perfil(form_data) -> tuple:
    """(alergias, dieta, excluidos) con que se eligen platos: los chips del formulario MÁS lo tecleado en «Otra
    alergia» / «Otro que no te gusta». Primero el perfil guardado; lo que mande el llamador, de respaldo."""
    fd = form_data if isinstance(form_data, dict) else {}
    hp = fd.get("health_profile") if isinstance(fd.get("health_profile"), dict) else {}
    hp_u, fd_u = _con_texto_libre(hp), _con_texto_libre(fd)
    alergias = [str(a) for a in (hp_u.get("allergies") or fd_u.get("allergies") or []) if a]
    dieta = hp.get("dietType") or fd.get("dietType") or fd.get("diet_type")
    excluidos = [str(x) for x in (hp_u.get("dislikes") or fd_u.get("dislikes") or []) if x]
    return alergias, dieta, excluidos


def riesgos_a_evitar(form_data) -> set:
    fd = form_data if isinstance(form_data, dict) else {}
    hp = fd.get("health_profile") if isinstance(fd.get("health_profile"), dict) else {}
    # [P1-PLAN-LOTE-166] una condición escrita a mano («Gota») también cuenta
    fd, hp = _con_texto_libre(fd), _con_texto_libre(hp)
    cond = hp.get("medicalConditions") or fd.get("medicalConditions") or []
    if isinstance(cond, str):
        cond = [cond]
    texto = " ".join(str(c).lower() for c in cond)
    out = set()
    for claves, riesgos in _RIESGOS_POR_CONDICION:
        if any(k in texto for k in claves):
            out.update(riesgos)
    return out


# Comprobaciones del escáner culinario que comparan la lista con los PASOS: sin pasos no miden nada.
_CHECKS_QUE_LEEN_PASOS = frozenset({"V3", "V5", "V6"})
# Qué parte de la meta del día es una comida «razonable» de cierre cuando ya es de noche.
FRACCION_CIERRE_NOCHE = 0.30
# Con menos de esto registrado a partir de la tarde, lo probable es que falte REGISTRAR, no comer.
FRACCION_DIARIO_CORTO = 0.35


def _num(v) -> float:
    try:
        return float(str(v).lower().replace("kcal", "").replace("g", "").replace(",", ".").strip().split()[0])
    except Exception:
        return 0.0


def _knob(nombre: str, defecto: bool = True) -> bool:
    try:
        from knobs import _env_bool
        return _env_bool(nombre, defecto)
    except Exception:
        return defecto


# ───────────────────────────── 1. Lo que falta hoy ─────────────────────────────

def metas_del_dia(form_data, plan_vigente) -> Optional[dict]:
    """Meta de kcal y macros de HOY, o `None` si no hay de dónde sacarla sin inventar.

    Mismo orden que el bloque DIARIO DE HOY (`form_data.target_calories` → plan vigente → metas del contador) y la misma
    guarda que `_daily_goal_context`: `get_nutrition_targets` rellena lo que falta con supuestos (25 años, 154 lb), así
    que solo se usa si el formulario trae peso, estatura y edad."""
    fd = form_data if isinstance(form_data, dict) else {}
    plan = plan_vigente if isinstance(plan_vigente, dict) else {}
    kcal = _num(fd.get("target_calories")) or _num(plan.get("calories"))
    macros = plan.get("macros") if isinstance(plan.get("macros"), dict) else {}
    fuente = "plan" if (kcal and macros) else ""
    _con_prot = bool(macros.get("protein") or macros.get("protein_g"))
    if not (kcal and _con_prot):
        if all(str(fd.get(k) or "").strip() for k in ("weight", "height", "age")):
            try:
                from nutrition_calculator import get_nutrition_targets
                t = get_nutrition_targets(fd) or {}
                kcal = kcal or _num(t.get("target_calories"))
                if not _con_prot:
                    macros = t.get("macros") if isinstance(t.get("macros"), dict) else {}
                fuente = fuente or "contador"
            except Exception as e:
                logger.warning(f"[P1-PLAN-LOTE-132] metas del contador ilegibles: {e!r}")
    if kcal <= 0:
        return None

    def _m(clave):   # el plan guarda «protein: "134g"»; `get_nutrition_targets`, «protein_g: 134»
        return _num(macros.get(clave)) or _num(macros.get(f"{clave}_g"))

    return {"kcal": kcal, "protein_g": _m("protein"), "carbs_g": _m("carbs"), "fats_g": _m("fats"),
            "fuente": fuente or "plan"}


def consumido_hoy(consumed_today) -> dict:
    filas = [m for m in (consumed_today or []) if isinstance(m, dict)]
    return {"kcal": sum(_num(m.get("calories")) for m in filas),
            "protein_g": sum(_num(m.get("protein")) for m in filas),
            "carbs_g": sum(_num(m.get("carbs")) for m in filas),
            "fats_g": sum(_num(m.get("healthy_fats")) for m in filas),
            "registros": len(filas)}


def falta_hoy(metas: dict, consumido: dict) -> dict:
    """Meta − consumido por macro. Negativo = se pasó. Una meta en 0 (desconocida) da `None`, nunca «falta 0»."""
    out = {}
    for k in ("kcal", "protein_g", "carbs_g", "fats_g"):
        meta = float((metas or {}).get(k) or 0)
        out[k] = (meta - float((consumido or {}).get(k) or 0)) if meta > 0 else None
    return out


# [P1-PLAN-LOTE-226 · 2026-09-25] El margen que le QUEDA, macro por macro, y la orden de mirarlo antes de aconsejar.
# Caso del dueño (24-sep, 22:31): cena de claras «sin yemas» y pregunta «¿me recomiendas comérmelas?». El coach contestó
# «no hay razón para sacarlas» cuando llevaba 52 de 57 g de grasa: las yemas le pasaban la meta. Tenía las cifras (el
# bloque de lo que falta y el total tras registrar), pero la grasa iba entre paréntesis y la resta la hacía el modelo,
# y ninguna regla le pedía comprobar si lo que recomienda CABE. Ahora la resta va hecha y la regla va al lado.
# tooltip-anchor: P1-PLAN-LOTE-226-MARGEN
_NOMBRES_MARGEN = (("kcal", "kcal", ""), ("protein_g", "proteína", " g"), ("carbs_g", "carbohidratos", " g"),
                   ("fats_g", "grasas", " g"))
# Por debajo de esta fracción de la meta, el margen de un macro se da por «casi agotado» y se nombra en la regla.
FRACCION_MARGEN_JUSTO = 0.15

REGLA_MARGEN = ("Antes de recomendarle AÑADIR, quitar o repetir un alimento (p. ej. «cómete las yemas», «ponle "
                "aguacate», «otra porción»), suma lo que aporta y comprueba que CABE en el margen de CADA macro, no solo "
                "en la proteína: si le haría pasarse (sobre todo de grasas o de kcal), díselo con la cifra y propón la "
                "versión que sí cabe; si cabe, dilo también con la cifra. La salud del alimento no cambia la cuenta.")


def margen_del_dia(metas: Optional[dict], consumido: dict) -> str:
    """«le quedan ~X kcal, ~Y g de proteína… (grasas: casi agotado)» con la resta hecha, o "" sin metas."""
    if not metas:
        return ""
    falta = falta_hoy(metas, consumido)
    partes, justos = [], []
    for k, nombre, u in _NOMBRES_MARGEN:
        v = falta.get(k)
        if v is None:
            continue
        cifra = f"~{int(round(abs(v)))}{u}"
        partes.append(f"{nombre}: quedan {cifra}" if v > 0 else f"{nombre}: YA SE PASÓ por {cifra}")
        if k != "protein_g" and (v <= 0 or v < FRACCION_MARGEN_JUSTO * float(metas.get(k) or 0)):
            justos.append(nombre)
    if not partes:
        return ""
    out = "MARGEN QUE LE QUEDA HOY — " + "; ".join(partes) + "."
    if justos:
        out += f" Sin margen (o casi) en: {', '.join(justos)}."
    return out


def momento_del_dia(hora: Optional[float], schedule_type=None) -> str:
    """`manana` | `mediodia` | `tarde` | `noche` | `madrugada`. Con turno nocturno el reloj del usuario va invertido:
    se devuelve `turno_nocturno` y la guía de noche NO aplica (la regla de ritmo circadiano ya lo cubre)."""
    if hora is None:
        return "desconocido"
    if str(schedule_type or "") == "night_shift":
        return "turno_nocturno"
    h = float(hora) % 24
    if h < 4.5:
        return "madrugada"
    if h < 12.0:
        return "manana"
    if h < 15.5:
        return "mediodia"
    if h < 20.0:
        return "tarde"
    return "noche"


# [P1-PLAN-LOTE-950 · 2026-09-30] [BB-049 · 2026-11-09] Qué comidas de hoy están anotadas y cuáles ya pasaron.
# Desayuno no pasa hasta las 12:00 PM (mediodía): en RD/LatAm a las 11:00-11:30 AM es desayuno/brunch, no almuerzo.
FIN_DE_FRANJA = {"desayuno": 12.0, "almuerzo": 16.0, "merienda": 19.0}   # la cena no «pasa» dentro del día
_FRANJA_DE = {"desayuno": "desayuno", "breakfast": "desayuno", "almuerzo": "almuerzo", "lunch": "almuerzo",
              "comida": "almuerzo", "merienda": "merienda", "snack": "merienda", "cena": "cena", "dinner": "cena"}


def franjas_anotadas(consumed_today) -> set:
    anotadas = set()
    for m in (consumed_today or []):
        if isinstance(m, dict):
            f = _FRANJA_DE.get(str(m.get("meal_type") or "").strip().lower())
            if f:
                anotadas.add(f)
    return anotadas


def estado_de_las_comidas(consumed_today, hora: Optional[float], schedule_type=None) -> str:
    """La línea «COMIDAS DE HOY» y la regla del cierre. "" sin hora o con turno nocturno (el reloj va invertido)."""
    if hora is None or str(schedule_type or "") == "night_shift":
        return ""
    h = float(hora) % 24
    if h < 4.5:
        return ""   # madrugada: el día que cuenta es el de antes; mejor no afirmar nada
    anotadas = franjas_anotadas(consumed_today)
    principales_cubiertas = {"desayuno", "almuerzo", "cena"}.issubset(anotadas)
    ahora = franja_por_hora(h)
    ahora = None if (ahora in anotadas or (ahora == "merienda" and principales_cubiertas)) else ahora   # la de esta hora ya está (o merienda opcional con día cubierto)
    partes, pasadas = [], []
    for f in FRANJAS:
        if f in anotadas:
            partes.append(f"{f} — anotado")
        elif f in FIN_DE_FRANJA and h >= FIN_DE_FRANJA[f]:
            extra = " (opcional; ya registró las 3 comidas principales, NO preguntes por ella)" if (f == "merienda" and principales_cubiertas) else ""
            partes.append(f"{f} — SIN anotar y su hora ya pasó{extra}")
            if f != "merienda":
                pasadas.append(f)
        else:
            partes.append(f"{f} — sin anotar, " + ("es la que toca ahora" if f == ahora else "más tarde"))
    out = "\n🍽️ COMIDAS DE HOY (según su diario y la hora): " + "; ".join(partes) + "."
    out += (" CÓMO CIERRAS: " + (f"si tu cierre nombra una comida, que sea la que toca AHORA ({ahora})" if ahora else
                                  "a esta hora no le toca ninguna comida: no le pidas «¿qué quieres anotar hoy?» ni invites a comer; si solo charla o cierra el día, responde breve sin repetir sus totales")
            + " — nunca «cuando almuerces» si el almuerzo ya pasó, y NUNCA copies el cierre de tu respuesta anterior: "
            "mira la hora de ESTE mensaje.")
    if ahora == "cena" and h >= 22.5:
        out += (" Como ya es tarde en la noche, no cierres con «cuando cenes» ni «¿te propongo algo para la cena?»: "
                "si preguntas por la cena, pregunta qué cenó para anotarlo.")
    if pasadas:
        p = pasadas[-1]
        out += (f" Del {p} (ya pasó sin anotar) se habla en PASADO y una sola vez en la conversación, si no se lo dijiste "
                f"ya: «No me mandaste tu {p} de hoy; si lo comiste, mándame una foto y lo contabilizo». Si ya se lo "
                "dijiste, no insistas.")
    return out


def _fmt_hora(hora: float) -> str:
    h = float(hora) % 24
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)) % 60:02d}"


def _g(v) -> str:
    return f"~{int(round(v))} g"


def build_day_gap_context(form_data, plan_vigente, consumed_today, hora_local: Optional[float],
                          schedule_type=None) -> str:
    """El bloque «LO QUE LE FALTA HOY» del system prompt. "" si no hay metas reales o el knob está apagado."""
    try:
        if not _knob("MEALFIT_CHAT_DAY_GAP_BLOCK", True):
            return ""
        metas = metas_del_dia(form_data, plan_vigente)
        if not metas:
            return ""
        cons = consumido_hoy(consumed_today)
        falta = falta_hoy(metas, cons)
        momento = momento_del_dia(hora_local, schedule_type)
        fk, fp = falta["kcal"], falta["protein_g"]

        meta_txt = f"{int(round(metas['kcal']))} kcal"
        if metas["protein_g"] > 0:
            meta_txt += f" y {int(round(metas['protein_g']))} g de proteína"
        out = ("\n\n🎯 LO QUE LE FALTA HOY (la resta ya está hecha por el sistema — usa ESTAS cifras tal cual, no las "
               f"recalcules; son estimadas porque el diario lo es): meta del día {meta_txt}. ")
        partes = []
        _tol_kcal = max(50, int(round(0.03 * metas["kcal"])))
        if fk is not None:
            if fk > 0:
                partes.append(f"le faltan ~{int(round(fk))} kcal")
            elif -fk <= _tol_kcal:
                partes.append("ya cubrió su meta calórica del día (en rango, sin exceso)")
            else:
                partes.append(f"ya se pasó de su meta por ~{int(round(-fk))} kcal")
        if fp is not None:
            partes.append(f"le faltan {_g(fp)} de proteína" if fp > 5 else
                          ("ya cubrió su proteína del día" if fp <= 0 else f"le faltan solo {_g(fp)} de proteína"))
        extras = []
        for k, nombre in (("carbs_g", "carbohidratos"), ("fats_g", "grasas")):
            v = falta.get(k)
            if v is not None:
                extras.append(f"{nombre}: {'faltan ' + _g(v) if v > 0 else 'se pasó por ' + _g(-v)}")
        out += "Con lo registrado hasta ahora, " + " y ".join(partes) + "."
        if extras:
            out += " (" + "; ".join(extras) + ".)"
        # [P1-PLAN-LOTE-226] El margen macro a macro y la regla de comprobar que lo recomendado cabe.
        _margen = margen_del_dia(metas, cons)
        if _margen:
            out += f"\n⚖️ {_margen} {REGLA_MARGEN}"
        if hora_local is not None:
            out += f" Son las {_fmt_hora(hora_local)}."
        out += estado_de_las_comidas(consumed_today, hora_local, schedule_type)   # [P1-PLAN-LOTE-950]

        _anotadas = franjas_anotadas(consumed_today)
        _principales_ok = {"desayuno", "almuerzo", "cena"}.issubset(_anotadas)
        _kcal_cubierta = metas["kcal"] > 0 and cons["kcal"] >= 0.85 * metas["kcal"]
        if _principales_ok or _kcal_cubierta:
            out += (
                " ✅ DÍA PRINCIPAL YA REGISTRADO: ya anotó sus comidas principales y/o cubrió su meta calórica de hoy; "
                "PROHIBIDO preguntarle «¿qué comiste hoy?» ni qué le falta por anotar (una merienda opcional o un pequeño margen restante NO son comidas faltantes)."
            )
        if momento == "madrugada" and cons["registros"] > 0:
            out += (
                " 🌙 MADRUGADA (jornada biológica en curso): los registros anteriores pertenecen a esta misma jornada "
                "todavía abierta hasta las 04:00 AM; a esta hora NO toca ninguna comida ni debes pedirle registrar nada si solo saluda o conversa."
            )

        corto = (fk is not None and metas["kcal"] > 0
                 and cons["kcal"] < FRACCION_DIARIO_CORTO * metas["kcal"]
                 and not _principales_ok
                 and momento in ("tarde", "noche", "madrugada"))
        if corto:
            out += (" OJO: a esta hora un diario tan corto casi siempre significa que le falta REGISTRAR, no que no "
                    "haya comido. Antes de recomendarle comer más, pregúntale en una frase qué ha comido hoy que no "
                    "esté anotado — y anótalo.")
        if momento == "noche" and fk is not None and fk > 0:
            tope = int(round(FRACCION_CIERRE_NOCHE * metas["kcal"] / 10.0) * 10)
            out += (
                "\n🌙 CÓMO SE CIERRA EL DÍA A ESTA HORA: (1) primero la PROTEÍNA, con algo ligero y fácil de digerir "
                "(batida de proteína, yogurt griego, huevos o claras, atún, pechuga, queso fresco). (2) No le pidas "
                f"meter de golpe todo lo que falta: una comida de cierre razonable ronda las ~{min(tope, max(int(round(fk)), 0)) or tope} kcal; "
                "si falta más, que cierre la proteína y esa parte — lo demás NO se «compensa» mañana comiendo de más, "
                "mañana se sigue normal. (3) Una batida de proteína de noche NO es una deshora ni hace daño: es ligera "
                "y es la forma más fácil de llegar a la meta; no la prohíbas ni la regañes. Lo que SÍ se desaconseja "
                "cerca de dormir: comida pesada, frita o muy grasosa; picante o mucha salsa si sufre reflujo; cafeína "
                "(pre-entrenos, «quemadores», batidas con café) y mucho líquido justo antes de acostarse (mejor ~1 hora "
                "antes). Dilo en UNA frase y solo lo que aplique a lo que él va a comer. Sus condiciones médicas mandan "
                "sobre todo esto (enfermedad renal ⇒ la proteína extra la decide su médico; diabetes ⇒ cuidado con el "
                "carbohidrato nocturno). (Usa esta guía SOLO si pide qué cenar o cómo cerrar sus macros; si ya registró "
                "la cena o solo está anotando su diario a última hora de la noche, NO le sugieras comer más ni tomar "
                "batidas antes de dormir.)")
        elif momento in ("manana", "mediodia", "tarde") and fp is not None and fp > 25:
            out += (" Si pide ideas, reparte la proteína que falta entre las comidas que quedan en vez de dejarla "
                    "toda para la noche.")
        return out
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-132] bloque de lo que falta hoy no calculado: {e!r}")
        return ""


# ───────────────────────────── 2. Una comida a medida ─────────────────────────────

def franja_por_hora(hora: Optional[float], faltan=None) -> str:
    """La franja más probable cuando el usuario no la nombra: la primera SIN registrar cuya hora ya llegó o está cerca;
    si no hay dato, la de la hora."""
    h = 12.0 if hora is None else float(hora) % 24
    pendientes = [f for f in (faltan or []) if f in FRANJAS]
    # [P1-LATE-BREAKFAST] [BB-049] Si aún no hay desayuno registrado y es antes de las 12:00 PM, es desayuno (ej. brunch a las 11:30)
    if pendientes and "desayuno" in pendientes and 4.5 <= h < 12.0:
        return "desayuno"
    por_hora = ("desayuno" if 4.5 <= h < 12.0 else "almuerzo" if 12.0 <= h < 16.0 else
                "merienda" if 16.0 <= h < 19.0 else "cena")
    if pendientes and por_hora not in pendientes:
        orden = list(FRANJAS)
        despues = [f for f in pendientes if orden.index(f) >= orden.index(por_hora)]
        return (despues or pendientes)[0]
    return por_hora


def objetivo_de_la_comida(franja: str, metas: Optional[dict], falta: Optional[dict], hora: Optional[float],
                          n_comidas: int = 4, kcal: Optional[float] = None,
                          proteina: Optional[float] = None, de_noche: bool = False, faltan=None,
                          hay_registros: bool = True) -> Optional[dict]:
    """Los macros a los que se escala la propuesta.

    Base: la fracción de la franja sobre la meta del día (`nutrition_calculator.MEAL_SLOT_SPLITS`, el SSOT del solver).
    Si el usuario VIENE REGISTRANDO hoy, lo que falta se reparte entre las comidas que quedan (esta y las posteriores
    sin registrar), sin salirse de [0,6×, 1,4×] de la ración normal: una cena no se convierte en 1.500 kcal porque el
    día vaya corto, ni en 80 porque vaya lleno. Con el diario VACÍO manda la ración normal: un diario vacío suele ser
    «no registró», no «no comió» — medido en la batería del lote: a las 8:10, con el día en blanco, el desayuno salía
    inflado a 1,4× (574 kcal en vez de 410) y el almuerzo de un diabético a 1.003 kcal.
    Lo que el llamador pide explícitamente (`kcal`, `proteina`) manda."""
    if not metas or not metas.get("kcal"):
        if not kcal:
            return None
        metas = {"kcal": float(kcal) * 4, "protein_g": 0, "carbs_g": 0, "fats_g": 0}
    reparto = {"desayuno": 0.25, "almuerzo": 0.35, "merienda": 0.10, "cena": 0.30}
    try:
        from nutrition_calculator import MEAL_SLOT_SPLITS
        reparto = MEAL_SLOT_SPLITS.get(int(n_comidas or 4)) or MEAL_SLOT_SPLITS[4]
        # «merienda» suma sus variantes (merienda_am/pm) partidas: una merienda es UNA de ellas, la mayor
        frac = max((float(v) for k, v in reparto.items() if str(k).startswith(franja)), default=None)
    except Exception:
        frac = None
    if not frac:   # p. ej. una merienda pedida en un día de 3 comidas
        frac = {"desayuno": 0.25, "almuerzo": 0.35, "merienda": 0.10, "cena": 0.30}.get(franja, 0.25)
    base = {k: float(metas.get(k) or 0) * frac for k in ("kcal", "protein_g", "carbs_g", "fats_g")}
    obj = dict(base)
    if falta and falta.get("kcal") is not None and hay_registros:
        # La parte de ESTA comida dentro de lo que queda por comer: ella + las posteriores sin registrar.
        parte = 1.0
        try:
            orden = list(FRANJAS)
            pend = [f for f in (faltan or []) if f in orden and orden.index(f) > orden.index(franja)]
            pesos = {f: max((float(v) for k, v in reparto.items() if str(k).startswith(f)), default=0.0) for f in pend}
            total = frac + sum(pesos.values())
            parte = frac / total if total > 0 else 1.0
        except Exception:
            parte = 1.0
        restante = max(float(falta["kcal"]), 0.0) * parte
        obj["kcal"] = min(max(restante, 0.6 * base["kcal"]), 1.4 * base["kcal"]) if restante else 0.6 * base["kcal"]
        esc = obj["kcal"] / base["kcal"] if base["kcal"] else 1.0
        for k in ("carbs_g", "fats_g"):
            obj[k] = base[k] * esc
        fp = falta.get("protein_g")
        # La proteína que falta (su parte) se persigue hasta 1,6× la de la franja: es el macro que más cuesta cerrar.
        obj["protein_g"] = (min(max(float(fp) * parte, base["protein_g"] * 0.8), base["protein_g"] * 1.6)
                            if fp is not None and fp > 0 else base["protein_g"] * esc)
    # De NOCHE la comida de cierre no pasa del mismo techo que el bloque del día le dicta al coach (medido en el humo
    # del lote: con 840 kcal por cerrar a las 21:05 la herramienta proponía yuca con 237 g de cerdo, 837 kcal — lo
    # contrario de lo que el propio bloque aconsejaba). Se recortan kcal, carbohidrato y grasa; la proteína se queda.
    if de_noche and not kcal:
        techo = FRACCION_CIERRE_NOCHE * float(metas.get("kcal") or 0)
        if techo and obj["kcal"] > techo:
            rec = techo / obj["kcal"]
            obj["kcal"] = techo
            obj["carbs_g"] *= rec
            obj["fats_g"] *= rec
    if kcal:
        esc = float(kcal) / obj["kcal"] if obj["kcal"] else 1.0
        obj = {k: v * esc for k, v in obj.items()}
        obj["kcal"] = float(kcal)
    if proteina:
        obj["protein_g"] = float(proteina)
    if obj["kcal"] <= 0:
        return None
    # Sin reparto de macros conocido: uno estándar, para que el scorer tenga contra qué medir.
    if obj["protein_g"] <= 0:
        obj["protein_g"] = obj["kcal"] * 0.25 / 4
    if obj["carbs_g"] <= 0:
        obj["carbs_g"] = obj["kcal"] * 0.45 / 4
    if obj["fats_g"] <= 0:
        obj["fats_g"] = obj["kcal"] * 0.30 / 9
    return obj


def _pasos_de_biblioteca(tid: str, country: str, factor: float) -> list:
    try:
        import recipe_library as rl
        pasos = (rl._library(country).get(tid) or {}).get("pasos")
        if not (isinstance(pasos, list) and pasos):
            return []
        from deterministic_day import escalar_agua_en_pasos
        escalados, _ = escalar_agua_en_pasos(list(pasos), float(factor))
        return list(escalados)
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-132] receta de biblioteca no disponible para {tid}: {e!r}")
        return []


def _armar(t: dict, factor: float, catalogo, franja: str, obj: dict) -> Optional[dict]:
    """La comida escalada, con los MISMOS ladrillos que `deterministic_day.construir_comida` (gramos escalados, los
    condimentos no crecen, inclinación hacia la proteína) — sin su exigencia de receta congelada, que aquí es opcional."""
    import deterministic_day as dd
    lineas = []
    for g, nombre in dd._de_plantilla(t):
        fila = catalogo.get(nombre)
        if not fila or fila.get("kcal_per_100g") is None:
            return None
        gg = g if dd._no_escala(nombre) else round(g * float(factor), 1)
        if gg > 0:
            lineas.append([gg, nombre, dd._clase(fila)])
    if not lineas:
        return None
    if float(obj.get("protein_g") or 0) > 0:
        lineas = dd._inclinar(lineas, catalogo, float(obj["protein_g"]))
    tot = dd._macros([(l[0], l[1]) for l in lineas], catalogo)
    if not tot:
        return None
    ings = [f"{l[0]:g} g de {l[1]}" for l in lineas]
    return {"meal": franja.capitalize(), "name": t.get("name"), "ingredients": ings, "ingredients_raw": list(ings),
            "calories": int(round(tot["kcal"])), "protein": f"{tot['protein_g']:.0f}g",
            "carbs": f"{tot['carbs_g']:.0f}g", "fats": f"{tot['fats_g']:.0f}g",
            "_lineas": [(l[0], l[1]) for l in lineas], "_template_id": t.get("template_id"),
            "_meal_source": "coach_proposal"}


def cobertura_nevera(lineas, nevera_nombres) -> tuple:
    """(tiene, falta): nombres de ingredientes de la comida que están / no están en la Nevera. Los condimentos que no
    escalan (sal, ajo en polvo…) no cuentan como faltantes: nadie va al colmado por una pizca de orégano."""
    try:
        from constants import pantry_names_match
        import deterministic_day as dd
    except Exception:
        return [], []
    tiene, falta = [], []
    for _g_, nombre in lineas or []:
        if any(pantry_names_match(nombre, n) for n in (nevera_nombres or [])):
            tiene.append(nombre)
        elif not dd._no_escala(nombre):
            falta.append(nombre)
    return tiene, falta


def proponer_comidas(form_data: dict, franja: str, objetivo: dict, nevera_nombres=None, excluir=None,
                     solo_nevera: bool = False, max_minutos: Optional[int] = None, n: int = 3,
                     rotacion: int = 0, gustos_tokens: Optional[set] = None) -> list:
    """Hasta `n` comidas del registro de platos, escaladas a `objetivo`, verificadas (escáner culinario + backstop
    clínico) y ordenadas: primero las que pasan el tiempo pedido, después por cobertura de Nevera (si `solo_nevera` o
    hay Nevera), afinidad con sus gustos aprendidos y ajuste de macros. Lista vacía si nada sirve."""
    import deterministic_day as dd
    import dish_registry as dr
    from constants import cultural_country_for_form_data, country_for_form_data
    from shopping_calculator import get_master_ingredients

    fd = form_data if isinstance(form_data, dict) else {}
    country = cultural_country_for_form_data(fd) or "DO"
    catalogo = dd._CatalogoPorNombre(get_master_ingredients() or [])
    por_id = dr.templates_by_id(country) or {}
    if not catalogo or not por_id:
        return []
    alergias, dieta, excluidos = restricciones_del_perfil(fd)   # [P1-PLAN-LOTE-166] con el texto libre
    try:
        from culinary_context import declared_equipment
        equipo = declared_equipment(fd)
    except Exception:
        equipo = None
    tids, _fuente = dd._candidatos(dr, country, franja, [], None, k=200, rotate=int(rotacion or 0),
                                   exclude_allergens=alergias, diet=dieta,
                                   market_country=country_for_form_data(fd), exclude_foods=excluidos,
                                   available_equipment=equipo)
    if not tids:
        return []
    excluir_list = [str(x) for x in (excluir or []) if str(x or "").strip()]
    evitar = {dd._norm(x) for x in excluir_list}

    _match_exc = None
    _toks_fn = None
    if gustos_tokens is not None:
        try:
            from rutina_del_coach import plato_coincide_con_exclusion, _tokens_plato
            _match_exc = plato_coincide_con_exclusion
            _toks_fn = _tokens_plato
        except Exception:
            _match_exc = None
            _toks_fn = None

    # `elegir_plantillas` corta en el empate; aquí se quiere el ranking entero para poder reordenar por Nevera y gustos.
    lo, hi = dd._banda(franja)
    op, oc, of = (max(float(objetivo.get(k) or 0), 1.0) for k in ("protein_g", "carbs_g", "fats_g"))
    ok = float(objetivo.get("kcal") or 0)
    puntuados = []
    _evitar_riesgos = riesgos_a_evitar(fd)
    for tid in tids:
        t = por_id.get(tid)
        if not t or dd._norm(t.get("name")) in evitar:
            continue
        if _match_exc and excluir_list and _match_exc(t.get("name") or "", excluir_list):
            continue
        base = dd._macros(dd._de_plantilla(t), catalogo)
        if not base or not base.get("kcal"):
            continue
        f = ok / base["kcal"]
        if not (lo <= f <= hi):
            continue
        score = (2.0 * abs(base["protein_g"] * f - op) / op + abs(base["carbs_g"] * f - oc) / oc
                 + abs(base["fats_g"] * f - of) / of)
        if _evitar_riesgos:
            _attrs = t.get("intrinsic_risk_attributes") or {}
            score += PENA_POR_RIESGO_CLINICO * sum(1 for r in _evitar_riesgos if _attrs.get(r))
        puntuados.append((score, str(tid), t, f))
    puntuados.sort(key=lambda x: (x[0], x[1]))

    # Lo BARATO primero (armar, minutos, Nevera), se ordena, y la verificación —0,3 s por plato: escáner culinario +
    # backstop clínico— corre solo sobre lo que se va a devolver. Medido en el humo del lote: verificar los 40 mejores
    # costaba 4-13 s por llamada; así son ~1-2 s.
    armadas = []
    for score, tid, t, f in puntuados[:60]:
        c = _armar(t, f, catalogo, franja, objetivo)
        if not c:
            continue
        _cp = _num(c.get("protein"))
        _cc = _num(c.get("carbs"))
        _cf = _num(c.get("fats"))
        _pena_riesgo = 0.0
        if _evitar_riesgos:
            _attrs = t.get("intrinsic_risk_attributes") or {}
            _pena_riesgo = PENA_POR_RIESGO_CLINICO * sum(1 for r in _evitar_riesgos if _attrs.get(r))
        score_real = (
            2.0 * abs(_cp - op) / op
            + 1.5 * max(0.0, op - _cp) / op
            + abs(_cc - oc) / oc
            + abs(_cf - of) / of
            + _pena_riesgo
        )
        minutos = None
        try:
            lg = t.get("logistics") or {}
            if str(lg.get("prep_minutes_source") or "") in ("receta", "tecnica") and int(lg.get("prep_minutes_est") or 0) > 0:
                minutos = int(lg["prep_minutes_est"])
        except (TypeError, ValueError):
            minutos = None
        tiene, falta = cobertura_nevera(c["_lineas"], nevera_nombres)
        n_afines = 0
        if gustos_tokens and _toks_fn:
            for _g_, _ing_nom in c.get("_lineas") or []:
                if _toks_fn(_ing_nom) & gustos_tokens:
                    n_afines += 1
        c.update({"_score": round(score_real, 4), "_minutos": minutos, "_tiene": tiene, "_falta": falta, "_tid": tid,
                  "_factor": f, "_afinidad": n_afines})
        armadas.append(c)

    def _orden(c):
        tarde = bool(max_minutos and c["_minutos"] and c["_minutos"] > int(max_minutos) * 1.25)
        n_falta = len(c["_falta"]) if nevera_nombres else 0
        bonus = -0.06 * min(int(c.get("_afinidad") or 0), 3) if gustos_tokens else 0.0
        if solo_nevera:                       # lo pidió: la Nevera manda, y dentro de ella el ajuste de macros
            return (tarde, n_falta, c["_score"] + bonus)
        return (tarde, c["_score"] + bonus + PENA_POR_INGREDIENTE_QUE_FALTA * n_falta)

    armadas.sort(key=_orden)
    if rotacion and len(armadas) > 1:
        _top0 = _orden(armadas[0])
        _techo = _top0[-1] + dd._empate_score()
        _emp = [c for c in armadas if _orden(c)[:-1] == _top0[:-1] and _orden(c)[-1] <= _techo]
        if len(_emp) > 1:
            _r = int(rotacion) % len(_emp)
            armadas = _emp[_r:] + _emp[:_r] + armadas[len(_emp):]
    out = []
    for c in armadas:
        # La receta va ANTES de verificar: V3 acusa al ingrediente que ningún paso menciona, y sin pasos los acusa a
        # todos (medido en el humo del lote: 42 armadas, 0 verificadas, 141 × V3). Sin receta escrita —bibliotecas que
        # no existen todavía fuera de DO— las comprobaciones que LEEN los pasos no informan y se apartan; el backstop
        # clínico, el retinol y el resto del escáner corren igual.
        c["recipe"] = _pasos_de_biblioteca(c["_tid"], country, c["_factor"])
        viol = dd.verifica_comida(c, fd, catalogo)
        if not c["recipe"]:
            viol = [v for v in viol if not (isinstance(v, dict) and str(v.get("check")) in _CHECKS_QUE_LEEN_PASOS)]
        if viol:
            continue
        out.append(c)
        if len(out) >= max(1, int(n)):
            break
    return out


def buscar_receta_de_plato(form_data: dict, textos_o_plato, franja: str, objetivo: dict,
                           nevera_nombres=None) -> list:
    """Busca un plato puntual del catálogo por nombre o mención previa en el chat y devuelve sus pasos exactos."""
    import deterministic_day as dd
    import dish_registry as dr
    from constants import cultural_country_for_form_data
    from shopping_calculator import get_master_ingredients

    fd = form_data if isinstance(form_data, dict) else {}
    country = cultural_country_for_form_data(fd) or "DO"
    por_id = dr.templates_by_id(country) or {}
    catalogo = dd._CatalogoPorNombre(get_master_ingredients() or [])
    if not por_id or not catalogo:
        return []

    candidatos_txt = [str(textos_o_plato)] if isinstance(textos_o_plato, str) else [str(x) for x in (textos_o_plato or []) if str(x or "").strip()]
    if not candidatos_txt:
        return []

    try:
        from rutina_del_coach import _tokens_plato, _METODOS_Y_CONTEXTO
    except Exception:
        _tokens_plato, _METODOS_Y_CONTEXTO = None, frozenset()

    norm_txts = [dd._norm(x) for x in candidatos_txt]
    mejor = None
    mejor_score = (0, 0.0, 0)  # (prioridad_exacta, sim_tokens, len_nombre)

    for tid, t in por_id.items():
        nombre = str(t.get("name") or "").strip()
        norm_nom = dd._norm(nombre)
        if len(norm_nom) < 4:
            continue
        exacta = 1 if any(norm_nom in txt or txt in norm_nom for txt in norm_txts) else 0
        sim = 0.0
        if exacta:
            sim = 1.0
        elif _tokens_plato:
            toks_nom = _tokens_plato(nombre) - _METODOS_Y_CONTEXTO
            if toks_nom:
                min_req = 2 if len(toks_nom) >= 2 else 1
                for raw in candidatos_txt:
                    toks_raw = _tokens_plato(raw) - _METODOS_Y_CONTEXTO
                    inter = len(toks_nom & toks_raw)
                    if inter >= min_req:
                        sim = max(sim, inter / float(len(toks_nom)))
        if exacta or sim >= 0.65:
            cand_score = (exacta, round(sim, 4), len(norm_nom))
            if cand_score > mejor_score:
                mejor_score = cand_score
                mejor = (tid, t)

    if not mejor:
        return []

    tid, t = mejor
    base = dd._macros(dd._de_plantilla(t), catalogo)
    if not base or not base.get("kcal"):
        return []
    slot_t = str(t.get("meal_type") or franja or "desayuno").lower()
    lo, hi = dd._banda(slot_t if slot_t in FRANJAS else franja)
    ok = float((objetivo or {}).get("kcal") or base["kcal"])
    f = min(max(ok / base["kcal"], lo), hi)
    c = _armar(t, f, catalogo, slot_t if slot_t in FRANJAS else franja, objetivo or {})
    if not c:
        return []
    tiene, falta = cobertura_nevera(c["_lineas"], nevera_nombres)
    c["recipe"] = _pasos_de_biblioteca(tid, country, f)
    c.update({"_tid": tid, "_factor": f, "_tiene": tiene, "_falta": falta, "_afinidad": 1})
    return [c]


def _linea_amigable(g: float, nombre: str) -> str:
    """«206.5 g de Yuca» → «205 g de Yuca»; «0.5 g de Sal» → «una pizca de Sal». Las macros ya se calcularon con los
    gramos exactos; esto es solo cómo se LEE (nadie pesa 206,5 g), y redondear a 5 g mueve el plato menos de un 2 %."""
    g = float(g)
    if g < 1.5:
        return f"una pizca de {nombre}"
    if g < 20:
        return f"{int(round(g))} g de {nombre}"
    return f"{int(round(g / 5.0) * 5)} g de {nombre}"


def _norm_casera(texto: str) -> str:
    txt = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in txt if not unicodedata.combining(c))


def _a_unidad_simple(g: float, g_por_unidad: float, sing: str, plur: str, nombre: str = "") -> str:
    ratio = float(g) / max(float(g_por_unidad), 1.0)
    if ratio < 0.35:
        lbl = f"1/4 de {sing}"
    elif ratio < 0.65:
        lbl = f"1/2 {sing}"
    elif ratio < 0.88:
        if sing == "lonja":
            lbl = "1 lonja pequeña"
        elif sing == "unidad mediana":
            lbl = "1 unidad pequeña"
        else:
            lbl = f"3/4 de {sing}"
    else:
        n = round(ratio * 2.0) / 2.0
        n_txt = f"{int(n)}" if n.is_integer() else f"{n:g}"
        unit = sing if n == 1 else plur
        lbl = f"{n_txt} {unit}"
    return f"{lbl} de {nombre}" if nombre else lbl


def _a_tazas(g: float, g_por_taza: float, nombre: str) -> str:
    ratio = float(g) / max(float(g_por_taza), 1.0)
    if ratio < 0.20:
        cdas = max(1, int(round(float(g) / (g_por_taza / 16.0))))
        return f"{cdas} {'cucharada' if cdas == 1 else 'cucharadas'} de {nombre}"
    fracs = (
        (0.25, "1/4 taza"), (0.33, "1/3 taza"), (0.5, "1/2 taza"),
        (0.67, "2/3 taza"), (0.75, "3/4 taza"), (1.0, "1 taza"),
        (1.25, "1 1/4 tazas"), (1.5, "1 1/2 tazas"), (2.0, "2 tazas"),
    )
    if ratio <= 2.1:
        etiq = min(fracs, key=lambda x: abs(x[0] - ratio))[1]
        return f"{etiq} de {nombre}"
    tazas = round(ratio * 2.0) / 2.0
    txt = f"{int(tazas)}" if tazas.is_integer() else f"{tazas:g}"
    return f"{txt} tazas de {nombre}"


_HOJAS_VERDES = ("espinaca", "lechuga", "rucula", "repollo", "berro", "acelga", "kale")

_TAZA_POR_ALIMENTO = (
    (("cottage", "yogur", "yogurt", "requeson", "ricotta", "skyr", "kefir"), 225.0),
    (("leche", "batida", "jugo"), 240.0),
    (("avena",), 80.0),
    (("arroz", "quinoa", "granola", "cereal"), 150.0),
    (("cebolla", "pimiento", "aji", "zanahoria", "pepino", "calabacin", "hongos", "champinon"), 120.0),
    (("lechoza", "papaya", "sandia", "pina", "fresa", "arandano", "blueberry", "frambuesa", "uva",
      "mango", "melon", "brocoli", "vegetal", "ensalada", "habichuela", "lenteja", "garbanzo", "maiz"), 150.0),
)

_UNIDAD_POR_ALIMENTO = (
    (("galleta de arroz", "tortita de arroz"), 10.0, "unidad", "unidades", True),
    (("casabe",), 35.0, "pieza", "piezas", True),
    (("pan pita",), 45.0, "unidad", "unidades", True),
    (("pan", "tostada"), 30.0, "rebanada", "rebanadas", True),
    (("tortilla", "arepa"), 40.0, "unidad", "unidades", True),
    (("queso", "jamon", "pavo"), 28.0, "lonja", "lonjas", True),
    (("aguacate",), 140.0, "aguacate mediano", "aguacates medianos", False),
    (("platano",), 180.0, "plátano mediano", "plátanos medianos", False),
    (("yuca", "batata", "auyama"), 150.0, "trozo mediano", "trozos medianos", True),
    (("papa", "yautia"), 150.0, "unidad mediana", "unidades medianas", True),
    (("guineo", "banano", "manzana", "pera", "naranja", "tomate"), 120.0, "unidad mediana", "unidades medianas", True),
    (("atun",), 115.0, "lata", "latas", True),
)


def _linea_casera(g: float, nombre: str) -> str:
    """Convierte gramos en crudo a porciones cotidianas comprensibles para un humano."""
    g = float(g)
    if g < 1.5:
        return f"una pizca de {nombre}"
    nom_l = _norm_casera(nombre)
    if "clara" in nom_l and ("huevo" in nom_l or nom_l.startswith("clara")):
        n = max(1, int(round(g / 33.0)))
        if "cocid" in nom_l:
            return f"{n} {'clara cocida' if n == 1 else 'claras cocidas'}"
        return f"{n} {'clara' if n == 1 else 'claras'} de huevo"
    if "huevo" in nom_l:
        n = max(1, int(round(g / 55.0)))
        return f"{n} {'huevo' if n == 1 else 'huevos'}"
    if any(k in nom_l for k in ("whey", "proteina en polvo", "caseina")):
        return _a_unidad_simple(g, 30.0, "scoop", "scoops", nombre)
    if any(k in nom_l for k in ("galleta de arroz", "tortita de arroz")):
        return _a_unidad_simple(g, 10.0, "galleta", "galletas", nombre)
    if any(k in nom_l for k in ("pasta", "espagueti", "fideo")):
        return _a_tazas(g, 65.0, nombre).replace("taza de", "taza cocida de").replace("tazas de", "tazas cocidas de")
    if any(k in nom_l for k in _HOJAS_VERDES):
        if g < 25.0:
            return f"1 puñado de {nombre}"
        return _a_tazas(g, 45.0, nombre)
    if any(k in nom_l for k in ("queso crema", "tahini", "hummus", "pesto", "cacao", "coco rallado")):
        cdas = max(1, int(round(g / 16.0)))
        return f"{cdas} {'cucharada' if cdas == 1 else 'cucharadas'} de {nombre}"
    for claves, g_taza in _TAZA_POR_ALIMENTO:
        if any(k in nom_l for k in claves):
            return _a_tazas(g, g_taza, nombre)
    if "aceite" in nom_l:
        if g < 10.0:
            cdtas = max(1, int(round(g / 5.0)))
            return f"{cdtas} {'cucharadita' if cdtas == 1 else 'cucharaditas'} de {nombre}"
        cdas = max(1, int(round(g / 14.0)))
        return f"{cdas} {'cucharada' if cdas == 1 else 'cucharadas'} de {nombre}"
    if any(k in nom_l for k in ("mantequilla", "crema de mani", "miel", "mermelada", "semilla", "chia", "linaza")):
        cdas = max(1, int(round(g / 16.0)))
        return f"{cdas} {'cucharada' if cdas == 1 else 'cucharadas'} de {nombre}"
    if any(k in nom_l for k in ("mani", "almendra", "nuez", "nueces", "cacahuate")):
        if 22.0 <= g <= 38.0:
            return f"1 puñado de {nombre}"
        cdas = max(1, int(round(g / 12.0)))
        return f"{cdas} {'cucharada' if cdas == 1 else 'cucharadas'} de {nombre}"
    for claves, g_u, sing, plur, con_nombre in _UNIDAD_POR_ALIMENTO:
        if any(k in nom_l for k in claves):
            if claves[0] == "yuca":
                if g < 110.0:
                    return f"1 trozo pequeño de {nombre}"
                if g <= 175.0:
                    return f"1 trozo mediano de {nombre}"
                if g <= 235.0:
                    return f"1 trozo grande de {nombre}"
                return _a_unidad_simple(g, g_u, sing, plur, nombre)
            return _a_unidad_simple(g, g_u, sing, plur, nombre if con_nombre else "")
    if any(k in nom_l for k in ("pollo", "pechuga", "res", "cerdo", "pescado", "salmon", "tilapia", "chivo", "lomo", "bistec")):
        if g < 95.0:
            return f"1 porción pequeña de {nombre}"
        if g <= 145.0:
            return f"1 porción mediana de {nombre}"
        if g <= 210.0:
            return f"1 porción grande de {nombre}"
        return f"2 porciones medianas de {nombre}"
    if g < 8.0:
        return f"1 cucharadita de {nombre}"
    if g < 20.0:
        return f"1 cucharada de {nombre}"
    if g <= 80.0:
        return f"1/2 porción de {nombre}"
    return f"1 porción de {nombre}"


def formatear_propuestas(propuestas: list, franja: str, objetivo: dict, con_nevera: bool, contexto: str = "",
                         solo_nevera: bool = False, falta: Optional[dict] = None, nevera_activa: bool = True,
                         rutina_franja_txt: str = "", es_para_manana: bool = False,
                         usar_medidas_caseras: bool = False, pidio_propuesta_nueva: bool = False,
                         es_receta_pedida: bool = False) -> str:
    """El texto que lee el modelo. Cifras y pasos EXACTOS; al final, cómo presentarlo.

    [P1-NEVERA-OPCIONAL · 2026-09-23] `nevera_activa=False` (apagada en modo contador, a mano o sola): ninguna línea
    nombra la Nevera —el coach tiene la orden de no mencionarla—. Con `True` (default), el texto de siempre."""
    con_nevera = con_nevera and nevera_activa
    rutina_limpia = str(rutina_franja_txt or "").strip()
    if not propuestas:
        base_vacia = (
            f"No encontré en el catálogo de recetas un plato de {franja} que encaje con ese objetivo y con su "
            "perfil (dieta, alergias, rechazos" + (", solo con su Nevera" if con_nevera else "") + "). "
            "(Para el asistente: díselo tal cual y propón TÚ una comida sencilla"
            + (" con lo que hay en su Nevera" if nevera_activa else "")
            + ", marcando las macros como estimadas con «~». No inventes una receta «del catálogo».)"
        )
        return f"{base_vacia}\n{rutina_limpia}" if rutina_limpia else base_vacia
    cab = (f"PROPUESTAS DE {franja.upper()} a su medida (objetivo de esta comida: ~{int(round(objetivo['kcal']))} kcal, "
           f"~{int(round(objetivo['protein_g']))} g de proteína){contexto}. Platos del catálogo de recetas; los gramos y "
           "las macros salen de la tabla de alimentos (no son estimaciones tuyas):")
    lineas = [cab]
    if rutina_limpia:
        lineas.append(rutina_limpia)
    if solo_nevera and not nevera_activa:   # sin inventario no se finge saber qué tiene, y sin nombrar la Nevera
        lineas.append("OJO: pidió cocinar SOLO con lo que tiene, pero no sabes qué hay en su casa: estas son ideas "
                      "generales. Pregúntale qué tiene para afinar.")
    elif solo_nevera and not con_nevera:
        lineas.append("OJO: pidió cocinar SOLO con lo que tiene, pero su Nevera está VACÍA en la app: estas son ideas "
                      "generales. Dile que no tienes registrado qué hay en su casa y pregúntale qué tiene (o que "
                      "escanee su Nevera) para afinar.")
    elif solo_nevera and all(c.get("_falta") for c in propuestas):
        lineas.append("OJO: ninguna receta del catálogo se hace SOLO con lo que tiene en la Nevera; estas son las que "
                      "MENOS le piden comprar. Dilo así.")
    if not con_nevera and not solo_nevera and nevera_activa:
        lineas.append("(Su Nevera está vacía EN LA APP: eso NO significa que no tenga comida en casa. No le digas que "
                      "«toca comprar»; como mucho, que no tienes registrado qué hay en su cocina.)")
    for i, c in enumerate(propuestas, start=1):
        t = f" · {c['_minutos']} min" if c.get("_minutos") else ""
        lineas.append(f"{i}) {c['name']} — {c['calories']} kcal · proteína {c['protein']} · carbohidratos {c['carbs']} · "
                      f"grasas {c['fats']}{t}")
        lineas.append("   Ingredientes (en crudo): " + "; ".join(_linea_amigable(g, nom) for g, nom in c["_lineas"]))
        if usar_medidas_caseras and c.get("_lineas"):
            lineas.append("   Porción casera aprox.: " + "; ".join(_linea_casera(g, nom) for g, nom in c["_lineas"]))
        if con_nevera:
            if c["_falta"]:
                lineas.append(f"   Nevera: tiene {len(c['_tiene'])} de {len(c['_tiene']) + len(c['_falta'])}; le falta: "
                              + ", ".join(c["_falta"]))
            else:
                lineas.append("   Nevera: lo tiene TODO en casa.")
        if falta and falta.get("kcal") is not None and not es_para_manana:
            # La resta hecha: en la batería del lote el modelo la hizo de cabeza y dio 1.560 donde eran 1.476.
            qk = float(falta["kcal"]) - float(c["calories"])
            txt = (f"le quedarían ~{int(round(qk))} kcal" if qk >= 0 else f"se pasaría por ~{int(round(-qk))} kcal")
            if falta.get("protein_g") is not None:
                qp = float(falta["protein_g"]) - _num(c["protein"])
                txt += (f" y ~{int(round(qp))} g de proteína por cubrir" if qp > 3 else " y la proteína del día cubierta")
            lineas.append(f"   Si se la come: {txt}.")
        if i == 1 and c.get("recipe"):
            lineas.append("   Pasos: " + " ".join(f"{j}. {p}" for j, p in enumerate(c["recipe"], start=1)))

    modo_default = not (rutina_limpia or es_para_manana or usar_medidas_caseras or pidio_propuesta_nueva or es_receta_pedida)
    if modo_default:
        lineas.append(
            "(Para el asistente: presenta la nº 1 con sus kcal y proteína y, en una frase, cómo deja su día (la línea «Si se "
            "la come», tal cual — no hagas la resta tú). Da los ingredientes con sus gramos; los PASOS solo si pidió receta o cómo se hace, "
            "resumidos sin cambiarles el sentido ni añadir ingredientes. Nombra la nº 2 como alternativa en media frase. "
            "No cambies cifras ni gramos. "
            + ("Si le falta algo de la Nevera, dilo. " if nevera_activa else "")
            + "NO la registres: todavía no se la ha comido "
            "— cierra ofreciendo anotarla cuando se la coma (entonces `log_consumed_meal` con ESTAS macros y estos "
            "ingredientes). Si pide otra opción, vuelve a llamar esta herramienta pasando en `excluir` los nombres ya "
            "propuestos.)"
        )
        return "\n".join(lineas)

    sin_activos_rutina = bool(
        rutina_limpia and "YA PROPUESTO EN ESTE CHAT" in rutina_limpia
        and "ya fueron propuestos en el turno anterior" in rutina_limpia
    )
    if es_receta_pedida:
        apertura = (
            "(Para el asistente: el usuario pide CÓMO PREPARAR o los pasos de este plato. Dale directamente los ingredientes "
            "en «Porción casera aprox.» (y gramos si ayuda), sus kcal/proteína y los Pasos numerados de preparación claros "
            "y directos — NUNCA digas «la herramienta no me mostró los pasos» ni propongas otro plato distinto sin dárselos."
        )
    elif pidio_propuesta_nueva or sin_activos_rutina:
        apertura = (
            "(Para el asistente: el usuario pidió una PROPUESTA NUEVA y distinta; PROHIBIDO repetir cualquier plato marcado "
            "con «⛔ YA PROPUESTO EN ESTE CHAT». Propón directamente la opción nº 1 del catálogo (o un plato de su rutina NO "
            "descartado arriba) usando su «Porción casera aprox.» y sus kcal y proteína."
        )
    elif rutina_limpia:
        apertura = (
            "(Para el asistente: PRIORIZA primero la rutina/plato real del usuario de arriba (en porciones caseras, "
            "con su ajuste o proteína extra si hace falta, y SIN recitar gramos en crudo salvo que él pida receta o "
            "gramaje), y nombra la nº 1 del catálogo en medidas caseras como alternativa práctica para variar."
        )
    else:
        apertura = (
            "(Para el asistente: presenta directamente la nº 1 con su «Porción casera aprox.» y sus kcal y proteína "
            "(sin recitar gramos en crudo salvo que él pida receta o gramaje)"
        )

    if es_para_manana:
        pref = "" if (rutina_limpia or pidio_propuesta_nueva or sin_activos_rutina or es_receta_pedida) else "."
        cierre_dia = (
            f"{pref} OJO: esta propuesta es para MAÑANA / su próxima jornada; indica solo cuánto aporta el plato, y "
            "PROHIBIDO decir «si lo comes te quedarían X kcal hoy» o restarlo de un presupuesto futuro. "
        )
    else:
        cierre_dia = (
            " Si presentas una opción para hoy y existe la línea «Si se la come», úsala tal cual — no hagas la resta tú. "
        )

    if es_receta_pedida:
        cuerpo = (
            "Si el plato no trajera pasos preescritos arriba, guíalo en 3-4 pasos culinarios sencillos usando únicamente sus "
            "ingredientes reales. PROHIBIDO mencionar «herramienta» o excusas técnicas. "
            + ("Si le falta algo de la Nevera, dilo. " if nevera_activa else "")
            + "NO la registres: cierra ofreciendo anotarla cuando la coma.)"
        )
    else:
        cuerpo = (
            "Presenta el plato conservando la medida casera EXACTA de CADA ingrediente (carbohidrato/vívere base, "
            "proteína, grasa y vegetales — PROHIBIDO omitir cantidades diciendo solo «la yuca», «el arroz» o «un poco de cebolla» "
            "si arriba aparece su «Porción casera aprox.»). Los gramos en crudo y los PASOS solo van si pidió receta o cómo se hace. "
            "Si aporta valor, menciona 1 alternativa breve con sus ~kcal y ~g de proteína. "
            "PROHIBIDO cerrar preguntando «¿Quieres que te dé la receta de...?». No cambies cifras ni macros. "
            + ("Si le falta algo de la Nevera, dilo. " if nevera_activa else "")
            + "NO la registres: todavía no se la ha comido "
            "— cierra ofreciendo anotarla cuando se la coma (entonces `log_consumed_meal` con ESTAS macros y estos "
            "ingredientes). Si pide otra opción («q más», «otra»), vuelve a llamar esta herramienta pasando en `excluir` los nombres ya "
            "propuestos.)"
        )
    lineas.append(apertura + cierre_dia + cuerpo)
    return "\n".join(lineas)
