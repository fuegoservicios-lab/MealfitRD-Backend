# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-960 · 2026-10-01] El coach usa la Nevera con INICIATIVA: sabe cuánta proteína le da cada cosa que tiene.

El dueño, 1-oct: «¿el agente está consciente al 100 % de mi inventario, para que pueda ser proactivo en momentos
oportunos? Tengo una proteína en la Nevera y a veces me gustaría que tomara la iniciativa, o si digo algo, que pueda
complementar con lo que ya tengo».

Lo que el coach tenía: la LISTA de la Nevera dos veces (📦 con caducidad, 🧊 con las cantidades reales) y una frase
genérica («prioriza cocinar con esto»). Lo que le faltaba para tomar la iniciativa con criterio:
  · cuánta proteína le da cada cosa que tiene — la cuenta la hacía de memoria, o no la hacía;
  · cruzarla con LO QUE LE FALTA HOY (el bloque del lote 132): «con 200 g de tu pechuga cierras los 45 g que faltan»;
  · cuándo es oportuno proponerla sin que la pida, y cuándo callarse (una vez por conversación, no en cada mensaje);
  · y el aviso de comida (el mensaje que sale justo antes de su hora) no miraba la Nevera en absoluto.

Todo determinista y sin LLM: la proteína y las kcal salen de `master_ingredients` (por 100 g), los gramos de
`db_inventory.convert_amount`, la caducidad de la misma regla que la anotación 📦 (`anotacion_nevera.plazo`) y la
compatibilidad del backstop clínico (`clinical_backstop_for_meal`: alergias, dieta, embarazo, medicación). Lo que el
backstop rechaza NO se propone nunca (puede ser de otra persona de la casa); si el backstop falla, tampoco (fail-secure).

Knob `MEALFIT_COACH_NEVERA_INICIATIVA` (True): apagado, ni el bloque del chat ni el del aviso.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

MAX_PROTEINAS = 8
# Por debajo de esto «le falta proteína» no merece una propuesta: es ruido de la estimación del diario.
FALTA_MINIMA_G = 15.0
# Con la caducidad dentro de esta ventana se prioriza; fuera de ella (más vieja) el dato suele ser una fila olvidada o
# algo congelado: no se afirma que caducó (la anotación 📦 ya lo dice a su manera), solo se deja de priorizar.
VENTANA_URGENTE = (-2, 2)

_LEGUMBRE_RX = re.compile(r"\b(habichuela|frijol|lenteja|garbanzo|gandul|judia|guisante|haba|soya|soja|edamame)",
                          re.I)
_LATA_RX = re.compile(r"\b(lata|en agua|en aceite|enlatad)", re.I)


def activo() -> bool:
    """tooltip-anchor: MEALFIT_COACH_NEVERA_INICIATIVA"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_COACH_NEVERA_INICIATIVA", True)
    except Exception:
        return True


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x if x == x else 0.0


def _clase(nombre: str, fila: dict) -> Optional[str]:
    """Qué TIPO de proteína es (decide la porción), o None si no es fuente de proteína para proponer."""
    cat = _norm(fila.get("category"))
    p, kcal = _f(fila.get("protein_g_per_100g")), _f(fila.get("kcal_per_100g"))
    n = _norm(nombre)
    if n.startswith("yema"):
        return None
    if "huevo" in n:
        return "huevo"
    if "texturizada" in n:
        return "seco"
    if cat.startswith("proteina") and p >= 10:
        if _LATA_RX.search(n) or _norm(fila.get("default_unit")) == "lata":
            return "lata"
        if fila.get("ready_to_eat") and kcal >= 250:
            return "embutido"
        return "carne"
    if cat.startswith("lacteo") and p >= 8:
        return "queso" if p >= 15 else "lacteo"
    if cat.startswith("despensa") and p >= 15 and _LEGUMBRE_RX.search(n):
        return "legumbre"
    return None


# Porción de referencia (gramos de la fila del catálogo) y tope de lo que se propone de una vez.
_PORCION_G = {"carne": 150, "lata": 100, "embutido": 50, "huevo": 100, "queso": 40, "lacteo": 200,
              "legumbre": 80, "seco": 50}
_TOPE_G = {"carne": 250, "lata": 200, "embutido": 80, "huevo": 200, "queso": 60, "lacteo": 300,
           "legumbre": 120, "seco": 80}
_NOTA_PORCION = {"carne": " en crudo", "legumbre": " en seco (≈1 taza cocida)", "seco": " en seco",
                 "lata": " escurrido"}


def _porcion_txt(clase: str, gramos: float) -> str:
    if clase == "huevo":
        n = max(1, int(round(gramos / 50.0)))
        return f"{n} huevo" + ("s" if n != 1 else "")
    return f"{int(round(gramos))} g" + _NOTA_PORCION.get(clase, "")


def _resolver(fila_inv: dict, por_id: dict, por_nombre: dict, catalogo: list) -> Optional[dict]:
    mid = str(fila_inv.get("master_ingredient_id") or "").strip()
    if mid and mid in por_id:
        return por_id[mid]
    nombre = fila_inv.get("ingredient_name") or ""
    fila = por_nombre.get(_norm(nombre))
    if fila:
        return fila
    try:
        from constants import pantry_names_match
        for c in catalogo:
            if pantry_names_match(nombre, c.get("name") or ""):
                return c
    except Exception:
        pass
    return None


def _gramos(fila_inv: dict, master: dict) -> Optional[float]:
    qty, unit = _f(fila_inv.get("quantity")), str(fila_inv.get("unit") or "")
    if qty <= 0:
        return None
    try:
        from db_inventory import convert_amount_container
        g = convert_amount_container(qty, unit, "g", master)   # «1 cartón (30 uds.)» de huevos también
        return float(g) if g and g > 0 else None
    except Exception:
        return None


def _dias_desde(valor, hoy: date) -> Optional[int]:
    if not valor:
        return None
    try:
        if isinstance(valor, datetime):
            d = valor.date()
        elif isinstance(valor, date):
            d = valor
        else:
            d = datetime.strptime(str(valor)[:10], "%Y-%m-%d").date()
        return (hoy - d).days
    except Exception:
        return None


def _dias_restantes(fila_inv: dict, master: dict, hoy: date) -> Optional[int]:
    """Misma regla que la anotación 📦 (`get_user_inventory`), con la señal más reciente de la fila (una edición
    manual reinicia el reloj)."""
    dias = [d for d in (_dias_desde(fila_inv.get("created_at"), hoy), _dias_desde(fila_inv.get("updated_at"), hoy))
            if d is not None]
    if not dias:
        return None
    try:
        from db_inventory import _infer_shelf_life_days
        import anotacion_nevera
        nombre, cat = fila_inv.get("ingredient_name") or "", master.get("category") or ""
        vida = master.get("shelf_life_days")
        if vida is None:
            vida = _infer_shelf_life_days(nombre, cat)
        vida = anotacion_nevera.plazo(nombre, cat, vida)
        return int(vida) - min(dias)
    except Exception:
        return None


def _violaciones(nombre: str, perfil: Optional[dict]) -> list:
    """Backstop clínico para UN alimento. Sin perfil, nada que comprobar. Si el escáner falla, se veta (fail-secure)."""
    if not isinstance(perfil, dict) or not perfil:
        return []
    try:
        from graph_orchestrator import clinical_backstop_for_meal
        alergias = perfil.get("allergies") or []
        if isinstance(alergias, str):
            alergias = [alergias]
        return clinical_backstop_for_meal({"name": nombre, "ingredients": [f"150 g de {nombre}"]},
                                          allergies=list(alergias), diet_type=perfil.get("dietType"),
                                          form_data=perfil)
    except Exception as e:
        return [f"backstop ilegible ({type(e).__name__})"]


def proteinas_de(filas_inv, catalogo, perfil: Optional[dict] = None, hoy: Optional[date] = None) -> list:
    """Las fuentes de proteína de la Nevera, con lo que aporta una porción y si se pueden proponer. Pura (salvo el
    backstop, que es determinista). Orden: aptas primero; dentro, la que caduca antes y después la de más proteína por
    caloría."""
    hoy = hoy or date.today()
    catalogo = [c for c in (catalogo or []) if isinstance(c, dict)]
    por_id = {str(c.get("id")): c for c in catalogo if c.get("id") is not None}
    por_nombre = {_norm(c.get("name")): c for c in catalogo}
    out = []
    for fi in filas_inv or []:
        if not isinstance(fi, dict) or _f(fi.get("quantity")) <= 0:
            continue
        nombre = str(fi.get("ingredient_name") or "").strip()
        master = _resolver(fi, por_id, por_nombre, catalogo)
        if not nombre or not master:
            continue
        clase = _clase(nombre, master)
        if not clase:
            continue
        p100, k100 = _f(master.get("protein_g_per_100g")), _f(master.get("kcal_per_100g"))
        porcion = float(_PORCION_G[clase])
        if clase == "huevo":
            porcion = 2 * (_f(master.get("density_g_per_unit")) or 50.0)
        gramos = _gramos(fi, master)
        dias = _dias_restantes(fi, master, hoy)
        viol = _violaciones(nombre, perfil)
        out.append({
            "nombre": nombre, "clase": clase, "p100": p100, "k100": k100,
            "cantidad": f"{_fmt_q(fi.get('quantity'))} {fi.get('unit') or ''}".strip(),
            "gramos": gramos, "porcion_g": porcion,
            "porcion_p": p100 * porcion / 100.0, "porcion_kcal": k100 * porcion / 100.0,
            "porciones": int(gramos // porcion) if gramos else None,
            "dias_restantes": dias,
            "urgente": dias is not None and VENTANA_URGENTE[0] <= dias <= VENTANA_URGENTE[1],
            "apta": not viol, "motivo": "; ".join(viol[:1]),
        })
    out.sort(key=lambda x: (not x["apta"], not x["urgente"], -(x["p100"] / max(x["k100"], 1.0))))
    return out[:MAX_PROTEINAS]


def _fmt_q(q) -> str:
    x = _f(q)
    return str(int(x)) if x.is_integer() else f"{x:g}"


# En desayuno y merienda, una pechuga de 200 g no es la idea natural: primero huevo, lácteo, queso o lata.
_LIGERAS = ("huevo", "lacteo", "queso", "lata")


def _ordenar_para_franja(proteinas: list, franja: Optional[str]) -> list:
    if franja not in ("desayuno", "merienda"):
        return list(proteinas)
    # estable: dentro de cada grupo se respeta el orden de `proteinas_de` (lo que caduca antes, primero)
    return sorted(proteinas, key=lambda x: (not x["urgente"], x["clase"] not in _LIGERAS))


def propuesta_para_falta(proteinas: list, falta_p: Optional[float], franja: Optional[str] = None) -> Optional[dict]:
    """La porción de SU proteína que cierra lo que falta (o lo más que una comida razonable admite). Lo que caduca
    pronto va primero; en desayuno y merienda, lo ligero antes que la carne."""
    if falta_p is None or falta_p < FALTA_MINIMA_G:
        return None
    for x in _ordenar_para_franja(proteinas, franja):
        if not x["apta"] or x["p100"] <= 0:
            continue
        g = falta_p / x["p100"] * 100.0
        paso = 50.0 if x["clase"] == "huevo" else 25.0
        g = max(paso, round(g / paso) * paso)
        g = min(g, float(_TOPE_G[x["clase"]]))
        if x["gramos"]:
            if x["gramos"] < paso:
                continue
            g = min(g, (x["gramos"] // paso) * paso)
        return {"nombre": x["nombre"], "porcion": _porcion_txt(x["clase"], g),
                "proteina": x["p100"] * g / 100.0, "kcal": x["k100"] * g / 100.0,
                "cubre": x["p100"] * g / 100.0 >= falta_p - 5}
    return None


def _linea(x: dict) -> str:
    t = (f"- {x['nombre']} ({x['cantidad']}): {_porcion_txt(x['clase'], x['porcion_g'])} ≈ "
         f"{int(round(x['porcion_p']))} g de proteína y ~{int(round(x['porcion_kcal']))} kcal")
    if x["porciones"]:
        t += f"; le alcanza para ~{x['porciones']} " + ("porción" if x["porciones"] == 1 else "porciones")
    if x["urgente"]:
        d = x["dias_restantes"]
        t += (" — ⏳ úsala PRIMERO: " + ("debería consumirse hoy" if d <= 0 else f"le quedan ~{d} día"
                                          + ("s" if d != 1 else "")) + " (si no la congeló)")
    return t + "."


def bloque_chat(proteinas: list, falta_p: Optional[float], franja: Optional[str] = None) -> str:
    """El bloque del system prompt del coach. "" sin proteínas (el bloque 🧊 ya dice lo que hay)."""
    aptas = [x for x in proteinas if x["apta"]]
    vetadas = [x for x in proteinas if not x["apta"]]
    if not aptas and not vetadas:
        return ("\n\n🥩 PROTEÍNA EN SU NEVERA [P1-PLAN-LOTE-960]: ninguna registrada. Si pide cocinar «con lo que "
                "tengo», díselo claro y propón la proteína más fácil de conseguir; no finjas que tiene.")
    out = ("\n\n🥩 PROTEÍNA QUE YA TIENE EN SU NEVERA [P1-PLAN-LOTE-960] (calculada por el sistema con la tabla de "
           "alimentos — usa ESTAS cifras tal cual, no las recalcules):")
    for x in aptas:
        out += "\n" + _linea(x)
    if vetadas:
        out += ("\n- NO la propongas (choca con sus alergias, dieta o condición; puede ser de otra persona de la casa): "
                + ", ".join(x["nombre"] for x in vetadas) + ".")
    prop = propuesta_para_falta(proteinas, falta_p, franja)
    if prop:
        out += (f"\n🎯 CON LO QUE LE FALTA HOY (~{int(round(falta_p))} g de proteína): {prop['porcion']} de "
                f"{prop['nombre']} (de su Nevera) ≈ {int(round(prop['proteina']))} g de proteína (~{int(round(prop['kcal']))} kcal)"
                + (" — lo cierra sin comprar nada." if prop["cubre"] else
                   " — es lo razonable en una comida; el resto, repartido en las que quedan."))
    out += (
        "\n🤝 INICIATIVA CON SU NEVERA [P1-PLAN-LOTE-960] — el usuario PIDIÓ que tomes la iniciativa y complementes con "
        "lo que ya tiene:"
        "\n1. COMPLEMENTA: cuando cuente lo que va a comer o lo que se está preparando y a esa comida (o al día) le "
        "falta proteína, propón en UNA frase añadir una porción concreta de SU proteína de arriba, con la cantidad y lo "
        "que suma («Ponle 150 g de la pechuga que tienes: +34 g de proteína»). Si ya lleva proteína suficiente, no la "
        "fuerces. Si ya COMIÓ, primero se registra lo que comió (regla de siempre); la sugerencia va para la próxima."
        "\n2. CUANDO PIDA IDEAS, diga que tiene hambre o que no sabe qué comer: arranca por lo que tiene (esta proteína "
        "primero, la marcada ⏳ antes que ninguna) antes de proponer nada que haya que comprar."
        "\n3. TOMA LA INICIATIVA sin que te lo pida SOLO en un momento oportuno: cuando toca una comida y le falta "
        "proteína, cuando algo marcado ⏳ se le va a dañar, o al cerrar un registro con el día corto de proteína. UNA "
        "frase concreta al final de tu respuesta, no al principio ni en cada mensaje; si ya se lo propusiste en esta "
        "conversación y no lo tomó, no insistas. Con su plan vigente, complementa el plato del plan; no lo cambies "
        "salvo que él lo pida."
        "\n4. Las cantidades de la Nevera son las de arriba: nunca prometas más de lo que tiene, y si dice que ya se la "
        "comió o que se le dañó, actualiza la Nevera con la herramienta.")
    return out


def bloque_aviso(proteinas: list, falta_p: Optional[float], franja: Optional[str] = None) -> str:
    """El bloque del prompt del aviso de comida: solo lo apto, máximo 3 (los de la franja primero) y la propuesta
    para ESTA comida — `falta_p` ya llega acotado a una comida por `para_aviso`."""
    aptas = _ordenar_para_franja([x for x in proteinas if x["apta"]], franja)[:3]
    if not aptas:
        return ""
    out = ("\nProteína que YA tiene en su Nevera (ya filtrada por sus alergias y dieta; son los ÚNICOS alimentos que "
           "puedes nombrar):")
    for x in aptas:
        out += "\n" + _linea(x)
    prop = propuesta_para_falta(aptas, falta_p, franja)
    if prop:
        out += (f"\n- Para esta comida: {prop['porcion']} de {prop['nombre']} (de su Nevera) ≈ "
                f"{int(round(prop['proteina']))} g de proteína.")
    out += ("\nToma la iniciativa: propón UNA idea sencilla para esta comida con lo que ya tiene (empieza por lo marcado "
            "⏳ si lo hay), con la porción. No le recites la Nevera ni la lista entera.\n")
    return out


# ───────────────────────────── Lectura (DB) ─────────────────────────────

def _filas(user_id: str) -> list:
    from db import execute_sql_query
    return execute_sql_query(
        "SELECT ingredient_name, quantity::float8 AS quantity, unit, created_at, updated_at, master_ingredient_id "
        "FROM user_inventory WHERE user_id = %s AND kind = 'food' AND quantity > 0 "
        "ORDER BY ingredient_name LIMIT 120",
        (user_id,), fetch_all=True,
    ) or []


def _perfil(form_data) -> dict:
    try:
        from coach_day_context import _con_texto_libre
        return _con_texto_libre(form_data if isinstance(form_data, dict) else {})
    except Exception:
        return form_data if isinstance(form_data, dict) else {}


def _proteinas_del_usuario(user_id: str, form_data) -> list:
    filas = _filas(user_id)
    if not filas:
        return []
    from shopping_calculator import get_master_ingredients
    return proteinas_de(filas, get_master_ingredients(), _perfil(form_data))


def para_chat(user_id, nevera_on: bool, form_data, plan_vigente, diario_de_hoy, tz_offset=None) -> str:
    """El bloque para los dos caminos del chat. "" para invitados, con la Nevera apagada, con el knob apagado, o si
    algo falla (con log): el coach sigue con los bloques 📦/🧊 de siempre."""
    if not user_id or user_id == "guest" or not nevera_on or not activo():
        return ""
    try:
        import coach_day_context as cdc
        try:
            from prompts.chat_agent import hora_local_del_chat
            franja = cdc.franja_por_hora(hora_local_del_chat(tz_offset))
        except Exception:
            franja = None
        falta_p = None
        if diario_de_hoy is not None:
            metas = cdc.metas_del_dia(form_data, plan_vigente)
            if metas:
                falta_p = cdc.falta_hoy(metas, cdc.consumido_hoy(diario_de_hoy)).get("protein_g")
        return bloque_chat(_proteinas_del_usuario(user_id, form_data), falta_p, franja)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-960] bloque de la Nevera con iniciativa no calculado: {e!r}")
        return ""


# Lo que una comida razonable aporta de la meta diaria de proteína: el aviso propone para ESA comida, no para el día
# entero (a las 7 a. m. «lo que falta hoy» es toda la meta, y 250 g de camarones no son un desayuno).
FRACCION_POR_COMIDA = 0.3
_FRANJA_DEL_AVISO = {"desayuno": "desayuno", "breakfast": "desayuno", "almuerzo": "almuerzo", "lunch": "almuerzo",
                     "merienda": "merienda", "snack": "merienda", "cena": "cena", "dinner": "cena"}


def para_aviso(user_id, health, consumed, meal: Optional[str] = None) -> str:
    """El bloque para el aviso de comida. "" con la Nevera apagada, sin proteína apta o si algo falla."""
    if not user_id or not activo():
        return ""
    try:
        from nevera_opcional import nevera_activa
        if not nevera_activa(user_id):
            return ""
        falta_p = None
        try:
            from aviso_del_dia import _metas
            metas = _metas(health)
            if metas and metas[1]:
                import coach_day_context as cdc
                falta_dia = metas[1] - cdc.consumido_hoy(consumed)["protein_g"]
                falta_p = min(falta_dia, max(25.0, FRACCION_POR_COMIDA * metas[1]))
        except Exception:
            falta_p = None
        return bloque_aviso(_proteinas_del_usuario(user_id, health), falta_p, _FRANJA_DEL_AVISO.get(_norm(meal)))
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-960] Nevera del aviso no calculada: {e!r}")
        return ""
