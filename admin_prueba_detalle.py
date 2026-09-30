# backend/admin_prueba_detalle.py
"""[P1-PLAN-LOTE-832 · 2026-09-29] Panel admin · Cuentas: el detalle de una cuenta de PRUEBA.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §4.4, §8 y §13.4 (raíz del
workspace); contrato 9 del plan. Una cuenta de prueba es una cuenta que el dueño marcó con un motivo y cuya persona ya
vio el aviso en la app (`cuentas_prueba.exigir_prueba`). Solo de ella, y solo mientras la marca siga viva, el panel
abre el CONTENIDO:

  · `formulario`     — `health_profile` presentado por grupos (las claves del asistente y de Configuración; lo que el
                       panel no conoce queda solo en el crudo) y lo que el coach recuerda (`user_facts` activos);
  · `comidas`        — los registros de `consumed_meals` de un rango de días del calendario del admin con los dos
                       extremos incluidos (los 30 últimos, hoy incluido, por defecto; 90 como máximo), con un margen
                       de horas que cubre cualquier huso (`ventana_de_comidas`);
  · `planes`, `plan` — los 20 últimos con sus bloques de `plan_chunk_queue`, y un plan día a día;
  · `conversaciones`, `conversacion` — los 50 hilos más recientes con el coach y uno entero;
  · `adjunto`        — una foto ENVIADA en uno de sus hilos (las del escáner no se guardan: no existen);
  · `actividad`      — la línea de tiempo (10 tipos, 500 eventos, del más nuevo al más viejo) y su gasto de IA.

Quién decide qué es de la cuenta (Review Focus 2), siempre dentro de la consulta:
  · un plan: `meal_plans.user_id`;
  · un hilo: el SSOT `USER_CHAT_THREAD_IDS_SQL` (el mismo conjunto que exportan y borran «Mis datos» y «Eliminar
    cuenta»); la respuesta del coach lleva `user_id` NULL y aun así es de su hilo;
  · una foto: su `session_id` es uno de esos hilos, su `user_id` (si lo lleva) es el de la cuenta, y fue ENVIADA
    (`message_id` no nulo: lo que se subió y no se mandó no es parte de ninguna conversación y se purga a las 24 h).
Un id de otra cuenta devuelve None —el router responde 404— aunque exista: lo mismo que si no existiera.

Aquí vive SOLO la lectura. El orden de cada vista (interruptor → `exigir_prueba` → rastro `ver_prueba` → consulta) y
los códigos HTTP son del router (`routers/admin.py`). Nada escribe, nada llama a la IA ni cobra cuota. Si una consulta
falla, la función LANZA y el router responde 503 sin datos: un hueco callado en la línea de tiempo diría «esta semana
no comió». Los parámetros se validan con funciones puras (`rango_de_comidas`, `tipos_de_actividad`) que lanzan
`ErrorDetalle(422, código)`: el router las corre ANTES de mirar la marca.
"""
from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Optional

import ajustes_cuenta
from admin_metricas import _ETIQUETA_FUNCION
from db import USER_CHAT_THREAD_IDS_SQL, execute_sql_query
from ficha_comida import origen_de_comida, plan_ref_limpio

logger = logging.getLogger(__name__)

GRUPOS_FORMULARIO = ("Objetivo", "Cuerpo", "Dieta y alergias", "Salud", "Horarios", "Hogar y presupuesto",
                     "País e idioma")
TIPOS_DE_EVENTO = ("comida", "plan", "mensaje", "bloque_fallido", "ia", "metrica", "alerta", "peso", "agua", "ajuste")
MAX_EVENTOS = 500
MAX_SESIONES = 50
MAX_PLANES = 20
MAX_PRIMER_MENSAJE = 140
MAX_MENSAJES = 500          # los últimos de un hilo, en orden cronológico
MAX_COMIDAS = 1000          # 90 días a 10 registros al día caben
MAX_MEMORIA = 500
MAX_BLOQUES = 1000
DIAS_COMIDAS = 30
MAX_DIAS_COMIDAS = 90
# La ventana de la consulta de comidas rebasa los días pedidos por estas horas (ver `ventana_de_comidas`).
MARGEN_ANTES_H = 12
MARGEN_DESPUES_H = 14
MAX_DIAS_ACTIVIDAD = 30
# Solo fotos, y solo los formatos que el chat acepta al subir (`routers/diary.py`, con su sniff de bytes). Sin SVG:
# abierta directamente en el navegador, una imagen SVG ejecutaría su script en el origen de la API.
TIPOS_DE_FOTO = {"image/jpeg": "image/jpeg", "image/jpg": "image/jpeg", "image/png": "image/png",
                 "image/webp": "image/webp", "image/heic": "image/heic", "image/heif": "image/heif"}

_RE_FECHA = re.compile(r"\d{4}-\d{2}-\d{2}")
_MAX_VALOR = 2000
_MAX_DETALLE = 240

_HILOS = USER_CHAT_THREAD_IDS_SQL           # tres %s: el uid de la cuenta en los tres
_VENTANA = "now() - make_interval(days => %s)"


class ErrorDetalle(Exception):
    """Un parámetro del detalle fuera de contrato: el router lo devuelve con su código (422) y su `detalle`
    (`fecha`, `rango`, `dias`, `tipos`)."""

    def __init__(self, status: int, detalle: str):
        super().__init__(detalle)
        self.status = status
        self.detalle = detalle


# ─────────────────────────────────────────────────────────────────────────────────────────────────────── utilidades
def _uuid(v) -> Optional[str]:
    """El id en forma canónica, o None si no es un uuid: no se consulta la base con basura."""
    try:
        return str(uuid.UUID(str(v).strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _iso(v) -> Optional[str]:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v) if v not in (None, "") else None


def _instante(v) -> Optional[datetime]:
    """Un instante comparable (UTC si viene sin zona), o None."""
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, date):
        return datetime.combine(v, time(0), tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return None


def _json(v):
    """jsonb que llega como texto (algunos drivers o filas viejas) → objeto; lo demás tal cual."""
    if isinstance(v, (str, bytes, bytearray)):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _num(v, defecto=None):
    """Una cifra para el JSON: entera si lo es, si no con un decimal (los `numeric` llegan como Decimal)."""
    if v is None or isinstance(v, bool):
        return defecto
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return defecto
    if not math.isfinite(f):
        return defecto
    return int(f) if f.is_integer() else round(f, 1)


def _entero(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _plural(n: int, palabra: str) -> str:
    return f"{n} {palabra}" + ("" if n == 1 else "s")


def _recorte(v, tope: int) -> Optional[str]:
    """El texto en una línea y como mucho `tope` caracteres (con «…» si se cortó). None si no hay nada."""
    if v is None:
        return None
    texto = " ".join(str(v).split())
    if not texto:
        return None
    return texto if len(texto) <= tope else texto[: tope - 1].rstrip() + "…"


def _plano(v, nivel: int = 0) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, (int, float, Decimal)):
        n = _num(v)
        return "" if n is None else str(n)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, dict):
        partes = []
        for clave, x in v.items():
            texto = _plano(x, nivel + 1)
            if texto:
                partes.append(f"{clave}: {texto}")
        if nivel == 0:
            return " · ".join(partes)
        return f"({', '.join(partes)})" if partes else ""
    if isinstance(v, (list, tuple)):
        return ", ".join(t for t in (_plano(x, nivel + 1) for x in v) if t)
    return str(v)


def _texto_de(v, tope: int = _MAX_DETALLE) -> str:
    """Cualquier valor como texto corto y legible («clave: valor · …»). El panel lo pinta como texto, nunca como
    HTML."""
    return _recorte(_plano(v), tope) or ""


def _textos(v) -> list:
    """Renglones de texto (ingredientes, pasos): lo que no es texto no entra; un texto suelto es un renglón."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [x.strip() for x in v if isinstance(x, str) and x.strip()]


def _unir(*partes) -> str:
    return " · ".join(p for p in (str(x).strip() for x in partes if x not in (None, "")) if p)


# ─────────────────────────────────────────────────────────────────────────────────────────── los parámetros (422)
def hoy_utc() -> date:
    return datetime.now(timezone.utc).date()


def _fecha(v) -> Optional[date]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if not isinstance(v, str) or not _RE_FECHA.fullmatch(v):
        raise ErrorDetalle(422, "fecha")
    try:
        return date.fromisoformat(v)
    except ValueError as e:                    # 2026-02-30: la forma es buena, la fecha no existe
        raise ErrorDetalle(422, "fecha") from e


def ventana_de_comidas(desde: date, hasta: date) -> tuple:
    """`(inicio, fin)` de la consulta de comidas, como instantes UTC con zona (no como `date`: así no dependen de la zona
    horaria de la sesión de Postgres): `consumed_at >= inicio` y `< fin`.

    [P1-PLAN-LOTE-832 · 2026-09-29] Los días son los del CALENDARIO del admin, pero las filas están en UTC y las personas
    viven en husos distintos (RD en UTC−4: su cena de las 23:00 del día `hasta` es del día siguiente en UTC; España en
    UTC+2: su desayuno de la 01:00 del día `desde` es del día anterior). Con la ventana exacta de días UTC, el usuario RD
    perdía la cena de hoy. Se ensancha para cubrir el día local de cualquiera: `MARGEN_ANTES_H` antes de `desde` 00:00 UTC
    y hasta las `MARGEN_DESPUES_H` del día siguiente a `hasta`. Puede colarse alguna comida de las horas vecinas; el `at`
    de cada una sale entero (en UTC) para que el panel la sitúe. Lanza `OverflowError` en el borde del calendario."""
    inicio = datetime.combine(desde, time(0), tzinfo=timezone.utc) - timedelta(hours=MARGEN_ANTES_H)
    fin = (datetime.combine(hasta + timedelta(days=1), time(0), tzinfo=timezone.utc)
           + timedelta(hours=MARGEN_DESPUES_H))
    return inicio, fin


def rango_de_comidas(desde=None, hasta=None, hoy: Optional[date] = None) -> tuple:
    """`(desde, hasta)` de la sección de comidas: días del calendario del admin, los DOS incluidos (el frontend pide
    `hasta` = hoy y espera las comidas de hoy; la consulta los cubre con el margen de `ventana_de_comidas`). Sin fechas:
    los 30 últimos, hoy (UTC) incluido; con una sola: 30 días desde ella o hasta ella. Una fecha que no es AAAA-MM-DD (o
    que no existe) ⇒ 422 `fecha`; `desde` posterior a `hasta` o más de 90 días contando los dos extremos ⇒ 422 `rango`
    (el límite es de las FECHAS: el margen de horas no cuenta)."""
    d1, d2 = _fecha(desde), _fecha(hasta)
    try:
        if d1 is None and d2 is None:
            d2 = hoy or hoy_utc()
        if d1 is None:
            d1 = d2 - timedelta(days=DIAS_COMIDAS - 1)
        elif d2 is None:
            d2 = d1 + timedelta(days=DIAS_COMIDAS - 1)
        ventana_de_comidas(d1, d2)             # la ventana (con su margen) de la consulta también tiene que existir
    except OverflowError as e:                 # 0001-01-01 / 9999-12-31: el borde del calendario, no un 500
        raise ErrorDetalle(422, "fecha") from e
    if d1 > d2 or (d2 - d1).days + 1 > MAX_DIAS_COMIDAS:
        raise ErrorDetalle(422, "rango")
    return d1, d2


def tipos_de_actividad(tipos=None) -> tuple:
    """Los tipos de evento pedidos (coma-separados o una lista), en el orden fijo de `TIPOS_DE_EVENTO`. Vacío = todos;
    uno que no está en la lista ⇒ 422 `tipos` (nada del cliente llega a una consulta sin pasar por aquí)."""
    if tipos is None:
        partes = []
    elif isinstance(tipos, str):
        partes = tipos.split(",")
    else:
        partes = [str(t) for t in tipos]
    pedidos = {t.strip() for t in partes if t.strip()}
    if not pedidos:
        return TIPOS_DE_EVENTO
    if pedidos - set(TIPOS_DE_EVENTO):
        raise ErrorDetalle(422, "tipos")
    return tuple(t for t in TIPOS_DE_EVENTO if t in pedidos)


def _dias_de_actividad(dias) -> int:
    if isinstance(dias, bool):
        raise ErrorDetalle(422, "dias")
    try:
        n = int(dias)
    except (TypeError, ValueError):
        raise ErrorDetalle(422, "dias") from None
    if not 1 <= n <= MAX_DIAS_ACTIVIDAD:
        raise ErrorDetalle(422, "dias")
    return n


def objeto_de_comidas(desde: date, hasta: date) -> str:
    """Lo que se abrió, para el rastro `ver_prueba`: el rango de días."""
    return f"{desde.isoformat()}..{hasta.isoformat()}"


def objeto_de_actividad(dias: int, tipos=None) -> str:
    """Lo que se abrió, para el rastro: `7d`, o `7d:comida,ia` si se filtró por tipo."""
    elegidos = tipos_de_actividad(tipos)
    return f"{dias}d" if elegidos == TIPOS_DE_EVENTO else f"{dias}d:{','.join(elegidos)}"


# ─────────────────────────────────────────────────────────────────────────────────────────────────────── formulario
# (clave de `health_profile`, grupo, etiqueta), en el orden de la pantalla. Las claves son las REALES del asistente
# (`initialFormData` de AssessmentContext y `components/assessment/questions`) y las de Configuración que describen a
# la persona (perfil clínico, súper personalización, básicos, horas de las comidas); `test_p1_plan_lote_832` comprueba
# que cada una existe en el frontend. `None` es la columna `user_profiles.locale`, no una clave del perfil. Lo que no
# está aquí (lo que escribe el servidor, lo interno `_*`, una clave nueva) sale solo en el crudo.
_CAMPOS = (
    ("appMode", "Objetivo", "Uso de la app"),
    ("mainGoal", "Objetivo", "Objetivo principal"),
    ("targetWeight", "Objetivo", "Peso meta"),
    ("targetWeightAuto", "Objetivo", "Sin meta de peso concreta"),
    ("goalPace", "Objetivo", "Ritmo"),
    ("motivation", "Objetivo", "Motivación"),
    ("struggles", "Objetivo", "Dificultades"),
    ("otherStruggles", "Objetivo", "Otras dificultades"),
    ("age", "Cuerpo", "Edad"),
    ("gender", "Cuerpo", "Sexo"),
    ("height", "Cuerpo", "Altura (cm)"),
    ("weight", "Cuerpo", "Peso"),
    ("weightUnit", "Cuerpo", "Unidad de peso"),
    ("bodyFat", "Cuerpo", "Grasa corporal (%)"),
    ("waistCm", "Cuerpo", "Cintura (cm)"),
    ("activityLevel", "Cuerpo", "Actividad física"),
    ("dietType", "Dieta y alergias", "Dieta"),
    ("allergies", "Dieta y alergias", "Alergias"),
    ("otherAllergies", "Dieta y alergias", "Otras alergias"),
    ("dislikes", "Dieta y alergias", "No le gusta"),
    ("otherDislikes", "Dieta y alergias", "Otros que no le gustan"),
    ("staple_foods", "Dieta y alergias", "Mis básicos"),
    ("stapleFoods", "Dieta y alergias", "Mis básicos"),           # el espejo del formulario: solo si falta la canónica
    ("stapleAnchors", "Dieta y alergias", "Anclas de los básicos"),
    ("super_personalization", "Dieta y alergias", "Súper personalización"),
    ("medicalConditions", "Salud", "Condiciones médicas"),
    ("otherConditions", "Salud", "Otras condiciones"),
    ("medications", "Salud", "Medicamentos"),
    ("otherMedications", "Salud", "Otros medicamentos"),
    ("clinical_profile", "Salud", "Perfil clínico avanzado"),
    ("habitAlcohol", "Salud", "Alcohol"),
    ("habitSmoking", "Salud", "Tabaco o vape"),
    ("habitCaffeine", "Salud", "Cafeína"),
    ("habitWater", "Salud", "Agua al día"),
    ("sleepHours", "Salud", "Horas de sueño"),
    ("stressLevel", "Salud", "Estrés"),
    ("includeSupplements", "Salud", "Quiere suplementos"),
    ("currentSupplements", "Salud", "Suplementos que toma"),
    ("recommendSupplements", "Salud", "Pide que le recomienden suplementos"),
    ("selectedSupplements", "Salud", "Suplementos elegidos"),
    ("scheduleType", "Horarios", "Jornada"),
    ("avisos_por_comida", "Horarios", "Horas de las comidas"),    # una fila por comida (ver `_horarios`)
    ("householdSize", "Hogar y presupuesto", "Personas en casa"),
    ("groceryDuration", "Hogar y presupuesto", "Compra cada"),
    ("budget", "Hogar y presupuesto", "Presupuesto"),
    ("budgetAmount", "Hogar y presupuesto", "Monto del presupuesto"),
    ("budgetCurrency", "Hogar y presupuesto", "Moneda"),
    ("cookingTime", "Hogar y presupuesto", "Tiempo para cocinar"),
    ("mealOrganization", "Hogar y presupuesto", "Organización de las comidas"),
    ("freshTopup", "Hogar y presupuesto", "Compra fresca entre semana"),
    ("freezerMode", "Hogar y presupuesto", "Congelador"),
    ("batchCooking", "Hogar y presupuesto", "Cocina por tandas"),
    ("planSource", "Hogar y presupuesto", "El plan parte de"),
    ("country", "País e idioma", "País"),
    (None, "País e idioma", "Idioma de la app"),
    ("cultureProfiles", "País e idioma", "Cocinas que le representan"),
)
_COMIDAS_DEL_DIA = (("desayuno", "Desayuno"), ("almuerzo", "Almuerzo"), ("merienda", "Merienda"), ("cena", "Cena"))


def claves_del_formulario() -> tuple:
    """Las claves de `health_profile` que el formulario presenta por campos (el resto va solo en el crudo)."""
    return tuple(dict.fromkeys(c for c, _, _ in _CAMPOS if c))


def _campo(grupo: str, etiqueta: str, valor) -> dict:
    return {"grupo": grupo, "etiqueta": etiqueta, "valor": valor}


def _valor(v):
    """El valor de un campo, apto para pintar como texto: los escalares tal cual (el texto, recortado), las listas de
    escalares como lista sin vacíos, y lo anidado como texto legible (su forma exacta está en el crudo)."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        return _num(v)
    if isinstance(v, str):
        return v if len(v) <= _MAX_VALOR else v[: _MAX_VALOR - 1] + "…"
    if isinstance(v, (list, tuple)):
        salida = []
        for x in v:
            if x is None or x == "":
                continue
            if isinstance(x, (str, bool, int, float, Decimal)):
                salida.append(_valor(x))
            else:
                texto = _texto_de(x, _MAX_VALOR)
                if texto:
                    salida.append(texto)
        return salida
    return _texto_de(v, _MAX_VALOR)


def _horarios(hp: dict) -> list:
    """Las horas de las comidas (`avisos_por_comida`: `{desayuno: {activo, hora}, …}`), una fila por comida."""
    todas = hp.get("avisos_por_comida")
    if todas is None:
        return []
    if not isinstance(todas, dict):
        return [_campo("Horarios", "Horas de las comidas", _valor(todas))]
    campos = []
    for clave, nombre in _COMIDAS_DEL_DIA:
        propia = todas.get(clave)
        if propia is None:
            continue
        if not isinstance(propia, dict):
            campos.append(_campo("Horarios", nombre, _valor(propia)))
            continue
        hora = propia.get("hora")
        texto = hora.strip() if isinstance(hora, str) and hora.strip() else "sin hora"
        if propia.get("activo") is False:
            texto += " (recordatorio apagado)"
        campos.append(_campo("Horarios", nombre, texto))
    return campos


def campos_del_formulario(hp, locale=None) -> list:
    """`[{grupo, etiqueta, valor}]` en el orden de `GRUPOS_FORMULARIO`: solo las claves conocidas que la persona tiene
    (con el valor que tengan, también null). Nunca revienta con tipos raros."""
    hp = hp if isinstance(hp, dict) else {}
    campos = []
    for clave, grupo, etiqueta in _CAMPOS:
        if clave is None:
            if isinstance(locale, str) and locale.strip():
                campos.append(_campo(grupo, etiqueta, locale.strip()))
        elif clave == "avisos_por_comida":
            campos.extend(_horarios(hp))
        elif clave == "stapleFoods" and "staple_foods" in hp:
            continue
        elif clave in hp:
            campos.append(_campo(grupo, etiqueta, _valor(hp[clave])))
    return campos


def _relevancia(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(f, 2) if math.isfinite(f) else None


_SQL_PERFIL = "SELECT health_profile, locale FROM public.user_profiles WHERE id = %s"
_SQL_MEMORIA = ("SELECT id::text AS id, fact, created_at, salience_score FROM public.user_facts "
                "WHERE user_id = %s AND is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT %s")


def formulario(user_id) -> Optional[dict]:
    """Contrato 9 · `formulario`: `{campos, crudo, memoria}`. None si la cuenta no existe."""
    uid = _uuid(user_id)
    if not uid:
        return None
    fila = execute_sql_query(_SQL_PERFIL, (uid,), fetch_one=True)
    if not fila:
        return None
    hp = _json(fila.get("health_profile"))
    hp = hp if isinstance(hp, dict) else {}
    hechos = execute_sql_query(_SQL_MEMORIA, (uid, MAX_MEMORIA), fetch_all=True) or []
    return {
        "campos": campos_del_formulario(hp, fila.get("locale")),
        "crudo": hp,
        "memoria": [{"id": str(h.get("id")), "dato": h.get("fact") or "", "creado": _iso(h.get("created_at")),
                     "relevancia": _relevancia(h.get("salience_score"))} for h in hechos],
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────── comidas
_SQL_COMIDAS = ("SELECT id::text AS id, consumed_at, meal_type, meal_name, ingredients, calories, protein, carbs, "
                "healthy_fats, source, plan_ref FROM public.consumed_meals "
                "WHERE user_id = %s AND consumed_at >= %s AND consumed_at < %s "
                "ORDER BY consumed_at DESC, id DESC LIMIT %s")


def _comida(f: dict) -> dict:
    return {
        "id": str(f.get("id")), "at": _iso(f.get("consumed_at")), "tipo": f.get("meal_type"),
        "plato": f.get("meal_name") or "", "ingredientes": _textos(_json(f.get("ingredients"))),
        "kcal": _num(f.get("calories"), 0), "proteina": _num(f.get("protein"), 0),
        "carbohidratos": _num(f.get("carbs"), 0), "grasas": _num(f.get("healthy_fats"), 0),
        # el vocabulario y la limpieza de la ficha del plato (lote 720): lo desconocido es None
        "origen": origen_de_comida(f.get("source")), "plan_ref": plan_ref_limpio(_json(f.get("plan_ref"))),
    }


def comidas(user_id, desde=None, hasta=None) -> dict:
    """Contrato 9 · `comidas`: `{desde, hasta, comidas}`, la más reciente primero. `desde`/`hasta` son las fechas pedidas
    (las del calendario del admin, ambas incluidas); la consulta lee la ventana ancha de `ventana_de_comidas`
    (`desde` 00:00 UTC − 12 h ≤ `consumed_at` < `hasta` + 1 día 00:00 UTC + 14 h), de modo que la comida de las 23:00 de
    un usuario en UTC−4 (las 03:00 UTC del día siguiente) entra. Lanza `ErrorDetalle` si el rango no vale."""
    d1, d2 = rango_de_comidas(desde, hasta)
    salida = {"desde": d1.isoformat(), "hasta": d2.isoformat(), "comidas": []}
    uid = _uuid(user_id)
    if not uid:
        return salida
    inicio, fin = ventana_de_comidas(d1, d2)
    filas = execute_sql_query(_SQL_COMIDAS, (uid, inicio, fin, MAX_COMIDAS), fetch_all=True) or []
    salida["comidas"] = [_comida(f) for f in filas]
    return salida


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────── planes
_SQL_PLANES = ("SELECT id::text AS id, created_at, name, calories, revision, "
               "plan_data ->> 'generation_status' AS estado, "
               "CASE WHEN jsonb_typeof(plan_data -> 'days') = 'array' THEN jsonb_array_length(plan_data -> 'days') "
               "ELSE 0 END AS dias FROM public.meal_plans WHERE user_id = %s "
               "ORDER BY created_at DESC, id DESC LIMIT %s")
_SQL_BLOQUES = ("SELECT id::text AS id, meal_plan_id::text AS plan_id, status, attempts, week_number, days_offset, "
                "dead_letter_reason FROM public.plan_chunk_queue "
                "WHERE user_id = %s AND meal_plan_id = ANY(%s::uuid[]) "
                "ORDER BY week_number, days_offset, created_at, id LIMIT %s")
_SQL_PLAN = ("SELECT id::text AS id, name, created_at, plan_data FROM public.meal_plans "
             "WHERE id = %s AND user_id = %s")


def _bloques(uid: str, plan_ids: list) -> list:
    """`[(plan_id, bloque)]` de la cola de esos planes de la cuenta, por semana y desplazamiento."""
    filas = execute_sql_query(_SQL_BLOQUES, (uid, list(plan_ids), MAX_BLOQUES), fetch_all=True) or []
    return [(str(b.get("plan_id")), {
        "id": str(b.get("id")), "estado": str(b.get("status") or "otro"), "intentos": _entero(b.get("attempts")),
        "semana": _num(b.get("week_number")), "dias_offset": _num(b.get("days_offset")),
        "motivo_fallo": _recorte(b.get("dead_letter_reason"), 300)}) for b in filas]


def planes(user_id) -> dict:
    """Contrato 9 · `planes`: los 20 últimos, el más nuevo primero, cada uno con sus bloques contados por estado y en
    detalle (intentos y el motivo del fallo)."""
    uid = _uuid(user_id)
    if not uid:
        return {"planes": []}
    filas = execute_sql_query(_SQL_PLANES, (uid, MAX_PLANES), fetch_all=True) or []
    if not filas:
        return {"planes": []}
    por_plan: dict = {}
    for plan_id, bloque in _bloques(uid, [str(f.get("id")) for f in filas]):
        por_plan.setdefault(plan_id, []).append(bloque)
    salida = []
    for f in filas:
        detalle = por_plan.get(str(f.get("id")), [])
        cuenta: dict = {}
        for b in detalle:
            cuenta[b["estado"]] = cuenta.get(b["estado"], 0) + 1
        salida.append({
            "id": str(f.get("id")), "creado": _iso(f.get("created_at")), "nombre": f.get("name") or None,
            "kcal": _num(f.get("calories")), "estado_generacion": _recorte(f.get("estado"), 64),
            "dias": _entero(f.get("dias")), "revision": _num(f.get("revision")),
            "bloques": cuenta, "bloques_detalle": detalle,
        })
    return {"planes": salida}


def _comida_del_plan(m: dict) -> dict:
    return {
        "tipo": m.get("meal") if isinstance(m.get("meal"), str) else None,
        "plato": m.get("name") if isinstance(m.get("name"), str) else "",
        "ingredientes": _textos(m.get("ingredients")),
        "kcal": _num(m.get("cals"), 0),
        "macros": {"proteina": _num(m.get("protein"), 0), "carbohidratos": _num(m.get("carbs"), 0),
                   "grasas": _num(m.get("fats"), 0)},
        "pasos": _textos(m.get("recipe")),
    }


def _comidas_del_dia(dia) -> list:
    lista = dia.get("meals") if isinstance(dia, dict) else None
    return [_comida_del_plan(m) for m in (lista if isinstance(lista, list) else []) if isinstance(m, dict)]


def plan(user_id, plan_id) -> Optional[dict]:
    """Contrato 9 · `planes/{plan_id}`: `{id, nombre, creado, dias: [{dia, comidas}], bloques}`. None si el plan no
    es de la cuenta (o no existe): se busca por su id Y el de la cuenta."""
    uid, pid = _uuid(user_id), _uuid(plan_id)
    if not uid or not pid:
        return None
    fila = execute_sql_query(_SQL_PLAN, (pid, uid), fetch_one=True)
    if not fila:
        return None
    datos = _json(fila.get("plan_data"))
    datos = datos if isinstance(datos, dict) else {}
    dias = datos.get("days") if isinstance(datos.get("days"), list) else []
    nombre = fila.get("name") or (datos.get("name") if isinstance(datos.get("name"), str) else None) or None
    return {
        "id": str(fila.get("id")), "nombre": nombre, "creado": _iso(fila.get("created_at")),
        # «Día N» es el N-ésimo de la ventana viva del plan (`plan_data.days`), el mismo índice que `plan_ref`
        "dias": [{"dia": i + 1, "comidas": _comidas_del_dia(d)} for i, d in enumerate(dias)],
        "bloques": [b for _, b in _bloques(uid, [pid])],
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────────── conversaciones
_SQL_SESIONES = (
    "SELECT s.id, s.inicio, s.ultimo, s.mensajes, s.pulgares_abajo, s.fotos, pm.texto AS primer_mensaje FROM ("
    "SELECT m.session_id AS sid, m.session_id::text AS id, min(m.created_at) AS inicio, "
    "max(m.created_at) AS ultimo, count(*) AS mensajes, count(*) FILTER (WHERE m.feedback = 'down') AS pulgares_abajo, "
    "COALESCE(sum(CASE WHEN jsonb_typeof(m.attachments) = 'array' THEN jsonb_array_length(m.attachments) ELSE 0 END), "
    "0) AS fotos FROM public.agent_messages m "
    f"WHERE m.session_id::text IN ({_HILOS}) GROUP BY m.session_id ORDER BY max(m.created_at) DESC LIMIT %s) s "
    "LEFT JOIN LATERAL (SELECT left(p.content, 200) AS texto FROM public.agent_messages p "
    "WHERE p.session_id = s.sid AND p.role = 'user' ORDER BY p.created_at, p.id LIMIT 1) pm ON true "
    "ORDER BY s.ultimo DESC, s.id"
)
_SQL_MENSAJES = ("SELECT m.id::text AS id, m.role, m.content, m.created_at, m.feedback, m.attachments "
                 "FROM public.agent_messages m "
                 f"WHERE m.session_id = %s AND m.session_id::text IN ({_HILOS}) "
                 "ORDER BY m.created_at DESC, m.id DESC LIMIT %s")
_SQL_ADJUNTO = ("SELECT a.content, a.content_type FROM public.chat_attachments a WHERE a.id = %s "
                f"AND a.message_id IS NOT NULL AND a.session_id::text IN ({_HILOS}) "
                "AND (a.user_id IS NULL OR a.user_id = %s)")


def conversaciones(user_id) -> dict:
    """Contrato 9 · `conversaciones`: los 50 hilos de la cuenta con mensajes, el más reciente primero, con cuántos
    mensajes, cuántos 👎 y cuántas fotos llevan, y su primer mensaje recortado a 140 caracteres."""
    uid = _uuid(user_id)
    if not uid:
        return {"sesiones": []}
    filas = execute_sql_query(_SQL_SESIONES, (uid, uid, uid, MAX_SESIONES), fetch_all=True) or []
    return {"sesiones": [{
        "id": str(f.get("id")), "inicio": _iso(f.get("inicio")), "ultimo": _iso(f.get("ultimo")),
        "mensajes": _entero(f.get("mensajes")), "pulgares_abajo": _entero(f.get("pulgares_abajo")),
        "fotos": _entero(f.get("fotos")), "primer_mensaje": _recorte(f.get("primer_mensaje"), MAX_PRIMER_MENSAJE),
    } for f in filas]}


def _fotos(adjuntos) -> list:
    """Las fotos de un mensaje (`agent_messages.attachments`, hasta 4): `[{id, tipo}]`. Sin su contenido: la foto se
    pide aparte a `adjunto`, que vuelve a comprobar de quién es."""
    adjuntos = _json(adjuntos)
    if not isinstance(adjuntos, list):
        return []
    salida = []
    for a in adjuntos:
        if not isinstance(a, dict):
            continue
        aid = _uuid(a.get("attachment_id") or a.get("id"))
        if aid:
            tipo = a.get("content_type")
            salida.append({"id": aid, "tipo": tipo if isinstance(tipo, str) else None})
    return salida


def _mensaje(f: dict) -> dict:
    contenido, feedback = f.get("content"), f.get("feedback")
    return {
        "id": str(f.get("id")), "rol": "persona" if f.get("role") == "user" else "coach",
        "texto": contenido if isinstance(contenido, str) else ("" if contenido is None else str(contenido)),
        "at": _iso(f.get("created_at")), "feedback": feedback if feedback in ("up", "down") else None,
        "fotos": _fotos(f.get("attachments")),
    }


def conversacion(user_id, session_id) -> Optional[dict]:
    """Contrato 9 · `conversaciones/{session_id}`: los mensajes del hilo en orden cronológico (los 500 últimos), con
    las respuestas del coach aunque lleven `user_id` NULL. None si el hilo no es de la cuenta (o no tiene mensajes)."""
    uid, sid = _uuid(user_id), _uuid(session_id)
    if not uid or not sid:
        return None
    filas = execute_sql_query(_SQL_MENSAJES, (sid, uid, uid, uid, MAX_MENSAJES), fetch_all=True) or []
    if not filas:
        return None
    return {"id": sid, "mensajes": [_mensaje(f) for f in reversed(filas)]}


def adjunto(user_id, attachment_id) -> Optional[tuple]:
    """Contrato 9 · `adjuntos/{attachment_id}`: `(bytes, content_type)` de una foto enviada en un hilo de la cuenta.
    None si no es de la cuenta, no se envió, no tiene contenido o no es una foto (el router responde 404)."""
    uid, aid = _uuid(user_id), _uuid(attachment_id)
    if not uid or not aid:
        return None
    fila = execute_sql_query(_SQL_ADJUNTO, (aid, uid, uid, uid, uid), fetch_one=True)
    if not fila:
        return None
    tipo = TIPOS_DE_FOTO.get(str(fila.get("content_type") or "").split(";")[0].strip().lower())
    contenido = fila.get("content")
    if isinstance(contenido, (memoryview, bytearray)):
        contenido = bytes(contenido)
    if not tipo or not isinstance(contenido, bytes) or not contenido:
        return None
    return contenido, tipo


# ─────────────────────────────────────────────────────────────────────────────────────────────── línea de tiempo
# Cada consulta: `[uid …] días límite` (el uid una vez, o tres en el SSOT de los hilos), la ventana de `now()` y los 500
# más recientes; la mezcla y el tope final los hace `actividad`. Los mensajes van SIN su texto (está en
# «conversaciones»); las alertas son las que llevan el id de la cuenta en su clave (la misma regla que los avisos
# abiertos de la ficha, lote 831).
_SQL_EVENTOS = {
    "comida": ("SELECT c.consumed_at AS at, c.meal_name, c.meal_type, c.calories, c.source "
               "FROM public.consumed_meals c "
               f"WHERE c.user_id = %s AND c.consumed_at >= {_VENTANA} ORDER BY c.consumed_at DESC LIMIT %s"),
    "plan": ("SELECT x.created_at AS at, x.name, x.calories, x.plan_data ->> 'generation_status' AS estado, "
             "CASE WHEN jsonb_typeof(x.plan_data -> 'days') = 'array' THEN jsonb_array_length(x.plan_data -> 'days') "
             "ELSE 0 END AS dias FROM public.meal_plans x "
             f"WHERE x.user_id = %s AND x.created_at >= {_VENTANA} ORDER BY x.created_at DESC LIMIT %s"),
    "mensaje": ("SELECT m.created_at AS at, CASE WHEN jsonb_typeof(m.attachments) = 'array' "
                "THEN jsonb_array_length(m.attachments) ELSE 0 END AS fotos FROM public.agent_messages m "
                f"WHERE m.role = 'user' AND m.session_id::text IN ({_HILOS}) AND m.created_at >= {_VENTANA} "
                "ORDER BY m.created_at DESC LIMIT %s"),
    "bloque_fallido": ("SELECT COALESCE(q.dead_lettered_at, q.updated_at) AS at, q.week_number, q.attempts, "
                       "q.dead_letter_reason FROM public.plan_chunk_queue q "
                       "WHERE q.user_id = %s AND q.status = 'failed' "
                       f"AND COALESCE(q.dead_lettered_at, q.updated_at) >= {_VENTANA} "
                       "ORDER BY COALESCE(q.dead_lettered_at, q.updated_at) DESC LIMIT %s"),
    "ia": ("SELECT e.created_at AS at, e.node, e.model, e.cost_usd_micros FROM public.llm_usage_events e "
           f"WHERE e.user_id = %s AND e.created_at >= {_VENTANA} ORDER BY e.created_at DESC LIMIT %s"),
    "metrica": ("SELECT pm.created_at AS at, pm.node, pm.duration_ms, pm.metadata FROM public.pipeline_metrics pm "
                f"WHERE pm.user_id = %s AND pm.created_at >= {_VENTANA} ORDER BY pm.created_at DESC LIMIT %s"),
    "alerta": ("SELECT a.triggered_at AS at, a.alert_key, a.title, a.severity, a.resolved_at "
               "FROM public.system_alerts a "
               f"WHERE strpos(a.alert_key, %s) > 0 AND a.triggered_at >= {_VENTANA} "
               "ORDER BY a.triggered_at DESC LIMIT %s"),
    "peso": ("SELECT w.created_at AS at, w.weight, w.unit FROM public.weight_log w "
             f"WHERE w.user_id = %s AND w.created_at >= {_VENTANA} ORDER BY w.created_at DESC LIMIT %s"),
    # `water_intake_log` no tiene created_at: el momento es su última actualización (el vaso más reciente del día)
    "agua": ("SELECT COALESCE(w.updated_at, w.log_date::timestamptz) AS at, w.log_date, w.glasses "
             "FROM public.water_intake_log w "
             f"WHERE w.user_id = %s AND w.log_date >= ({_VENTANA})::date ORDER BY w.log_date DESC LIMIT %s"),
}
_SQL_GASTO = ("SELECT COALESCE(sum(e.cost_usd_micros), 0) AS micros FROM public.llm_usage_events e "
              f"WHERE e.user_id = %s AND e.created_at >= {_VENTANA}")

_ORIGEN_COMIDA = {"photo": "foto (escáner)", "manual": "a mano", "estimate": "texto libre", "plan_meal": "del plan",
                  "chat": "desde el chat", "repeat": "repetida"}
_ORIGEN_AJUSTE = {"app": "la persona", "coach": "el coach", "sistema": "el sistema"}
_TITULO_METRICA = {"scan_outcome": "Registro de un plato escaneado", "plan_meal_deviation": "Desvío declarado del plan",
                   "vision_scan_resultado": "Resultado del escáner", "wizard_funnel": "Formulario (embudo)"}


def _ev_comida(f):
    kcal = _num(f.get("calories"))
    return {"titulo": f"Registró «{_recorte(f.get('meal_name'), 80) or 'sin nombre'}»",
            "detalle": _unir(f"{kcal} kcal" if kcal is not None else None, _recorte(f.get("meal_type"), 32),
                             _ORIGEN_COMIDA.get(origen_de_comida(f.get("source")) or ""))}


def _ev_plan(f):
    nombre, kcal = _recorte(f.get("name"), 80), _num(f.get("calories"))
    return {"titulo": f"Plan creado «{nombre}»" if nombre else "Plan creado",
            "detalle": _unir(_plural(_entero(f.get("dias")), "día"), _recorte(f.get("estado"), 64),
                             f"{kcal} kcal al día" if kcal is not None else None)}


def _ev_mensaje(f):
    fotos = _entero(f.get("fotos"))
    return {"titulo": "Mensaje al coach", "detalle": f"con {_plural(fotos, 'foto')}" if fotos else ""}


def _ev_bloque(f):
    semana = _num(f.get("week_number"))
    return {"titulo": "Bloque de plan fallido",
            "detalle": _unir(f"semana {semana}" if semana is not None else None,
                             _plural(_entero(f.get("attempts")), "intento"),
                             _recorte(f.get("dead_letter_reason"), 200))}


def _ev_ia(f):
    nodo = _recorte(f.get("node"), 64) or ""
    micros = _entero(f.get("cost_usd_micros"))
    # un solo uso cuesta fracciones de centavo: con 4 decimales se ve (el total del periodo va con 2)
    return {"titulo": _ETIQUETA_FUNCION.get(nodo) or (f"Uso de IA: {nodo}" if nodo else "Uso de IA"),
            "detalle": _unir(nodo, _recorte(f.get("model"), 64), f"US${micros / 1_000_000:.4f}")}


def _ev_metrica(f):
    nodo = _recorte(f.get("node"), 64) or ""
    ms = _entero(f.get("duration_ms"))
    return {"titulo": _TITULO_METRICA.get(nodo) or nodo or "Métrica del flujo",
            "detalle": _unir(f"{ms} ms" if ms > 0 else None, _texto_de(_json(f.get("metadata"))))}


def _ev_alerta(f):
    tipo = _recorte(str(f.get("alert_key") or "").split(":", 1)[0], 64)
    return {"titulo": _recorte(f.get("title"), 120) or tipo or "Aviso del sistema",
            "detalle": _unir(_recorte(f.get("severity"), 32), "resuelta" if f.get("resolved_at") else "abierta", tipo)}


def _ev_peso(f):
    partes = (_num(f.get("weight")), _recorte(f.get("unit"), 16))
    return {"titulo": "Peso registrado", "detalle": " ".join(str(x) for x in partes if x not in (None, ""))}


def _ev_agua(f):
    fecha = _iso(f.get("log_date"))
    vasos = _plural(_entero(f.get("glasses")), "vaso")
    return {"titulo": "Agua del día", "detalle": vasos + (f" el {fecha}" if fecha else "")}


def _ev_ajuste(f):
    antes, despues = _texto_de(f.get("antes"), 80) or "—", _texto_de(f.get("despues"), 80) or "—"
    return {"titulo": f"Cambió «{f.get('etiqueta') or f.get('clave') or 'un ajuste'}»",
            "detalle": f"{antes} → {despues} · lo cambió {_ORIGEN_AJUSTE.get(f.get('origen'), 'la persona')}"}


_EVENTOS = {"comida": _ev_comida, "plan": _ev_plan, "mensaje": _ev_mensaje, "bloque_fallido": _ev_bloque,
            "ia": _ev_ia, "metrica": _ev_metrica, "alerta": _ev_alerta, "peso": _ev_peso, "agua": _ev_agua,
            "ajuste": _ev_ajuste}


def _filas_de(tipo: str, uid: str, dias: int) -> list:
    if tipo == "ajuste":                       # el historial de ajustes (lote 837) ya trae su «at»
        return ajustes_cuenta.historial(uid, dias) or []
    sql = _SQL_EVENTOS[tipo]
    params = (uid,) * (sql.count("%s") - 2) + (dias, MAX_EVENTOS)
    return execute_sql_query(sql, params, fetch_all=True) or []


def actividad(user_id, dias: int = 7, tipos=None) -> dict:
    """Contrato 9 · `actividad`: `{dias, gasto_ia_usd, eventos: [{at, tipo, titulo, detalle}]}`. Junta los tipos
    pedidos (todos por defecto), del más nuevo al más viejo, 500 como mucho. El gasto de IA es el del periodo entero,
    se filtre o no por tipo. Lanza `ErrorDetalle` con `dias` fuera de 1..30 o un tipo desconocido, y cualquier error de
    la base tal cual (el router: 503)."""
    n = _dias_de_actividad(dias)
    elegidos = tipos_de_actividad(tipos)
    uid = _uuid(user_id)
    if not uid:
        return {"dias": n, "gasto_ia_usd": 0.0, "eventos": []}
    eventos = []
    for tipo in elegidos:
        orden = TIPOS_DE_EVENTO.index(tipo)
        for f in _filas_de(tipo, uid, n):
            instante = _instante(f.get("at"))
            if instante is not None:           # sin momento no hay sitio en la línea de tiempo
                eventos.append((instante, orden, {"at": _iso(f.get("at")), "tipo": tipo, **_EVENTOS[tipo](f)}))
    eventos.sort(key=lambda e: (e[0], -e[1]), reverse=True)
    gasto = execute_sql_query(_SQL_GASTO, (uid, n), fetch_one=True) or {}
    return {"dias": n, "gasto_ia_usd": round(_entero(gasto.get("micros")) / 1_000_000, 2),
            "eventos": [e for _, _, e in eventos[:MAX_EVENTOS]]}
