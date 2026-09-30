# backend/ajustes_de_la_app.py
"""[P1-PLAN-LOTE-900 · 2026-09-29] El coach cambia los ajustes de la app y abre sus pantallas.

El dueño, en modo voz: «Activa la hidratación». El coach no tenía herramienta para eso y ANOTÓ UN VASO DE AGUA
(`log_water_glass`, 17:39 UTC): contestó «Marqué un vaso de agua, ya llevas 1 de 9». Su petición: «quiero que si le
pido cualquier cosa como esa que lo haga, quiero que tenga 100 % acceso a todo de la app».

Aquí vive la regla de cada ajuste, reutilizando las MISMAS funciones que los interruptores de Configuración (una regla
que vive en dos puertas se cumple en una):

  · En el servidor (se escriben aquí): hidratación, Nevera, memoria a largo plazo, generador de planes, recordatorios
    de comida y de agua.
  · En el dispositivo (los aplica la pantalla): tema e idioma — viven en el teléfono (localStorage + catálogo).
  · Pantallas: la app navega a la que el usuario pida.

Lo que la pantalla tiene que saber viaja en un marcador `<<AJUSTE_APP_JSON: {...}>>` al final del resultado de la
tool; `agent.execute_tools` lo quita antes de que lo lea el modelo (patrón de `<<PANTRY_DEPLETED_JSON>>`) y lo acumula
en el estado del turno; el evento `done` del stream lo entrega como `ajustes_de_app` y `AgentPage` lo aplica.

Fuera A PROPÓSITO: el consentimiento de entrenamiento de IA y las políticas (un consentimiento lo da la persona en su
pantalla, no un reconocimiento de voz), borrar la cuenta y cancelar la suscripción (irreversibles o de dinero): el
coach abre la pantalla y la persona decide. El permiso de notificaciones lo concede el sistema operativo, no la app.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

MARCADOR = "<<AJUSTE_APP_JSON:"
_RE_MARCADOR = re.compile(r"<<AJUSTE_APP_JSON:\s*(\{.*?\})\s*>>", re.DOTALL)


def _norm(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").strip().lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", t)


# ── catálogo ────────────────────────────────────────────────────────────────────────────────────────────────────
# ajuste canónico → sinónimos que el modelo (o el reconocimiento de voz) puede mandar.
AJUSTES = {
    "hidratacion": ("hidratacion", "agua", "tarjeta_de_agua", "water_tracker", "vasos"),
    "nevera": ("nevera", "despensa", "inventario", "pantry"),
    "memoria": ("memoria", "memoria_a_largo_plazo", "long_term_memory", "recordar"),
    "generador_de_planes": ("generador_de_planes", "generador", "planes", "plan", "modo_plan", "plan_mode",
                            "generacion_de_planes", "modo_automatico"),
    "recordatorios_de_comida": ("recordatorios_de_comida", "avisos_de_comida", "avisos_comida"),
    "recordatorios_de_agua": ("recordatorios_de_agua", "avisos_de_agua", "avisos_agua"),
    "tema": ("tema", "apariencia", "modo_oscuro", "modo_claro", "theme"),
    "idioma": ("idioma", "lenguaje", "language", "locale"),
}
_SINONIMO = {s: canon for canon, sins in AJUSTES.items() for s in sins}

TEMAS = {"claro": "light", "modo_claro": "light", "light": "light", "blanco": "light", "oscuro": "dark",
         "modo_oscuro": "dark", "dark": "dark", "negro": "dark",
         "sistema": "system", "automatico": "system", "system": "system", "auto": "system"}
IDIOMAS = {"es_do": "es-DO", "espanol": "es-DO", "es": "es-DO", "en_us": "en-US", "ingles": "en-US", "en": "en-US",
           "english": "en-US", "pt_br": "pt-BR", "portugues": "pt-BR", "pt": "pt-BR", "fr_fr": "fr-FR",
           "frances": "fr-FR", "fr": "fr-FR", "it_it": "it-IT", "italiano": "it-IT", "it": "it-IT"}
_SI = {"true", "1", "si", "on", "encender", "encendido", "encendida", "activar", "activo", "activa", "activado",
       "activada", "mostrar", "ver", "prender", "reanudar"}
_NO = {"false", "0", "no", "off", "apagar", "apagado", "apagada", "desactivar", "desactivado", "desactivada",
       "ocultar", "quitar", "pausar", "pausa", "pausado"}

# pantalla canónica → sinónimos. La RUTA la decide la app (`utils/ajustesDelCoach.js`): «Progreso» es `/dashboard`
# en modo seguimiento y `/dashboard/progress` con plan. Configuración abre la sección por su `#id` (Settings.jsx).
PANTALLAS = {
    "inicio": ("inicio", "dashboard", "panel", "hoy", "plan", "menu"),
    "progreso": ("progreso", "contador", "macros", "micros"),
    "agente": ("agente", "chat", "coach"),
    "nevera": ("nevera", "despensa", "inventario", "alacena"),
    "recetas": ("recetas",),
    "historial": ("historial", "dias_anteriores", "historia"),
    "configuracion": ("configuracion", "ajustes", "settings", "preferencias"),
}
_PANTALLA_SIN = {s: canon for canon, sins in PANTALLAS.items() for s in sins}
SECCIONES = {"general": "profile", "cuenta": "profile", "notificaciones": "profile", "apariencia": "profile",
             "idioma": "profile", "alergias": "health", "dieta": "health", "capacidades": "preferences",
             "privacidad": "privacy", "super_personalizacion": "superpers", "perfil_clinico": "clinical",
             "objetivo": "plan", "calorias": "plan", "suscripcion": "subscription", "pagos": "subscription",
             "plan_y_objetivo": "plan"}


def _a_bool(valor) -> Optional[bool]:
    if isinstance(valor, bool):
        return valor
    v = _norm(valor)
    if v in _SI:
        return True
    if v in _NO:
        return False
    return None


def _con_marcador(texto: str, cambio: dict) -> str:
    return f"{texto} {MARCADOR} {json.dumps(cambio, ensure_ascii=False)}>>"


def extraer_marcador(resultado) -> tuple:
    """(texto sin el marcador, dict del cambio o None). Lo usa `agent.execute_tools`: el modelo no ve el JSON."""
    if not isinstance(resultado, str) or MARCADOR not in resultado:
        return resultado, None
    m = _RE_MARCADOR.search(resultado)
    cambio = None
    if m:
        try:
            cambio = json.loads(m.group(1))
        except ValueError:
            cambio = None
    return _RE_MARCADOR.sub("", resultado).strip(), (cambio if isinstance(cambio, dict) else None)


def _es_invitado(user_id) -> bool:
    return not user_id or str(user_id) == "guest"


# ── cambiar un ajuste ───────────────────────────────────────────────────────────────────────────────────────────

def cambiar_ajuste(user_id: str, ajuste: str, valor) -> str:
    canon = _SINONIMO.get(_norm(ajuste))
    if not canon:
        return ("No reconozco ese ajuste. Los que puedo cambiar: hidratación, Nevera, memoria, generador de planes, "
                "recordatorios de comida o de agua, tema e idioma. Para otra cosa, ofrece abrirle la pantalla con "
                "`abrir_pantalla_de_la_app`.")
    if canon == "tema":
        return _cambiar_tema(valor)
    if canon == "idioma":
        return _cambiar_idioma(valor)
    if _es_invitado(user_id):
        return "Este ajuste se guarda en la cuenta y el usuario aún no tiene una: dile que cree su cuenta para usarlo."
    activar = _a_bool(valor)
    if activar is None:
        return f"Para «{canon}» el valor es encender o apagar (true/false); recibí {valor!r}. Pregúntale qué prefiere."
    try:
        return {
            "hidratacion": _hidratacion,
            "nevera": _nevera,
            "memoria": _memoria,
            "generador_de_planes": _generador,
            "recordatorios_de_comida": lambda uid, a: _recordatorio(uid, "avisos_comida", a),
            "recordatorios_de_agua": lambda uid, a: _recordatorio(uid, "avisos_agua", a),
        }[canon](user_id, activar)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-900] cambiar «{canon}»={activar} falló para {user_id}: {type(e).__name__}: {e}")
        return "ERROR: no se pudo guardar el cambio ahora. No digas que se hizo: dile que lo intente de nuevo en un momento."


def _hidratacion(user_id: str, activar: bool) -> str:
    from db_profiles import update_water_tracker_enabled
    if not update_water_tracker_enabled(user_id, activar):
        raise RuntimeError("sin fila")
    if activar:
        import hydration_reminders
        hydration_reminders.al_encender(user_id)
    texto = ("Hecho: la tarjeta de Hidratación ya se ve en su panel (no anotaste ningún vaso: encender la tarjeta no "
             "es beber agua)." if activar else "Hecho: la tarjeta de Hidratación quedó oculta; su historial de vasos se conserva.")
    return _con_marcador(texto, {"hidratacion": activar})


def encender_hidratacion_al_anotar_agua(user_id: str) -> str:
    """[P1-PLAN-LOTE-907 · 2026-09-30] Anotar agua con la tarjeta apagada la enciende. El dueño, en voz: «me bebí 3
    vasos» → «sumé tus 3 vasos» y la Hidratación seguía apagada: los vasos quedaban donde no se ven. Quien anota agua
    quiere verla. Devuelve lo que se añade al resultado de `log_water_glass` (con el marcador para la pantalla) o «»."""
    if _es_invitado(user_id):
        return ""
    try:
        from db_profiles import get_water_tracker_enabled, update_water_tracker_enabled
        if get_water_tracker_enabled(user_id) or not update_water_tracker_enabled(user_id, True):
            return ""
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-907] no se pudo encender la Hidratación de {user_id}: {type(e).__name__}: {e}")
        return ""
    try:
        import hydration_reminders
        hydration_reminders.al_encender(user_id)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-907] recordatorios al encender la Hidratación: {type(e).__name__}: {e}")
    return " " + _con_marcador("La tarjeta de Hidratación estaba apagada y la encendí para que vea sus vasos: díselo.",
                               {"hidratacion": True})


def _nevera(user_id: str, activar: bool) -> str:
    import nevera_opcional
    if not nevera_opcional.interruptor_disponible():
        return "La opción de apagar o encender la Nevera no está disponible ahora mismo. Díselo sin rodeos."
    if not nevera_opcional.fijar_nevera(user_id, activar):
        raise RuntimeError("sin fila")
    if not activar:
        try:
            from cron_tasks import try_unfreeze_plan_for_user
            try_unfreeze_plan_for_user(user_id)
        except Exception as _e:
            logger.debug(f"[P1-PLAN-LOTE-900] descongelar tras apagar la Nevera: no-op ({_e})")
    activa = bool(nevera_opcional.estado_nevera(user_id).get("activa"))
    if activa != activar:
        return _con_marcador("La Nevera no cambió: en su modo actual no se puede apagar. Díselo así.", {"nevera": activa})
    texto = ("Hecho: la Nevera está activa; aparece su pestaña." if activa
             else "Hecho: la Nevera quedó oculta; su inventario se conserva.")
    return _con_marcador(texto, {"nevera": activa})


def _memoria(user_id: str, activar: bool) -> str:
    from db_profiles import update_long_term_memory_enabled
    if not update_long_term_memory_enabled(user_id, activar):
        raise RuntimeError("sin fila")
    texto = ("Hecho: la memoria a largo plazo está activa; aprenderás de sus conversaciones." if activar
             else "Hecho: la memoria a largo plazo quedó en pausa; lo ya guardado se conserva y no aprendes nada nuevo.")
    return _con_marcador(texto, {"memoria": activar})


def _tiene_plan(user_id: str) -> bool:
    from db import get_latest_usable_meal_plan_with_id
    try:
        return bool(get_latest_usable_meal_plan_with_id(user_id))
    except Exception:
        return False


def _generador(user_id: str, activar: bool) -> str:
    from plan_mode import get_plan_mode, pause_plan_generation, resume_plan_generation
    if not activar:
        out = pause_plan_generation(user_id)
        if out.get("skipped"):
            return "Ahora mismo no se puede cambiar la generación de planes. Díselo y que lo intente más tarde."
        return _con_marcador(
            "Hecho: la generación de planes quedó apagada; la app queda como contador (macros y diario). Su plan, si "
            "tenía, queda en el Historial.",
            {"generador_de_planes": "tracking", "tenia_plan": _tiene_plan(user_id)},
        )
    if get_plan_mode(user_id).get("plan_mode") == "plan":
        return _con_marcador("La generación de planes ya estaba encendida.", {"generador_de_planes": "plan", "tenia_plan": True})
    if not _tiene_plan(user_id):
        # Sin plan no hay nada que reanudar: el camino es el formulario (misma puerta que Configuración). Abrirlo no
        # gasta; el crédito se gasta al final, en «Finalizar y Generar».
        return _con_marcador(
            "No tiene plan que reanudar: le abrí el formulario para generarlo (solo pregunta lo que falta; el crédito se "
            "usa al final, al pulsar «Finalizar y Generar»). Díselo en una frase.",
            {"pantalla": "formulario"},
        )
    out = resume_plan_generation(user_id)
    if out.get("skipped"):
        return "Ahora mismo no se puede cambiar la generación de planes. Díselo y que lo intente más tarde."
    texto = ("Hecho: planes reanudados, pero su plan venció la ventana: que genere uno nuevo cuando quiera."
             if out.get("plan_expired") else "Hecho: planes reanudados; la generación sigue donde quedó.")
    return _con_marcador(texto, {"generador_de_planes": "plan", "tenia_plan": True})


def _recordatorio(user_id: str, clave: str, activar: bool) -> str:
    from db import update_user_health_profile_atomic

    def _fusionar(hp):
        hp = dict(hp or {})
        hp[clave] = activar
        return hp

    if update_user_health_profile_atomic(user_id, _fusionar) is None:
        raise RuntimeError("sin fila")
    que = "de comida" if clave == "avisos_comida" else "de agua"
    texto = (f"Hecho: recordatorios {que} {'encendidos' if activar else 'apagados'}. "
             + ("Solo llegan si tiene las notificaciones del teléfono permitidas para la app." if activar else ""))
    return _con_marcador(texto.strip(), {clave: activar})


def _cambiar_tema(valor) -> str:
    tema = TEMAS.get(_norm(valor))
    if not tema:
        return "El tema puede ser claro, oscuro o el del sistema. Pregúntale cuál quiere."
    nombre = {"light": "claro", "dark": "oscuro", "system": "el del sistema"}[tema]
    return _con_marcador(f"Hecho: la app pasa al tema {nombre}.", {"tema": tema})


def _cambiar_idioma(valor) -> str:
    idioma = IDIOMAS.get(_norm(valor))
    if not idioma:
        return "Los idiomas de la app son español, inglés, portugués, francés e italiano. Pregúntale cuál quiere."
    return _con_marcador(
        "Hecho: la app cambia de idioma ahora; desde el siguiente mensaje le hablas en ese idioma.",
        {"idioma": idioma},
    )


# ── abrir una pantalla ──────────────────────────────────────────────────────────────────────────────────────────

def abrir_pantalla(pantalla: str, seccion: Optional[str] = None) -> str:
    canon = _PANTALLA_SIN.get(_norm(pantalla))
    if not canon:
        return ("No reconozco esa pantalla. Existen: inicio (progreso del día), agente, nevera, recetas, historial y "
                "configuración (secciones: general, alergias y dieta, capacidades, privacidad, súper personalización, "
                "perfil clínico, objetivo, suscripción).")
    cambio = {"pantalla": canon}
    if canon == "configuracion" and seccion:
        sec = SECCIONES.get(_norm(seccion))
        if sec:
            cambio["seccion"] = sec
    return _con_marcador(f"Hecho: la app muestra {canon}.", cambio)
