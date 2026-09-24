# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-133 · 2026-09-20] Los recordatorios de comida, como DATO: a qué hora toca cada uno y qué dicen.

El encargo del dueño: «revisa a profundidad el sistema de notificaciones: quiero que le avise al usuario si no ha
desayunado, almorzado, merendado… déjalo 100 % listo para producción, y quítale ese beta».

Medido en producción antes de tocar nada (solo lectura, 20-sep): el motor del servidor FUNCIONA —38 avisos en 5 días, a
las 10:30 / 14:30 / 17:30 / 21:00— y casi nadie los ve en su pantalla: `push_subscriptions` tiene UNA fila. La app nativa
de iOS es un WKWebView: no hay Service Worker ni `PushManager`, así que ahí el interruptor «Alertas Inteligentes» no podía
funcionar (tocarlo daba «Tu navegador no soporta notificaciones Push»), y el dueño —que usa esa app— solo veía sus avisos
al abrir el chat.

Este módulo sirve a los dos canales con UNA cuenta:
  · el cron (`proactive_agent.run_proactive_checks`) usa `proactive_agent.hora_de_aviso`, la misma que aquí;
  · la app nativa pide `GET /api/notifications/meal-reminders` y programa esos avisos EN EL TELÉFONO (notificaciones
    locales): salen aunque la app esté cerrada, sin APNs ni servidor, y se cancelan al registrar la comida.

El minuto: [P1-PLAN-LOTE-220] la notificación local suena a la hora EXACTA del aviso (la que la persona eligió en
Configuración o, si no eligió, la normal de esa comida menos 15 min) y el cron, que corre cada 15 min, escribe el mensaje
del chat en su último tick ANTES de esa hora: al tocar la notificación, el mensaje del coach ya está en el chat. (Hasta
el 208 el cron corría a y media y escribía en el tick de la HORA del aviso: con un aviso a las 2:15 el chat llegaba a
las 2:30, después del teléfono.)

Los textos son fijos y cortos (los lee una pantalla de bloqueo), en los 5 idiomas de la app. El mensaje largo y
personalizado sigue siendo el del chat.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# [P1-PLAN-LOTE-150 · 2026-09-21] El minuto SALE de la hora calculada; esta constante es solo el respaldo de cuando
# no se pudo calcular. Con la espera de 1,5 h daba igual clavarlo en :35 (nadie nota 5 minutos en un aviso que llega
# una hora tarde), pero con 15 minutos de antelación sí importa: una cena habitual a las 19:30 avisada a las «19:35»
# llegaba DESPUÉS de la cena. Ahora da 19:15.
MINUTO_DEL_AVISO_DE_RESPALDO = 35
COMIDAS = ("Desayuno", "Almuerzo", "Merienda", "Cena")

_TITULO = {
    "es-DO": "Bioboros", "en-US": "Bioboros", "pt-BR": "Bioboros", "fr-FR": "Bioboros", "it-IT": "Bioboros",
}
_CUERPO = {
    "Desayuno": {
        "es-DO": "Es tu hora de desayunar. Cuando comas, cuéntamelo y lo anoto.",
        "en-US": "Time for breakfast. When you eat, tell me and I'll log it.",
        "pt-BR": "É sua hora do café da manhã. Quando comer, me conta e eu anoto.",
        "fr-FR": "C'est l'heure de votre petit-déjeuner. Quand vous mangez, dites-le-moi et je le note.",
        "it-IT": "È la tua ora di colazione. Quando mangi, dimmelo e lo segno.",
    },
    "Almuerzo": {
        "es-DO": "Es tu hora de almorzar. Cuando comas, cuéntamelo y lo anoto.",
        "en-US": "Time for lunch. When you eat, tell me and I'll log it.",
        "pt-BR": "É sua hora do almoço. Quando comer, me conta e eu anoto.",
        "fr-FR": "C'est l'heure de votre déjeuner. Quand vous mangez, dites-le-moi et je le note.",
        "it-IT": "È la tua ora di pranzo. Quando mangi, dimmelo e lo segno.",
    },
    "Merienda": {
        "es-DO": "Es tu hora de merendar. Si comes algo, cuéntamelo y lo anoto.",
        "en-US": "Time for a snack. If you eat something, tell me and I'll log it.",
        "pt-BR": "É sua hora do lanche. Se comer algo, me conta e eu anoto.",
        "fr-FR": "C'est l'heure de votre goûter. Si vous mangez quelque chose, dites-le-moi et je le note.",
        "it-IT": "È la tua ora di merenda. Se mangi qualcosa, dimmelo e lo segno.",
    },
    "Cena": {
        "es-DO": "Es tu hora de cenar. Cuando comas, cuéntamelo y cierro tu día.",
        "en-US": "Time for dinner. When you eat, tell me and I'll close out your day.",
        "pt-BR": "É sua hora do jantar. Quando comer, me conta e eu fecho o seu dia.",
        "fr-FR": "C'est l'heure de votre dîner. Quand vous mangez, dites-le-moi et je clôture votre journée.",
        "it-IT": "È la tua ora di cena. Quando mangi, dimmelo e chiudo la tua giornata.",
    },
    # El aviso de las 23:00 del cron, para el suscriptor sin chat reciente (no se programa en el teléfono).
    "Resumen del día": {
        "es-DO": "Hoy no registraste ninguna comida. ¿Se te pasó anotar? Cuéntame qué comiste.",
        "en-US": "You didn't log any meals today. Did it slip your mind? Tell me what you ate.",
        "pt-BR": "Hoje você não registrou nenhuma refeição. Esqueceu de anotar? Me conta o que comeu.",
        "fr-FR": "Vous n'avez enregistré aucun repas aujourd'hui. Un oubli\u202f? Dites-moi ce que vous avez mangé.",
        "it-IT": "Oggi non hai registrato nessun pasto. Te ne sei dimenticato? Dimmi cosa hai mangiato.",
    },
}


def texto_del_aviso(meal: str, locale: Optional[str]) -> tuple:
    """`(título, cuerpo)` del recordatorio de `meal` en el idioma del usuario; español si no se conoce."""
    loc = locale if isinstance(locale, str) and locale in _TITULO else "es-DO"
    cuerpos = _CUERPO.get(meal) or _CUERPO["Resumen del día"]
    return _TITULO[loc], cuerpos.get(loc) or cuerpos["es-DO"]


def etiqueta_del_aviso(meal: str) -> str:
    """La etiqueta (`tag`) de la notificación: una por comida, para que la nueva sustituya a la vieja."""
    return f"comida-{str(meal).lower().split()[0]}"


def _hora_y_minuto(user_id: str, meal: str, health: Optional[dict]) -> tuple:
    """`(hora, minuto)` locales del aviso de `meal`: la cuenta de `proactive_agent.hora_del_aviso`, la misma del cron."""
    import proactive_agent as pa
    def_hour = pa.HORAS_POR_DEFECTO_DE_COMIDA[meal]
    try:
        hora_aviso = pa.hora_del_aviso(user_id, meal, def_hour, health)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-133] hora del aviso de {meal} no calculada ({e!r}); se usa la de por defecto")
        hora_aviso = (math.floor(def_hour) + MINUTO_DEL_AVISO_DE_RESPALDO / 60.0) % 24
    # [P1-PLAN-LOTE-150] Hora y minuto salen LOS DOS de la hora calculada. Redondear a minutos de una vez evita que un
    # 59,7 acabe en «:60». [P1-PLAN-LOTE-220] con la misma función que el cron: el chat y el teléfono, el mismo minuto.
    return divmod(pa.minuto_del_dia(hora_aviso), 60)


def horario_de_avisos(user_id: str, locale: Optional[str] = None, consumed_today=None,
                      health: Optional[dict] = None) -> list:
    """Un recordatorio por comida, en orden del día: `{meal, hour, minute, title, body, tag, logged_today}`.

    `hour`/`minute` son hora LOCAL del usuario. Un aviso que caería dentro de las horas de silencio (quien cena a las
    23:30 → la 1:00) no se programa: el cron tampoco lo manda (`MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR`).
    [P1-PLAN-LOTE-220] Tampoco el de una comida que la persona apagó en Configuración, y la hora es la que eligió."""
    import proactive_agent as pa
    silencio = pa._hora_de_silencio()
    out = []
    for meal in COMIDAS:
        if not pa.comida_con_aviso(health, meal):
            continue
        hora, minuto = _hora_y_minuto(user_id, meal, health)
        if hora < silencio:
            continue
        titulo, cuerpo = texto_del_aviso(meal, locale)
        out.append({
            "meal": meal.lower(), "hour": hora, "minute": minuto, "title": titulo, "body": cuerpo,
            "tag": etiqueta_del_aviso(meal),
            "logged_today": bool(pa._comida_ya_registrada(consumed_today, meal)),
        })
    return out


def comidas_para_configuracion(user_id: str, health: Optional[dict] = None) -> list:
    """[P1-PLAN-LOTE-220] Las CUATRO comidas como las pinta Configuración: `{meal, active, hour, minute, chosen,
    default_hour, default_minute}`. Aquí sí salen las apagadas: la pantalla necesita su interruptor para volver a
    encenderlas. La hora es la efectiva (la que suena); `default_*` es la normal, para «volver a la de siempre»."""
    import proactive_agent as pa
    out = []
    for meal in COMIDAS:
        hora, minuto = _hora_y_minuto(user_id, meal, health)
        def_hora, def_minuto = divmod(pa.minuto_del_dia(pa.HORAS_POR_DEFECTO_DE_COMIDA[meal] - pa._antelacion_del_aviso_h()), 60)
        out.append({
            "meal": meal.lower(), "active": bool(pa.comida_con_aviso(health, meal)),
            "hour": hora, "minute": minuto, "chosen": pa.hora_elegida(health, meal) is not None,
            "default_hour": def_hora, "default_minute": def_minuto,
        })
    return out
