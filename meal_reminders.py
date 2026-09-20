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

El minuto: el cron corre a y media, así que el aviso del chat de una comida cuyo `nudge_hour` es 10,8 sale a las 10:30.
La notificación local se programa a `floor(nudge_hour)`:35 — cinco minutos después del tick, para que al tocarla el
mensaje del coach ya esté en el chat.

Los textos son fijos y cortos (los lee una pantalla de bloqueo), en los 5 idiomas de la app. El mensaje largo y
personalizado sigue siendo el del chat.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

MINUTO_DEL_AVISO_LOCAL = 35
COMIDAS = ("Desayuno", "Almuerzo", "Merienda", "Cena")

_TITULO = {
    "es-DO": "Bioboros", "en-US": "Bioboros", "pt-BR": "Bioboros", "fr-FR": "Bioboros", "it-IT": "Bioboros",
}
_CUERPO = {
    "Desayuno": {
        "es-DO": "¿Ya desayunaste? Cuéntame qué comiste y lo anoto en tu diario.",
        "en-US": "Had breakfast yet? Tell me what you ate and I'll log it.",
        "pt-BR": "Já tomou café da manhã? Me conta o que comeu e eu anoto no seu diário.",
        "fr-FR": "Tu as pris ton petit-déjeuner ? Dis-moi ce que tu as mangé et je le note.",
        "it-IT": "Hai già fatto colazione? Dimmi cosa hai mangiato e lo segno nel diario.",
    },
    "Almuerzo": {
        "es-DO": "¿Ya almorzaste? Cuéntame qué comiste y lo anoto en tu diario.",
        "en-US": "Had lunch yet? Tell me what you ate and I'll log it.",
        "pt-BR": "Já almoçou? Me conta o que comeu e eu anoto no seu diário.",
        "fr-FR": "Tu as déjeuné ? Dis-moi ce que tu as mangé et je le note.",
        "it-IT": "Hai già pranzato? Dimmi cosa hai mangiato e lo segno nel diario.",
    },
    "Merienda": {
        "es-DO": "¿Ya merendaste? Si comiste algo, cuéntamelo y lo anoto.",
        "en-US": "Had a snack? If you ate something, tell me and I'll log it.",
        "pt-BR": "Já lanchou? Se comeu algo, me conta e eu anoto.",
        "fr-FR": "Tu as pris un goûter ? Si tu as mangé quelque chose, dis-le-moi et je le note.",
        "it-IT": "Hai fatto merenda? Se hai mangiato qualcosa, dimmelo e lo segno.",
    },
    "Cena": {
        "es-DO": "¿Ya cenaste? Cuéntame qué comiste y cierro tu día.",
        "en-US": "Had dinner yet? Tell me what you ate and I'll close out your day.",
        "pt-BR": "Já jantou? Me conta o que comeu e eu fecho o seu dia.",
        "fr-FR": "Tu as dîné ? Dis-moi ce que tu as mangé et je clôture ta journée.",
        "it-IT": "Hai già cenato? Dimmi cosa hai mangiato e chiudo la tua giornata.",
    },
    # El aviso de las 23:00 del cron, para el suscriptor sin chat reciente (no se programa en el teléfono).
    "Resumen del día": {
        "es-DO": "Hoy no registraste ninguna comida. ¿Se te pasó anotar? Cuéntame qué comiste.",
        "en-US": "You didn't log any meals today. Did it slip your mind? Tell me what you ate.",
        "pt-BR": "Hoje você não registrou nenhuma refeição. Esqueceu de anotar? Me conta o que comeu.",
        "fr-FR": "Tu n'as enregistré aucun repas aujourd'hui. Un oubli ? Dis-moi ce que tu as mangé.",
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


def horario_de_avisos(user_id: str, locale: Optional[str] = None, consumed_today=None) -> list:
    """Un recordatorio por comida, en orden del día: `{meal, hour, minute, title, body, tag, logged_today}`.

    `hour`/`minute` son hora LOCAL del usuario. Un aviso que caería dentro de las horas de silencio (quien cena a las
    23:30 → la 1:00) no se programa: el cron tampoco lo manda (`MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR`)."""
    import proactive_agent as pa
    silencio = pa._hora_de_silencio()
    out = []
    for meal in COMIDAS:
        def_hour = pa.HORAS_POR_DEFECTO_DE_COMIDA[meal]
        try:
            nudge_hour, _rate, _total = pa.hora_de_aviso(user_id, meal, def_hour)
        except Exception as e:
            logger.warning(f"[P1-PLAN-LOTE-133] hora del aviso de {meal} no calculada ({e!r}); se usa la de por defecto")
            nudge_hour = (def_hour + 1.5) % 24
        hora = int(math.floor(float(nudge_hour))) % 24
        if hora < silencio:
            continue
        titulo, cuerpo = texto_del_aviso(meal, locale)
        out.append({
            "meal": meal.lower(), "hour": hora, "minute": MINUTO_DEL_AVISO_LOCAL, "title": titulo, "body": cuerpo,
            "tag": etiqueta_del_aviso(meal),
            "logged_today": bool(pa._comida_ya_registrada(consumed_today, meal)),
        })
    return out
