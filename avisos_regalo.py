"""[P1-PLAN-LOTE-773 · 2026-09-28] El aviso de un regalo de la cuenta (spec 2026-09-28-admin-cuentas-regalos-design §5).

Push best-effort con el texto YA en el idioma de la persona: `push_i18n.translate_push_text` traduce por texto español
exacto y no admite cifras ni fechas, así que la frase se arma aquí para cada idioma (y allí pasa tal cual). El aviso
DENTRO de la app lo pinta el cliente con `regalos_recientes` de `GET /api/user/credits`.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from db import execute_sql_query

logger = logging.getLogger(__name__)

_ZONA_RD = ZoneInfo("America/Santo_Domingo")
_IDIOMAS = ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT")
# Los mismos nombres que la app (`frontend/src/config/plans.js` + catálogos): Ultra se llama «Max».
_NOMBRE_PLAN = {
    "es-DO": {"basic": "Básico", "plus": "Plus", "ultra": "Max"},
    "en-US": {"basic": "Basic", "plus": "Plus", "ultra": "Max"},
    "pt-BR": {"basic": "Básico", "plus": "Plus", "ultra": "Max"},
    "fr-FR": {"basic": "Basic", "plus": "Plus", "ultra": "Max"},
    "it-IT": {"basic": "Base", "plus": "Plus", "ultra": "Max"},
}
_TITULO = {"es-DO": "Tienes un regalo 🎁", "en-US": "You have a gift 🎁", "pt-BR": "Você ganhou um presente 🎁",
           "fr-FR": "Vous avez un cadeau 🎁", "it-IT": "Hai un regalo 🎁"}
# (una unidad, varias)
_CREDITOS = {
    "es-DO": ("Te regalamos {n} crédito para crear planes, válido hasta el {fecha}.",
              "Te regalamos {n} créditos para crear planes, válidos hasta el {fecha}."),
    "en-US": ("We gave you {n} credit to create plans, valid until {fecha}.",
              "We gave you {n} credits to create plans, valid until {fecha}."),
    "pt-BR": ("Você ganhou {n} crédito para criar planos, válido até {fecha}.",
              "Você ganhou {n} créditos para criar planos, válidos até {fecha}."),
    "fr-FR": ("Nous vous offrons {n} crédit pour créer des plans, valable jusqu’au {fecha}.",
              "Nous vous offrons {n} crédits pour créer des plans, valables jusqu’au {fecha}."),
    "it-IT": ("Ti regaliamo {n} credito per creare piani, valido fino al {fecha}.",
              "Ti regaliamo {n} crediti per creare piani, validi fino al {fecha}."),
}
_COACH = {
    "es-DO": ("Te regalamos {n} mensaje más con tu coach, válido hasta el {fecha}.",
              "Te regalamos {n} mensajes más con tu coach, válidos hasta el {fecha}."),
    "en-US": ("We gave you {n} more message with your coach, valid until {fecha}.",
              "We gave you {n} more messages with your coach, valid until {fecha}."),
    "pt-BR": ("Você ganhou mais {n} mensagem com seu coach, válida até {fecha}.",
              "Você ganhou mais {n} mensagens com seu coach, válidas até {fecha}."),
    "fr-FR": ("Nous vous offrons {n} message de plus avec votre coach, valable jusqu’au {fecha}.",
              "Nous vous offrons {n} messages de plus avec votre coach, valables jusqu’au {fecha}."),
    "it-IT": ("Ti regaliamo {n} messaggio in più con il tuo coach, valido fino al {fecha}.",
              "Ti regaliamo altri {n} messaggi con il tuo coach, validi fino al {fecha}."),
}
_PLAN_HASTA = {"es-DO": "Tienes {nombre} de cortesía hasta el {fecha}.", "en-US": "You have {nombre} on us until {fecha}.",
               "pt-BR": "Você tem o {nombre} de cortesia até {fecha}.",
               "fr-FR": "Vous bénéficiez de {nombre} offert jusqu’au {fecha}.",
               "it-IT": "Hai {nombre} in omaggio fino al {fecha}."}
_PLAN_SIN_FIN = {"es-DO": "Ahora tienes {nombre} de cortesía.", "en-US": "You now have {nombre} on us.",
                 "pt-BR": "Agora você tem o {nombre} de cortesia.", "fr-FR": "Vous bénéficiez désormais de {nombre} offert.",
                 "it-IT": "Ora hai {nombre} in omaggio."}


def _idioma(v) -> str:
    return v if v in _IDIOMAS else "es-DO"


def fecha_corta(fin: datetime, idioma: str) -> str:
    """El último día en que vale el regalo (el fin es EXCLUSIVO), en la hora de RD: dd/mm, o mm/dd en inglés."""
    ultimo = (fin - timedelta(microseconds=1)).astimezone(_ZONA_RD)
    return f"{ultimo.month:02d}/{ultimo.day:02d}" if idioma == "en-US" else f"{ultimo.day:02d}/{ultimo.month:02d}"


def texto_del_aviso(regalo: dict, idioma) -> tuple:
    idioma = _idioma(idioma)
    fin = regalo.get("ends_at")
    if regalo.get("kind") == "plan":
        nombre = _NOMBRE_PLAN[idioma].get(regalo.get("plan"), str(regalo.get("plan") or ""))
        cuerpo = (_PLAN_HASTA[idioma].format(nombre=nombre, fecha=fecha_corta(fin, idioma)) if fin
                  else _PLAN_SIN_FIN[idioma].format(nombre=nombre))
    else:
        n = int(regalo.get("amount") or 0)
        uno, varios = (_COACH if regalo.get("kind") == "creditos_coach" else _CREDITOS)[idioma]
        cuerpo = (uno if n == 1 else varios).format(n=n, fecha=fecha_corta(fin, idioma))
    return _TITULO[idioma], cuerpo


def avisar(user_id: str, regalo: dict) -> bool:
    """Envía la push. Nunca lanza: un aviso que no sale no deshace el regalo."""
    try:
        fila = execute_sql_query("SELECT locale FROM public.user_profiles WHERE id = %s", (user_id,),
                                 fetch_one=True) or {}
        titulo, cuerpo = texto_del_aviso(regalo, fila.get("locale"))
        from utils_push import send_push_notification
        return bool(send_push_notification(user_id, titulo, cuerpo, url="/dashboard", tag=f"regalo-{regalo.get('id')}"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-773] aviso de regalo no enviado a {user_id}: {e!r}")
        return False


def avisar_en_segundo_plano(user_id: str, regalo: dict) -> None:
    threading.Thread(target=avisar, args=(user_id, regalo), daemon=True, name="aviso-regalo").start()
