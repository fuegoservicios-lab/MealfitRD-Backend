# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1059 · plan 1035.4 · 2026-11-17] El aviso del domingo: «Empieza tu Semana 10. La 9 cerró con 2★».

- Solo a quien lo encendió en Ajustes (`health_profile.aviso_semanal == true`): APAGADO por defecto (plan §7.1, la
  recomendación). El knob `MEALFIT_PROGRESO_AVISO_DOMINGO` (on por defecto) lo apaga para todos.
- Sin modelo (D-G): plantilla con SUS cifras de `progreso_semanas.resumen`, en su idioma.
- Una vez por semana y persona (`app_kv_store` `progreso_aviso_domingo:{uid}`), el domingo a las
  `MEALFIT_PROGRESO_AVISO_HORA` (9) locales. El cron corre cada hora.

Logs: solo recuentos. Nunca lanza.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_TEXTOS = {
    "es-DO": ("Tu semana en Bioboros", "Empieza tu Semana {n}. La {p} cerró con {e}.", "Empieza tu Semana {n}. La {p} cerró con {d} de 7 días en meta."),
    "en-US": ("Your week in Bioboros", "Week {n} starts today. Week {p} closed with {e}.", "Week {n} starts today. Week {p} closed with {d} of 7 days on goal."),
    "pt-BR": ("Sua semana no Bioboros", "Começa sua Semana {n}. A {p} fechou com {e}.", "Começa sua Semana {n}. A {p} fechou com {d} de 7 dias na meta."),
    "fr-FR": ("Votre semaine sur Bioboros", "Votre semaine {n} commence. La {p} s’est terminée avec {e}.", "Votre semaine {n} commence. La {p} s’est terminée avec {d} jours sur 7 dans l’objectif."),
    "it-IT": ("La tua settimana su Bioboros", "Inizia la tua settimana {n}. La {p} si è chiusa con {e}.", "Inizia la tua settimana {n}. La {p} si è chiusa con {d} giorni su 7 in obiettivo."),
}


def texto_del_aviso(locale: Optional[str], numero: int, pasada: Optional[dict]) -> Optional[tuple[str, str]]:
    """Puro: (título, cuerpo) o None si no hay semana pasada que contar."""
    if not pasada or not pasada.get("numero"):
        return None
    titulo, con_estrellas, sin_estrellas = _TEXTOS.get(str(locale or ""), _TEXTOS["es-DO"])
    e = int(pasada.get("estrellas") or 0)
    if e > 0:
        cuerpo = con_estrellas.format(n=numero, p=pasada["numero"], e="★" * e)
    else:
        cuerpo = sin_estrellas.format(n=numero, p=pasada["numero"], d=int(pasada.get("dias_en_meta") or 0))
    return titulo, cuerpo


def toca_ahora(ahora_utc: datetime, off_reloj_min: int, hora: int) -> bool:
    """¿Es domingo a la `hora` en el reloj LOCAL de la persona? `off_reloj_min` = JS getTimezoneOffset (RD = 240)."""
    local = ahora_utc - timedelta(minutes=off_reloj_min)
    return local.weekday() == 6 and local.hour == hora


def _activo_global() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PROGRESO_AVISO_DOMINGO", True)
    except Exception:  # noqa: BLE001
        return True


def _hora() -> int:
    try:
        from knobs import _env_int
        return max(6, min(int(_env_int("MEALFIT_PROGRESO_AVISO_HORA", 9)), 21))
    except Exception:  # noqa: BLE001
        return 9


def enviar_avisos_del_domingo(ahora_utc: Optional[datetime] = None) -> int:
    """Cron horario. Devuelve cuántos avisos salieron."""
    if not _activo_global():
        return 0
    try:
        from db_core import execute_sql_query, execute_sql_write
        from diary_targets import resolve_user_effective_off_min
        from db_facts import _diary_rollover_hour
        import progreso_semanas as ps
        from utils_push import send_push_notification
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1059] aviso del domingo sin dependencias: {type(exc).__name__}")
        return 0
    ahora = ahora_utc or datetime.now(timezone.utc)
    hora = _hora()
    enviados = 0
    try:
        filas = execute_sql_query(
            "SELECT id::text AS id, locale FROM user_profiles WHERE health_profile->>'aviso_semanal' = 'true'",
            fetch_all=True,
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1059] aviso del domingo: no se leyeron las cuentas ({type(exc).__name__})")
        return 0
    for f in filas:
        uid = f.get("id")
        try:
            off = resolve_user_effective_off_min(uid, None) - int(_diary_rollover_hour()) * 60
            if not toca_ahora(ahora, off, hora):
                continue
            res = ps.resumen(uid, off, con_eventos=False)
            numero = int(((res or {}).get("semana") or {}).get("numero") or 0)
            par = texto_del_aviso(f.get("locale"), numero, (res or {}).get("semana_pasada"))
            if not par:
                continue
            clave = f"progreso_aviso_domingo:{uid}"
            ya = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (clave,), fetch_one=True) or {}
            v = ya.get("value")
            v = json.loads(v) if isinstance(v, str) else (v or {})
            if int((v or {}).get("semana") or 0) == numero:
                continue
            if send_push_notification(uid, par[0], par[1], url="/dashboard/progress", tag="progreso-semana"):
                enviados += 1
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (clave, json.dumps({"semana": numero})),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[LOTE-1059] aviso del domingo omitido para una cuenta: {type(exc).__name__}")
    if enviados:
        logger.info(f"[LOTE-1059] avisos del domingo enviados: {enviados}")
    return enviados
