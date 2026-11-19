# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1107 · 2026-11-18] Peso estancado: el coach propone un ajuste pequeño en vez de que lo descubra solo.

El dueño eligió «Estancamiento de peso: 4 semanas sin cambio con buen cumplimiento → el coach propone un ajuste
pequeño». Solo PROPONE (un mensaje en el chat y un push): no cambia ninguna meta por su cuenta.

Toca cuando TODO se cumple:
  · su objetivo es bajar grasa (`lose_fat`) o ganar músculo (`gain_muscle`);
  · `weight_history` cubre 28 días o más, y entre el primer pesaje de esa ventana y el último hay menos del 0,5 % de
    su peso en la dirección de su objetivo (bajar: no bajó; ganar: no subió);
  · cumplió: en las 3 últimas semanas cerradas, 4 o más días en meta de media (si no cumple, el problema no es la meta);
  · no se le propuso en los últimos 28 días (`app_kv_store` `estancamiento_peso:{uid}`).

Corre cada hora y solo actúa los lunes a las 9 locales (tras el resumen del domingo). Sin modelo. Logs sin cifras.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

VENTANA_DIAS = 28
UMBRAL = 0.005
DIAS_EN_META_MIN = 4.0
ENTRE_PROPUESTAS = 28

_T = {
    "es-DO": ("Tu peso se ha quedado quieto",
              "Llevas 4 semanas casi igual aunque estás cumpliendo. ¿Ajustamos un poco?",
              {"lose_fat": "Tu peso lleva unas 4 semanas casi igual ({a} → {b} {u}), y eso que cumpliste tu meta {d} "
                           "días por semana de media: lo estás haciendo bien, el cuerpo se adaptó. Te propongo un "
                           "ajuste pequeño: bajar unas 150 kcal al día tu meta, o sumar 20 minutos de caminata. "
                           "¿Cuál prefieres? Dime y lo ajusto.",
               "gain_muscle": "Tu peso lleva unas 4 semanas casi igual ({a} → {b} {u}), y eso que cumpliste tu meta {d} "
                              "días por semana de media. Para seguir ganando, te propongo subir unas 150 kcal al día tu "
                              "meta (una merienda más). ¿Lo ajusto?"}),
    "en-US": ("Your weight has stalled",
              "About 4 weeks nearly unchanged while you're on track. Shall we adjust a little?",
              {"lose_fat": "Your weight has been almost the same for about 4 weeks ({a} → {b} {u}), even though you hit "
                           "your goal {d} days a week on average: you're doing it right, your body adapted. I suggest a "
                           "small change: lower your daily goal by about 150 kcal, or add a 20-minute walk. Which do "
                           "you prefer? Tell me and I'll adjust it.",
               "gain_muscle": "Your weight has been almost the same for about 4 weeks ({a} → {b} {u}), even though you "
                              "hit your goal {d} days a week on average. To keep gaining, I suggest raising your daily "
                              "goal by about 150 kcal (one more snack). Shall I adjust it?"}),
    "pt-BR": ("Seu peso estacionou",
              "Umas 4 semanas quase igual mesmo cumprindo. Vamos ajustar um pouco?",
              {"lose_fat": "Seu peso está quase igual há umas 4 semanas ({a} → {b} {u}), mesmo você cumprindo a meta "
                           "{d} dias por semana em média: você está indo bem, o corpo se adaptou. Proponho um ajuste "
                           "pequeno: baixar umas 150 kcal por dia na meta, ou somar 20 minutos de caminhada. Qual você "
                           "prefere? Me diga e eu ajusto.",
               "gain_muscle": "Seu peso está quase igual há umas 4 semanas ({a} → {b} {u}), mesmo você cumprindo a meta "
                              "{d} dias por semana em média. Para continuar ganhando, proponho subir umas 150 kcal por "
                              "dia na meta (um lanche a mais). Posso ajustar?"}),
    "fr-FR": ("Votre poids stagne",
              "Environ 4 semaines presque sans changement alors que vous suivez le plan. On ajuste un peu ?",
              {"lose_fat": "Votre poids est presque le même depuis environ 4 semaines ({a} → {b} {u}), alors que vous "
                           "avez atteint votre objectif {d} jours par semaine en moyenne : vous faites bien, le corps "
                           "s’est adapté. Je vous propose un petit ajustement : baisser votre objectif d’environ 150 "
                           "kcal par jour, ou ajouter 20 minutes de marche. Que préférez-vous ? Dites-le-moi et "
                           "j’ajuste.",
               "gain_muscle": "Votre poids est presque le même depuis environ 4 semaines ({a} → {b} {u}), alors que "
                              "vous avez atteint votre objectif {d} jours par semaine en moyenne. Pour continuer à "
                              "prendre, je vous propose d’augmenter votre objectif d’environ 150 kcal par jour (une "
                              "collation de plus). J’ajuste ?"}),
    "it-IT": ("Il tuo peso si è fermato",
              "Circa 4 settimane quasi uguale anche se stai seguendo il piano. Regoliamo un po’?",
              {"lose_fat": "Il tuo peso è quasi uguale da circa 4 settimane ({a} → {b} {u}), anche se hai rispettato "
                           "l’obiettivo {d} giorni a settimana in media: stai andando bene, il corpo si è adattato. Ti "
                           "propongo un piccolo aggiustamento: abbassare l’obiettivo di circa 150 kcal al giorno, o "
                           "aggiungere 20 minuti di camminata. Cosa preferisci? Dimmelo e lo regolo.",
               "gain_muscle": "Il tuo peso è quasi uguale da circa 4 settimane ({a} → {b} {u}), anche se hai rispettato "
                              "l’obiettivo {d} giorni a settimana in media. Per continuare a crescere, ti propongo di "
                              "alzare l’obiettivo di circa 150 kcal al giorno (uno spuntino in più). Lo regolo?"}),
}


def _pesajes(hp: dict) -> list[tuple[date, float, str]]:
    out = []
    for e in (hp or {}).get("weight_history") or []:
        if not isinstance(e, dict):
            continue
        try:
            out.append((date.fromisoformat(str(e.get("date"))[:10]), float(e.get("weight")), str(e.get("unit") or "lb")))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _kg(p: float, u: str) -> float:
    return p * 0.45359237 if u.lower().startswith("lb") else p


def estancado(hp: dict, hoy: date) -> Optional[dict]:
    """Puro: {a, b, u} (primer y último pesaje de la ventana) si el peso no se movió hacia su objetivo, o None."""
    objetivo = str((hp or {}).get("mainGoal") or "")
    if objetivo not in ("lose_fat", "gain_muscle"):
        return None
    ps = _pesajes(hp)
    if len(ps) < 2:
        return None
    ult = ps[-1]
    if (hoy - ult[0]).days > 10:          # sin un pesaje reciente no se sabe cómo está
        return None
    ventana = [p for p in ps if (ult[0] - p[0]).days >= VENTANA_DIAS]
    if not ventana:
        return None
    ini = ventana[-1]                      # el pesaje más reciente con 28+ días de antigüedad
    a, b = _kg(ini[1], ini[2]), _kg(ult[1], ult[2])
    cambio = (b - a) / a if a else 0.0
    movio = cambio <= -UMBRAL if objetivo == "lose_fat" else cambio >= UMBRAL
    if movio:
        return None
    lb = ult[2].lower().startswith("lb")
    conv = (lambda kg: kg / 0.45359237) if lb else (lambda kg: kg)
    return {"a": f"{conv(a):.1f}", "b": f"{conv(b):.1f}", "u": "lb" if lb else "kg", "objetivo": objetivo}


def cumplio(semanas: list) -> Optional[float]:
    """Media de días en meta de las 3 últimas semanas cerradas si es buena (≥ 4), o None."""
    cerradas = [s for s in semanas or [] if s.get("cerrada")][-3:]
    if len(cerradas) < 3:
        return None
    media = sum(int(s.get("dias_en_meta") or 0) for s in cerradas) / 3.0
    return media if media >= DIAS_EN_META_MIN else None


def textos(locale: Optional[str], est: dict, media: float) -> tuple[str, str, str]:
    t = _T.get(str(locale or ""), _T["es-DO"])
    d = f"{media:.0f}" if abs(media - round(media)) < 0.05 else f"{media:.1f}"
    return t[0], t[1], t[2][est["objetivo"]].format(a=est["a"], b=est["b"], u=est["u"], d=d)


def toca_ahora(ahora_utc: datetime, off_reloj_min: int, hora: int = 9) -> bool:
    local = ahora_utc - timedelta(minutes=off_reloj_min)
    return local.weekday() == 0 and local.hour == hora


def revisar_estancamientos(ahora_utc: Optional[datetime] = None) -> int:
    """Cron horario. Devuelve cuántas propuestas salieron."""
    try:
        from knobs import _env_bool
        if not _env_bool("MEALFIT_ESTANCAMIENTO_PESO", True):
            return 0
        from db_core import execute_sql_query, execute_sql_write
        from diary_targets import resolve_user_effective_off_min
        from db_facts import _diary_rollover_hour
        from utils_push import send_push_notification
        import progreso_semanas as ps
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1107] estancamiento sin dependencias: {type(exc).__name__}")
        return 0
    ahora = ahora_utc or datetime.now(timezone.utc)
    enviados = 0
    try:
        filas = execute_sql_query(
            "SELECT id::text AS id, locale, health_profile AS hp FROM user_profiles "
            "WHERE health_profile->>'mainGoal' IN ('lose_fat', 'gain_muscle') "
            "AND jsonb_array_length(COALESCE(health_profile->'weight_history', '[]'::jsonb)) >= 2",
            fetch_all=True,
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1107] estancamiento: no se leyeron las cuentas ({type(exc).__name__})")
        return 0
    for f in filas:
        uid = f.get("id")
        try:
            off = resolve_user_effective_off_min(uid, None)
            if not toca_ahora(ahora, off):
                continue
            hp = f.get("hp") if isinstance(f.get("hp"), dict) else json.loads(f.get("hp") or "{}")
            hoy = (ahora - timedelta(minutes=off)).date()
            est = estancado(hp, hoy)
            if not est:
                continue
            clave = f"estancamiento_peso:{uid}"
            ya = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (clave,), fetch_one=True) or {}
            v = ya.get("value")
            v = json.loads(v) if isinstance(v, str) else (v or {})
            if v.get("propuesto"):
                try:
                    if (hoy - date.fromisoformat(str(v["propuesto"])[:10])).days < ENTRE_PROPUESTAS:
                        continue
                except ValueError:
                    pass
            res = ps.resumen(uid, off - int(_diary_rollover_hour()) * 60, con_semanas=True, con_eventos=False) or {}
            media = cumplio(res.get("semanas") or [])
            if media is None:
                continue
            titulo, cuerpo, mensaje = textos(f.get("locale"), est, media)
            try:
                from aviso_peso import _sesion_del_dia
                sesion = _sesion_del_dia(uid, ahora, off)
                if sesion:
                    from db_chat import save_message
                    save_message(sesion, "model", mensaje)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LOTE-1107] estancamiento sin chat: {type(exc).__name__}")
            if send_push_notification(uid, titulo, cuerpo, url="/dashboard/agent", tag="estancamiento-peso"):
                enviados += 1
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (clave, json.dumps({"propuesto": hoy.isoformat()})),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[LOTE-1107] estancamiento omitido para una cuenta: {type(exc).__name__}")
    if enviados:
        logger.info(f"[LOTE-1107] propuestas por estancamiento: {enviados}")
    return enviados
