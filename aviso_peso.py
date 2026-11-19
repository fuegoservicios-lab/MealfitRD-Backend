# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1105 · 2026-11-18] El coach pide el peso cada cierto tiempo.

El dueño: «quiero que también me pida cambiar el peso cada cierto tiempo, ejemplo cada 2 semanas». Cada 14 días por
defecto (el peso de un día a otro es agua y sal; dos semanas es lo que tarda en verse un cambio real y lo que sirve para
ajustar metas), en Ajustes: cada semana, cada 2 semanas, cada mes o apagado (`health_profile.aviso_peso_dias` = 7 | 14 |
30 | 0; ausente = 14).

- Por la mañana (`MEALFIT_AVISO_PESO_HORA`, 8 locales): en ayunas es cuando el peso se compara con el anterior.
- Toca si su último pesaje (`weight_history`) tiene la frecuencia o más; si se pesó por su cuenta, no se le pregunta.
- UN recordatorio 2 días después si no respondió; luego nada hasta el ciclo siguiente.
- Va al teléfono (push) y al chat del día como mensaje del coach: así la respuesta («182 libras») le llega al coach con
  la pregunta delante y la anota con `update_form_field`, que ahora también la suma al historial.
- Sin modelo: plantilla en su idioma. Estado por persona en `app_kv_store` `aviso_peso:{uid}`.

Logs: solo recuentos, nunca un peso. Nunca lanza.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

FRECUENCIAS = (0, 7, 14, 30)
POR_DEFECTO = 14
DIAS_HASTA_RECORDAR = 2

_TEXTOS = {
    "es-DO": ("¿Cuánto pesas hoy?",
              "Hace {d} días de tu último peso. Pésate en ayunas y dime cuánto marcó.",
              "Cuando puedas, pésate en ayunas y dime cuánto marcó la báscula.",
              "Hace {d} días que no anotamos tu peso. Cuando puedas, pésate en ayunas (después de ir al baño y antes de "
              "comer) y dime cuánto marcó: lo anoto y te digo cómo va tu tendencia.",
              "¿Me dices tu peso de hoy? Pésate en ayunas y dime cuánto marcó: lo anoto y te digo cómo va tu tendencia."),
    "en-US": ("How much do you weigh today?",
              "It's been {d} days since your last weigh-in. Weigh yourself before breakfast and tell me the number.",
              "When you can, weigh yourself before breakfast and tell me the number.",
              "It's been {d} days since we logged your weight. When you can, weigh yourself before breakfast (after the "
              "bathroom, before eating) and tell me the number: I'll log it and tell you how your trend looks.",
              "Can you tell me your weight today? Weigh yourself before breakfast and tell me the number: I'll log it "
              "and tell you how your trend looks."),
    "pt-BR": ("Quanto você pesa hoje?",
              "Faz {d} dias do seu último peso. Pese-se em jejum e me diga quanto deu.",
              "Quando puder, pese-se em jejum e me diga quanto deu.",
              "Faz {d} dias que não anotamos seu peso. Quando puder, pese-se em jejum (depois de ir ao banheiro e antes "
              "de comer) e me diga quanto deu: eu anoto e te digo como vai sua tendência.",
              "Pode me dizer seu peso de hoje? Pese-se em jejum e me diga quanto deu: eu anoto e te digo como vai sua "
              "tendência."),
    "fr-FR": ("Combien pesez-vous aujourd’hui ?",
              "Votre dernière pesée date de {d} jours. Pesez-vous à jeun et dites-moi le chiffre.",
              "Quand vous pourrez, pesez-vous à jeun et dites-moi le chiffre.",
              "Cela fait {d} jours que nous n’avons pas noté votre poids. Quand vous pourrez, pesez-vous à jeun (après "
              "être passé aux toilettes, avant de manger) et dites-moi le chiffre : je le note et je vous dis "
              "comment évolue votre tendance.",
              "Pouvez-vous me dire votre poids d’aujourd’hui ? Pesez-vous à jeun et dites-moi le chiffre : je "
              "le note et je vous dis comment évolue votre tendance."),
    "it-IT": ("Quanto pesi oggi?",
              "Sono passati {d} giorni dall’ultima pesata. Pesati a digiuno e dimmi quanto segna.",
              "Quando puoi, pesati a digiuno e dimmi quanto segna.",
              "Sono {d} giorni che non annotiamo il tuo peso. Quando puoi, pesati a digiuno (dopo il bagno e prima di "
              "mangiare) e dimmi quanto segna: lo annoto e ti dico come va la tua tendenza.",
              "Mi dici il tuo peso di oggi? Pesati a digiuno e dimmi quanto segna: lo annoto e ti dico come va la tua "
              "tendenza."),
}


def frecuencia(hp: Optional[dict]) -> int:
    """Días entre pesajes que eligió (0 = apagado); ausente o inválido = 14."""
    v = (hp or {}).get("aviso_peso_dias")
    try:
        v = int(v)
    except (TypeError, ValueError):
        return POR_DEFECTO
    return v if v in FRECUENCIAS else POR_DEFECTO


def ultimo_pesaje(hp: Optional[dict]) -> Optional[date]:
    fechas = []
    for e in (hp or {}).get("weight_history") or []:
        if isinstance(e, dict) and e.get("date"):
            try:
                fechas.append(date.fromisoformat(str(e["date"])[:10]))
            except ValueError:
                continue
    return max(fechas) if fechas else None


def que_toca(hoy: date, hp: Optional[dict], estado: Optional[dict], alta: Optional[date] = None) -> Optional[str]:
    """Puro: "pregunta", "recordatorio" o None. `estado` = {"preguntado": "YYYY-MM-DD", "recordado": bool}. Sin
    historial, la referencia es el ALTA de la cuenta (su peso del formulario): el 18-nov ninguna de las 30 cuentas
    tenía `weight_history`, y sin esto el primer día se le habría preguntado a todas a la vez."""
    n = frecuencia(hp)
    if n <= 0 or not (hp or {}).get("weight"):
        return None
    ultimo = ultimo_pesaje(hp) or alta
    if ultimo is not None and (hoy - ultimo).days < n:
        return None
    estado = estado or {}
    try:
        preguntado = date.fromisoformat(str(estado.get("preguntado"))[:10]) if estado.get("preguntado") else None
    except ValueError:
        preguntado = None
    if preguntado is None or (ultimo is not None and ultimo >= preguntado) or (hoy - preguntado).days >= n:
        return "pregunta"
    if not estado.get("recordado") and (hoy - preguntado).days >= DIAS_HASTA_RECORDAR:
        return "recordatorio"
    return None


def textos(locale: Optional[str], dias: Optional[int]) -> tuple[str, str, str]:
    """(título del push, cuerpo del push, mensaje del coach en el chat)."""
    t = _TEXTOS.get(str(locale or ""), _TEXTOS["es-DO"])
    if dias:
        return t[0], t[1].format(d=dias), t[3].format(d=dias)
    return t[0], t[2], t[4]


def _hora() -> int:
    try:
        from knobs import _env_int
        return max(5, min(int(_env_int("MEALFIT_AVISO_PESO_HORA", 8)), 12))
    except Exception:  # noqa: BLE001
        return 8


def _activo_global() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_AVISO_PESO", True)
    except Exception:  # noqa: BLE001
        return True


def _sesion_del_dia(uid: str, ahora: datetime, off: int) -> Optional[str]:
    """El chat de hoy de la persona (el mismo criterio que los avisos de comida), o None si nunca abrió el chat."""
    from db_core import execute_sql_query
    fila = execute_sql_query(
        "SELECT id::text AS id FROM agent_sessions WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
        (uid,), fetch_one=True,
    )
    if not fila or not fila.get("id"):
        return None
    from proactive_agent import _sesion_del_dia_para_aviso
    return _sesion_del_dia_para_aviso(fila["id"], uid, ahora, off)


def enviar_avisos_de_peso(ahora_utc: Optional[datetime] = None) -> int:
    """Cron horario. Devuelve cuántos avisos salieron."""
    if not _activo_global():
        return 0
    try:
        from db_core import execute_sql_query, execute_sql_write
        from diary_targets import resolve_user_effective_off_min
        from utils_push import send_push_notification
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1105] aviso de peso sin dependencias: {type(exc).__name__}")
        return 0
    ahora = ahora_utc or datetime.now(timezone.utc)
    hora = _hora()
    enviados = 0
    try:
        filas = execute_sql_query(
            "SELECT id::text AS id, locale, health_profile AS hp, created_at FROM user_profiles "
            "WHERE COALESCE(health_profile->>'aviso_peso_dias', '14') <> '0' "
            "AND health_profile ? 'weight'",
            fetch_all=True,
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1105] aviso de peso: no se leyeron las cuentas ({type(exc).__name__})")
        return 0
    for f in filas:
        uid = f.get("id")
        try:
            off = resolve_user_effective_off_min(uid, None)
            local = ahora - timedelta(minutes=off)
            if local.hour != hora:
                continue
            hp = f.get("hp") if isinstance(f.get("hp"), dict) else json.loads(f.get("hp") or "{}")
            clave = f"aviso_peso:{uid}"
            ya = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (clave,), fetch_one=True) or {}
            estado = ya.get("value")
            estado = json.loads(estado) if isinstance(estado, str) else (estado or {})
            hoy = local.date()
            alta = f.get("created_at")
            alta = (alta - timedelta(minutes=off)).date() if isinstance(alta, datetime) else None
            accion = que_toca(hoy, hp, estado, alta)
            if not accion:
                continue
            ultimo = ultimo_pesaje(hp) or alta
            titulo, cuerpo, mensaje = textos(f.get("locale"), (hoy - ultimo).days if ultimo else None)
            try:
                sesion = _sesion_del_dia(uid, ahora, off)
                if sesion:
                    from db_chat import save_message
                    save_message(sesion, "model", mensaje)
            except Exception as exc:  # noqa: BLE001 — sin chat, el push solo
                logger.debug(f"[LOTE-1105] aviso de peso sin chat: {type(exc).__name__}")
            if send_push_notification(uid, titulo, cuerpo, url="/dashboard/agent", tag="aviso-peso"):
                enviados += 1
            nuevo = ({"preguntado": hoy.isoformat(), "recordado": False} if accion == "pregunta"
                     else {**estado, "recordado": True})
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (clave, json.dumps(nuevo)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[LOTE-1105] aviso de peso omitido para una cuenta: {type(exc).__name__}")
    if enviados:
        logger.info(f"[LOTE-1105] avisos de peso enviados: {enviados}")
    return enviados


# ── lo que hace el coach con la respuesta ────────────────────────────────────────────────────────────────────────────

def anotar_en_historial(hp: dict, peso, unidad: Optional[str], hoy: Optional[date] = None) -> None:
    """Suma el pesaje a `weight_history` (el mismo formato y tope de 30 que `POST /api/diary/progress`). Antes, el peso
    que se le decía al coach cambiaba el perfil pero NO el historial: sin historial no hay tendencia ni aviso que sepa
    cuándo se pesó por última vez."""
    hoy_txt = (hoy or datetime.now(timezone.utc).date()).isoformat()
    wh = hp.get("weight_history") if isinstance(hp.get("weight_history"), list) else []
    wh = [e for e in wh if isinstance(e, dict) and e.get("date")]
    mismo = next((e for e in wh if e.get("date") == hoy_txt), None)
    if mismo:
        mismo["weight"], mismo["unit"] = peso, unidad
    else:
        wh.append({"date": hoy_txt, "weight": peso, "unit": unidad})
    hp["weight_history"] = sorted(wh, key=lambda e: e["date"])[-30:]


def _en_kg(peso, unidad) -> Optional[float]:
    try:
        p = float(peso)
    except (TypeError, ValueError):
        return None
    return p * 0.45359237 if str(unidad or "lb").lower().startswith("lb") else p


def tendencia(hp: Optional[dict]) -> str:
    """Para el coach: «−1.2 lb en 14 días (desde 183 lb)», contra el pesaje anterior de hace 7 días o más; "" si no hay
    con qué comparar. En la unidad del último pesaje."""
    wh = sorted([e for e in (hp or {}).get("weight_history") or [] if isinstance(e, dict) and e.get("date")],
                key=lambda e: e["date"])
    if len(wh) < 2:
        return ""
    ult = wh[-1]
    try:
        d_ult = date.fromisoformat(str(ult["date"])[:10])
    except ValueError:
        return ""
    previo = None
    for e in reversed(wh[:-1]):
        try:
            if (d_ult - date.fromisoformat(str(e["date"])[:10])).days >= 7:
                previo = e
                break
        except ValueError:
            continue
    if not previo:
        return ""
    a, b = _en_kg(previo.get("weight"), previo.get("unit")), _en_kg(ult.get("weight"), ult.get("unit"))
    if a is None or b is None:
        return ""
    lb = str(ult.get("unit") or "lb").lower().startswith("lb")
    delta = (b - a) / 0.45359237 if lb else (b - a)
    unidad = "lb" if lb else "kg"
    dias = (d_ult - date.fromisoformat(str(previo["date"])[:10])).days
    desde = _en_kg(previo.get("weight"), previo.get("unit"))
    desde = desde / 0.45359237 if lb else desde
    return f"{delta:+.1f} {unidad} en {dias} días (desde {desde:.1f} {unidad})".replace("+-", "−")
