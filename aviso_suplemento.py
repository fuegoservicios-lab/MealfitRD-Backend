# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1106 · 2026-11-18] «Te quedan ~5 días de tu suplemento»: el aviso antes de que se acabe el pote.

El dueño eligió «Suplemento por acabarse: aviso ~5 días antes de que se acabe un pote de la Alacena». Hasta ahora solo
se avisaba AL REGISTRAR una toma con ≤ 3 porciones (`tools.log_consumed_meal`); esto avisa con días de margen aunque
no esté hablando con el coach.

- El pote: `user_inventory` con `kind = 'supplement'` y porciones conocidas (`quantity > 0`, que es lo que descuenta
  cada toma registrada con `suplemento=<nombre>`).
- Su ritmo: las tomas registradas (`consumed_meals`, `meal_type = 'suplemento'`, el nombre del pote en `meal_name`)
  de los últimos 14 días; con menos de 3 no hay ritmo y no se avisa. Una toma = una porción (lo que descuenta).
- Avisa si quedan 5 días o menos, a las 10 locales, UNA vez por pote hasta que se rellene (`app_kv_store`
  `aviso_suplemento:{id}` guarda con cuántas porciones se avisó; si sube, se rellenó y vuelve a contar).
- Push + mensaje del coach en el chat del día. Sin modelo. Logs sin nombres ni cifras.

[P1-PLAN-LOTE-1109] El dueño decidió que las vitaminas SÍ se anotan (0 kcal, `meal_type='suplemento'`) y descuentan;
cada toma guarda sus porciones (`anotar_toma`), así 2 de 4 cápsulas cuentan 0,5 y no 1.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DIAS_DE_MARGEN = 5
VENTANA_DIAS = 14
TOMAS_MIN = 3
HORA = 10

_T = {
    "es-DO": ("Tu {p} se está acabando", "Te quedan unos {d} días de {p}.",
              "Te quedan unos {d} días de {p} (unas {q} porciones a tu ritmo de las últimas 2 semanas). Para no "
              "quedarte sin él, inclúyelo en tu próxima compra; cuando abras el pote nuevo, dime cuántas porciones "
              "trae y lo actualizo."),
    "en-US": ("Your {p} is running low", "About {d} days of {p} left.",
              "You have about {d} days of {p} left (around {q} servings at your pace of the last 2 weeks). So you "
              "don't run out, add it to your next shopping trip; when you open the new one, tell me how many "
              "servings it has and I'll update it."),
    "pt-BR": ("Seu {p} está acabando", "Restam uns {d} dias de {p}.",
              "Restam uns {d} dias de {p} (umas {q} porções no seu ritmo das últimas 2 semanas). Para não ficar sem, "
              "inclua na próxima compra; quando abrir o pote novo, me diga quantas porções ele tem e eu atualizo."),
    "fr-FR": ("Votre {p} se termine", "Il vous reste environ {d} jours de {p}.",
              "Il vous reste environ {d} jours de {p} (environ {q} portions à votre rythme des 2 dernières "
              "semaines). Pour ne pas en manquer, ajoutez-le à vos prochaines courses ; quand vous ouvrirez le "
              "nouveau pot, dites-moi combien de portions il contient et je mets à jour."),
    "it-IT": ("Il tuo {p} sta finendo", "Ti restano circa {d} giorni di {p}.",
              "Ti restano circa {d} giorni di {p} (circa {q} porzioni al tuo ritmo delle ultime 2 settimane). Per "
              "non restarne senza, aggiungilo alla prossima spesa; quando apri il barattolo nuovo, dimmi quante "
              "porzioni contiene e lo aggiorno."),
}


def dias_restantes(porciones: float, tomas_en_ventana: int, porciones_tomadas: Optional[float] = None) -> Optional[float]:
    """Puro: días que quedan a su ritmo, o None si no hay ritmo (menos de 3 tomas en 14 días). Con
    `porciones_tomadas` (lo que de verdad se descontó: 2 de 4 cápsulas = 0,5) el ritmo es ese; sin él, 1 por toma."""
    if tomas_en_ventana < TOMAS_MIN or not porciones or porciones <= 0:
        return None
    gastado = float(porciones_tomadas) if porciones_tomadas and porciones_tomadas > 0 else float(tomas_en_ventana)
    return float(porciones) / (gastado / float(VENTANA_DIAS))


# ── [P1-PLAN-LOTE-1109] las porciones de cada toma (la fila del diario no las guarda) ────────────────────────────────
_TOPE_TOMAS = 60


def tomas_en_ventana(registro: Optional[list], ahora: datetime) -> tuple[int, float]:
    """Puro: (cuántas tomas, cuántas porciones) del registro `[[iso, porciones], …]` en los últimos 14 días."""
    desde = ahora - timedelta(days=VENTANA_DIAS)
    n, suma = 0, 0.0
    for it in registro or []:
        try:
            t = datetime.fromisoformat(str(it[0]))
            t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
            if t >= desde:
                n, suma = n + 1, suma + max(0.0, float(it[1]))
        except (TypeError, ValueError, IndexError):
            continue
    return n, suma


def anotar_toma(pote_id, porciones, ahora: Optional[datetime] = None) -> None:
    """Guarda la toma con sus porciones (`app_kv_store` `suplemento_tomas:{id}`, las 60 últimas). Nunca rompe."""
    try:
        from db_core import execute_sql_query, execute_sql_write
        clave = f"suplemento_tomas:{pote_id}"
        fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (clave,), fetch_one=True) or {}
        reg = fila.get("value")
        reg = json.loads(reg) if isinstance(reg, str) else (reg or [])
        reg = (list(reg) if isinstance(reg, list) else []) + [
            [(ahora or datetime.now(timezone.utc)).isoformat(), float(porciones or 1)]]
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (clave, json.dumps(reg[-_TOPE_TOMAS:])),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1109] toma sin anotar: {type(exc).__name__}")


def toca(dias: Optional[float], porciones: float, estado: Optional[dict]) -> bool:
    """Puro: avisar si quedan ≤ 5 días y no se avisó ya con este pote (si se rellenó, vuelve a contar)."""
    if dias is None or dias > DIAS_DE_MARGEN:
        return False
    avisado = (estado or {}).get("avisado_con")
    try:
        return avisado is None or float(porciones) > float(avisado)
    except (TypeError, ValueError):
        return True


def textos(locale: Optional[str], nombre: str, dias: float, porciones: float) -> tuple[str, str, str]:
    t = _T.get(str(locale or ""), _T["es-DO"])
    d, q = max(1, int(round(dias))), max(1, int(round(porciones)))
    return t[0].format(p=nombre), t[1].format(d=d, p=nombre), t[2].format(d=d, p=nombre, q=q)


def enviar_avisos_de_suplemento(ahora_utc: Optional[datetime] = None) -> int:
    """Cron horario. Devuelve cuántos avisos salieron."""
    try:
        from knobs import _env_bool
        if not _env_bool("MEALFIT_AVISO_SUPLEMENTO", True):
            return 0
        from db_core import execute_sql_query, execute_sql_write
        from diary_targets import resolve_user_effective_off_min
        from utils_push import send_push_notification
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1106] aviso de suplemento sin dependencias: {type(exc).__name__}")
        return 0
    ahora = ahora_utc or datetime.now(timezone.utc)
    enviados = 0
    try:
        potes = execute_sql_query(
            # [SUPLEMENTOS-OK: lee los potes para avisar antes de que se acaben]
            "SELECT i.id::text AS id, i.user_id::text AS user_id, i.ingredient_name AS nombre, "
            "i.quantity::float8 AS porciones, p.locale FROM user_inventory i "
            "JOIN user_profiles p ON p.id = i.user_id WHERE i.kind = 'supplement' AND i.quantity > 0",
            fetch_all=True,
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LOTE-1106] aviso de suplemento: no se leyeron los potes ({type(exc).__name__})")
        return 0
    for p in potes:
        uid, nombre = p.get("user_id"), str(p.get("nombre") or "").strip()
        if not uid or not nombre:
            continue
        try:
            off = resolve_user_effective_off_min(uid, None)
            if (ahora - timedelta(minutes=off)).hour != HORA:
                continue
            fila = execute_sql_query(
                "SELECT COUNT(*) AS n FROM consumed_meals WHERE user_id = %s AND meal_type = 'suplemento' "
                "AND meal_name ILIKE %s AND consumed_at >= %s",
                (uid, "%" + nombre[:40] + "%", ahora - timedelta(days=VENTANA_DIAS)), fetch_one=True,
            ) or {}
            porciones = float(p.get("porciones") or 0)
            reg = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s",
                                    (f"suplemento_tomas:{p['id']}",), fetch_one=True) or {}
            reg = reg.get("value")
            n_reg, gastado = tomas_en_ventana(json.loads(reg) if isinstance(reg, str) else reg, ahora)
            if n_reg >= TOMAS_MIN:      # [P1-PLAN-LOTE-1109] con su registro, las porciones reales de cada toma
                dias = dias_restantes(porciones, n_reg, gastado)
            else:
                dias = dias_restantes(porciones, int(fila.get("n") or 0))
            clave = f"aviso_suplemento:{p['id']}"
            ya = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (clave,), fetch_one=True) or {}
            estado = ya.get("value")
            estado = json.loads(estado) if isinstance(estado, str) else (estado or {})
            if dias is not None and dias > DIAS_DE_MARGEN and estado.get("avisado_con") is not None:
                execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (clave,))   # se rellenó: vuelve a contar
                continue
            if not toca(dias, porciones, estado):
                continue
            titulo, cuerpo, mensaje = textos(p.get("locale"), nombre, dias, porciones)
            try:
                from aviso_peso import _sesion_del_dia
                sesion = _sesion_del_dia(uid, ahora, off)
                if sesion:
                    from db_chat import save_message
                    save_message(sesion, "model", mensaje)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LOTE-1106] aviso de suplemento sin chat: {type(exc).__name__}")
            if send_push_notification(uid, titulo, cuerpo, url="/dashboard/agent", tag=f"suplemento-{p['id'][:8]}"):
                enviados += 1
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (clave, json.dumps({"avisado_con": porciones})),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[LOTE-1106] aviso de suplemento omitido: {type(exc).__name__}")
    if enviados:
        logger.info(f"[LOTE-1106] avisos de suplemento enviados: {enviados}")
    return enviados
