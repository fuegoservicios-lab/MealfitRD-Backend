# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1145 · 2026-11-21] Tu día termina cuando duermes, no a las 4:00 fijas.

El dueño (21-nov, 4:23): «yo no he dormido pero me comí esa galleta de Koko y se reinició el contador calórico; ¿cómo
mejoramos esa área para la gente que tiene el horario desorganizado?». El corte biológico es fijo a las 4:00
(`hora_de_la_franja.CORTE_BIOLOGICO_H`, ~30 sitios del servidor y `diaBiologico.js`): a las 4:01 de una noche en vela
el contador volvía a 0 y lo comido caía en el día nuevo.

Regla (entre las 4:00 y el mediodía, `TOPE_H`): sigues en el día ANTERIOR si
  1. lo pediste («Seguir en mi viernes», `accion=seguir` en `app_kv_store`), o
  2. tu día termina más tarde («Mi día termina a las…», `health_profile.dia_termina_h`, 4-8) y aún no es esa hora, o
  3. sigues despierto: hubo actividad (registros, mensajes al coach, presencia en la app) en las 4 h antes del corte y,
     desde ella hasta ahora, ningún hueco de `HUECO_SUENO_H` (4 h) sin actividad (el sueño).
  «Empezar mi sábado» (`accion=empezar`) gana a 2 y 3.

Cómo se aplica SIN tocar los ~30 sitios del corte: lo que se registra mientras sigues en el día anterior se guarda al
FINAL de ese día (`fin − 1 min`, la hora real queda en `created_at`), así toda ventana de día que ya existe lo cuenta
donde toca. Y el «hoy» del coach y del diario lo da `dia_de_hoy` (la app lo pide a `GET /api/diary/dia-activo`).
Knob `MEALFIT_DIA_TERMINA_AL_DORMIR` (True).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PREFIJO = "dia_extendido:"
TOPE_H = 12
HUECO_SUENO_H = 4.0
ANTES_DEL_CORTE_H = 4.0
CORTE_MIN_H, CORTE_MAX_H = 4, 8
_CACHE_S = 45.0
_cache: dict = {}
_lock = threading.Lock()


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DIA_TERMINA_AL_DORMIR", True)
    except Exception:  # noqa: BLE001
        return True


def _corte_h() -> int:
    try:
        from hora_de_la_franja import CORTE_BIOLOGICO_H
        return int(CORTE_BIOLOGICO_H)
    except Exception:  # noqa: BLE001
        return 4


def corte_personal(valor: Any) -> int:
    """`health_profile.dia_termina_h` acotado a 4-8 (lo que no se entiende, el corte de siempre)."""
    try:
        h = int(float(valor))
    except (TypeError, ValueError):
        return _corte_h()
    return max(CORTE_MIN_H, min(CORTE_MAX_H, h))


def _a_utc(local: datetime, off_min: int) -> datetime:
    return (local + timedelta(minutes=int(off_min))).replace(tzinfo=timezone.utc)


def _aware(t: Any) -> Optional[datetime]:
    if t is None:
        return None
    if isinstance(t, str):
        try:
            t = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(t, datetime):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def decidir(ahora_utc: datetime, off_min: int, *, actividad: list, accion: Optional[dict] = None,
            termina_h: int = 4) -> dict:
    """La regla, pura (sin base de datos). `actividad`: instantes UTC de uso de la app (cualquier orden).
    `accion`: `{"dia": iso, "accion": "seguir"|"empezar"}` guardada por el usuario, o None."""
    ahora_utc = _aware(ahora_utc) or datetime.now(timezone.utc)
    corte = _corte_h()
    local = ahora_utc.replace(tzinfo=None) - timedelta(minutes=int(off_min))
    natural = (local - timedelta(hours=corte)).date()
    base = {"extendido": False, "dia": natural.isoformat(), "dia_natural": natural.isoformat(),
            "fin": None, "hasta": None, "motivo": None, "puede_seguir": False, "anterior": None}
    if not activo() or not (corte <= local.hour < TOPE_H):
        return base
    anterior = natural - timedelta(days=1)
    corte_local = datetime(natural.year, natural.month, natural.day, corte)
    corte_utc = _a_utc(corte_local, off_min)
    tope_utc = _a_utc(datetime(natural.year, natural.month, natural.day, TOPE_H), off_min)
    base["anterior"] = anterior.isoformat()

    def _extendido(motivo: str, hasta: datetime) -> dict:
        return {**base, "extendido": True, "dia": anterior.isoformat(), "fin": corte_utc.isoformat(),
                "hasta": min(hasta, tope_utc).isoformat(), "motivo": motivo}

    pts = sorted(t for t in (_aware(x) for x in (actividad or [])) if t is not None and t <= ahora_utc)
    de_noche = [t for t in pts if corte_utc - timedelta(hours=ANTES_DEL_CORTE_H) <= t < corte_utc]
    base["puede_seguir"] = bool(de_noche)

    acc = accion if isinstance(accion, dict) and accion.get("dia") == anterior.isoformat() else None
    if acc and acc.get("accion") == "empezar":
        return {**base, "puede_seguir": False}
    if acc and acc.get("accion") == "seguir":
        return _extendido("pedido", tope_utc)
    if termina_h > corte and local.hour < termina_h:
        return _extendido("corte_personal", _a_utc(datetime(natural.year, natural.month, natural.day, termina_h), off_min))
    if de_noche:
        cadena = [t for t in pts if t >= de_noche[0]] + [ahora_utc]
        hueco = max((b - a for a, b in zip(cadena, cadena[1:])), default=timedelta(0))
        if hueco < timedelta(hours=HUECO_SUENO_H):
            ultima = pts[-1] if pts else ahora_utc
            return _extendido("despierto", max(ultima, ahora_utc) + timedelta(hours=HUECO_SUENO_H))
    return base


def _leer(user_id: str, ahora_utc: datetime, off_min: int, q: Callable) -> tuple:
    # [P1-PLAN-LOTE-1146 · 2026-11-21] Solo cuentan ACTOS del usuario (registrar, escribirle al coach), NO la presencia
    # (`presencia:<uid>`): la marca cualquier petición con sesión, y una pestaña abierta toda la noche sondea sola
    # (el dueño: «si dejo una pestaña abierta mientras duermo, ¿qué pasa?») — no habría nunca un hueco de sueño y el
    # día seguiría extendido hasta el mediodía, con el desayuno cayendo en el día de ayer.
    desde = ahora_utc - timedelta(hours=TOPE_H + ANTES_DEL_CORTE_H + 2)
    filas = q(
        "SELECT created_at AS t FROM consumed_meals WHERE user_id = %s::uuid AND created_at >= %s "
        "UNION ALL SELECT created_at FROM agent_messages WHERE user_id = %s::uuid AND role = 'user' AND created_at >= %s",
        (user_id, desde.isoformat(), user_id, desde.isoformat()), fetch_all=True) or []
    actividad = [f.get("t") for f in filas if isinstance(f, dict)]
    fila = q("SELECT value FROM app_kv_store WHERE key = %s", (PREFIJO + user_id,), fetch_one=True) or {}
    accion = fila.get("value") if isinstance(fila, dict) else None
    termina_h = _corte_h()
    try:
        from db import get_user_profile
        hp = (get_user_profile(user_id) or {}).get("health_profile") or {}
        termina_h = corte_personal(hp.get("dia_termina_h", termina_h))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1145] corte personal sin leer: {exc!r}")
    return actividad, accion, termina_h


def estado(user_id: Optional[str], ahora_utc: Optional[datetime] = None, *, tz_offset_min: Optional[int] = None,
           execute_sql_query: Any = None, usar_cache: bool = True) -> dict:
    """La decisión para este usuario ahora. Fail-open: ante cualquier error, el día de siempre."""
    ahora_utc = _aware(ahora_utc) or datetime.now(timezone.utc)
    try:
        off = tz_offset_min
        if off is None:
            from db_facts import user_tz_offset_min
            off = user_tz_offset_min(user_id) if user_id else 240
        off = int(off)
    except Exception:  # noqa: BLE001
        off = 240
    vacio = decidir(ahora_utc, off, actividad=[])
    if not user_id or user_id == "guest" or not activo():
        return vacio
    clave = (str(user_id), off)
    if usar_cache:
        with _lock:
            hit = _cache.get(clave)
        if hit and time.monotonic() - hit[0] < _CACHE_S:
            return hit[1]
    try:
        q = execute_sql_query
        if q is None:
            from db_core import execute_sql_query as q
        actividad, accion, termina_h = _leer(str(user_id), ahora_utc, off, q)
        res = decidir(ahora_utc, off, actividad=actividad, accion=accion, termina_h=termina_h)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-1145] día extendido sin decidir (se usa el corte de siempre): {exc!r}")
        return vacio
    with _lock:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[clave] = (time.monotonic(), res)
    return res


def olvidar(user_id: Optional[str]) -> None:
    with _lock:
        for k in [k for k in _cache if k[0] == str(user_id)]:
            _cache.pop(k, None)


def fijar(user_id: str, accion: str, ahora_utc: Optional[datetime] = None, *, execute_sql_write: Any = None) -> dict:
    """«Seguir en mi viernes» (`seguir`) o «Empezar mi sábado» (`empezar`): vale para la mañana de hoy."""
    if accion not in ("seguir", "empezar"):
        raise ValueError("accion debe ser 'seguir' o 'empezar'")
    ahora_utc = _aware(ahora_utc) or datetime.now(timezone.utc)
    previo = estado(user_id, ahora_utc, usar_cache=False)
    dia = previo.get("anterior") or (date.fromisoformat(previo["dia_natural"]) - timedelta(days=1)).isoformat()
    import json
    w = execute_sql_write
    if w is None:
        from db_core import execute_sql_write as w
    w("INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
      "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
      (PREFIJO + str(user_id), json.dumps({"dia": dia, "accion": accion, "en": ahora_utc.isoformat()})))
    olvidar(user_id)
    return estado(user_id, ahora_utc, usar_cache=False)


def dia_de_hoy(user_id: Optional[str], tz_offset_min: Optional[int] = None) -> Optional[str]:
    """El día que el usuario está viviendo (YYYY-MM-DD) si sigue en el anterior; si no, None (usa el de siempre)."""
    try:
        e = estado(user_id, tz_offset_min=tz_offset_min)
        return e["dia"] if e.get("extendido") else None
    except Exception:  # noqa: BLE001
        return None


def desayuno_empieza_el_dia(user_id: Optional[str], meal_type: Any, ahora_utc: Optional[datetime] = None) -> bool:
    """[P1-PLAN-LOTE-1146] Registrar un DESAYUNO de ahora mientras sigue en el día anterior solo por estar «despierto»
    es la señal más clara de que el día nuevo empezó (durmió menos de 4 h, o la regla se equivocó): se empieza el día
    nuevo, como si hubiera tocado «Empezar mi sábado». Lo que él eligió («Seguir…», su hora de corte) no se toca.
    True si cerró la extensión."""
    if not user_id or str(meal_type or "").strip().lower() != "desayuno":
        return False
    try:
        e = estado(user_id, ahora_utc)
        if e.get("extendido") and e.get("motivo") == "despierto":
            fijar(str(user_id), "empezar", ahora_utc)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-1146] el desayuno no pudo empezar el día: {exc!r}")
    return False


def hora_para_registro(user_id: Optional[str], ahora_utc: Optional[datetime] = None, meal_type: Any = None) -> Optional[str]:
    """Si sigue en el día anterior: la hora (UTC ISO) con la que se guarda lo que come AHORA — un minuto antes del
    corte, dentro de ese día. Si no, None (la hora de ahora, como siempre). Un desayuno «despierto» empieza el día."""
    try:
        if desayuno_empieza_el_dia(user_id, meal_type, ahora_utc):
            return None
        e = estado(user_id, ahora_utc)
        if not e.get("extendido") or not e.get("fin"):
            return None
        return (datetime.fromisoformat(e["fin"]) - timedelta(minutes=1)).isoformat()
    except Exception:  # noqa: BLE001
        return None
