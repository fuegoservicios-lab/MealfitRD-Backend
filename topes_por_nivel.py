# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1116 · 2026-11-20] Los topes de uso por nivel, en UNA tabla: generación, coach, fotos, voz y gasto.

Pedido del dueño (20-nov): «topes al 100 % en todos los sentidos —voz, etc.—: no tenemos mucho presupuesto», con los
precios bajados (Básico 3,99 · Plus 6,99 · Max 11,99 USD/mes). Antes había medidores sueltos y dos agujeros:
la VOZ EN VIVO (US$ 0,05/min) no tenía tope por usuario ni por sesión, y el escáner de fotos solo un anti-spam de
10/min. El tope diario de US$ 0,25 era igual para todos los niveles.

La tabla está calibrada para que lo PEOR que pueda gastar un usuario de pago (todo al máximo) quede por debajo de ~45 %
de lo que deja su cuota tras la comisión de PayPal, y un gratis por debajo de US$ 0,50 al mes. El último candado es
`usd_mes`: el gasto de IA del mes (todos los nodos) — si se alcanza, se frena lo que cuesta, nunca lo que ya es suyo.

Cada número se puede mover sin redeploy: `MEALFIT_TOPE_<NIVEL>_<CLAVE>` (ej. `MEALFIT_TOPE_GRATIS_FOTOS_DIA=5`).
`admin` no tiene topes prácticos. Los nombres de clave son contrato con el frontend (`config/plans.js` → TIER_LIMITS),
que los pinta en la página de planes; `test_p1_plan_lote_1116_*` ancla la paridad. tooltip-anchor: P1-PLAN-LOTE-1116-TOPES
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("mealfit")

NIVELES = ("gratis", "basic", "plus", "ultra")
CLAVES = ("generacion_mes", "coach_mes", "coach_dia", "fotos_dia", "voz_min_mes", "voz_min_sesion", "usd_dia", "usd_mes")

TABLA = {
    #            planes  coach/mes coach/día fotos/día voz min/mes min/sesión  US$/día  US$/mes
    # medido 20-nov: el gratis más activo llega a 6 fotos y 15 mensajes al día y US$ 1,50 al mes → el uso normal cabe,
    # el intensivo pasa a Básico
    "gratis": dict(generacion_mes=4, coach_mes=60, coach_dia=20, fotos_dia=4, voz_min_mes=5, voz_min_sesion=3,
                   usd_dia=0.10, usd_mes=0.50),
    "basic": dict(generacion_mes=12, coach_mes=300, coach_dia=60, fotos_dia=10, voz_min_mes=20, voz_min_sesion=8,
                  usd_dia=0.20, usd_mes=1.40),
    "plus": dict(generacion_mes=30, coach_mes=900, coach_dia=120, fotos_dia=20, voz_min_mes=45, voz_min_sesion=12,
                 usd_dia=0.40, usd_mes=2.80),
    "ultra": dict(generacion_mes=60, coach_mes=2000, coach_dia=250, fotos_dia=40, voz_min_mes=100, voz_min_sesion=20,
                  usd_dia=0.80, usd_mes=5.00),
}
_SIN_TOPE = dict(generacion_mes=999999, coach_mes=999999, coach_dia=999999, fotos_dia=999999, voz_min_mes=99999,
                 voz_min_sesion=60, usd_dia=1000.0, usd_mes=10000.0)


def activo() -> bool:
    return str(os.environ.get("MEALFIT_TOPES_POR_NIVEL", "on")).strip().lower() not in ("off", "0", "false", "no")


def nivel_de(plan_tier) -> str:
    t = str(plan_tier or "gratis").strip().lower()
    if t == "admin":
        return "admin"
    return t if t in NIVELES else "gratis"


def limite(plan_tier, clave: str):
    """El tope de `clave` para el nivel, con el override `MEALFIT_TOPE_<NIVEL>_<CLAVE>` si es un número válido."""
    if clave not in CLAVES:
        raise KeyError(clave)
    nivel = nivel_de(plan_tier)
    base = _SIN_TOPE[clave] if nivel == "admin" else TABLA[nivel][clave]
    crudo = os.environ.get(f"MEALFIT_TOPE_{nivel.upper()}_{clave.upper()}")
    if crudo not in (None, ""):
        try:
            v = float(crudo)
            if v >= 0:
                return v if isinstance(base, float) else int(v)
        except ValueError:
            pass
    return base


def tabla_publica() -> dict:
    """Lo que la página de planes enseña (ya con overrides)."""
    return {n: {c: limite(n, c) for c in CLAVES} for n in NIVELES}


# ─────────────────────────────────────────────────────────────── lecturas de uso (fail-open: 0 si no se puede leer)
def _q(sql: str, params: tuple):
    try:
        from db_core import execute_sql_query, connection_pool
        if not connection_pool:
            return None
        return execute_sql_query(sql, params, fetch_one=True) or {}
    except Exception as e:                                             # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1116] uso ilegible: {type(e).__name__}")
        return None


def _inicio_de_mes() -> datetime:
    try:
        from regalos_cuenta import inicio_de_mes
        return inicio_de_mes()
    except Exception:                                                  # noqa: BLE001
        n = datetime.now(timezone.utc)
        return datetime(n.year, n.month, 1, tzinfo=timezone.utc)


def gasto_mes_usd(user_id: str) -> float:
    f = _q("SELECT COALESCE(SUM(cost_usd_micros), 0)::bigint AS m FROM llm_usage_events WHERE user_id = %s AND created_at >= %s",
           (str(user_id), _inicio_de_mes()))
    return int((f or {}).get("m") or 0) / 1e6


def fotos_24h(user_id: str) -> int:
    f = _q("SELECT count(*)::int AS n FROM llm_usage_events WHERE user_id = %s AND node = 'vision_scan' "
           "AND created_at >= NOW() - INTERVAL '24 hours'", (str(user_id),))
    return int((f or {}).get("n") or 0)


def coach_24h(user_id: str) -> int:
    f = _q("SELECT count(*)::int AS n FROM api_usage WHERE user_id = %s AND endpoint = 'llm_chat' "
           "AND created_at >= NOW() - INTERVAL '24 hours'", (str(user_id),))
    return int((f or {}).get("n") or 0)


def voz_segundos_mes(user_id: str) -> float:
    f = _q("SELECT COALESCE(SUM((metadata->>'segundos')::float + COALESCE((metadata->>'segundos_espera')::float, 0) / 2.5), 0) AS s "
           "FROM llm_usage_events WHERE user_id = %s AND node = 'coach_live_voice' AND created_at >= %s",
           (str(user_id), _inicio_de_mes()))
    return float((f or {}).get("s") or 0)


def _tier(user_id: str) -> str:
    try:
        from db import get_user_profile
        return nivel_de((get_user_profile(user_id) or {}).get("plan_tier"))
    except Exception:                                                  # noqa: BLE001
        return "gratis"


# ─────────────────────────────────────────────────────────────── decisiones
def frenado(user_id: Optional[str], que: str) -> Optional[dict]:
    """None si puede seguir; si no, {codigo, limite, usado, mensaje}. `que` ∈ foto | coach | voz | generacion.
    Fail-open: sin cuenta, knob off, admin o lectura fallida ⇒ None."""
    if not activo() or not user_id or user_id == "guest":
        return None
    tier = _tier(user_id)
    if tier == "admin":
        return None
    mes = gasto_mes_usd(user_id)
    tope_mes = float(limite(tier, "usd_mes"))
    if mes >= tope_mes:
        return {"codigo": "tope_mes_usd", "limite": tope_mes, "usado": round(mes, 4), "nivel": tier,
                "mensaje": "Llegaste al uso de IA incluido en tu plan este mes. Se renueva el día 1; "
                           "mientras, el contador y lo que ya registraste siguen funcionando."}
    if que == "foto":
        n, lim = fotos_24h(user_id), int(limite(tier, "fotos_dia"))
        if n >= lim:
            return {"codigo": "tope_fotos_dia", "limite": lim, "usado": n, "nivel": tier,
                    "mensaje": f"Usaste tus {lim} análisis de foto de hoy. Puedes registrar la comida escribiéndola "
                               "o desde el buscador; mañana vuelven."}
    if que == "coach":
        n, lim = coach_24h(user_id), int(limite(tier, "coach_dia"))
        if n >= lim:
            return {"codigo": "tope_coach_dia", "limite": lim, "usado": n, "nivel": tier,
                    "mensaje": f"Llegaste a los {lim} mensajes con el coach de hoy. Mañana sigue la conversación."}
    if que == "voz":
        usados = voz_segundos_mes(user_id) / 60.0
        lim = float(limite(tier, "voz_min_mes"))
        if usados >= lim:
            return {"codigo": "tope_voz_mes", "limite": lim, "usado": round(usados, 1), "nivel": tier,
                    "mensaje": "Usaste tus minutos de voz en vivo de este mes. Puedes seguir hablándole al coach "
                               "con el micrófono del chat."}
    return None


def segundos_de_sesion_voz(user_id: str) -> int:
    """Lo que puede durar la próxima sesión en vivo: el máximo por sesión de su nivel, sin pasar de lo que le queda
    del mes (mínimo 30 s para que el corte no llegue a mitad de la primera frase)."""
    tier = _tier(user_id)
    por_sesion = int(float(limite(tier, "voz_min_sesion")) * 60)
    if tier == "admin" or not activo():
        return por_sesion
    queda = max(0.0, float(limite(tier, "voz_min_mes")) * 60 - voz_segundos_mes(user_id))
    return max(30, int(min(por_sesion, queda)))
