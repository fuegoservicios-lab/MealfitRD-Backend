# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1037 · A-34 · 2026-11-15] Casos de borde del turno del coach: comportamiento DECIDIDO y escrito.

El plan v2 (A-34) pedía «un caso de prueba por cada uno; decidir el comportamiento y dejarlo escrito». Al leer el
camino del turno salieron tres huecos reales y una decisión que no estaba en ningún sitio:

1. TOPE DIARIO × CUOTA MENSUAL. Cuando el tope diario de gasto bloquea el turno, el coach contesta con un texto fijo
   SIN llamar al modelo — pero ese texto viajaba como un `chunk` normal, así que el router lo cobraba contra la cuota
   mensual (`log_api_usage`) y lanzaba la extracción de hechos, que sí gasta IA. Una cuenta gratis (60 al mes) que
   llegaba al tope y seguía escribiendo perdía mensajes del mes sin recibir respuesta. DECISIÓN: un turno sin modelo
   no cuenta contra la cuota ni dispara trabajo de fondo con IA. Los eventos del tope llevan `budget_cap: true` y el
   router pregunta aquí (`turno_sin_modelo`).

2. INVITADOS. Sin cuenta no hay memoria a largo plazo ni caché de contexto (ya era así y queda fijado en pruebas), ni
   cuota mensual, ni tope diario de gasto: lo único que acotaba a un invitado era el limitador de 30 por minuto por IP
   — unos 43.000 turnos al día desde una sola IP. Medido el 15-nov: 0 sesiones de invitado en 30 días, así que acotarlo
   no le quita nada a nadie. DECISIÓN: tope de turnos de invitado por IP y día (`MEALFIT_CHAT_GUEST_TURNS_PER_DAY`,
   40; 0 = sin tope). En memoria del proceso, como el resto del estado del turno (A-30 lo llevará a Redis si hay más
   de un proceso).

3. CAMBIO DE DÍA Y DE HUSO A MITAD DE SESIÓN. El «hoy» se resuelve EN CADA TURNO con la fecha y el huso que manda el
   cliente (no se guarda en la sesión), y las entradas de la caché de contexto que dependen del día lo llevan en la
   clave. Lo que no había era validación: la fecha y el huso del cuerpo pasaban tal cual. DECISIÓN: el huso debe ser un
   entero en [-840, 840] y la fecha un `YYYY-MM-DD` a no más de un día del día biológico que calcula el servidor para
   ese huso; lo que no cumple se descarta y el servidor lo deriva (`_resolve_chat_local_time`): una pestaña abierta
   desde anteayer o un cliente con un bug ya no hacen que el coach lea —o anote— el diario de otro día.

4. DOS DISPOSITIVOS (O DOS ENVÍOS) A LA VEZ EN LA MISMA SESIÓN. Los dos turnos corrían en paralelo sobre el mismo hilo
   del grafo (mismo `thread_id`): gana el último que escribe el checkpoint y el otro turno desaparece de la memoria
   del modelo, o queda una llamada a herramienta sin su resultado. También pasa en un solo teléfono: «Detener» y
   enviar otra cosa antes de que el turno anterior termine de cortar. DECISIÓN: los turnos de una misma sesión van EN
   SERIE: el segundo espera al primero (hasta `MEALFIT_CHAT_TURNO_EN_SERIE_S`, 45 s; 0 = apagado) y después corre con
   el historial completo. Si la espera se agota, sigue (falla abierto, contado en `fail_open`). No se rechaza con un
   error: los paquetes viejos de la app no sabrían qué hacer con él.

Sin datos de salud ni texto del usuario en los logs. tooltip-anchor: P1-PLAN-LOTE-1037-BORDES-DEL-TURNO
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


def _entero_de_env(nombre: str, por_defecto: int, minimo: int, maximo: int) -> int:
    try:
        v = int(str(os.getenv(nombre, "") or "").strip() or por_defecto)
    except (TypeError, ValueError):
        return por_defecto
    return v if minimo <= v <= maximo else por_defecto


# ───────────────────────────────────────────────────────────── 1. turno sin modelo (tope diario)
MARCA_SIN_MODELO = "budget_cap"


def turno_sin_modelo(evento) -> bool:
    """True si el evento del stream viene de un turno que NO llamó al modelo (el tope diario lo contestó con texto fijo)."""
    return isinstance(evento, dict) and evento.get(MARCA_SIN_MODELO) is True


# ───────────────────────────────────────────────────────────── 2. invitados: tope por IP y día
_DIA_S = 86_400.0
_MAX_IPS = 4_096
_cerrojo_inv = threading.Lock()
_TURNOS_DE_INVITADO: dict[str, deque] = {}


def tope_de_invitado() -> int:
    # En la suite todos los invitados comparten la IP del cliente de pruebas: apagado salvo que la prueba fije el knob.
    if os.environ.get("PYTEST_CURRENT_TEST") and "MEALFIT_CHAT_GUEST_TURNS_PER_DAY" not in os.environ:
        return 0
    return _entero_de_env("MEALFIT_CHAT_GUEST_TURNS_PER_DAY", 40, 0, 100_000)


def invitado_puede(ip: Optional[str], ahora: Optional[float] = None) -> bool:
    """Anota un turno de invitado para esa IP y dice si cabe en el tope de 24 h. Un turno rechazado NO se anota (el que
    insiste no se alarga el castigo). Sin IP no se puede contar: pasa (lo acota el limitador por minuto)."""
    tope = tope_de_invitado()
    if tope <= 0 or not ip:
        return True
    t = time.time() if ahora is None else float(ahora)
    with _cerrojo_inv:
        cola = _TURNOS_DE_INVITADO.get(ip)
        if cola is None:
            if len(_TURNOS_DE_INVITADO) >= _MAX_IPS:      # barrido barato: fuera las IP sin turnos en 24 h
                for k in [k for k, c in _TURNOS_DE_INVITADO.items() if not c or t - c[-1] >= _DIA_S]:
                    _TURNOS_DE_INVITADO.pop(k, None)
                if len(_TURNOS_DE_INVITADO) >= _MAX_IPS:
                    _TURNOS_DE_INVITADO.clear()
            cola = _TURNOS_DE_INVITADO.setdefault(ip, deque())
        while cola and t - cola[0] >= _DIA_S:
            cola.popleft()
        if len(cola) >= tope:
            return False
        cola.append(t)
        return True


def exigir_cupo_de_invitado(ip: Optional[str], verified_user_id: Optional[str]) -> None:
    """Para el router: 429 si un invitado (sin token) agotó su tope del día. Con cuenta no aplica (tiene su cuota)."""
    if verified_user_id or invitado_puede(ip):
        return
    from fastapi import HTTPException
    logger.warning("[P1-PLAN-LOTE-1037] invitado en su tope diario de turnos: 429")
    raise HTTPException(
        status_code=429,
        detail="Llegaste al límite de mensajes de prueba por hoy. Crea tu cuenta gratis para seguir conversando.",
        headers={"Retry-After": "3600", "X-Coach-Guest-Cap": str(tope_de_invitado())},
    )


# ───────────────────────────────────────────────────────────── 3. fecha y huso del cliente
_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HUSO_MIN, HUSO_MAX = -840, 840      # getTimezoneOffset(): UTC+14 .. UTC-14


def huso_del_cliente(tz_offset) -> Optional[int]:
    """El huso del cuerpo como entero en [-840, 840], o None si no es un huso (un booleano, un epoch, un texto)."""
    if tz_offset is None or isinstance(tz_offset, bool):
        return None
    try:
        v = int(float(tz_offset))
    except (TypeError, ValueError, OverflowError):
        return None
    return v if HUSO_MIN <= v <= HUSO_MAX else None


def fecha_del_cliente(local_date, tz_offset: Optional[int], ahora_utc=None) -> Optional[str]:
    """La fecha del cuerpo si es un `YYYY-MM-DD` real a ≤ 1 día del día biológico del servidor para ese huso; si no, None."""
    if not isinstance(local_date, str) or not _FECHA.match(local_date):
        return None
    try:
        d = date.fromisoformat(local_date)
    except ValueError:
        return None
    try:
        from db_facts import diary_local_date_str
        del_servidor = date.fromisoformat(diary_local_date_str(tz_offset, now_utc=ahora_utc))
    except Exception:  # noqa: BLE001 — sin referencia no se puede juzgar: se deja como llegó
        return local_date
    return local_date if abs((d - del_servidor).days) <= 1 else None


def fecha_y_huso_del_turno(local_date, tz_offset, ahora_utc=None):
    """(fecha, huso) saneados. Lo descartado vuelve como None para que el servidor lo derive del perfil."""
    huso = huso_del_cliente(tz_offset)
    fecha = fecha_del_cliente(local_date, huso, ahora_utc)
    if (local_date is not None and fecha is None) or (tz_offset is not None and huso is None):
        logger.warning(
            "[P1-PLAN-LOTE-1037] fecha u huso del cliente descartados (fecha_ok=%s, huso_ok=%s): los deriva el servidor",
            local_date is None or fecha is not None, tz_offset is None or huso is not None,
        )
    return fecha, huso


# ───────────────────────────────────────────────────────────── 4. una sesión, un turno a la vez
_cerrojo_serie = threading.Lock()
_SESIONES: dict[str, list] = {}     # session_id -> [Lock, cuántos lo esperan o lo tienen]


def espera_en_serie_s() -> int:
    return _entero_de_env("MEALFIT_CHAT_TURNO_EN_SERIE_S", 45, 0, 300)


class _Paso:
    """El turno de una sesión. `salir()` es idempotente y puede llamarse desde otro hilo (el stream cambia de hilo)."""

    def __init__(self, sid: Optional[str], lock, tomado: bool, espero_ms: int):
        self._sid, self._lock, self.tomado, self.espero_ms = sid, lock, tomado, espero_ms
        self._hecho = False

    def salir(self) -> None:
        if self._hecho:
            return
        self._hecho = True
        if self._sid is None:
            return
        if self.tomado:
            try:
                self._lock.release()
            except RuntimeError:
                pass
        with _cerrojo_serie:
            e = _SESIONES.get(self._sid)
            if e is not None:
                e[1] -= 1
                if e[1] <= 0:
                    _SESIONES.pop(self._sid, None)


def entrar_en_serie(session_id, espera_s: Optional[float] = None) -> _Paso:
    """Espera a que termine el turno anterior de ESTA sesión y toma el suyo. Nunca lanza ni bloquea para siempre: si la
    espera se agota, el turno sigue sin el cerrojo (y queda contado). Hay que llamar a `salir()` en el `finally`."""
    t = espera_en_serie_s() if espera_s is None else espera_s
    if not session_id or t <= 0:
        return _Paso(None, None, False, 0)
    sid = str(session_id)
    with _cerrojo_serie:
        e = _SESIONES.setdefault(sid, [threading.Lock(), 0])
        e[1] += 1
    t0 = time.monotonic()
    tomado = e[0].acquire(timeout=float(t))
    ms = int((time.monotonic() - t0) * 1000)
    if not tomado:
        try:
            from fail_open import fail_open
            fail_open("chat_turno_en_serie", TimeoutError(f"el turno anterior no terminó en {t} s"))
        except Exception:  # noqa: BLE001
            pass
    elif ms >= 50:
        logger.info("[P1-PLAN-LOTE-1037] turno en serie: esperó %d ms al turno anterior de la sesión %s", ms, sid[:8])
    return _Paso(sid, e[0], tomado, ms)
