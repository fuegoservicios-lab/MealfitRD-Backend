# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1059 · plan 1031/1035 · 2026-11-17] El coach conoce el progreso por semanas.

Un bloque corto (~70 tokens) en la zona VOLÁTIL del turno (después de `_VOLATILE_SYS_SPLIT_MARKER`: no rompe el prefijo
cacheado), con solo cifras de `progreso_semanas.resumen` (D-E). Para otras semanas está la herramienta
`consultar_progreso` (`tools.py`). El día que ocurre un hito o cierra una semana, una línea «HOY:» para que lo celebre
UNA vez (lote 1035).

Canario `progreso_coach` (`coach_canario`). Invitado, sin día 1 o cualquier fallo ⇒ cadena vacía (`fail_open`).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

_MES = ("", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
_RANGO = {"bronce": "Bronce", "plata": "Plata", "oro": "Oro", "platino": "Platino", "diamante": "Diamante",
          "maestro": "Maestro"}
_ORD = {1: "1.ª", 2: "2.ª", 3: "3.ª"}


def _rango_txt(clave, division=None) -> str:
    """[P1-PLAN-LOTE-1161] «Plata II» (o solo «Plata» si no hay división). [1163] Divisiones IV→I: la `division`
    interna asciende (1 = entrada) y se ve en romano al revés (Plata IV es la entrada, Plata I la cima)."""
    import juego_puntos as jp
    if isinstance(clave, dict):
        clave, division = clave.get("clave"), clave.get("division")
    base = _RANGO.get(clave, clave)
    romano = jp.division_romana(division)
    return f"{base} {romano}" if romano else str(base)


def _mult(m) -> str:
    return ("×" + (f"{float(m):g}").replace(".", ",")) if m else "×1"


def _fecha(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {_MES[d.month]}"


def _estrellas(n: int) -> str:
    return f"{n}★"


def _linea_de_hoy(claves: list[str], res: dict) -> str:
    partes = []
    for c in claves:
        tipo, _, valor = c.partition(":")
        if tipo == "semana_cerrada":
            sp = res.get("semana_pasada") or {}
            if str(sp.get("numero")) == valor:
                partes.append(f"cerró la S{valor} con {_estrellas(sp.get('estrellas', 0))}")
        elif tipo == "primera_estrella":
            partes.append("1.ª estrella de su vida en la app")
        elif tipo == "primera_semana_perfecta":
            partes.append("1.ª semana perfecta")
        elif tipo == "racha":
            partes.append(f"racha de {valor} días registrando (multiplicador {_mult(__import__('juego_puntos').tramo_de_racha(int(valor) if valor.isdigit() else 0)['mult'])})")
        elif tipo == "division":
            clave, _, div = valor.partition(":")
            partes.append(f"subió a {_rango_txt(clave, int(div) if div.isdigit() else None)}")
        elif tipo == "maestro":
            partes.append("¡llegó a MAESTRO, el rango más difícil de la app (14.500 pts, 8 semanas perfectas y racha de 100 días)!")
        elif tipo == "escudo":
            partes.append("ganó un escudo de racha")
        elif tipo == "salvo":
            partes.append("un escudo salvó su racha")
        elif tipo == "record":
            partes.append(f"récord de puntos en un día ({(res.get('hoy_puntos') or {}).get('puntos', 0)} pts)")
        elif tipo == "semanas_seguidas_con_estrella":
            partes.append(f"{valor} semanas seguidas con estrella")
        elif tipo == "rango":
            partes.append(f"subió a rango {_RANGO.get(valor, valor)}")
        elif tipo == "semana_numero":
            partes.append(f"llegó a la Semana {valor}")
    return (" HOY: " + "; ".join(partes) + " (celébralo una vez).") if partes else ""


def texto_del_bloque(res: dict) -> str:
    """Puro: el bloque a partir del `resumen`. Vacío si no hay resumen."""
    if not res or not res.get("semana"):
        return ""
    s = res["semana"]
    est = s["estrellas"]
    # [P1-PLAN-LOTE-1161→1164] juego por PUNTOS de calidad: registrar 0 (solo racha), día completo +5, calorías en banda +20 (cerca +8, pasado +3), proteína +10, día perfecto +10,
    # agua +5; la racha de días registrando multiplica; estrellas de la semana y rango (con división) por puntos.
    if est >= 3:
        falta = "semana perfecta"
    elif "faltan_puntos_para_otra_estrella" in s:
        falta = f"faltan {s['faltan_puntos_para_otra_estrella']} pts para la {_ORD[est + 1]} estrella"
    else:
        falta = f"faltan {s['faltan_para_otra_estrella']} días en meta para la {_ORD[est + 1]} estrella"
    hp = res.get("hoy_puntos") or {}
    t = (f"\n\n🏆 PROGRESO (cifras del servidor, juego por puntos): Semana {s['numero']} "
         f"({_fecha(s['empieza'])}–{_fecha(s['termina'])}), día {s['dia']} de 7. ")
    if hp:
        t += f"Hoy: {hp.get('puntos', 0)} pts ({_mult(hp.get('mult'))}). "
    if "puntos" in s:
        t += f"Esta semana: {s['puntos']} pts, {_estrellas(est)}; {falta}. "
    else:
        t += f"Esta semana: {s['dias_en_meta']} de 7 días en meta, {_estrellas(est)}; {falta}. "
    tramo = res.get("racha_tramo") or {}
    t += f"Racha: {res['racha']} días registrando ({_mult(tramo.get('mult'))}; mejor: {res['mejor_racha']})"
    if "escudos" in res:
        t += f"; escudos {res['escudos']}/{res.get('escudos_max', 2)}"
    t += "."
    sp = res.get("semana_pasada")
    if sp:
        t += f" Semana pasada (S{sp['numero']}): {sp.get('puntos', 0)} pts, {_estrellas(sp['estrellas'])}."
    tot, rg = res["total"], res["rango"]
    t += f" Total: {tot.get('puntos', 0)} pts · rango {_rango_txt(rg)}"
    if rg.get("siguiente") and rg.get("puntos_para_la_siguiente") is not None:
        t += f" (faltan {rg['puntos_para_la_siguiente']} pts para {_rango_txt(rg['siguiente'])})"
    m = res.get("maestro") or {}
    if m and not m.get("conseguido") and rg.get("clave") in ("platino", "diamante"):
        req = {r["clave"]: r for r in m.get("requisitos") or []}
        t += (f"; Maestro pide 14.500 pts + {req.get('semanas_perfectas', {}).get('meta', 8)} semanas perfectas "
              f"(lleva {req.get('semanas_perfectas', {}).get('actual', 0)}) + racha de 100 días "
              f"(mejor {req.get('racha', {}).get('actual', 0)})")
    t += "."
    t += _linea_de_hoy(res.get("eventos_de_hoy") or [], res)
    return t


def medicion_para_admin(horas: int = 168) -> dict:
    """[plan 1036.5] ¿Cuántas respuestas del coach hablan del progreso? Debe ser BAJO (regla R5: no en cada turno).
    Solo recuentos: nunca el texto."""
    from db_core import execute_sql_query
    h = max(1, min(int(horas or 168), 720))
    f = execute_sql_query(
        "SELECT count(*)::int AS total, count(*) FILTER (WHERE content ~* %s)::int AS con_progreso "
        "FROM public.agent_messages WHERE role = 'model' AND created_at > now() - make_interval(hours => %s)",
        (r"(estrellas?\M|★|\mrachas?\M|\msemana [0-9]+)", h), fetch_one=True,
    ) or {}
    total, con = int(f.get("total") or 0), int(f.get("con_progreso") or 0)
    return {"respuestas": total, "con_progreso": con, "pct": round(100 * con / total, 1) if total else 0.0}


def activo(user_id: Optional[str]) -> bool:
    if not user_id or str(user_id) == "guest":
        return False
    try:
        from coach_canario import activa
        return activa(user_id, "progreso_coach")
    except Exception:  # noqa: BLE001
        return False


def precargar(user_id: Optional[str], tz_offset: Optional[int]) -> None:
    """Para el prefetch del turno: llena la caché de `progreso_semanas` en paralelo con el resto del contexto."""
    if activo(user_id):
        __import__("progreso_semanas").resumen(str(user_id), tz_offset, con_eventos=False)


# El progreso al EMPEZAR el turno, por usuario: la caché del turno no sirve (`clear_reads()` la vacía antes de las
# herramientas que escriben, justo antes del registro que queremos comparar).
_ANTES: dict[str, tuple[float, dict]] = {}
_ANTES_LOCK = threading.Lock()
_ANTES_TTL_S = 600.0
_ANTES_MAX = 500


def _guardar_antes(user_id: str, res: dict) -> None:
    with _ANTES_LOCK:
        if len(_ANTES) >= _ANTES_MAX:
            _ANTES.pop(next(iter(_ANTES)), None)
        _ANTES[user_id] = (time.monotonic(), res or {})


def _antes_de(user_id: str) -> dict:
    with _ANTES_LOCK:
        v = _ANTES.get(user_id)
    return v[1] if v and time.monotonic() - v[0] < _ANTES_TTL_S else {}


def nota_de_estrella(antes: dict, despues: dict) -> str:
    """Puro: si entre `antes` y `despues` (misma semana) subió una estrella, la nota para el asistente; si no, ''."""
    a, d = (antes or {}).get("semana") or {}, (despues or {}).get("semana") or {}
    logros = []
    if a and d and a.get("numero") == d.get("numero") and d.get("estrellas", 0) > a.get("estrellas", 0):
        if d["estrellas"] >= 3:
            logros.append(f"completó la Semana {d['numero']} perfecta: 3★ ({d.get('puntos', 0)} pts)")
        else:
            logros.append(f"ganó la {_ORD[d['estrellas']]} estrella de la Semana {d['numero']} ({d.get('puntos', 0)} pts)")
    # [P1-PLAN-LOTE-1161] subir de división también se dice al confirmar.
    ra, rd = (antes or {}).get("rango") or {}, (despues or {}).get("rango") or {}
    if "division" in ra and "division" in rd:
        import juego_puntos as jp
        if jp.nivel_indice(rd["clave"], rd["division"] or 0) > jp.nivel_indice(ra["clave"], ra["division"] or 0):
            logros.append(f"subió a {_rango_txt(rd)}")
    if not logros:
        return ""
    logro = " y ".join(logros)
    return f"\n\n🏆 Aviso para el asistente: con este registro {logro}. Menciónalo en UNA frase al confirmar."


def nota_tras_registrar(user_id: Optional[str]) -> str:
    """Tras un registro del coach: ¿ganó una estrella? Compara con el progreso de ANTES del turno (el del bloque 🏆,
    guardado en la caché del turno). Sin ese «antes» no dice nada (no se puede saber si es nueva)."""
    if not activo(user_id):
        return ""
    try:
        antes = _antes_de(str(user_id))
        if not antes:
            return ""
        despues = __import__("progreso_semanas").resumen(str(user_id), None, con_eventos=False)
        _guardar_antes(str(user_id), despues)
        return nota_de_estrella(antes, despues)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1059] nota de progreso omitida: {type(exc).__name__}")
        return ""


def bloque_para_chat(user_id: Optional[str], tz_offset: Optional[int] = None) -> str:
    if not activo(user_id):
        return ""
    try:
        from turn_cache import get_or_load
        res = get_or_load("progreso", f"{user_id}:{tz_offset}",
                          lambda: __import__("progreso_semanas").resumen(str(user_id), tz_offset, con_eventos=False))
        _guardar_antes(str(user_id), res)
        return texto_del_bloque(res)
    except Exception as exc:  # noqa: BLE001
        try:
            from fail_open import fail_open
            fail_open("progreso_coach", exc)
        except Exception:  # noqa: BLE001
            logger.debug(f"[LOTE-1059] bloque de progreso omitido: {type(exc).__name__}")
        return ""
