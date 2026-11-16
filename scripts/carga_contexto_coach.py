# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1040 · A-14 · 2026-11-16] ¿Cuántos turnos del coach a la vez aguanta la fase de contexto?

El turno del coach es casi todo espera de E/S: modelo 3,3 s p50 (fuera del GIL) y contexto ~1 s p50, que son lecturas a
Neon lanzadas EN PARALELO (`_ChatContextPrefetcher`: hasta 6 hilos por turno, más preflight y RAG). El proceso vive
con UN worker y un pool de 12 conexiones, así que el primer cuello candidato es el pool, no la CPU ni el modelo.

Este script lanza N prefetches REALES del contexto a la vez (los del turno, mismas funciones) y mide cuánto tarda cada
uno y cuánto se espera al pool. Sin IA y SIN ESCRIBIR: la conexión se abre con `default_transaction_read_only=on`
(cualquier INSERT/UPDATE falla en la base, y el código del contexto es fail-open), la caché de contexto se apaga para
que cada turno lea de verdad. Corre en un proceso aparte del backend (su propio pool, del mismo tamaño que el de
producción). Uso, en el servidor:

    python scripts/carga_contexto_coach.py --usuario <uuid> --niveles 1,5,10,20,40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _solo_lectura(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}options=-c%20default_transaction_read_only%3Don"


def percentil(xs: list, p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None


def un_turno(user_id: str, out: list, errores: list) -> None:
    import agent
    t0 = time.monotonic()
    try:
        pf = agent._ChatContextPrefetcher(
            session_id=str(uuid.uuid4()), user_id=user_id, current_plan=None,
            local_date=time.strftime("%Y-%m-%d"), tz_offset=-240, memoria_on=True, nevera_on=True,
        ).start()
        pf.resolve_all()
        out.append({"ms": int((time.monotonic() - t0) * 1000), "bloques": dict(pf.blocks_ms)})
    except Exception as e:  # noqa: BLE001
        errores.append(type(e).__name__)


def nivel(n: int, user_id: str) -> dict:
    out, errores = [], []
    hilos = [threading.Thread(target=un_turno, args=(user_id, out, errores)) for _ in range(n)]
    t0, cpu0 = time.monotonic(), time.process_time()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = int((time.monotonic() - t0) * 1000)
    cpu_por_turno = round((time.process_time() - cpu0) * 1000 / max(1, n), 1)   # CPU de TODO el proceso / turnos
    ms = [o["ms"] for o in out]
    peor_bloque = {}
    for o in out:
        for k, v in o["bloques"].items():
            peor_bloque[k] = max(peor_bloque.get(k, 0), v)
    return {"turnos": n, "ok": len(out), "errores": len(errores), "pared_ms": pared, "cpu_ms_por_turno": cpu_por_turno,
            "turno_p50_ms": percentil(ms, .5), "turno_p90_ms": percentil(ms, .9), "turno_max_ms": max(ms) if ms else None,
            "bloque_mas_lento_ms": max(peor_bloque.items(), key=lambda kv: kv[1]) if peor_bloque else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usuario", required=True)
    ap.add_argument("--niveles", default="1,5,10,20,40")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    load_dotenv(Path.cwd() / ".env")   # copiado fuera del backend: el .env del directorio desde el que se corre
    for clave in ("NEON_DATABASE_URL", "NEON_DATABASE_URL_POOLED", "DATABASE_URL"):
        if os.environ.get(clave):
            os.environ[clave] = _solo_lectura(os.environ[clave])
    os.environ["MEALFIT_CHAT_CTX_CACHE_TTL_S"] = "0"
    import db_core
    if db_core.connection_pool:
        db_core.connection_pool.open(wait=True)
    print(json.dumps({"pool_max": getattr(db_core, "DB_POOL_MAX_SIZE", None)}))
    un_turno(args.usuario, [], [])   # calentar imports y catálogos: no cuenta
    for n in [int(x) for x in args.niveles.split(",") if x.strip()]:
        print(json.dumps(nivel(n, args.usuario), ensure_ascii=False), flush=True)
        time.sleep(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
