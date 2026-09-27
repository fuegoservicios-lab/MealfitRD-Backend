# backend/scripts/banco_analizador_correr.py
"""[P1-PLAN-LOTE-572 · 2026-09-27] Corre el analizador REAL sobre el banco congelado y mide (spec §2-§3).

Llama a vision_agent.process_image_with_vision con los bytes de cada foto (convertida a JPEG ≤ 1024 px, como la manda
ScanMealModal), sin HTTP ni usuario. NO registra gasto de usuario ni toca el diario. Con --guardar inserta UNA fila en
analyzer_benchmark_runs (necesita NEON_DATABASE_URL). Va en el VPS: la clave de visión solo vive allí.

  cd /opt/mealfit/backend && /home/ubuntu/miniforge3/envs/mealfit/bin/python scripts/banco_analizador_correr.py \
      --cache /home/ubuntu/banco_analizador/cache --salida /home/ubuntu/banco_analizador/corrida.json [--guardar]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

import banco_analizador as ba  # noqa: E402

ESPERA_REINTENTO_S = 3.0


def _vision():
    import vision_agent
    return vision_agent


def a_jpeg(png: bytes, lado_max: int = 1024, calidad: int = 85) -> bytes:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    im.thumbnail((lado_max, lado_max))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=calidad)
    return out.getvalue()


async def _uno(va, plato: dict, png: bytes, sem: asyncio.Semaphore) -> dict:
    async with sem:
        jpeg = a_jpeg(png)
        res, lat, uso = None, None, {}
        for intento in (1, 2):
            va._reset_last_vision_usage()
            t0 = time.perf_counter()
            res = await va.process_image_with_vision(jpeg)
            lat = round(time.perf_counter() - t0, 3)
            uso = va.get_last_vision_usage() or {}
            if not (isinstance(res, dict) and res.get("analysis_failed")) or intento == 2:
                break
            await asyncio.sleep(ESPERA_REINTENTO_S)
    fila = ba.evaluar_plato(plato, res, lat)
    fila["tokens"] = {"input": int(uso.get("input_tokens") or 0), "output": int(uso.get("output_tokens") or 0)}
    if isinstance(res, dict):
        fila["respuesta"] = {"meal_name": res.get("meal_name"), "photo_kind": res.get("photo_kind"),
                             "items": [it.get("name") for it in (res.get("items") or []) if isinstance(it, dict)]}
    return fila


async def correr(man: dict, leer, concurrencia: int = 4, va=None) -> list[dict]:
    va = va or _vision()
    sem = asyncio.Semaphore(max(1, concurrencia))
    return list(await asyncio.gather(*[_uno(va, p, leer(p["dish_id"]), sem) for p in man["platos"]]))


def _guardar(corrida: dict) -> None:
    import psycopg
    with psycopg.connect(os.environ["NEON_DATABASE_URL"]) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO public.analyzer_benchmark_runs "
            "(model, prompt_sha, manifest_sha, n, ok, failed, metrics, per_dish, tokens, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)",
            (corrida["model"], corrida["prompt_sha"], corrida["manifest_sha"], corrida["metrics"]["n"],
             corrida["metrics"]["ok"], corrida["metrics"]["n"] - corrida["metrics"]["ok"],
             json.dumps(corrida["metrics"]), json.dumps(corrida["per_dish"]), json.dumps(corrida["tokens"]),
             corrida["notes"]),
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Corre el banco del analizador.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--concurrencia", type=int, default=4)
    ap.add_argument("--guardar", action="store_true")
    ap.add_argument("--notas", default="")
    args = ap.parse_args(argv)
    texto_man = ba.MANIFIESTO.read_text(encoding="utf-8")
    man = json.loads(texto_man)
    cache = Path(args.cache)

    def leer(dish_id):
        f = cache / f"{dish_id}.png"
        return f.read_bytes() if f.exists() else None

    malos = ba.verificar_cache(man, leer)
    if malos:
        # [P2-LOGGER-EXEMPT: CLI del banco, salida a stdout a propósito]
        print(f"ABORTADO: {len(malos)} fotos no coinciden con el manifiesto (primera: {malos[0]})")
        return 2
    va = _vision()
    if hasattr(va, "is_vision_enabled") and not va.is_vision_enabled():
        # [P2-LOGGER-EXEMPT: CLI del banco, salida a stdout a propósito]
        print("ABORTADO: la vision esta apagada (MEALFIT_VISION_PROVIDER)")
        return 2
    filas = asyncio.run(correr(man, leer, args.concurrencia, va=va))
    metricas = ba.agregar(filas)
    modelo = va._vision_model_name() if hasattr(va, "_vision_model_name") else "desconocido"
    tokens = {"input": sum(f["tokens"]["input"] for f in filas), "output": sum(f["tokens"]["output"] for f in filas)}
    from db import compute_llm_cost_micros
    micros = compute_llm_cost_micros(modelo, tokens["input"], tokens["output"])
    tokens["coste_usd"] = round(micros / 1_000_000, 4) if micros is not None else None
    corrida = {
        "ran_at": datetime.now(timezone.utc).isoformat(), "model": modelo,
        "prompt_sha": hashlib.sha256(str(getattr(va, "_MEAL_VISION_PROMPT", "")).encode()).hexdigest()[:12],
        "manifest_sha": hashlib.sha256(texto_man.encode()).hexdigest()[:12],
        "metrics": metricas, "per_dish": filas, "tokens": tokens, "notes": args.notas,
    }
    Path(args.salida).write_text(json.dumps(corrida, ensure_ascii=False, indent=1), encoding="utf-8")
    # [P2-LOGGER-EXEMPT: CLI del banco, salida a stdout a propósito]
    print(f"n={metricas['n']} ok={metricas['ok']} valida={metricas['valida']} "
          f"kcal_med={metricas['kcal']['mediana']} prot_med={metricas['proteina_g']['mediana']} "
          f"recall={metricas['recall_componentes']} coste_usd={tokens['coste_usd']}")
    if args.guardar:
        _guardar(corrida)
    return 0 if metricas["valida"] else 3


if __name__ == "__main__":
    sys.exit(main())
