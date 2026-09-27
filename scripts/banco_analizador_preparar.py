# backend/scripts/banco_analizador_preparar.py
"""[P1-PLAN-LOTE-571 · 2026-09-27] Prepara el banco congelado del analizador (se corre UNA vez, en local).

Baja la metadata de Nutrition5k y la lista de platos con foto cenital (`rgb_test_ids`, la partición reservada para
prueba), muestrea con semilla fija (banco_analizador.muestrear), descarga las PNG a --cache y escribe el manifiesto con
el sha256 de cada PNG (se commitea; las imágenes no). Con --solo-cache rellena/verifica la caché de otra máquina (el
VPS) contra el manifiesto ya commiteado.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

import banco_analizador as ba  # noqa: E402

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"


def descargar(url: str, cliente) -> bytes:
    r = cliente.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def url_png(dish_id: str) -> str:
    return f"{BASE}/imagery/realsense_overhead/{dish_id}/rgb.png"


def existe(url: str, cliente) -> bool:
    try:
        return cliente.head(url, timeout=30).status_code == 200
    except httpx.HTTPError:
        return False


def _platos(cliente) -> list[dict]:
    por_id: dict = {}
    for cafe in ("cafe1", "cafe2"):
        texto = descargar(f"{BASE}/metadata/dish_metadata_{cafe}.csv", cliente).decode("utf-8")
        for campos in csv.reader(io.StringIO(texto)):
            p = ba.parse_fila_n5k(campos)
            if p:
                por_id.setdefault(p["dish_id"], p)
    return list(por_id.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Congela el banco del analizador (Nutrition5k).")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--semilla", type=int, default=ba.SEMILLA)
    ap.add_argument("--solo-cache", action="store_true")
    args = ap.parse_args(argv)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True) as cliente:
        if args.solo_cache:
            man = json.loads(ba.MANIFIESTO.read_text(encoding="utf-8"))
            elegidos = man["platos"]
        else:
            con_foto = set(descargar(f"{BASE}/dish_ids/splits/rgb_test_ids.txt", cliente).decode().split())
            platos = _platos(cliente)
            # No toda la partición de prueba trae la foto cenital (medido: dish_1550705477 da 404). Se comprueba ANTES
            # de muestrear, así la muestra sigue siendo determinista sobre lo que sí existe.
            candidatos = [p for p in platos if p["dish_id"] in con_foto and ba.plato_valido(p)]
            con_foto = {p["dish_id"] for p in candidatos if existe(url_png(p["dish_id"]), cliente)}
            elegidos = ba.muestrear(platos, con_foto, n=args.n, semilla=args.semilla)
        hashes = {}
        for p in elegidos:
            destino = cache / f"{p['dish_id']}.png"
            if not destino.exists():
                destino.write_bytes(descargar(url_png(p["dish_id"]), cliente))
            hashes[p["dish_id"]] = ba.sha256_de(destino.read_bytes())
    if args.solo_cache:
        malos = ba.verificar_cache(man, lambda d: (cache / f"{d}.png").read_bytes() if (cache / f"{d}.png").exists() else None)
        # [P2-LOGGER-EXEMPT: CLI de preparación, salida a stdout a propósito]
        print(f"cache: {len(elegidos) - len(malos)} ok, {len(malos)} distintos")
        return 1 if malos else 0
    ba.MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    ba.MANIFIESTO.write_text(json.dumps(ba.manifiesto(elegidos, hashes, args.semilla, BASE), ensure_ascii=False,
                                        indent=1) + "\n", encoding="utf-8")
    # [P2-LOGGER-EXEMPT: CLI de preparación, salida a stdout a propósito]
    print(f"manifiesto: {len(elegidos)} platos -> {ba.MANIFIESTO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
