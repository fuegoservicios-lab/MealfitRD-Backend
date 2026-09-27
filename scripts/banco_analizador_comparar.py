# backend/scripts/banco_analizador_comparar.py
"""[P1-PLAN-LOTE-601 · 2026-09-27] ¿Entra una corrida candidata del banco? Regla PAREADA contra ≥ 2 corridas de la base.

    python scripts/banco_analizador_comparar.py --base linea_base.json,linea_base_r2.json --candidata corrida.json

Sin red ni DB: lee los JSON que escribe banco_analizador_correr.py. Imprime, por macro, la diferencia media de error
(candidata − media de las bases, en fracción) con su intervalo al 90 % y cuántos platos mejoran o empeoran más de 2
puntos. Sale 0 si la candidata entra y 1 si no.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import banco_analizador as ba  # noqa: E402


def _filas(ruta: str) -> list[dict]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))["per_dish"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compara una corrida del banco contra la base (regla pareada).")
    ap.add_argument("--base", required=True, help="corridas de la base separadas por comas (al menos 2)")
    ap.add_argument("--candidata", required=True)
    args = ap.parse_args(argv)
    r = ba.comparar_pareado([_filas(p) for p in args.base.split(",") if p.strip()], _filas(args.candidata))
    # [P2-LOGGER-EXEMPT: CLI del banco, salida a stdout a propósito]
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0 if r["acepta"] else 1


if __name__ == "__main__":
    sys.exit(main())
