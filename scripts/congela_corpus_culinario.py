# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-18 · 2026-09-12 · C0] Congela el corpus FIJO de la medición culinaria (sólo SELECTs).

La línea base del 6-sep se midió sobre la ventana VIVA de `plan_data` y dejó de ser reproducible en 14 h (los mismos
96 planes daban 1.186 comidas y luego 1.182: el shift del cron encogió dos planes). Y la purga de cuentas del 09-11
se llevó la flota entera: hoy no queda ni uno de aquellos 96. Un corpus que vive en la base no es un corpus.

Este guion escribe el fichero que `culinary_baseline.py --corpus` mide: `days`, `_culinary_judge_history` y estado de
cada plan con días, más el catálogo del índice culinario, todo con huella (`culinary_corpus.py`).

    python scripts/congela_corpus_culinario.py --motivo "por qué se congela hoy"
        → scripts/data/culinary_corpus_<YYYY_MM_DD>.json (o --out)
    python scripts/congela_corpus_culinario.py --verificar scripts/data/culinary_corpus_2026_09_12.json
        → exit 0 si el fichero está íntegro, 3 si alguna huella no recalcula

Después:

    python scripts/culinary_baseline.py --corpus <fichero> --congelar    # línea base reproducible
    python scripts/culinary_baseline.py --corpus <fichero> --verificar   # ¿da hoy las mismas cifras?

Congelado el 2026-09-12: 5 planes con días, 100 comidas, catálogo de 349 filas (flota pequeña tras la purga; el
instrumento vale igual y se re-congela cuando haya flota — cada corpus lleva su fecha y su huella).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(_BACKEND))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)

from culinary_corpus import COLUMNAS_CATALOGO, cargar, congelar, verificar_integridad  # noqa: E402


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes: Any) -> None:
    print(*partes)


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_BACKEND),
                                       text=True, timeout=10).strip() or None
    except Exception:
        return None


def leer_flota(cur) -> tuple[list[dict], list[dict]]:
    """Sólo SELECTs: los planes con días y el catálogo del índice."""
    cur.execute(
        "SELECT id::text AS id, created_at, updated_at, revision, plan_data FROM meal_plans "
        "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') > 0 ORDER BY id"
    )
    filas = [dict(zip(("id", "created_at", "updated_at", "revision", "plan_data"), r)) for r in cur.fetchall()]
    cur.execute(f"SELECT {', '.join(COLUMNAS_CATALOGO)} FROM master_ingredients ORDER BY name")
    cat = [dict(zip(COLUMNAS_CATALOGO, r)) for r in cur.fetchall()]
    return filas, cat


def escribir(corpus: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--motivo", default="corpus fijo de la medición culinaria (C0)")
    ap.add_argument("--out", help="destino; por defecto scripts/data/culinary_corpus_<YYYY_MM_DD>.json")
    ap.add_argument("--verificar", metavar="FICHERO", help="recalcula las huellas de un corpus ya congelado")
    args = ap.parse_args(argv)

    if args.verificar:
        doc = json.loads(Path(args.verificar).read_text(encoding="utf-8"))
        fallos = verificar_integridad(doc)
        if fallos:
            _say("CORPUS NO ÍNTEGRO:")
            for f in fallos:
                _say("  -", f)
            return 3
        _say(f"íntegro: {doc.get('planes_n')} planes · {doc.get('comidas')} comidas · huella {doc.get('huella')} · "
             f"catálogo {doc.get('catalogo', {}).get('filas')} filas ({doc.get('catalogo', {}).get('huella')})")
        return 0

    from dotenv import load_dotenv  # noqa: E402 — el import vive aquí para que importar el módulo sea puro

    load_dotenv(_BACKEND / ".env")
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        _say("falta NEON_DATABASE_URL (backend/.env)")
        return 2
    import psycopg  # noqa: E402

    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            filas, cat = leer_flota(cur)
    ahora = datetime.now(timezone.utc)
    corpus = congelar(filas, cat, motivo=args.motivo, git_sha=_git_sha(), ahora=ahora)
    out = Path(args.out) if args.out else _BACKEND / "scripts" / "data" / f"culinary_corpus_{ahora:%Y_%m_%d}.json"
    escribir(corpus, out)
    # se relee y se verifica: lo que se congela es lo que se va a medir, no lo que había en memoria
    cargar(out)
    _say(f"corpus congelado en {out.relative_to(_BACKEND).as_posix()}: {corpus['planes_n']} planes · "
         f"{corpus['comidas']} comidas · catálogo {corpus['catalogo']['filas']} filas · huella {corpus['huella']} · "
         f"reglas {corpus['codigo']['reglas_huella']} · git {corpus['codigo']['git_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
