"""[P2-SYNTHETIC-CORPUS · 2026-07-04] Genera planes SINTÉTICOS de calidad para
el corpus de entrenamiento (`ai_training_corpus`, source='synthetic').

ROL (y límites — ver COMMENT de la tabla):
  • SÍ: probar/construir el pipeline del corpus de punta a punta y servir como
    set de EVALUACIÓN reproducible.
  • NO: dieta principal de entrenamiento. Entrenar solo con salidas del LLM
    actual = destilación circular (se heredan sus sesgos). El oro es
    source='user' con consentimiento (P2-AI-TRAINING-CONSENT).

GATE DE CALIDAD (solo entran planes "de calidad"):
  - el revisor NO lo rechazó (`_review_failed_but_delivered` falsy — contrato
    P3-NEW-8 del plan_result final; `review_passed` vive solo en el state),
  - NO fallback matemático (`_is_fallback`),
  - los 4 macros (kcal/P/C/F) entregados dentro de ±band del target (default 10%),
  - >= 3 días (el pipeline síncrono produce el chunk inicial de 3 días; los días
    4-7 llegan por chunk workers atados a un plan persistido — fuera del alcance
    de este generador. Cada día ya es un ejemplo válido perfil→menú).

NOTA (estado 2026-07-04): bajo el override GEMINI-TEMP el revisor puede no
aprobar (flash-lite es débil como reviewer) → el gate descarta. Correr el batch
con DeepSeek (saldo) para obtener review_passed reales.

Perfiles: reusa el set FIJO y diverso del benchmark (M2-MACRO-BENCHMARK,
scripts/benchmark_macro_compliance.py) — 20 perfiles held-out no cherry-picked
(género × edad × objetivo × actividad × condiciones).

Uso (desde backend/, con el venv mealfit):
    python scripts/generate_synthetic_corpus.py --limit 2            # dry-run (NO escribe)
    python scripts/generate_synthetic_corpus.py --commit             # genera 20 e inserta los que pasan el gate
    python scripts/generate_synthetic_corpus.py --export-jsonl corpus.jsonl   # dump del corpus a JSONL

Costo: cada perfil corre el pipeline LLM completo (~decenas de calls). Correr
cuando el provider tenga capacidad (DeepSeek con saldo / Gemini con cuota).

# [P2-LOGGER-EXEMPT: script CLI one-shot — salida humana a stdout]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

BANDS_DEFAULT = 0.10
MACROS = ("kcal", "protein", "carbs", "fats")


def _num(x) -> float:
    try:
        s = str(x).lower().replace("kcal", "").replace("g", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


def _engine_version() -> str:
    """Marker del árbol al momento de generar (sin importar app.py, que es pesado)."""
    try:
        src = (_BACKEND / "app.py").read_text(encoding="utf-8")
        m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def _quality(result: dict, band: float) -> dict:
    """Métricas de calidad del plan entregado vs su propio target (patrón benchmark)."""
    macros = result.get("macros") or {}
    target = {
        "kcal": _num(result.get("calories")),
        "protein": _num(macros.get("protein")),
        "carbs": _num(macros.get("carbs")),
        "fats": _num(macros.get("fats")),
    }
    days = result.get("days") or []
    worst_dev = 0.0
    for d in days:
        delivered = {k: 0.0 for k in MACROS}
        # Mismo acceso que el benchmark: macros al TOPE de la comida (no anidados);
        # kcal vive en `cals` (o `calories` legacy).
        for meal in (d.get("meals") or []):
            delivered["kcal"] += _num(meal.get("cals") if meal.get("cals") is not None else meal.get("calories"))
            delivered["protein"] += _num(meal.get("protein"))
            delivered["carbs"] += _num(meal.get("carbs"))
            delivered["fats"] += _num(meal.get("fats"))
        for k in MACROS:
            if target[k] > 0:
                worst_dev = max(worst_dev, abs(delivered[k] - target[k]) / target[k])
    # Contrato del plan_result final (P3-NEW-8): el resultado NO trae
    # `review_passed`; trae `_review_failed_but_delivered` (True = el revisor
    # lo rechazó pero se entregó igual) y `_review_severity`. Review OK ⟺ el
    # flag es falsy.
    review_failed = bool(result.get("_review_failed_but_delivered"))
    is_fallback = bool(result.get("_is_fallback"))
    passes = (
        not review_failed
        and not is_fallback
        and len(days) >= 3
        and worst_dev <= band
    )
    return {
        "passes_gate": passes,
        "band": band,
        "worst_macro_deviation": round(worst_dev, 4),
        "review_failed_but_delivered": review_failed,
        "review_severity": result.get("_review_severity"),
        "is_fallback": is_fallback,
        "days": len(days),
    }


def _strip_internals(plan: dict) -> dict:
    """Copia del plan sin claves internas de pipeline (prefijo '_')."""
    return {k: v for k, v in plan.items() if not str(k).startswith("_")}


def _open_pools_sync() -> None:
    """Abre el pool sync de Neon (patrón benchmark P0-CLINICAL-VALIDATION:
    los pools nacen open=False fuera del lifespan de FastAPI). Idempotente."""
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()


async def _generate(limit: int | None, band: float, commit: bool) -> None:
    # Imports pesados DENTRO (permite --export-jsonl sin cargar el orquestador).
    from graph_orchestrator import arun_plan_pipeline
    from scripts.benchmark_macro_compliance import PROFILES

    # Sin pools abiertos, el pipeline no lee master_ingredients y degrada a
    # fallback (que el gate descarta) — mismo bug latente que arregló el
    # benchmark. Idempotente.
    import db_core
    _open_pools_sync()
    if getattr(db_core, "async_connection_pool", None):
        await db_core.async_connection_pool.open()
    await asyncio.sleep(1.5)

    profiles = PROFILES[:limit] if limit else PROFILES
    engine = _engine_version()
    try:
        from llm_provider import resolve_model_for_user
        model_id = resolve_model_for_user(None)
    except Exception:
        model_id = "unknown"

    print(f"Generando {len(profiles)} planes sinteticos (engine={engine}, model={model_id}, "
          f"band=+/-{band:.0%}, commit={commit})")

    sem = asyncio.Semaphore(3)
    inserted = 0
    gated_out = 0
    errors = 0

    async def _one(profile: dict):
        nonlocal inserted, gated_out, errors
        async with sem:
            pid = profile["_id"]
            fd = {k: v for k, v in profile.items() if k != "_id"}
            try:
                result = await arun_plan_pipeline(dict(fd))
            except Exception as e:
                errors += 1
                print(f"  [{pid:>2}] ERROR {type(e).__name__}: {e}")
                return
            q = _quality(result, band)
            # Prints ASCII-only: la consola Windows (cp1252) no soporta flechas.
            verdict = "OK -> corpus" if q["passes_gate"] else "descartado (gate)"
            print(f"  [{pid:>2}] dev={q['worst_macro_deviation']:.1%} "
                  f"review_failed={q['review_failed_but_delivered']} "
                  f"fallback={q['is_fallback']} days={q['days']} -> {verdict}")
            if not q["passes_gate"]:
                gated_out += 1
                return
            if commit:
                from db import execute_sql_write
                execute_sql_write(
                    """
                    INSERT INTO ai_training_corpus
                        (source, user_id, profile, plan, quality, engine_version, model_id)
                    VALUES ('synthetic', NULL, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        json.dumps(fd, ensure_ascii=False),
                        json.dumps(_strip_internals(result), ensure_ascii=False, default=str),
                        json.dumps(q, ensure_ascii=False),
                        engine,
                        model_id,
                    ),
                )
            inserted += 1

    await asyncio.gather(*[_one(p) for p in profiles])
    action = "insertados" if commit else "pasarían el gate (dry-run, nada escrito — usa --commit)"
    print(f"\nResumen: {inserted} {action} | {gated_out} descartados por gate | {errors} errores")


def _export_jsonl(path: str) -> None:
    from db import execute_sql_query
    _open_pools_sync()
    rows = execute_sql_query(
        "SELECT id, source, profile, plan, quality, engine_version, model_id, created_at "
        "FROM ai_training_corpus ORDER BY created_at",
        fetch_all=True,
    ) or []
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"Exportadas {len(rows)} filas del corpus a {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="máx perfiles a generar")
    parser.add_argument("--band", type=float, default=BANDS_DEFAULT, help="banda de macros del gate (0.10 = ±10%%)")
    parser.add_argument("--commit", action="store_true", help="insertar en DB (default: dry-run)")
    parser.add_argument("--export-jsonl", metavar="PATH", default=None, help="exportar el corpus existente a JSONL y salir")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")

    if args.export_jsonl:
        _export_jsonl(args.export_jsonl)
        return
    asyncio.run(_generate(args.limit, args.band, args.commit))


if __name__ == "__main__":
    main()
