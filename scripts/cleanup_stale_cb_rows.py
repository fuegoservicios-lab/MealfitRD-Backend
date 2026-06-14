"""[P2-CB-FOSSIL · 2026-05-26] One-shot manual para eliminar rows fósiles de
`llm_circuit_breaker:*` en `app_kv_store` que pertenecen a modelos deprecados
o renombrados.

Contexto:
    El cron `_sweep_stale_llm_circuit_breakers` (P2-NEW-D · 2026-05-11) RESETEA
    rows stale `(failures, last_failure) → (0, 0)` pero NO las borra (porque
    una row en zero-canonical es estado VÁLIDO post-reset). El sweep está bien
    diseñado para CBs vivos.

    Modos de fallo cubiertos por el sweep:
      - CB stuck in is_open=true → resetea.
      - CB con last_failure viejo + failures=0 → no-op (correcto).

    Modo NO cubierto: rows de modelos que fueron renombrados o deprecados.
    Caso real (audit 2026-05-26):
        `llm_circuit_breaker:gemini-3.1-pro-preview` (preview model deprecado
        por Google) — row stale 14 días en zero canonical. Sweep la ignora (no
        está abierta), pero el modelo ya NO está en `_KNOBS_REGISTRY`. Es un
        fósil. Acumula linealmente con cada migración de modelo.

    Por qué NO añadirlo al cron automatizado:
      - El criterio "modelo deprecado" no es derivable solo del KV. Requiere
        comparar contra `MEALFIT_*_MODEL` env vars + defaults hardcoded en
        `_<feature>_model_name()` (graph_orchestrator.py).
      - Auto-DELETE incorrecta puede destruir state de un breaker que mañana
        vuelve a activarse cuando el modelo se re-introduzca (e.g. A/B test).
      - El operador es el único que sabe "estos modelos quedan deprecados".

Uso operacional:

    # Preview (default, NO modifica DB):
    py backend/scripts/cleanup_stale_cb_rows.py --min-age-days 7

    # Solo modelos específicos (preview):
    py backend/scripts/cleanup_stale_cb_rows.py --models gemini-3.1-pro-preview,gemini-2.5-pro

    # APLICAR DELETE (irreversible):
    py backend/scripts/cleanup_stale_cb_rows.py --apply --models gemini-3.1-pro-preview

Argumentos:
    --min-age-days N    Solo considera rows con `updated_at < NOW() - N days`.
                        Default 7. Defensa contra borrar rows recién reset.
    --models LIST       CSV de model IDs específicos a borrar (e.g.
                        `gemini-3.1-pro-preview,gemini-2.5-flash`). Si omites,
                        el script lista TODOS los `llm_circuit_breaker:*` en
                        zero canonical stale para que decidas manualmente.
    --apply             Aplica DELETE. Sin esta flag, solo preview (SELECT).
    --connection-string Override de NEON_DATABASE_URL.

Defensa-en-profundidad:
    - Solo borra rows con `failures=0 AND last_failure=0` (zero canonical, NO
      abierta).
    - Solo borra rows con `updated_at < NOW() - N days` (NO recién reset).
    - Si --models especificado, solo borra esos. Sin --models, lista pero NO
      borra (operador debe seleccionar explícitamente).
    - Logs cada DELETE individualmente para audit trail.

Tooltip-anchor: P2-CB-FOSSIL.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Hacer backend/ importable para usar el connection_pool del repo.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _connection_string(override: str | None) -> str | None:
    return override or os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-age-days", type=int, default=7,
                        help="Edad mínima en días (default 7).")
    parser.add_argument("--models", type=str, default=None,
                        help="CSV de model IDs a borrar (e.g. "
                             "`gemini-3.1-pro-preview,foo-bar`).")
    parser.add_argument("--apply", action="store_true",
                        help="Aplica DELETE. Sin esta flag, solo preview.")
    parser.add_argument("--connection-string", type=str, default=None,
                        help="Override de NEON_DATABASE_URL.")
    args = parser.parse_args(argv)

    conn_str = _connection_string(args.connection_string)
    if not conn_str:
        print(
            "[P2-CB-FOSSIL] ERROR: NEON_DATABASE_URL no configurada y "
            "--connection-string no provisto.",
            file=sys.stderr,
        )
        return 2

    try:
        import psycopg
    except ImportError as e:
        print(
            f"[P2-CB-FOSSIL] ERROR: psycopg no disponible ({e}). "
            f"Instala con `pip install psycopg[binary]` o asegúrate "
            f"de correr desde el venv del backend.",
            file=sys.stderr,
        )
        return 2

    selected_models = None
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Step 1: SELECT preview de candidates.
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    key,
                    value,
                    EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0 AS age_days,
                    updated_at
                FROM app_kv_store
                WHERE key LIKE 'llm_circuit_breaker:%%'
                  AND updated_at < NOW() - make_interval(days => %s)
                  AND (value->>'failures')::int = 0
                  AND (value->>'last_failure')::float = 0
                  AND COALESCE((value->>'is_open')::bool, false) = false
                ORDER BY updated_at ASC
                """,
                (int(args.min_age_days),),
            )
            rows = cur.fetchall()

            print(
                f"[P2-CB-FOSSIL] Candidates ({len(rows)} rows, "
                f"min_age_days={args.min_age_days}):"
            )
            for key, value, age_days, updated_at in rows:
                key_str = str(key)
                # Extract model_id del key `llm_circuit_breaker:<model>`.
                model_id = key_str.split(":", 1)[1] if ":" in key_str else "(legacy)"
                marker = ""
                if selected_models and model_id in selected_models:
                    marker = "  ← seleccionado"
                print(
                    f"  - {key_str}  age={float(age_days):.1f}d  "
                    f"updated={updated_at.isoformat()}{marker}"
                )

            if not rows:
                print(
                    "[P2-CB-FOSSIL] OK: 0 candidates. "
                    "Sweep de `_sweep_stale_llm_circuit_breakers` los está "
                    "reseteando correctamente."
                )
                return 0

            # Step 2: DELETE si --apply y --models están especificados.
            if not args.apply:
                print(
                    "\n[P2-CB-FOSSIL] Preview (no DELETE aplicado). "
                    "Re-corre con `--apply --models <id1>,<id2>` para borrar."
                )
                return 0

            if not selected_models:
                print(
                    "[P2-CB-FOSSIL] ERROR: --apply requiere --models. "
                    "Sin --models no borramos en masa (defensa).",
                    file=sys.stderr,
                )
                return 2

            keys_to_delete = [
                f"llm_circuit_breaker:{m}" for m in selected_models
            ]
            cur.execute(
                """
                DELETE FROM app_kv_store
                WHERE key = ANY(%s)
                  AND updated_at < NOW() - make_interval(days => %s)
                  AND (value->>'failures')::int = 0
                  AND (value->>'last_failure')::float = 0
                  AND COALESCE((value->>'is_open')::bool, false) = false
                RETURNING key
                """,
                (keys_to_delete, int(args.min_age_days)),
            )
            deleted = cur.fetchall()
            conn.commit()

            print(f"\n[P2-CB-FOSSIL] DELETE aplicado: {len(deleted)} rows.")
            for (key,) in deleted:
                print(f"  - {key}")

            not_deleted = set(keys_to_delete) - {k for (k,) in deleted}
            if not_deleted:
                print(
                    f"\n[P2-CB-FOSSIL] {len(not_deleted)} key(s) NO borradas "
                    f"(no matcheaban filtros zero-canonical o age):"
                )
                for k in sorted(not_deleted):
                    print(f"  - {k}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
