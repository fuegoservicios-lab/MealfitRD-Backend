"""[P1-HARDEN-POOLS-CANARY-GATING · 2026-07-24] Lectura del A/B del canario A1 (pool hardening).

El canario reparte a los usuarios en cohortes 'on'/'off' con un bucket sha256 estable y la métrica
`clinical_band` de `pipeline_metrics` lleva el tag `metadata->>'harden_pools_cohort'`. Este script
hace el slice: es la contraparte de lectura sin la cual encender el canario no informa nada.

Qué compara, y por qué esas tres columnas:
  - `mecanismo`  — `same_day_protein_repeats`: ¿la restricción dura ELIMINÓ la clase? Si no baja,
                   el enforcer no está haciendo lo que dice y el resto del análisis sobra.
  - `costo`      — % de planes con `retries > 0`: cada rechazo del revisor = regenerar día(s) +
                   otra pasada. El forensic corr=d57ffe04 midió que el 49% de los rechazos en 72h
                   eran "misma proteína repetida el mismo día"; esta columna dice si eso cayó.
  - `calidad`    — band score medio y % de planes entregados degradados: la restricción dura
                   estrecha la canasta, así que hay que confirmar que no paga el precio en precisión.

Uso:
    PYTHONPATH=backend python backend/scripts/harden_pools_ab.py            # ventana 14 días
    PYTHONPATH=backend python backend/scripts/harden_pools_ab.py --days 30
    PYTHONPATH=backend python backend/scripts/harden_pools_ab.py --json

Read-only (un solo SELECT). Lee `NEON_DATABASE_URL` del `.env` del backend.
"""
# [P2-LOGGER-EXEMPT: CLI subcommand — la salida a stdout ES el producto del script]
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Las filas SIN el tag son histórico anterior al canario (el campo se emite desde 2026-07-09):
# etiquetarlas 'on' por COALESCE mezclaría 302 planes pre-canario con el grupo de tratamiento y
# haría ver una diferencia donde solo hay cambio de época. Se reportan aparte como 'sin-tag'.
_SQL = """
SELECT
    COALESCE(metadata->>'harden_pools_cohort', 'sin-tag')        AS cohorte,
    COUNT(*)                                                     AS planes,
    AVG(confidence)                                              AS band_medio,
    AVG(CASE WHEN retries > 0 THEN 1.0 ELSE 0.0 END)             AS tasa_rechazo,
    AVG((metadata->>'same_day_protein_repeats')::float)          AS repeticiones_mismo_dia,
    AVG(CASE WHEN metadata->>'review_passed' = 'false' THEN 1.0 ELSE 0.0 END) AS tasa_degradado
FROM pipeline_metrics
WHERE node = 'clinical_band'
  AND created_at > NOW() - (%s || ' days')::interval
GROUP BY 1
ORDER BY 1
"""


def _fmt(v, pct=False, nd=3):
    if v is None:
        return "—"
    return f"{float(v) * 100:.1f}%" if pct else f"{float(v):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice A/B del canario A1 (pool hardening)")
    ap.add_argument("--days", type=int, default=14, help="ventana en días (default 14)")
    ap.add_argument("--json", action="store_true", help="salida JSON para tooling")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    dsn = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: falta NEON_DATABASE_URL en el entorno/.env", file=sys.stderr)
        return 2

    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_SQL, (str(args.days),))
        rows = cur.fetchall()

    cols = ["cohorte", "planes", "band_medio", "tasa_rechazo", "repeticiones_mismo_dia", "tasa_degradado"]
    data = [dict(zip(cols, r)) for r in rows]

    if args.json:
        print(json.dumps(data, default=float, indent=2, ensure_ascii=False))
        return 0

    print(f"\nCanario A1 · pool hardening — últimos {args.days} días (node=clinical_band)\n")
    print(f"{'cohorte':<10}{'planes':>8}{'band':>9}{'rechazos':>11}{'rep/día':>10}{'degradados':>13}")
    print("-" * 61)
    for d in data:
        print(f"{d['cohorte']:<10}{d['planes']:>8}{_fmt(d['band_medio']):>9}"
              f"{_fmt(d['tasa_rechazo'], pct=True):>11}{_fmt(d['repeticiones_mismo_dia'], nd=2):>10}"
              f"{_fmt(d['tasa_degradado'], pct=True):>13}")

    cohortes_reales = [d for d in data if d["cohorte"] in ("on", "off")]
    if any(d["cohorte"] == "sin-tag" for d in data):
        print("\n'sin-tag' = planes anteriores al canario (el campo se emite desde 2026-07-09); "
              "no son grupo de control.")
    if len(cohortes_reales) < 2:
        print("\nSolo una cohorte en la ventana. Para un A/B real hace falta "
              "MEALFIT_HARDEN_POOLS_CANARY_PCT entre 1 y 99 (es el % que queda en CONTROL).")
    else:
        on = next((d for d in data if d["cohorte"] == "on"), None)
        off = next((d for d in data if d["cohorte"] == "off"), None)
        if on and off and on["planes"] and off["planes"]:
            d_rech = (float(on["tasa_rechazo"]) - float(off["tasa_rechazo"])) * 100
            print(f"\nDelta de rechazos (on − off): {d_rech:+.1f} puntos. "
                  f"Negativo = el endurecimiento evita rechazos.")
            print("Ojo con la n: por debajo de ~30 planes por cohorte el ruido domina.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
