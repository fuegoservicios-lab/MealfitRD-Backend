"""[P1-DAYGEN-LUNA-CANARY · 2026-07-26] Lectura del A/B del canario de MODELO del day-gen.

Contraparte de lectura del canario: encenderlo sin esto no informa nada. Responde las dos
preguntas que deciden si el modelo caro se queda:

    ¿genera mejores días?   → menos reintentos, mejor banda, menos planes degradados
    ¿a qué precio?          → USD por llamada de day-gen, por modelo

## Cómo lee cada mitad, y por qué NO son la misma fuente

  · **calidad** sale de `pipeline_metrics.node='clinical_band'`, que lleva el tag
    `metadata->>'daygen_model_cohort'` y la columna `retries`.
  · **costo** sale de `llm_usage_events` filtrando `node='day_generator'` y agrupando por
    `model`. NO usa el tag: agrupa por el modelo que REALMENTE corrió. Un plan puede quedar en
    la cohorte 'on' y aun así generarse con DeepSeek — si el circuit breaker del canario está
    abierto, `_build_day_llm` cae al siguiente del chain. El tag dice a quién se le asignó; el
    modelo dice qué pasó. Cuando difieren, manda el modelo.

## Limitación real de la atribución (no es un descuido de este script)

`llm_usage_events` NO persiste `plan_id` ni `user_id` en los nodos de generación (medido
2026-07-26: 0 de 134 filas en 7 días los traen). Por eso **no existe costo por plan ni por
usuario**: el costo sólo se puede partir por modelo, y el "USD por plan" es una división entre
el total y el número de planes de la ventana, no una suma atribuida. Si algún día hace falta
costo por cohorte, hay que propagar `plan_id`/`user_id` al emit de `llm_usage_events` primero.

Uso:
    PYTHONPATH=backend python backend/scripts/daygen_canary_ab.py            # ventana 14 días
    PYTHONPATH=backend python backend/scripts/daygen_canary_ab.py --days 30
    PYTHONPATH=backend python backend/scripts/daygen_canary_ab.py --json

Read-only. Lee `NEON_DATABASE_URL` del `.env` del backend.
"""
# [P2-LOGGER-EXEMPT: CLI subcommand — la salida a stdout ES el producto del script]
import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Las filas SIN el tag son de antes del canario (se emite desde 2026-07-26). Etiquetarlas 'off'
# por COALESCE mezclaría cientos de planes de otra época con el grupo de control y haría ver una
# diferencia donde sólo hay cambio de calendario. Van aparte como 'sin-tag'.
# [P1-BAND-COHORT-ON-FINAL · 2026-07-26] DOS nodos, a propósito, porque ninguno tiene las dos
# cosas:
#   · `clinical_band`       → `retries` (el costo). Su `confidence` es PRE-finalize: en los dos
#                             primeros planes con Luna decía 0.833 sobre planes entregados en 1.00.
#   · `clinical_band_final` → la banda ENTREGADA (lo que el usuario recibe), pero `retries` es 0
#                             fijo porque se emite fuera del grafo.
# Leer la banda del primero era comparar cohortes con un número que nadie recibió.
_SQL_COSTO_REINTENTOS = """
SELECT
    COALESCE(metadata->>'daygen_model_cohort', 'sin-tag')                     AS cohorte,
    COUNT(*)                                                                  AS planes,
    -- [P1-BAND-METRIC-NO-SILENT-DROP · 2026-07-26] Las corridas sin banda calculable ahora SÍ
    -- emiten fila (antes se descartaban enteras y con ellas su `retries` — y la corrida perdida
    -- del 14:55 era justo una que reintentó, o sea sesgo NO aleatorio). Cuentan en el denominador
    -- de reintentos y se excluyen del promedio de banda.
    COUNT(*) FILTER (WHERE metadata->>'band_unavailable' = 'true')             AS sin_banda,
    AVG(CASE WHEN retries > 0 THEN 1.0 ELSE 0.0 END)                          AS tasa_reintento,
    AVG(retries::float)                                                       AS reintentos_medios,
    AVG(CASE WHEN metadata->>'review_passed' = 'false' THEN 1.0 ELSE 0.0 END) AS tasa_degradado
FROM pipeline_metrics
WHERE node = 'clinical_band'
  AND created_at > {_DESDE}
GROUP BY 1
ORDER BY 1
"""

_SQL_BANDA_ENTREGADA = """
SELECT
    COALESCE(metadata->>'daygen_model_cohort', 'sin-tag')                       AS cohorte,
    COUNT(*)                                                                    AS lecturas,
    AVG(confidence)                                                             AS band_entregada,
    AVG(CASE WHEN metadata->>'quality_degraded' = 'true' THEN 1.0 ELSE 0.0 END) AS tasa_degradado
FROM pipeline_metrics
WHERE node = 'clinical_band_final'
  AND created_at > {_DESDE}
  AND COALESCE(metadata->>'band_unavailable', 'false') <> 'true'
GROUP BY 1
ORDER BY 1
"""

_SQL_RAZONES = """
SELECT COALESCE(metadata->>'daygen_model_cohort', 'sin-tag') AS cohorte, r AS razon
FROM pipeline_metrics,
     -- `jsonb_typeof` y no `COALESCE`: cuando un plan no tuvo rechazos el emit guarda JSON `null`,
     -- que NO es SQL NULL — COALESCE lo dejaba pasar y `jsonb_array_elements_text` reventaba con
     -- "cannot extract elements from a scalar".
     LATERAL jsonb_array_elements_text(
         CASE WHEN jsonb_typeof(metadata->'rejection_reasons') = 'array'
              THEN metadata->'rejection_reasons' ELSE '[]'::jsonb END) r
WHERE node = 'clinical_band'
  AND created_at > {_DESDE}
"""

# Agrupa por el modelo que corrió de verdad, no por el tag (ver docstring).
_SQL_COSTO = """
SELECT
    model,
    COUNT(*)                                    AS llamadas,
    SUM(input_tokens)                           AS tokens_in,
    SUM(output_tokens)                          AS tokens_out,
    SUM(cost_usd_micros) / 1e6                  AS usd,
    -- [P1-DAYGEN-LUNA-CANARY] Un modelo que no está en `_DEFAULT_LLM_PRICING_MICROS_PER_M`
    -- persiste tokens pero NO costo. Sin esta columna, SUM(NULL)=NULL se imprimiría como
    -- 0.0000 y el modelo caro parecería gratis — el error más caro que podría cometer esta
    -- tabla. Se distingue "no cuesta nada" de "no sabemos cuánto cuesta".
    COUNT(cost_usd_micros)                      AS con_precio
FROM llm_usage_events
WHERE node = 'day_generator'
  AND created_at > {_DESDE}
GROUP BY 1
ORDER BY 5 DESC NULLS LAST
"""


def _fmt(v, pct=False, nd=3):
    if v is None:
        return "—"
    return f"{float(v) * 100:.1f}%" if pct else f"{float(v):.{nd}f}"


def _resumir(razon: str, ancho: int = 58) -> str:
    """Primera oración de la razón, recortada. Las razones son texto libre del revisor: no hay
    taxonomía canónica, así que se agrupan por su comienzo y las lee un humano."""
    r = " ".join(str(razon).split())
    for sep in (":", ".", " — "):
        if sep in r[:ancho + 12]:
            r = r.split(sep)[0]
            break
    return r[:ancho]


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice A/B del canario de modelo del day-gen")
    ap.add_argument("--days", type=int, default=14, help="ventana en días (default 14)")
    # [P1-CATALOG-VARIETY-OPENED · 2026-07-26] Sin este corte la comparación MIENTE cuando un fix
    # entra a mitad de la ventana: la cohorte 'on' quedó mezclando 3 planes anteriores al arreglo
    # del contrato de fruta con 1 posterior, y el % de reintentos de ese grupo no describía ninguna
    # de las dos configuraciones. `--since` recorta AMBAS cohortes al mismo momento.
    ap.add_argument("--since", default=None,
                    help="ISO 'YYYY-MM-DD HH:MM' — recorta ambas cohortes desde ahí (p.ej. el "
                         "arranque del binario con el fix). Ignora --days.")
    ap.add_argument("--json", action="store_true", help="salida JSON para tooling")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    dsn = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: falta NEON_DATABASE_URL en el entorno/.env", file=sys.stderr)
        return 2

    import psycopg

    def _q(sql):
        """Sustituye el marcador de ventana. Se hace aquí y no con un parámetro porque el default
        es una EXPRESIÓN SQL (`now() - interval`), no un valor."""
        lit = ("'" + d.replace("'", "''") + "'::timestamptz") if args.since else d
        return sql.replace("{_DESDE}", lit)
    # Un solo parámetro para las 4 consultas: instante de inicio de la ventana.
    if args.since:
        d = args.since
        _etiqueta = f"desde {args.since}"
    else:
        d = f"now() - interval '{int(args.days)} days'"
        _etiqueta = f"últimos {args.days} días"
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_q(_SQL_COSTO_REINTENTOS))
        calidad = [dict(zip(["cohorte", "planes", "sin_banda", "tasa_reintento",
                             "reintentos_medios", "tasa_degradado"], r)) for r in cur.fetchall()]
        cur.execute(_q(_SQL_BANDA_ENTREGADA))
        entregada = {r[0]: dict(zip(["lecturas", "band_entregada", "tasa_degradado"], r[1:]))
                     for r in cur.fetchall()}
        for c in calidad:
            c["band_entregada"] = (entregada.get(c["cohorte"]) or {}).get("band_entregada")
            c["lecturas_entregadas"] = (entregada.get(c["cohorte"]) or {}).get("lecturas")
        cur.execute(_q(_SQL_RAZONES))
        razones_raw = cur.fetchall()
        cur.execute(_q(_SQL_COSTO))
        costo = [dict(zip(["model", "llamadas", "tokens_in", "tokens_out", "usd", "con_precio"], r))
                 for r in cur.fetchall()]

    razones = {}
    for coh, raz in razones_raw:
        razones.setdefault(coh, Counter())[_resumir(raz)] += 1

    if args.json:
        print(json.dumps({"calidad": calidad, "costo": costo,
                          "razones": {k: v.most_common(8) for k, v in razones.items()}},
                         default=float, indent=2, ensure_ascii=False))
        return 0

    print(f"\nCanario de modelo · day generator — {_etiqueta}\n")
    print("CALIDAD por cohorte  (reintentos: node='clinical_band' · banda: node='clinical_band_final')\n")
    print(f"{'cohorte':<10}{'corridas':>10}{'sin banda':>11}{'band entr.':>12}{'% reint.':>10}"
          f"{'reint/plan':>12}{'degradados':>13}")
    print("-" * 78)
    for c in calidad:
        print(f"{c['cohorte']:<10}{c['planes']:>10}{c.get('sin_banda') or 0:>11}"
              f"{_fmt(c.get('band_entregada')):>12}"
              f"{_fmt(c['tasa_reintento'], pct=True):>10}{_fmt(c['reintentos_medios'], nd=2):>12}"
              f"{_fmt(c['tasa_degradado'], pct=True):>13}")
    print("\n'corridas' = corridas del PIPELINE, no planes persistidos: una regeneración sobreescribe\n"
          "el plan pero es una corrida más, y es el denominador correcto para una tasa de reintentos.\n"
          "Comparar estas filas contra `meal_plans` da un '362% de cobertura' que no significa nada.\n"
          "'sin banda' = corridas donde el score no se pudo calcular: cuentan para reintentos y NO\n"
          "para el promedio de banda (antes se descartaba la fila entera y se perdía la corrida).\n"
          "'band entr.' = la banda que el usuario RECIBIÓ. La de `clinical_band` es pre-finalize y\n"
          "en los 2 primeros planes con Luna marcaba 0.833 sobre planes entregados en 1.00.")

    print("\n\nCOSTO (llm_usage_events, node='day_generator', por modelo que CORRIÓ)\n")
    print(f"{'modelo':<22}{'llamadas':>10}{'tok in':>12}{'tok out':>10}{'USD':>10}{'USD/call':>11}")
    print("-" * 75)
    total_usd = 0.0
    sin_precio = []
    for c in costo:
        n = c["llamadas"] or 0
        if not c.get("con_precio"):
            sin_precio.append((c["model"], n, c["tokens_in"] or 0, c["tokens_out"] or 0))
            print(f"{c['model']:<22}{n:>10}{c['tokens_in'] or 0:>12,}"
                  f"{c['tokens_out'] or 0:>10,}{'sin precio':>10}{'—':>11}")
            continue
        usd = float(c["usd"] or 0)
        total_usd += usd
        print(f"{c['model']:<22}{n:>10}{c['tokens_in'] or 0:>12,}"
              f"{c['tokens_out'] or 0:>10,}{usd:>10.4f}{usd / n if n else 0:>11.5f}")
    print(f"{'TOTAL con precio':<22}{'':>10}{'':>12}{'':>10}{total_usd:>10.4f}")
    for mdl, n, ti, to in sin_precio:
        print(f"\n  ⚠ '{mdl}' no está en la tabla de precios: {n} llamadas con {ti:,} tokens de "
              f"entrada y {to:,} de salida quedaron SIN costo.\n"
              f"    Los tokens sí están guardados, así que el costo se puede backfillar. Para que "
              f"se registre de aquí en adelante:\n"
              f"    MEALFIT_LLM_PRICING_JSON='{{\"{mdl}\": {{\"input\": <micros/M>, "
              f"\"output\": <micros/M>}}}}'  (USD/1M × 1e6)")

    if razones:
        print("\n\nPOR QUÉ SE REINTENTÓ (razones acumuladas, top por cohorte)\n")
        for coh in sorted(razones):
            print(f"  [{coh}]")
            for raz, n in razones[coh].most_common(6):
                print(f"    {n:>3}×  {raz}")
            print()

    reales = [c for c in calidad if c["cohorte"] in ("on", "off")]
    if any(c["cohorte"] == "sin-tag" for c in calidad):
        print("'sin-tag' = planes anteriores al canario (el campo se emite desde 2026-07-26); "
              "no son grupo de control.")
    if len(reales) < 2:
        print("\nSolo una cohorte en la ventana. Para un A/B real hace falta "
              "MEALFIT_DAYGEN_CANARY_MODEL puesto y MEALFIT_DAYGEN_CANARY_PCT entre 1 y 99.")
    else:
        on = next(c for c in reales if c["cohorte"] == "on")
        off = next(c for c in reales if c["cohorte"] == "off")
        d_re = (float(on["tasa_reintento"]) - float(off["tasa_reintento"])) * 100
        d_bd = float(on.get("band_entregada") or 0) - float(off.get("band_entregada") or 0)
        print(f"\nDelta reintentos (on − off): {d_re:+.1f} puntos  ·  delta banda: {d_bd:+.3f}")
        print("Negativo en reintentos y positivo en banda = el modelo caro se está ganando el sueldo.")
        print("Ojo con la n: por debajo de ~30 planes por cohorte el ruido domina.")
        if not any("gpt" in (c["model"] or "") for c in costo):
            print("\n⚠ Hay cohorte 'on' pero NINGUNA llamada al modelo del canario en la ventana: "
                  "el chain cayó al fallback (circuit breaker abierto o error del proveedor). "
                  "El A/B de calidad NO está midiendo el modelo — revisa los logs del day-gen.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
