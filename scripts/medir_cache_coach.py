# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-997 / P1-PLAN-LOTE-999] Mide uso de caché implícito, turnos y latencia del coach en producción.

Uso:
    python scripts/medir_cache_coach.py [horas=24] [--json]
    python scripts/medir_cache_coach.py --horas 48 --json

Vistas:
    1) Resumen diario por modelo (`llamadas`, `in medio`, `%cacheado`, `out medio`, `s medios`, `USD`).
    2) Por turno (`metadata->>'corr'`): costo total, 1.ª vs 2.ª llamada del turno, `reasoning_tokens`
       y desglose compacto si `metadata->'prompt_breakdown'` está presente.
    3) 1.ª llamada del turno según segundos desde la llamada anterior de la misma sesión/modelo
       (`sin_previa`, `<=30s`, `31-60s`, `61-180s`, `181-300s`, `>300s`).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - entorno sin python-dotenv
    load_dotenv = None

_SQL_EVENTOS = """
SELECT
  id::text AS id,
  created_at,
  user_id::text AS user_id,
  model,
  COALESCE(input_tokens, 0) AS input_tokens,
  COALESCE(cached_tokens, 0) AS cached_tokens,
  COALESCE(output_tokens, 0) AS output_tokens,
  COALESCE(cost_usd_micros, 0) AS cost_usd_micros,
  metadata
FROM llm_usage_events
WHERE node = 'chat_call_model'
  AND created_at >= NOW() - (%s || ' hours')::interval
ORDER BY created_at ASC, id ASC
"""

_BUCKET_ORDER: tuple[str, ...] = (
    "sin_previa",
    "<=30s",
    "31-60s",
    "61-180s",
    "181-300s",
    ">300s",
)


def _cargar_env_fallback() -> None:
    """Carga `.env` del backend si las variables Neon aún no están en `os.environ`."""
    if os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL"):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(env_path, override=False)
        if os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL"):
            return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in ("NEON_DATABASE_URL_POOLED", "NEON_DATABASE_URL") and val and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


def _resolve_neon_db_url() -> str:
    """Devuelve la URL Neon (prioriza pooler) sin caer en la antigua `DATABASE_URL`."""
    _cargar_env_fallback()
    return (
        os.environ.get("NEON_DATABASE_URL_POOLED")
        or os.environ.get("NEON_DATABASE_URL")
        or ""
    ).strip()


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _to_datetime(val: Any) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str) and val.strip():
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _pct(num: int | float, den: int | float) -> float:
    if not den or den <= 0:
        return 0.0
    return round(100.0 * float(num) / float(den), 1)


def _format_compact_breakdown(pb: Any) -> str:
    if not isinstance(pb, dict) or not pb:
        return ""
    parts: list[str] = []
    tot = _safe_int(pb.get("total_sys_chars"))
    pref = _safe_int(pb.get("prefijo_estatico"))
    if tot:
        parts.append(f"sys={tot / 1000.0:.1f}k(pref={pref / 1000.0:.1f}k)")
    hm = _safe_int(pb.get("historial_msgs"))
    hc = _safe_int(pb.get("historial_chars"))
    if hm or hc:
        parts.append(f"hist={hm}m/{hc / 1000.0:.1f}k")
    tools_n = pb.get("tools_n")
    if tools_n is not None:
        parts.append(f"tools={tools_n}")
    lean = _safe_int(pb.get("lean_ahorro"))
    if lean:
        parts.append(f"lean=-{lean / 1000.0:.1f}k")
    return " ".join(parts)


def resumir_diario_modelos(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrupa eventos por día UTC y modelo (misma métrica base que LOTE-997)."""
    grupos: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dt = _to_datetime(row.get("created_at"))
        fecha = dt.astimezone(timezone.utc).date().isoformat() if dt else "sin_fecha"
        model = str(row.get("model") or "desconocido")
        key = (fecha, model)
        g = grupos.setdefault(
            key,
            {
                "fecha": fecha,
                "model": model,
                "llamadas": 0,
                "sum_in": 0,
                "sum_cached": 0,
                "sum_out": 0,
                "durs": [],
                "sum_micros": 0,
            },
        )
        g["llamadas"] += 1
        g["sum_in"] += _safe_int(row.get("input_tokens"))
        g["sum_cached"] += _safe_int(row.get("cached_tokens"))
        g["sum_out"] += _safe_int(row.get("output_tokens"))
        g["sum_micros"] += _safe_int(row.get("cost_usd_micros"))
        meta = _parse_metadata(row.get("metadata"))
        dur = _safe_float(meta.get("duration_s"))
        if dur is not None:
            g["durs"].append(dur)

    out: list[dict[str, Any]] = []
    for (_, _), g in sorted(grupos.items(), key=lambda kv: (kv[0][0], -kv[1]["llamadas"], kv[0][1]), reverse=True):
        n = g["llamadas"]
        avg_dur = round(sum(g["durs"]) / len(g["durs"]), 2) if g["durs"] else None
        out.append(
            {
                "fecha": g["fecha"],
                "model": g["model"],
                "llamadas": n,
                "in_medio": int(round(g["sum_in"] / n)) if n else 0,
                "pct_cache": _pct(g["sum_cached"], g["sum_in"]),
                "out_medio": int(round(g["sum_out"] / n)) if n else 0,
                "dur_s_medio": avg_dur,
                "usd": round(g["sum_micros"] / 1_000_000.0, 4),
            }
        )
    return out


def _call_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    in_tok = _safe_int(row.get("input_tokens"))
    cached_tok = _safe_int(row.get("cached_tokens"))
    out_tok = _safe_int(row.get("output_tokens"))
    micros = _safe_int(row.get("cost_usd_micros"))
    meta = _parse_metadata(row.get("metadata"))
    dur = _safe_float(meta.get("duration_s"))
    return {
        "input_tokens": in_tok,
        "cached_tokens": cached_tok,
        "pct_cache": _pct(cached_tok, in_tok),
        "output_tokens": out_tok,
        "dur_s": round(dur, 2) if dur is not None else None,
        "usd": round(micros / 1_000_000.0, 6),
        "reasoning_tokens": _safe_int(meta.get("reasoning_tokens")),
    }


def _agg_call_slot(calls: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not calls:
        return None
    n = len(calls)
    sum_in = sum(c["input_tokens"] for c in calls)
    sum_cached = sum(c["cached_tokens"] for c in calls)
    sum_out = sum(c["output_tokens"] for c in calls)
    sum_usd = sum(c["usd"] for c in calls)
    durs = [c["dur_s"] for c in calls if c["dur_s"] is not None]
    return {
        "n": n,
        "in_medio": int(round(sum_in / n)),
        "pct_cache": _pct(sum_cached, sum_in),
        "out_medio": int(round(sum_out / n)),
        "dur_s_medio": round(sum(durs) / len(durs), 2) if durs else None,
        "usd_medio": round(sum_usd / n, 6),
    }


def resumir_por_turno(rows: Iterable[dict[str, Any]], *, max_turnos: int = 20) -> dict[str, Any]:
    """Agrupa por `(corr, model)` para comparar 1.ª frente a 2.ª llamada del turno y reasoning tokens."""
    ordered = sorted(
        rows,
        key=lambda r: (_to_datetime(r.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), str(r.get("id") or "")),
    )
    por_corr: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sin_corr = 0
    for row in ordered:
        meta = _parse_metadata(row.get("metadata"))
        corr = str(meta.get("corr") or "").strip()
        if not corr:
            sin_corr += 1
            continue
        model = str(row.get("model") or "desconocido")
        por_corr[(corr, model)].append(row)

    turnos: list[dict[str, Any]] = []
    por_modelo_raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"call1": [], "call2": [], "usd_turnos": [], "reasoning": [], "turnos_con_2": 0}
    )

    for (corr, model), calls in por_corr.items():
        first_dt = _to_datetime(calls[0].get("created_at"))
        c1 = _call_snapshot(calls[0])
        c2 = _call_snapshot(calls[1]) if len(calls) >= 2 else None
        usd_total = round(sum(_safe_int(c.get("cost_usd_micros")) for c in calls) / 1_000_000.0, 6)
        reasoning_total = sum(_safe_int(_parse_metadata(c.get("metadata")).get("reasoning_tokens")) for c in calls)
        desglose = ""
        for c in calls:
            pb = _parse_metadata(c.get("metadata")).get("prompt_breakdown")
            if isinstance(pb, dict) and pb:
                desglose = _format_compact_breakdown(pb)
                break

        m_bucket = por_modelo_raw[model]
        m_bucket["call1"].append(c1)
        if c2 is not None:
            m_bucket["call2"].append(c2)
            m_bucket["turnos_con_2"] += 1
        m_bucket["usd_turnos"].append(usd_total)
        m_bucket["reasoning"].append(reasoning_total)

        turnos.append(
            {
                "created_at": first_dt.astimezone(timezone.utc).isoformat() if first_dt else "",
                "corr": corr,
                "model": model,
                "llamadas": len(calls),
                "usd_total": usd_total,
                "reasoning_tokens": reasoning_total,
                "call1": c1,
                "call2": c2,
                "desglose": desglose,
            }
        )

    turnos.sort(key=lambda t: t["created_at"], reverse=True)
    por_modelo: list[dict[str, Any]] = []
    for model, data in sorted(por_modelo_raw.items(), key=lambda kv: len(kv[1]["usd_turnos"]), reverse=True):
        n_t = len(data["usd_turnos"])
        por_modelo.append(
            {
                "model": model,
                "turnos": n_t,
                "turnos_con_2_llamadas": data["turnos_con_2"],
                "usd_medio_turno": round(sum(data["usd_turnos"]) / n_t, 6) if n_t else 0.0,
                "reasoning_medio_turno": int(round(sum(data["reasoning"]) / n_t)) if n_t else 0,
                "call1": _agg_call_slot(data["call1"]),
                "call2": _agg_call_slot(data["call2"]),
            }
        )

    return {
        "por_modelo": por_modelo,
        "turnos": turnos[:max_turnos],
        "turnos_totales": len(turnos),
        "llamadas_sin_corr": sin_corr,
    }


def _bucket_segundos(gap_s: Optional[float]) -> str:
    if gap_s is None:
        return "sin_previa"
    if gap_s <= 30.0:
        return "<=30s"
    if gap_s <= 60.0:
        return "31-60s"
    if gap_s <= 180.0:
        return "61-180s"
    if gap_s <= 300.0:
        return "181-300s"
    return ">300s"


def resumir_gap_primera_llamada(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evalúa caché de la 1.ª llamada del turno según el gap con la llamada anterior de la misma sesión/modelo.

    Equivalente en lógica al SQL:
        `LAG(created_at) OVER (PARTITION BY session_key, model ORDER BY created_at)`
        con `session_key = COALESCE(NULLIF(metadata->>'session_id',''), user_id::text, 'anon')`,
    reteniendo la 1.ª llamada de cada `corr` (o toda llamada sin `corr`).
    """
    ordered = sorted(
        rows,
        key=lambda r: (_to_datetime(r.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), str(r.get("id") or "")),
    )
    prev_ts_by_stream: dict[tuple[str, str], datetime] = {}
    seen_corrs: set[tuple[str, str]] = set()
    buckets: dict[tuple[str, str], dict[str, Any]] = {}

    for row in ordered:
        dt = _to_datetime(row.get("created_at"))
        if dt is None:
            continue
        meta = _parse_metadata(row.get("metadata"))
        model = str(row.get("model") or "desconocido")
        session_id = str(meta.get("session_id") or "").strip()
        user_id = str(row.get("user_id") or "").strip()
        session_key = session_id or user_id or "anon"

        stream_key = (session_key, model)
        prev_dt = prev_ts_by_stream.get(stream_key)
        prev_ts_by_stream[stream_key] = dt

        corr = str(meta.get("corr") or "").strip()
        if corr:
            corr_key = (corr, model)
            if corr_key in seen_corrs:
                continue
            seen_corrs.add(corr_key)

        gap_s = max(0.0, (dt - prev_dt).total_seconds()) if prev_dt is not None else None
        bucket_name = _bucket_segundos(gap_s)
        acc = buckets.setdefault(
            (model, bucket_name),
            {"llamadas": 0, "sum_in": 0, "sum_cached": 0, "sum_micros": 0},
        )
        acc["llamadas"] += 1
        acc["sum_in"] += _safe_int(row.get("input_tokens"))
        acc["sum_cached"] += _safe_int(row.get("cached_tokens"))
        acc["sum_micros"] += _safe_int(row.get("cost_usd_micros"))

    order_idx = {name: i for i, name in enumerate(_BUCKET_ORDER)}
    out: list[dict[str, Any]] = []
    for (model, bucket_name), acc in sorted(
        buckets.items(), key=lambda kv: (kv[0][0], order_idx.get(kv[0][1], 99))
    ):
        n = acc["llamadas"]
        out.append(
            {
                "model": model,
                "bucket": bucket_name,
                "llamadas": n,
                "in_medio": int(round(acc["sum_in"] / n)) if n else 0,
                "pct_cache": _pct(acc["sum_cached"], acc["sum_in"]),
                "usd_medio": round((acc["sum_micros"] / 1_000_000.0) / n, 6) if n else 0.0,
            }
        )
    return out


def _fetch_events(horas: int) -> list[dict[str, Any]]:
    url = _resolve_neon_db_url()
    if not url:
        raise RuntimeError("Falta NEON_DATABASE_URL_POOLED / NEON_DATABASE_URL.")
    with psycopg.connect(url, row_factory=dict_row, prepare_threshold=None) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(_SQL_EVENTOS, (str(int(horas)),))
            return [dict(r) for r in cur.fetchall()]


def _fmt_dur(val: Optional[float]) -> str:
    return "-" if val is None else f"{val:.2f}"


def _imprimir_resumen_diario(diario: list[dict[str, Any]]) -> None:
    print("=== 1) Resumen diario por modelo ===")
    if not diario:
        print("Sin llamadas de chat_call_model en esa ventana.")
        return
    print("fecha      | modelo             | llam | in medio | %cache | out med | s med |    USD")
    print("-" * 86)
    for r in diario:
        print(
            f"{r['fecha']} | {r['model']:<18} | {r['llamadas']:>4} | {r['in_medio']:>8} | "
            f"{r['pct_cache']:>5.1f}% | {r['out_medio']:>7} | {_fmt_dur(r['dur_s_medio']):>5} | {r['usd']:.4f}"
        )


def _imprimir_por_turno(turnos_res: dict[str, Any]) -> None:
    print("\n=== 2) Por turno (metadata->>'corr'): 1.a vs 2.a llamada ===")
    por_modelo = turnos_res.get("por_modelo") or []
    if not por_modelo:
        print("Sin turnos etiquetados con metadata->>'corr' en esa ventana.")
        return
    print("modelo             | turnos | >=2 | USD/turno | reason | 1a(in / %c / out / s)      | 2a(in / %c / out / s)")
    print("-" * 112)
    for m in por_modelo:
        c1 = m["call1"] or {}
        c2 = m["call2"] or {}
        c1_str = f"{c1.get('in_medio', 0)}/{c1.get('pct_cache', 0.0):.0f}%/{c1.get('out_medio', 0)}/{_fmt_dur(c1.get('dur_s_medio'))}s"
        c2_str = (
            f"{c2.get('in_medio', 0)}/{c2.get('pct_cache', 0.0):.0f}%/{c2.get('out_medio', 0)}/{_fmt_dur(c2.get('dur_s_medio'))}s"
            if c2
            else "-"
        )
        print(
            f"{m['model']:<18} | {m['turnos']:>6} | {m['turnos_con_2_llamadas']:>3} | "
            f"{m['usd_medio_turno']:>9.5f} | {m['reasoning_medio_turno']:>6} | {c1_str:<26} | {c2_str}"
        )

    recientes = turnos_res.get("turnos") or []
    if recientes:
        print("\nÚltimos turnos:")
        for t in recientes[:10]:
            c1 = t["call1"]
            c2 = t["call2"]
            c1_txt = f"1a:{c1['input_tokens']}@{c1['pct_cache']:.0f}%({c1['output_tokens']}o,{_fmt_dur(c1['dur_s'])}s)"
            c2_txt = (
                f"2a:{c2['input_tokens']}@{c2['pct_cache']:.0f}%({c2['output_tokens']}o,{_fmt_dur(c2['dur_s'])}s)"
                if c2
                else "2a:-"
            )
            extra = f" | {t['desglose']}" if t["desglose"] else ""
            print(
                f"  {t['corr']:<10} | {t['model']:<18} | n={t['llamadas']} | ${t['usd_total']:.5f} | "
                f"reas={t['reasoning_tokens']} | {c1_txt} | {c2_txt}{extra}"
            )


def _imprimir_gap_sesion(gaps: list[dict[str, Any]]) -> None:
    print("\n=== 3) 1.a llamada del turno segun segundos desde llamada previa de la sesion ===")
    if not gaps:
        print("Sin llamadas suficientes para medir distancia temporal entre llamadas.")
        return
    print("modelo             | intervalo | llam | in medio | %cache | USD medio")
    print("-" * 73)
    for g in gaps:
        print(
            f"{g['model']:<18} | {g['bucket']:<9} | {g['llamadas']:>4} | {g['in_medio']:>8} | "
            f"{g['pct_cache']:>5.1f}% | {g['usd_medio']:.5f}"
        )


def main(horas: int = 24, *, as_json: bool = False) -> int:
    try:
        rows = _fetch_events(horas)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diario = resumir_diario_modelos(rows)
    turnos_res = resumir_por_turno(rows)
    gaps = resumir_gap_primera_llamada(rows)

    if as_json:
        print(
            json.dumps(
                {
                    "horas": int(horas),
                    "total_eventos": len(rows),
                    "resumen_diario": diario,
                    "por_turno": turnos_res,
                    "gap_primera_llamada": gaps,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _imprimir_resumen_diario(diario)
    _imprimir_por_turno(turnos_res)
    _imprimir_gap_sesion(gaps)
    return 0


def _parse_cli_args(argv: list[str]) -> tuple[int, bool]:
    horas = 24
    as_json = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--json":
            as_json = True
        elif arg in ("--horas", "-H") and i + 1 < len(argv):
            horas = int(argv[i + 1])
            i += 1
        elif arg.isdigit():
            horas = int(arg)
        i += 1
    return horas, as_json


if __name__ == "__main__":
    _horas, _as_json = _parse_cli_args(sys.argv[1:])
    sys.exit(main(_horas, as_json=_as_json))
