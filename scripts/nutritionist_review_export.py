"""[P0-CLINICAL-VALIDATION] Export del artefacto de revisión para NUTRICIONISTA — habilita la
validación HUMANA que el audit clínico nombró como gap (la parte que NO es código: un profesional
revisa). Renderiza una muestra de planes REALES persistidos (comidas + ingredientes + macros + target
vs entregado color-codeado + campos de revisión) a un HTML imprimible. Read-only (no persiste nada).

Complementa `clinical_validation_export.py` (que produce el CSV de precisión/integridad por día): este
produce el artefacto LEGIBLE por un humano (con el contenido real de las comidas, no solo números).

Uso:  python scripts/nutritionist_review_export.py [--n 5] [--out /tmp/nutritionist_review.html]
"""
import argparse
import asyncio
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _num(x) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).lower().replace("g", "").replace("kcal", ""))
    return float(m.group(0)) if m else 0.0


def _band(delivered: float, target: float, lo: float = 0.90, hi: float = 1.12) -> str:
    if target <= 0:
        return "na"
    r = delivered / target
    return "ok" if lo <= r <= hi else ("low" if r < lo else "high")


async def _open_pools():
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    if getattr(db_core, "async_connection_pool", None):
        await db_core.async_connection_pool.open()
    await asyncio.sleep(1.5)


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;color:#1a1a1a;padding:0 16px}
h1{font-size:22px} h2{font-size:17px;margin-top:32px;border-bottom:2px solid #eee;padding-bottom:6px}
.meta{color:#666;font-size:13px} table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top} th{background:#f5f5f5}
.ok{background:#e6f4ea} .low{background:#fdecea} .high{background:#fff4e5}
.macros{font-size:12px;color:#444} .ing{font-size:12px;color:#333}
.review{background:#f0f7ff;border:1px solid #cfe3ff;border-radius:8px;padding:12px;margin:12px 0}
.review label{display:block;margin:6px 0;font-size:13px} .review input[type=text]{width:60%}
.legend span{display:inline-block;padding:2px 8px;border-radius:4px;margin-right:8px;font-size:12px}
@media print{.review{break-inside:avoid}}
"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="nº de planes a muestrear")
    ap.add_argument("--out", default="/tmp/nutritionist_review.html")
    args = ap.parse_args()

    await _open_pools()
    from db_core import execute_sql_query
    rows = execute_sql_query(
        "SELECT id::text AS id, plan_data, created_at::text AS created_at FROM meal_plans "
        "WHERE jsonb_array_length(plan_data->'days') > 0 AND plan_data ? 'macros' "
        "  AND COALESCE(plan_data->>'generation_status','') <> 'failed' "
        "ORDER BY created_at DESC LIMIT %s", (args.n,), fetch_all=True) or []

    parts = [f"<!doctype html><meta charset='utf-8'><style>{_CSS}</style>",
             "<h1>Revisión clínica de planes — MealfitRD</h1>",
             "<p class='meta'>Muestra de planes reales generados en producción. Para revisión por "
             "nutricionista certificado: verifica que las comidas, porciones y macros sean clínicamente "
             "sensatas y que el plan entregue su target. Marca cada plan y anota observaciones.</p>",
             "<p class='legend'><span class='ok'>en banda (90–112% del target)</span>"
             "<span class='low'>por debajo</span><span class='high'>por encima</span></p>"]

    for r in rows:
        pd = r["plan_data"]
        if isinstance(pd, str):
            pd = json.loads(pd)
        tgt = {"P": _num((pd.get("macros") or {}).get("protein")), "C": _num((pd.get("macros") or {}).get("carbs")),
               "G": _num((pd.get("macros") or {}).get("fats")), "kcal": _num(pd.get("calories"))}
        parts.append(f"<h2>Plan {html.escape(r['id'][:8])} · {html.escape(r['created_at'][:10])}</h2>")
        parts.append(f"<p class='meta'>Target diario: <b>{tgt['kcal']:.0f} kcal</b> · P {tgt['P']:.0f}g · "
                     f"C {tgt['C']:.0f}g · G {tgt['G']:.0f}g · fallback: {bool(pd.get('_is_fallback'))}</p>")
        for di, day in enumerate(pd.get("days") or [], 1):
            dp = {"P": 0.0, "C": 0.0, "G": 0.0, "kcal": 0.0}
            mrows = []
            for m in (day.get("meals") or []):
                mp, mc, mg = _num(m.get("protein")), _num(m.get("carbs")), _num(m.get("fats"))
                mk = _num(m.get("cals") if m.get("cals") is not None else m.get("calories"))
                dp["P"] += mp; dp["C"] += mc; dp["G"] += mg; dp["kcal"] += mk
                ings = "<br>".join(html.escape(str(i)) for i in (m.get("ingredients") or []))
                mrows.append(f"<tr><td><b>{html.escape(str(m.get('name', '?')))}</b></td>"
                             f"<td class='ing'>{ings}</td>"
                             f"<td class='macros'>P {mp:.0f} · C {mc:.0f} · G {mg:.0f}<br>{mk:.0f} kcal</td></tr>")
            parts.append(f"<p class='meta'><b>Día {di}</b> — entregado: "
                         f"<span class='{_band(dp['kcal'], tgt['kcal'], 0.95, 1.05)}'>{dp['kcal']:.0f} kcal</span> · "
                         f"<span class='{_band(dp['P'], tgt['P'])}'>P {dp['P']:.0f}g "
                         f"({dp['P'] / tgt['P'] * 100 if tgt['P'] else 0:.0f}%)</span> · "
                         f"<span class='{_band(dp['C'], tgt['C'])}'>C {dp['C']:.0f}g</span> · "
                         f"<span class='{_band(dp['G'], tgt['G'])}'>G {dp['G']:.0f}g</span></p>")
            parts.append("<table><tr><th>Comida</th><th>Ingredientes</th><th>Macros</th></tr>"
                         + "".join(mrows) + "</table>")
        parts.append(
            "<div class='review'><b>Revisión del nutricionista</b>"
            "<label>¿Las comidas y porciones son clínicamente sensatas? &nbsp; Sí ☐ &nbsp; No ☐ &nbsp; Con reservas ☐</label>"
            "<label>¿El plan entrega su target de macros razonablemente? &nbsp; Sí ☐ &nbsp; No ☐</label>"
            "<label>Observaciones: <input type='text'></label></div>")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"[nutritionist-review] {len(rows)} planes -> {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
