# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-17 · 2026-09-12 · D4] Medición READ-ONLY de cantidades absurdas en `ingredients_raw` (lo que la
lista COMPRA), comida a comida, sobre toda la flota.

Hallazgo del 09-08 (`docs/hallazgo_abierto_cantidades_absurdas_en_compra.md`): «30 cdas de aceite de oliva» en la
compra para un plato que lee «¾ cdta». El productor está identificado y reproducido (rama «polvo de queso» del piso,
por ÍNDICE, `tests/test_p1_plan_lote_17.py`); esta sonda mide el RESIDUO para decidir si hace falta un barrido, y
queda para re-medir cuando la flota crezca.

Criterio — DOS condiciones, para no repetir el falso positivo del 09-08 («2¼ tazas» = «34,75 cdas»: conversión
CORRECTA que la sonda de entonces acusó por mirar sólo el número):

  1. la línea de raw supera el tope de su unidad de cocina (> 6 cdtas, > 8 cdas, > 6 tazas, > 12 dientes);
  2. y su gemela del display (misma línea, por alimento) o no existe, o pide menos de un tercio en ml
     (cdta 5 · cda 15 · taza 240; los dientes se comparan como conteo): la compra supera `FACTOR_ABSURDO`× la receta.
     Si la gemela está en otra magnitud (gramos, piezas) no se puede comparar: se informa aparte, no se acusa.

Recorre `days` y `_archived_days`: la lista se agrega desde el ciclo entero (P0-SHOPPING-CYCLE-DAYS).

Sólo SELECTs; abre la conexión en `read_only`. Uso (desde `backend/`):

    python scripts/medir_cantidades_absurdas_raw.py [--json]

Medido el 2026-09-12: 6 planes (5 con días), 100 comidas, 0 líneas absurdas; los 3 planes del hallazgo
(`d476023a`, `7c545d59`, `e2bbb280`) ya no existen.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(_BACKEND))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)

#: La compra tiene que superar este múltiplo de la receta (en ml o conteo) para acusar la línea.
FACTOR_ABSURDO = 3.0

#: unidad canónica → (tope por encima del cual la cantidad es absurda, ml por unidad; `None` = se compara como conteo)
UNIDADES_DE_COCINA: dict[str, tuple[float, Optional[float]]] = {
    "cdta": (6.0, 5.0),
    "cda": (8.0, 15.0),
    "taza": (6.0, 240.0),
    "diente": (12.0, None),
}
_UNIDAD_CANONICA = {
    "cdta": "cdta", "cdtas": "cdta", "cucharadita": "cdta", "cucharaditas": "cdta",
    "cda": "cda", "cdas": "cda", "cucharada": "cda", "cucharadas": "cda",
    "taza": "taza", "tazas": "taza",
    "diente": "diente", "dientes": "diente",
}
_FRACCION_UNICODE = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_LIDER_RE = re.compile(
    r"^\s*(?P<ent>\d+(?:[.,]\d+)?)?\s*(?:(?P<uni>[½¼¾⅓⅔⅛])|(?P<num>\d+)\s*/\s*(?P<den>\d+))?\s*"
    r"(?:de\s+)?(?P<unidad>[a-záéíóú]+)\b",
    re.I,
)
_STOP = {"picada", "picado", "picadas", "picados", "fresca", "fresco", "molido", "molida", "rallado", "rallada",
         "laminas", "láminas", "cubos", "rodajas", "trozos", "hojas", "grande", "mediana", "mediano", "pequeña",
         "pequeño", "verde", "roja", "rojo", "blanca", "blanco", "para", "con", "sin", "del", "las", "los", "una", "uno"}


def cantidad_lider(linea: str) -> Optional[tuple[float, str]]:
    """«2¼ tazas de x» → (2.25, 'taza'); «0.5 cdas» → (0.5, 'cda'); «1/2 cda» → (0.5, 'cda'). `None` si la línea no
    empieza por cantidad + unidad de cocina conocida (gramos, piezas, «al gusto»…)."""
    m = _LIDER_RE.match(str(linea or ""))
    if not m or not (m.group("ent") or m.group("uni") or m.group("num")):
        return None
    unidad = _UNIDAD_CANONICA.get(m.group("unidad").lower())
    if not unidad:
        return None
    q = 0.0
    if m.group("ent"):
        q += float(m.group("ent").replace(",", "."))
    if m.group("uni"):
        q += _FRACCION_UNICODE[m.group("uni")]
    elif m.group("num"):
        den = float(m.group("den"))
        if den <= 0:
            return None
        q += float(m.group("num")) / den
    return (q, unidad) if q > 0 else None


def _en_ml(q: float, unidad: str) -> Optional[float]:
    ml = UNIDADES_DE_COCINA[unidad][1]
    return q * ml if ml is not None else None


def _alimento(linea: str) -> Optional[str]:
    """Primer sustantivo útil tras la cantidad y la unidad: «30 cdas de cebolla picada» → «cebolla»."""
    s = str(linea or "").lower()
    s = re.sub(r"^\s*[\d.,/½¼¾⅓⅔⅛\s]+[a-záéíóú]*\.?\s*(?:de\s+)?", "", s, count=1)
    s = re.sub(r"\([^)]*\)", " ", s)
    for tok in re.findall(r"[a-záéíóúñ]{4,}", s):
        if tok not in _STOP and tok not in _UNIDAD_CANONICA:
            return tok
    return None


def gemela_en_display(linea_raw: str, display: list) -> Optional[str]:
    """La línea del display del MISMO alimento (léxico, sin catálogo): una sola coincidencia, o la que comparte
    unidad de cocina si hay varias; `None` si el alimento no aparece."""
    tok = _alimento(linea_raw)
    if not tok:
        return None
    hits = [str(d) for d in display or [] if tok in str(d).lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        cl = cantidad_lider(linea_raw)
        misma = [h for h in hits if cl and (cantidad_lider(h) or (None, None))[1] == cl[1]]
        return misma[0] if misma else hits[0]
    return None


def juzgar_linea(linea_raw: str, display: list) -> Optional[dict]:
    """`None` si la línea de raw es normal. Si supera el tope de su unidad: dict con `veredicto` ∈
    {'absurda', 'no_comparable'} y el par raw↔display con el factor en ml (o conteo) cuando se puede calcular."""
    cl = cantidad_lider(linea_raw)
    if not cl:
        return None
    q, unidad = cl
    tope, _ = UNIDADES_DE_COCINA[unidad]
    if q <= tope:
        return None
    gem = gemela_en_display(linea_raw, display)
    caso: dict[str, Any] = {"raw": str(linea_raw), "cantidad": q, "unidad": unidad, "display": gem, "factor": None}
    if gem is None:
        caso["veredicto"] = "absurda"
        caso["motivo"] = "sin gemela en el display"
        return caso
    cg = cantidad_lider(gem)
    if not cg:
        caso["veredicto"] = "no_comparable"
        caso["motivo"] = "la gemela no está en unidad de cocina"
        return caso
    raw_ml, gem_ml = _en_ml(q, unidad), _en_ml(cg[0], cg[1])
    if raw_ml is None or gem_ml is None:
        if unidad == cg[1]:
            factor = q / cg[0] if cg[0] > 0 else None
        else:
            caso["veredicto"] = "no_comparable"
            caso["motivo"] = "conteo frente a volumen"
            return caso
    else:
        factor = raw_ml / gem_ml if gem_ml > 0 else None
    caso["factor"] = round(factor, 2) if factor is not None else None
    if factor is not None and factor > FACTOR_ABSURDO:
        caso["veredicto"] = "absurda"
        caso["motivo"] = f"la compra pide {factor:.1f}× la receta"
    else:
        caso["veredicto"] = "conversion_correcta"
    return caso


def medir_plan(plan_id: str, plan_data: dict) -> tuple[int, list[dict]]:
    comidas, casos = 0, []
    dias = list(plan_data.get("days") or []) + list(plan_data.get("_archived_days") or [])
    for di, d in enumerate(dias):
        for mi, m in enumerate((d or {}).get("meals") or []):
            if not isinstance(m, dict):
                continue
            comidas += 1
            display = m.get("ingredients") or []
            for ri, linea in enumerate(m.get("ingredients_raw") or []):
                c = juzgar_linea(str(linea), display)
                if c and c["veredicto"] != "conversion_correcta":
                    c.update({"plan": plan_id[:8], "dia": di, "comida": mi, "plato": m.get("name") or m.get("meal"),
                              "raw_idx": ri, "piso": bool(m.get("_portion_floor_adjusted"))})
                    casos.append(c)
    return comidas, casos


def medir(cur) -> dict:
    cur.execute(
        "SELECT id::text AS id, plan_data FROM meal_plans "
        "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') > 0"
    )
    filas = cur.fetchall()
    cur.execute("SELECT count(*) FROM meal_plans")
    total = int(cur.fetchone()[0])
    comidas, casos = 0, []
    for pid, pd in filas:
        c, k = medir_plan(pid, pd or {})
        comidas += c
        casos.extend(k)
    absurdas = [c for c in casos if c["veredicto"] == "absurda"]
    return {
        "planes_total": total, "planes_con_dias": len(filas), "comidas": comidas,
        "lineas_absurdas": len(absurdas), "lineas_no_comparables": len(casos) - len(absurdas),
        "planes_afectados": sorted({c["plan"] for c in absurdas}), "casos": casos,
        "criterio": {"topes": {u: t for u, (t, _) in UNIDADES_DE_COCINA.items()}, "factor_absurdo": FACTOR_ABSURDO},
    }


def veredicto(res: dict) -> str:
    if res["comidas"] == 0:
        return "NO CONCLUYENTE: sin comidas en la flota"
    if res["lineas_absurdas"] == 0:
        return "SIN RESIDUO: 0 líneas absurdas en la compra ⇒ nada que barrer"
    return (f"RESIDUO: {res['lineas_absurdas']} línea(s) absurda(s) en {len(res['planes_afectados'])} plan(es) "
            f"⇒ decidir barrido (dueño)")


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes: Any) -> None:
    print(*partes)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="salida JSON completa")
    args = ap.parse_args(argv)
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
            res = medir(cur)
    if args.json:
        _say(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        return 0
    _say(f"planes: {res['planes_total']} ({res['planes_con_dias']} con días) · comidas: {res['comidas']} · "
         f"líneas absurdas en raw: {res['lineas_absurdas']} · no comparables: {res['lineas_no_comparables']}")
    for c in res["casos"]:
        _say(f"  [{c['veredicto']}] {c['plan']} d{c['dia']} m{c['comida']} {c['plato']!r}: COMPRA {c['raw']!r} ↔ "
             f"LEE {c['display']!r} · factor {c['factor']} · {c.get('motivo', '')}")
    _say(veredicto(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
