# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-60 · 2026-09-15] (lote 38 del plan · C1 cierre) Refresca las columnas de la MÁQUINA del golden set.

`docs/culinary_golden_set.json` guarda lo que dijo cada capa el 2026-09-06 (`maquina_determinista`, `maquina_juez`): los
hallazgos PERSISTIDOS de 96 planes, no un escaneo nuevo. Desde entonces nacieron V6, V7a-e, V8 y V9 (lotes 22-31) y el
juez ganó el estado `[dudosa]` (lote 28): medir el instrumento de hoy contra esas columnas es medir el de hace nueve
días. Este script vuelve a pasar la máquina ACTUAL por los 80 casos y escribe su resultado AL LADO, con fecha:

    python scripts/culinary_golden_refresh.py                                  # determinista: resumen, NO escribe
    python scripts/culinary_golden_refresh.py --escribir                       # escribe maquina_determinista_<hoy>
    python scripts/culinary_golden_refresh.py --escribir --con-juez            # + maquina_juez_<hoy> (LLM, tope $0,50)
    python scripts/culinary_golden_refresh.py --escribir --con-juez --solo-juez   # sólo la del juez (la otra ya está)
    python scripts/culinary_golden_refresh.py --catalogo scripts/data/culinary_corpus_2026_09_12.json   # sin base

Después: `python scripts/culinary_golden_score.py --estricto --anotaciones <fichero> --comparar-maquina <fecha>`.

## Qué hace, qué no

- **Nunca borra ni reescribe** `maquina_determinista` / `maquina_juez` (las del 09-06): la comparación antes/después ES el
  resultado. Una columna de la misma fecha no se pisa salvo `--reemplazar`, y `aplicar` lanza si una base cambiara.
- **Determinista**: `culinary_contract_scan_status` sobre la comida tal cual la guarda el caso (franja, nombre,
  ingredientes y pasos) con el catálogo de hoy (`master_ingredients`, SELECT con `read_only`) o el de un fichero. El
  golden set no guarda la asignación de receta de biblioteca ni el formulario: las cuentas exactas de `recipe_usage` y
  V8b (equipo declarado) no aplican, y el refresco lo dice en `no_aplica` en vez de fingirlo.
- **El alimento acusado viaja en el texto**: `V7e: <detalle> (alimento: Casabe)`, el campo `food` de la violación (sólo
  en V1-V7e, donde es un alimento: V8a dice «tiempo», V8b el equipo, V9 el tipo de estructura), y
  `<tipo>: <detalle> (componente: X)` en el juez. El builder del 09-06 los tiraba, y un detalle que cita el paso nombra
  OTROS alimentos: el marcador compara con el declarado cuando está.
- **Juez** (`--con-juez`): `graph_orchestrator.run_culinary_judge` con el MISMO modelo que producción (glm-5.3-flash,
  medido en `llm_usage_events` el 2026-09-15: 67 llamadas del juez en 14 días, todas de ese modelo), UNA comida por
  llamada (el golden set no guarda el resto del día; en producción el juez ve el plan entero), país DO y sin formulario.
  `MEALFIT_CULINARY_JUDGE_GUARD=warn` sólo en ESTE proceso: no es el flip de C6, que es del dueño. Las escrituras a la
  base se sustituyen por dobles que cuentan (`bench_superficies_culinarias.bloquear_escrituras`) y el coste se suma EN
  PROCESO con la tarifa del emisor (`contar_coste`). La primera llamada va sola: si no registró coste, se para (sin
  contador no hay tope). Con `--presupuesto-usd` (0,50 por defecto; nunca más de 1,00, decisión del dueño del 14-sep)
  no se lanza una llamada que pudiera pasarlo, y el refresco dice cuántos casos quedaron sin juzgar. Un caso sin juzgar
  no lleva la columna: el marcador lo deja fuera de la capa, no lo cuenta como «limpio».

Solo lectura sobre la base. tooltip-anchor: P1-PLAN-LOTE-60-REFRESH
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.append(str(_BACKEND))

GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
#: Las columnas del 2026-09-06. Este script las LEE; jamás las escribe.
COLUMNAS_BASE = ("maquina_determinista", "maquina_juez")
PRESUPUESTO_USD = 0.50      # plan 38-44 §0.2: el mismo tope que el bench real del lote 30
TOPE_USD = 1.00             # decisiones del dueño 2026-09-14: «juez sí, con tope de $1»
CONCURRENCIA = 2           # con 4 a la vez la primera corrida del 2026-09-15 dio 5 timeouts en 5 minutos
NO_APLICA = [
    "recipe_usage (cuentas exactas de V3/V6/V7a/V7e): el caso no guarda la asignación de receta de biblioteca",
    "V8b (equipo): el caso no guarda el formulario, así que no hay equipo declarado",
]


def columna(capa: str, fecha: str) -> str:
    """`maquina_<capa>_<fecha>`. Una fecha vacía es un error: nunca devuelve una columna del 09-06."""
    if not str(fecha or "").strip():
        raise ValueError("un refresco lleva fecha")
    col = f"maquina_{capa}_{str(fecha).strip()}"
    if col in COLUMNAS_BASE:
        raise ValueError(col)
    return col


def comida_de(caso: dict) -> dict:
    """La comida tal como la guarda el caso, con las claves que leen el escáner y el juez (`meal`, `name`,
    `ingredients`, `recipe` — las mismas de `plan_data`)."""
    return {"meal": caso.get("franja") or "", "name": caso.get("nombre") or "",
            "ingredients": [str(x) for x in (caso.get("ingredientes") or [])],
            "recipe": [str(x) for x in (caso.get("pasos") or [])]}


def plan_de(caso: dict) -> dict:
    dia = caso.get("dia") if isinstance(caso.get("dia"), int) else 1
    return {"days": [{"day": dia, "meals": [comida_de(caso)]}]}


#: Los checks cuyo `food` es un alimento (V8a dice «tiempo», V8b el equipo, V9 el tipo de estructura).
CHECKS_CON_ALIMENTO = frozenset({"V1", "V2", "V3", "V4", "V5", "V6", "V7a", "V7b", "V7c", "V7d", "V7e"})


def _sin_parentesis(x) -> str:
    return " ".join(str(x or "").replace("(", " ").replace(")", " ").split())


def texto_determinista(v: dict) -> str:
    """El formato de `culinary_golden_sample.indexar_hallazgos` (el marcador lee el código antes de «:») más el alimento
    que el detector acusa, al final: `(alimento: X)`."""
    base = f"{v.get('check')}: {str(v.get('detail'))[:130]}"
    food = _sin_parentesis(v.get("food"))
    return f"{base} (alimento: {food})" if food and str(v.get("check")) in CHECKS_CON_ALIMENTO else base


def texto_juez(v: dict) -> str:
    """El formato del golden set del 09-06, con la marca `[dudosa]` del lote 28 y el `componente` al final."""
    marca = " [dudosa]" if str(v.get("certeza") or "") == "dudosa" else ""
    base = f"{v.get('tipo')}{marca}: {str(v.get('detalle'))[:130]}"
    comp = _sin_parentesis(v.get("componente"))
    return f"{base} (componente: {comp})" if comp else base


def por_codigo(d: dict, col: str) -> dict:
    """Cuántos hallazgos de cada código trae una columna (sin deduplicar: es lo que está escrito)."""
    cnt = collections.Counter()
    for c in d.get("casos") or []:
        for t in c.get(col) or []:
            cnt[str(t).split(":", 1)[0].replace(" [dudosa]", "").strip()] += 1
    return dict(sorted(cnt.items()))


def escanear(casos: list, catalogo: list) -> tuple[dict, dict]:
    """El escáner determinista ACTUAL sobre cada caso. Devuelve `({id: [textos]}, resumen)`. Puro: sin base."""
    from culinary_coherence import culinary_contract_scan_status, rules_fingerprint
    out, estados, checks = {}, collections.Counter(), collections.Counter()
    for c in casos:
        viols, est = culinary_contract_scan_status(plan_de(c), catalogo)
        estados[str(est.get("status"))] += 1
        out[str(c.get("id"))] = [texto_determinista(v) for v in viols]
        for v in viols:
            checks[str(v.get("check"))] += 1
    return out, {"estado_scan": dict(estados), "por_check": dict(sorted(checks.items())),
                 "hallazgos": sum(checks.values()), "casos_con_hallazgo": sum(1 for x in out.values() if x),
                 "reglas_huella": rules_fingerprint(), "no_aplica": NO_APLICA}


def aplicar(d: dict, capa: str, fecha: str, por_caso: dict, meta: dict, reemplazar: bool = False) -> dict:
    """Una COPIA del golden set con `maquina_<capa>_<fecha>` en cada caso de `por_caso` (colocada tras `maquina_juez`) y el
    refresco anotado en `refrescos`. Las columnas del 09-06 salen idénticas o lanza."""
    col = columna(capa, fecha)
    if not reemplazar and any(col in c for c in d.get("casos") or []):
        raise ValueError(f"{col} ya existe: un refresco con fecha no se pisa (--reemplazar, a sabiendas)")
    nuevo = json.loads(json.dumps(d, ensure_ascii=False))
    casos = []
    for c in nuevo.get("casos") or []:
        cid = str(c.get("id"))
        antes = {k: c.get(k) for k in COLUMNAS_BASE}
        orden = {}
        for k, v in c.items():
            if k == col:
                continue
            orden[k] = v
            if k == "maquina_juez" and cid in por_caso:
                orden[col] = list(por_caso[cid])
        if cid in por_caso and col not in orden:
            orden[col] = list(por_caso[cid])
        if {k: orden.get(k) for k in COLUMNAS_BASE} != antes:
            raise AssertionError(f"{cid}: una columna del 09-06 cambió")
        casos.append(orden)
    refrescos = [r for r in (nuevo.get("refrescos") or []) if r.get("columna") != col]
    refrescos.append({"columna": col, "capa": capa, "fecha": fecha, "casos": len(por_caso), **meta})
    top = {}
    for k, v in nuevo.items():
        if k == "casos":
            top["refrescos"] = refrescos
            top["casos"] = casos
        elif k != "refrescos":
            top[k] = v
    return top


# ─────────────────────────────────────────────────────────────────────────────────────────────
# catálogo: la base (SELECT con read_only) o un fichero
# ─────────────────────────────────────────────────────────────────────────────────────────────

def catalogo_de_la_base(env_path: Path) -> tuple[list, dict]:
    """`master_ingredients` con las columnas del índice (`culinary_corpus.COLUMNAS_CATALOGO`), en una conexión read_only."""
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row
    from culinary_corpus import COLUMNAS_CATALOGO, filas_catalogo, huella_catalogo
    load_dotenv(env_path)
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        conn.read_only = True
        filas = conn.execute(f"SELECT {', '.join(COLUMNAS_CATALOGO)} FROM master_ingredients ORDER BY name").fetchall()
    cat = filas_catalogo([dict(r) for r in filas])
    return cat, {"fuente": "master_ingredients (Neon, SELECT con read_only)", "filas": len(cat),
                 "huella": huella_catalogo(cat)}


def catalogo_de_fichero(path) -> tuple[list, dict]:
    """Un corpus fijo (`catalogo_filas`) o una lista de filas."""
    from culinary_corpus import filas_catalogo, huella_catalogo
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    filas = (doc.get("catalogo_filas") or doc.get("filas")) if isinstance(doc, dict) else doc
    if not isinstance(filas, list) or not filas:
        raise ValueError(f"{path}: sin filas de catálogo")
    cat = filas_catalogo(filas)
    return cat, {"fuente": Path(path).name, "filas": len(cat), "huella": huella_catalogo(cat)}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# el juez: el de producción, con tope y sin escribir en la base
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _cargar_script(nombre: str):
    spec = importlib.util.spec_from_file_location(Path(nombre).stem + "_refresh", _BACKEND / "scripts" / nombre)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def juzgar(casos: list, presupuesto_usd: float, env_path: Path, concurrencia: int = CONCURRENCIA,
           timeout_s: "int | None" = None) -> tuple[dict, dict]:
    """El juez de producción sobre cada caso. Devuelve `({id: [textos]} SÓLO de los juzgados, resumen)`.

    `timeout_s` fija `MEALFIT_CULINARY_JUDGE_TIMEOUT_S` en ESTE proceso: es la paciencia del cliente, no el juicio. Un
    fail-open por timeout no devuelve uso, así que su coste no se cuenta; el proveedor pudo cobrar la llamada y su
    reintento, y el resumen publica esa cota (`coste_no_contado_max_usd`)."""
    if not (0 < float(presupuesto_usd) <= TOPE_USD):
        raise ValueError(f"presupuesto fuera de (0, {TOPE_USD}]")
    os.environ["MEALFIT_CULINARY_JUDGE_GUARD"] = "warn"      # sólo ESTE proceso, y ANTES de importar el orquestador
    if timeout_s:
        os.environ["MEALFIT_CULINARY_JUDGE_TIMEOUT_S"] = str(int(timeout_s))
    from dotenv import load_dotenv
    load_dotenv(env_path)
    bench = _cargar_script("bench_superficies_culinarias.py")
    escrituras = bench.bloquear_escrituras()                  # ANTES del orquestador: los módulos enlazan el nombre al importar
    import graph_orchestrator as go
    if go.CULINARY_JUDGE_GUARD != "warn":
        raise SystemExit("el orquestador se importó antes de fijar el guard: el juez devolvería None en cada llamada")
    _log, coste = bench.contar_coste()
    resultados, fallos, sin_juzgar = {}, {}, []
    maximo = [0.0]
    en_vuelo = [0]
    t0 = time.time()

    hechos = [0]

    def _progreso(c, t_ini, que):
        hechos[0] += 1
        print(f"  juez {hechos[0]}/{len(casos)} · {c.get('id')} · {time.time() - t_ini:.1f} s · "
              f"${coste['micros'] / 1e6:.4f} · {que}", file=sys.stderr, flush=True)

    async def _uno(c):
        antes, t_ini = coste["micros"], time.time()
        rep = await go.run_culinary_judge(plan_de(c), country="DO", form_data=None)
        maximo[0] = max(maximo[0], (coste["micros"] - antes) / 1e6)
        if rep is None:
            fallos[str(c.get("id"))] = "fail-open: el juez devolvió None"
            _progreso(c, t_ini, "fail-open")
            return
        viol = [v.model_dump() if hasattr(v, "model_dump") else dict(v) for v in (getattr(rep, "violations", None) or [])]
        resultados[str(c.get("id"))] = [texto_juez(v) for v in viol]
        _progreso(c, t_ini, f"{len(viol)} quejas")

    async def _con_tope(sem, c):
        async with sem:
            # no se lanza una llamada que, con las que ya vuelan, pudiera pasar el tope
            if coste["micros"] / 1e6 + (en_vuelo[0] + 1) * maximo[0] > presupuesto_usd:
                sin_juzgar.append(str(c.get("id")))
                return
            en_vuelo[0] += 1
            try:
                await _uno(c)
            except Exception as e:  # noqa: BLE001
                fallos[str(c.get("id"))] = f"{type(e).__name__}: {str(e)[:160]}"
                _progreso(c, time.time(), f"error {type(e).__name__}")
            finally:
                en_vuelo[0] -= 1

    async def _todo():
        pendientes = list(casos)
        if not pendientes:
            return
        primero = pendientes.pop(0)
        await _uno(primero)                                   # SOLA: comprueba que el contador cuenta
        if coste["llamadas"] == 0 or coste["micros"] <= 0:
            raise SystemExit(f"la primera llamada no registró coste ({coste['llamadas']} llamadas, {coste['micros']} micros; "
                             f"fallos: {fallos}): sin contador no hay tope y no se sigue")
        sem = asyncio.Semaphore(max(1, int(concurrencia)))
        await asyncio.gather(*(_con_tope(sem, c) for c in pendientes))

    asyncio.run(_todo())
    return resultados, {
        "modelo": go.CULINARY_JUDGE_MODEL, "modelos_llamados": list(coste.get("modelos") or []),
        "thinking": bool(go.CULINARY_JUDGE_THINKING), "pais": "DO", "formulario": None, "una_comida_por_llamada": True,
        "llamadas": coste["llamadas"], "coste_usd_est": round(coste["micros"] / 1e6, 5), "presupuesto_usd": presupuesto_usd,
        # un fail-open no devuelve uso: el proveedor pudo cobrar la llamada y su reintento (CULINARY_JUDGE_MAX_RETRIES)
        "coste_no_contado_max_usd": round((1 + int(getattr(go, "CULINARY_JUDGE_MAX_RETRIES", 1) or 0)) * len(fallos)
                                          * maximo[0], 5),
        "timeout_s": int(getattr(go, "CULINARY_JUDGE_TIMEOUT_S", 0) or 0), "concurrencia": int(concurrencia),
        "casos_juzgados": len(resultados), "casos_sin_juzgar": sorted(sin_juzgar), "fallos": fallos,
        "escrituras_suprimidas": dict(escrituras), "segundos": round(time.time() - t0, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────

def _commit() -> "str | None":
    try:
        sha = subprocess.run(["git", "-C", str(_BACKEND), "rev-parse", "--short=8", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        sucio = subprocess.run(["git", "-C", str(_BACKEND), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha + ("+cambios_sin_commit" if sucio else "")) or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fecha", default=date.today().isoformat(), help="sufijo de las columnas (por defecto, hoy)")
    ap.add_argument("--escribir", action="store_true", help="escribe docs/culinary_golden_set.json (sin él, sólo resumen)")
    ap.add_argument("--reemplazar", action="store_true", help="pisa una columna de la MISMA fecha (nunca las del 09-06)")
    ap.add_argument("--catalogo", help="JSON con catalogo_filas (corpus fijo) en vez de la base")
    ap.add_argument("--env", default=str(_BACKEND / ".env"), help="fichero de entorno (URL de la base, clave del LLM)")
    ap.add_argument("--con-juez", action="store_true", help="refresca también el juez (LLM, con tope)")
    ap.add_argument("--solo-juez", action="store_true", help="no rehace la columna determinista")
    ap.add_argument("--presupuesto-usd", type=float, default=PRESUPUESTO_USD, help=f"tope del juez (<= {TOPE_USD})")
    ap.add_argument("--concurrencia", type=int, default=CONCURRENCIA, help="llamadas del juez a la vez")
    ap.add_argument("--timeout-s", type=int, help="MEALFIT_CULINARY_JUDGE_TIMEOUT_S de ESTE proceso (10-120; producción 45)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.con_juez and not (0 < a.presupuesto_usd <= TOPE_USD):
        print(f"--presupuesto-usd debe estar en (0, {TOPE_USD}]: sin tope no se gasta")
        return 2
    d = json.loads(GOLDEN.read_text(encoding="utf-8"))
    casos = d.get("casos") or []
    comun = {"commit": _commit(), "generado_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    informe = {"fecha": a.fecha, "casos": len(casos), "escrito": bool(a.escribir)}
    if not a.solo_juez:
        cat, meta_cat = catalogo_de_fichero(a.catalogo) if a.catalogo else catalogo_de_la_base(Path(a.env))
        det, resumen = escanear(casos, cat)
        # sin --escribir no se escribe nada: «no pisar» no aplica y el resumen se puede ver aunque la columna ya exista
        d = aplicar(d, "determinista", a.fecha, det, {**resumen, "catalogo": meta_cat, **comun}, a.reemplazar or not a.escribir)
        col = columna("determinista", a.fecha)
        informe["determinista"] = {**resumen, "catalogo": meta_cat, "columna": col,
                                   "antes_por_codigo": por_codigo(d, "maquina_determinista"),
                                   "despues_por_codigo": por_codigo(d, col)}
    if a.con_juez:
        jz, meta_j = juzgar(casos, a.presupuesto_usd, Path(a.env), a.concurrencia, a.timeout_s)
        d = aplicar(d, "juez", a.fecha, jz, {**meta_j, **comun}, a.reemplazar or not a.escribir)
        col = columna("juez", a.fecha)
        informe["juez"] = {**meta_j, "columna": col, "antes_por_codigo": por_codigo(d, "maquina_juez"),
                           "despues_por_codigo": por_codigo(d, col)}
    else:
        informe["juez"] = "sin --con-juez: la columna del juez se queda como está (2026-09-06) y el marcador lo declara"
    if a.escribir:
        GOLDEN.write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(informe, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
