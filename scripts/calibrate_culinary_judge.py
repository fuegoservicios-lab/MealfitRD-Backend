"""[P1-CULINARY-JUDGE · 2026-08-01] Calibración del sistema culinario (capa 1
determinista + juez LLM, F3) contra el golden set (spec §6,
`docs/superpowers/specs/2026-07-31-culinary-coherence-design.md`). MANUAL,
reproducible; hace llamadas LLM REALES a DeepSeek flash — NO corre en CI (el
golden set en CI solo ejerce la capa 1 determinista, ver
`tests/test_p1_culinary_golden.py`, que hace `pytest.skip` sin DB y jamás
invoca `run_culinary_judge`).

TRAMPA Nº1 de este script — LÉELA ANTES DE TOCAR EL ORDEN DE LOS IMPORTS:
`CULINARY_JUDGE_GUARD` / `CULINARY_JUDGE_MODEL` / `CULINARY_JUDGE_THINKING`
en `graph_orchestrator.py` son constantes MÓDULO-LEVEL — se leen de
`os.environ` UNA SOLA VEZ, a IMPORT-TIME del módulo (`_env_str(...)` corre en
el cuerpo del módulo, no dentro de `run_culinary_judge`). Si algo importa
`graph_orchestrator` (directa o TRANSITIVAMENTE — p.ej. un futuro import en
`culinary_coherence.py`/`shopping_calculator.py` que hoy no existe, verificado
con grep) ANTES de que este script escriba `os.environ["MEALFIT_CULINARY_
JUDGE_GUARD"] = "warn"`, el knob queda CONGELADO en `"off"` y
`run_culinary_judge` retorna `None` SIEMPRE — sin excepción, sin log de
error, indistinguible a simple vista de "el juez corrió y no encontró nada"
(recall del juez saldría 0% y el script mentiría con un fail silencioso, no
un crash). Por eso: (1) `argparse` + las escrituras a `os.environ` viven a
nivel de MÓDULO, antes de cualquier `import` local; (2) `graph_orchestrator`
se importa DESPUÉS de esas escrituras, nunca antes; (3) `main()` verifica en
caliente que `go.CULINARY_JUDGE_GUARD == "warn"` y aborta fuerte si no —
cinturón y tirantes: si la trampa vuelve a morder (por un refactor futuro que
mueva un import), este script FALLA RUIDOSO en vez de reportar números falsos.

Uso (desde backend/):
    python scripts/calibrate_culinary_judge.py [--model deepseek-v4-flash] [--thinking]

Salida por capa y por clase (stdout, se copia a mano a
`docs/culinary_coherence.md` sección "Calibración <fecha>"):
    capa1: recall sobre defects expected_by=capa1:* (DEBE ser 1.00, ya cubierto
           por CI vía `test_capa1_atrapa_100pct_de_sus_clases`; este script lo
           re-confirma en el mismo run que el juez) + FPs en los 5 buenos
           (DEBE ser 0, también re-confirmado).
    juez:  recall sobre defects expected_by=juez (criterio spec §6: ≥0.80 para
           autorizar la escalada OFF→warn) + FPs en los ~60 platos de los 5
           buenos (criterio: <5%, ~3 platos).

Matching de recall del juez: `v.day == df["day"] and v.tipo == df["class"]` —
las 3 clases con `expected_by="juez"` en el manifest (combo_absurdo,
nombre_no_corresponde, tecnica_impropia) son EXACTAMENTE 3 de los 5 valores
canónicos que `CulinaryViolation.tipo` acepta (los otros 2, paso_incoherente/
slot_inapropiado, no tienen fixture dedicado en este golden set — el schema
los soporta para violaciones que el juez detecte espontáneamente, no hay
ground-truth etiquetado para ellos hoy).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Parseo de args + escritura de os.environ ANTES de cualquier import local
# (ver TRAMPA Nº1 arriba). NO mover este bloque después de un `import
# graph_orchestrator` ni de un import de un módulo que pueda arrastrarlo.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--model", default=None,
                  help="override MEALFIT_CULINARY_JUDGE_MODEL (default: flash, directiva del owner)")
_ap.add_argument("--thinking", action="store_true",
                  help="activa MEALFIT_CULINARY_JUDGE_THINKING (DeepSeek-only, ignorado en modelos OpenAI)")
_args = _ap.parse_args()

if _args.model:
    os.environ["MEALFIT_CULINARY_JUDGE_MODEL"] = _args.model
os.environ["MEALFIT_CULINARY_JUDGE_GUARD"] = "warn"   # habilita la llamada LLM (nace "off" en prod)
if _args.thinking:
    os.environ["MEALFIT_CULINARY_JUDGE_THINKING"] = "1"

import db_core
db_core.connection_pool.open()          # ⚠️ sin esto el catálogo sale vacío (fuera de FastAPI)

from culinary_coherence import culinary_contract_scan
from shopping_calculator import get_master_ingredients
import graph_orchestrator as go

assert go.CULINARY_JUDGE_GUARD == "warn", (
    f"CULINARY_JUDGE_GUARD={go.CULINARY_JUDGE_GUARD!r}, esperaba 'warn' — algo importó "
    f"graph_orchestrator ANTES de que este script escribiera os.environ (TRAMPA Nº1 del "
    f"docstring). El juez retornaría None en cada llamada y los números serían falsos."
)

_FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "culinary_golden"


def _load(n):
    return json.loads((_FIX / f"{n}.json").read_text(encoding="utf-8"))


async def main():
    cat = get_master_ingredients()
    assert cat, "catálogo vacío — ¿pool DB abierto? ¿NEON_DATABASE_URL en backend/.env?"
    man = _load("golden_manifest")

    print(f"Modelo juez: {go.CULINARY_JUDGE_MODEL}  thinking={go.CULINARY_JUDGE_THINKING}  "
          f"guard={go.CULINARY_JUDGE_GUARD}  timeout={go.CULINARY_JUDGE_TIMEOUT_S}s")

    fp_capa1 = fp_juez = 0
    n_meals_buenos = 0
    for i in range(1, 6):
        bueno = _load(f"golden_{i:02d}_bueno")
        n_meals_buenos += sum(len(d.get("meals") or []) for d in bueno.get("days") or [])
        fp_capa1 += len(culinary_contract_scan(bueno, cat))
        rep = await go.run_culinary_judge(bueno)
        fp_juez += len(rep.violations) if rep else 0

    stats: dict[tuple[str, str], list[int]] = {}   # (clase, capa) -> [atrapados, total]
    for nombre, entry in man["mutados"].items():
        plan = _load(nombre)
        v1 = culinary_contract_scan(plan, cat)
        rep = await go.run_culinary_judge(plan)
        vj = rep.violations if rep else []
        for df in entry["defects"]:
            key = (df["class"], df["expected_by"].split(":")[0])
            hit = (any(x["check"] == df["expected_by"].split(":")[1] and x["day"] == df["day"] for x in v1)
                   if df["expected_by"].startswith("capa1")
                   else any(v.day == df["day"] and v.tipo == df["class"] for v in vj))
            s = stats.setdefault(key, [0, 0])
            s[1] += 1
            s[0] += 1 if hit else 0

    fp_juez_pct = (fp_juez / n_meals_buenos) if n_meals_buenos else 0.0
    print(f"\nFP capa1 sobre buenos: {fp_capa1} (criterio: 0)")
    print(f"FP juez  sobre buenos: {fp_juez} de {n_meals_buenos} meals = {fp_juez_pct:.1%} (criterio: <5%)")
    for (clase, capa), (hit, tot) in sorted(stats.items()):
        print(f"  {capa:6s} {clase:24s} recall {hit}/{tot} = {hit/tot:.0%}")

    # Recall agregado por capa (spec §6: capa1 debe ser 1.00; juez >= 0.80 para autorizar OFF->warn).
    agregados = {}
    for capa_objetivo in ("capa1", "juez"):
        hits = sum(h for (_, cp), (h, t) in stats.items() if cp == capa_objetivo)
        tots = sum(t for (_, cp), (h, t) in stats.items() if cp == capa_objetivo)
        if tots:
            r = hits / tots
            agregados[capa_objetivo] = r
            print(f"  TOTAL {capa_objetivo:6s} recall {hits}/{tots} = {r:.0%}")

    print("\nCriterios (spec §6): capa1 recall 1.00 + 0 FP (contrato F1, ya en CI — este script "
          "lo re-confirma). Juez: recall >= 0.80 en clases-juez para autorizar OFF->warn; "
          "FP juez < 5% sobre los buenos. Escalada warn->block requiere además >=1 semana warn "
          "limpio en prod (spec §7, T14).")

    capa1_ok = agregados.get("capa1", 0.0) >= 1.0 and fp_capa1 == 0
    juez_ok = agregados.get("juez", 0.0) >= 0.80 and fp_juez_pct < 0.05
    print(f"\nVeredicto: capa1={'OK' if capa1_ok else 'FALLA'}  juez={'OK' if juez_ok else 'FALLA'}")


if __name__ == "__main__":
    asyncio.run(main())
