"""[P1-PLAN-LOTE-1014 · A-04] Ejecuta las LECTURAS del chat contra la base real y falla si alguna protección cae en fail-open.

Por qué existe: el 14-nov el tope diario de gasto y la alerta de pico de coste llevaban todo el día consultando columnas que no
existen; las pruebas simulaban la base y el error se tragaba en silencio. Esto corre DESPUÉS del deploy, en el VPS:

    /home/ubuntu/miniforge3/envs/mealfit/bin/python scripts/verificar_chat_contra_base.py [user_id]

Solo lectura: las escrituras (alertas) van a un sumidero. Sale con 0 si todo va bien, 1 si alguna consulta falló o algún sitio
`fail_open` contó un fallo. Es un AVISO del deploy, no revierte nada.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SECCIONES_SALUD = ("reliability", "feedback", "fallback_hourly_burst", "cost_spike_3x")


def verificar(q, owner: str) -> list[str]:
    """Devuelve la lista de problemas (vacía = todo bien). `q` es `execute_sql_query`; se inyecta para poder probarla."""
    import fail_open
    import chat_hard_turn
    import chat_observability_alerts as coa
    import chat_phase_metrics as cpm

    fail_open.reset_para_tests()
    problemas: list[str] = []

    def _intenta(nombre, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            problemas.append(f"{nombre}: {type(exc).__name__}: {str(exc)[:200]}")
            return None

    presupuesto = _intenta("check_user_daily_chat_budget",
                           lambda: chat_hard_turn.check_user_daily_chat_budget(owner, execute_sql_query=q))
    if isinstance(presupuesto, dict) and "status" not in presupuesto:
        problemas.append("check_user_daily_chat_budget: respuesta sin 'status'")

    salud = _intenta("evaluar_salud_coach_y_alertar",
                     lambda: coa.evaluar_salud_coach_y_alertar(execute_sql_query=q, execute_sql_write=lambda *a, **k: None,
                                                               include_hourly_and_cost=True))
    if isinstance(salud, dict) and salud.get("enabled", True):
        for sec in SECCIONES_SALUD:
            if sec not in salud:
                problemas.append(f"evaluar_salud_coach_y_alertar: falta la sección '{sec}' (su consulta cayó)")

    for nombre in ("query_chat_phase_percentiles", "query_chat_efficiency_breakdown", "query_chat_ops_insights",
                   "query_chat_cache_summary", "query_chat_by_arm"):
        _intenta(nombre, lambda n=nombre: getattr(cpm, n)(24))

    for sitio, v in fail_open.snapshot().items():
        problemas.append(f"fail_open[{sitio}] contó {v['n']} fallo(s): {v['ultimo_error']}")
    return problemas


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(BACKEND / ".env")
    import db_core

    db_core.connection_pool.open(wait=True, timeout=20)
    owner = (sys.argv[1] if len(sys.argv) > 1 else "") or (os.environ.get("MEALFIT_ADMIN_USER_IDS", "").split(",")[0].strip())
    if not owner:
        print("falta el user_id (argumento o MEALFIT_ADMIN_USER_IDS)")
        return 2
    problemas = verificar(db_core.execute_sql_query, owner)
    if problemas:
        print("❌ LAS LECTURAS DEL CHAT TIENEN PROBLEMAS CONTRA LA BASE REAL:")
        for p in problemas:
            print("  -", p)
        return 1
    print("✅ lecturas del chat contra la base real: sin fallos ni fail-open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
