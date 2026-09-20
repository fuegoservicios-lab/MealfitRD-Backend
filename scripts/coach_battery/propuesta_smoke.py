"""[P1-PLAN-LOTE-132] Humo SIN LLM de `proponer_comida` y del bloque «LO QUE LE FALTA HOY», contra los datos reales de la
cuenta de la batería, en SOLO LECTURA (hereda la guardia y los parches de `run_battery`). Cero gasto de IA.

    MEALFIT_BATTERY_ENV=<.env>  MEALFIT_BATTERY_USER_ID=<uuid>  python scripts/coach_battery/propuesta_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_battery as rb  # noqa: E402  (guardia de solo lectura + pool + reloj y diario simulados)

import coach_day_context as cdc  # noqa: E402

DIA_A_MEDIAS = [
    {"meal_type": "desayuno", "meal_name": "Mangú con huevos", "calories": 520, "protein": 24, "carbs": 62, "healthy_fats": 19},
    {"meal_type": "almuerzo", "meal_name": "Arroz con pollo guisado", "calories": 690, "protein": 38, "carbs": 85, "healthy_fats": 18},
]


def main() -> int:
    if not rb.OWNER:
        print("Falta MEALFIT_BATTERY_USER_ID")
        return 2
    persona = rb.load_persona("dueno")
    plan = persona["plan"] if isinstance(persona["plan"], dict) else None
    plan_data = (plan or {}).get("plan_data") if isinstance((plan or {}).get("plan_data"), dict) else plan
    for hora, diario, llamadas in (
        ("21:05", DIA_A_MEDIAS, [{}, {"solo_con_nevera": True}, {"meal_type": "cena", "proteina_objetivo": 45}]),
        ("08:10", [], [{"meal_type": "desayuno"}, {"meal_type": "desayuno", "max_minutos": 10}]),
        ("21:10", [], [{}]),
    ):
        rb.set_hora(hora)
        rb.DIARIO_SIM["rows"] = diario
        print("=" * 110)
        print(f"HORA {hora} · diario simulado: {len(diario)} registros")
        print(cdc.build_day_gap_context(persona["form_data"], plan_data, diario, rb.SIM["hora"],
                                        (persona["form_data"] or {}).get("scheduleType")))
        for kw in llamadas:
            t0 = time.monotonic()
            out = rb.tools.proponer_comida.func(user_id=rb.OWNER, **kw)
            print("-" * 110)
            print(f"proponer_comida({kw})  [{time.monotonic() - t0:.2f}s]")
            print(out)
    print("escrituras bloqueadas:", [b for b in rb.BLOCKED if not rb._TELEMETRY_RE.match(b)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
