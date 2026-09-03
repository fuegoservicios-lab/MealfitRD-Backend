# -*- coding: utf-8 -*-
"""[P1-FREEZE-OWNS-ITS-CHUNKS · 2026-08-14] Un plan congelado es dueño de sus
chunks: nadie más los revive.

Caso real (plan cb361844, cuenta de prueba): nevera vacía 48h → P1-PLAN-FREEZE
congeló el plan y pasó TODOS los pending → pending_user_action, para
reanudarlos al reponer. Pero `pending_user_action` tiene OTRO dueño más viejo:
`_recover_pantry_paused_chunks` (cron de 1 min) que a las 12-24h «rescata» esos
chunks activando modo flexible con `execute_after = NOW()`. Resultado medido:
24h después del congelado, el rescate revivió las SIETE semanas en cadena
(exec = done del anterior + 1 min) y quemó 7 generaciones LLM en UNA HORA —
21:08 → 22:06 del 13-ago — para un plan cuyos días «no corren», con semanas
que no vencían hasta septiembre.

Es la trampa que P1-PLAN-MODE esquivó a conciencia: su análisis documenta que
la pausa eligió `cancelled` PORQUE «el recovery resucita los
pending_user_action a las 12h». El congelado (julio) eligió ese mismo estado
sin auditar a sus dueños — la lección de P1-PAUSE-GC-SURVIVAL, del otro lado.

Dos capas (la receta de P1-PLAN-MODE):
  1. El rescate EXCLUYE chunks de planes congelados en su SELECT — un filtro
     en el punto de estrangulamiento cubre las ~6 ramas que escriben
     exec=NOW().
  2. El pickup del worker tampoco los toma (gate junto al de plan_mode): si
     cualquier otro mecanismo los flipea a pending, no hay gasto LLM.

La reanudación no cambia: `_resume_frozen_plan` limpia `_frozen_at` ANTES de
flipear los chunks a pending, así que el gate no bloquea lo reanudado.
"""
from __future__ import annotations

import re
from pathlib import Path

import cron_tasks as ct

SRC = Path(ct.__file__).read_text(encoding="utf-8")


def _cuerpo(desde: str) -> str:
    i = SRC.index(desde)
    return SRC[i:SRC.index("\ndef ", i + 10)]


# ── Capa 1: el rescate no toca lo congelado ─────────────────────────────────

def test_el_rescate_excluye_planes_congelados_en_el_select():
    """El filtro va en el SELECT (punto de estrangulamiento): las ~6 ramas del
    cron que escriben execute_after=NOW() quedan cubiertas de una vez."""
    cuerpo = _cuerpo("def _recover_pantry_paused_chunks")
    m = re.search(r"SELECT[\s\S]*?FROM plan_chunk_queue[\s\S]*?LIMIT 50", cuerpo)
    assert m, "no se encontró el SELECT de paused_rows"
    sel = m.group(0)
    assert re.search(r"NOT EXISTS[\s\S]*?meal_plans[\s\S]*?'_frozen_at'", sel), (
        "el rescate vuelve a ver chunks de planes CONGELADOS: a las 12-24h los "
        "revive con exec=NOW() y quema todas las semanas en cadena (burst real "
        "de 7 chunks LLM en 1h, plan cb361844)"
    )


# ── Capa 2: el pickup tampoco (la capa que detiene el gasto) ────────────────

def test_el_pickup_gatea_planes_congelados_en_ambas_ramas():
    """Mismo mecanismo que el gate de plan_mode: fragmento constante inyectado
    donde vive el token __PLAN_MODE_GATE__ (presente en las DOS CTEs)."""
    assert "_FREEZE_GATE_SQL" in SRC, "falta el fragmento del gate de congelado"
    m = re.search(r'_FREEZE_GATE_SQL\s*=\s*"""([\s\S]*?)"""', SRC)
    assert m, "el fragmento debe ser una constante (cero entrada de usuario)"
    frag = m.group(1)
    assert re.search(r"NOT EXISTS[\s\S]*?meal_plans[\s\S]*?q1\.meal_plan_id[\s\S]*?'_frozen_at'", frag), (
        "el gate debe filtrar por el flag _frozen_at contra la CTE q1"
    )
    # Y se inyecta JUNTO al gate de plan_mode: la composición `_gates_sql`
    # concatena ambos fragmentos y es lo que sustituye al token.
    linea = next((l for l in SRC.splitlines()
                  if "_gates_sql = " in l and "_pm_gate_sql" in l), "")
    assert "_FREEZE_GATE_SQL" in linea, (
        "el fragmento existe pero no entra en la composición del gate: el "
        "pickup seguiría tomando chunks de planes congelados"
    )


def test_el_gate_respeta_el_knob_del_freeze():
    """MEALFIT_PLAN_FREEZE_ENABLED apaga el feature entero: sin freeze no hay
    flag que leer y el gate debe desaparecer con él (no bloquear por un
    _frozen_at huérfano de una era anterior)."""
    linea = next((l for l in SRC.splitlines()
                  if "_gates_sql = " in l and "_FREEZE_GATE_SQL" in l), "")
    assert "_freeze_on" in linea, (
        "el gate del congelado debe estar condicionado al knob del freeze"
    )
    assert 'MEALFIT_PLAN_FREEZE_ENABLED", True' in SRC.split("_freeze_on = ")[1][:80], (
        "_freeze_on debe leer MEALFIT_PLAN_FREEZE_ENABLED (el knob del propio feature)"
    )


# ── La reanudación sigue funcionando ────────────────────────────────────────

def test_resume_limpia_el_flag_antes_de_flipear_chunks():
    """Orden vital con el gate nuevo: si los chunks pasaran a pending ANTES de
    limpiar _frozen_at, el pickup los ignoraría hasta el siguiente sweep. El
    orden actual (flag primero) ya era correcto — este test lo vuelve contrato."""
    cuerpo = _cuerpo("def _resume_frozen_plan")
    i_flag = cuerpo.index("'_frozen_at'")
    i_chunks = cuerpo.index("SET status = 'pending'")
    assert i_flag < i_chunks, (
        "la reanudación debe limpiar _frozen_at ANTES de devolver los chunks a "
        "pending, o el gate del pickup los deja invisibles"
    )
