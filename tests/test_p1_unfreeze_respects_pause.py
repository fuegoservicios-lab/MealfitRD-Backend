"""[P1-UNFREEZE-RESPECTS-PAUSE · 2026-08-14] El deshielo inmediato de `/restock` no
puede resucitar un plan que el usuario PAUSÓ.

LA CADENA. Con el plan congelado (Nevera bajo mínimos 48 h) queda `_frozen_at` en
`plan_data`. Si el usuario pasa entonces a modo contador, `generation_status` se
vuelve `paused_by_user` pero **`_frozen_at` sobrevive**: `plan_mode.py` no lo toca
—de hecho su propio comentario CITA ese flag como precedente de «una bandera que
el pickup no lee»—. Después el usuario repone la nevera y pulsa «Ya compré la
lista»: `/restock` llama a `try_unfreeze_plan_for_user`, cuyo SELECT pedía
únicamente `_frozen_at` y jamás `generation_status`. Resultado, sobre un plan que
el usuario apagó:

  1. `_shift_plan_dates_for_freeze` corre las CUATRO anclas del plan
     (`_plan_start_date`, `plan_start_date`, `grocery_start_date`,
     `cycle_start_date`) — justo el dato del que depende la promesa «reanudarlo
     retoma exactamente donde quedó».
  2. `UPDATE plan_chunk_queue SET status='pending' … WHERE status='pending_user_action'`
     resucita la cola POR DETRÁS de la pausa. Es literalmente la condición que la
     alerta `plan_paused_with_live_queue` vigila.
  3. Push al teléfono: «¡Tu plan está de vuelta! 🧊→▶️ … ¡A cocinar!» — de un plan
     que el usuario apagó, con la app cerrada.

LA ASIMETRÍA. La defensa YA EXISTÍA en el hermano: `_plan_freeze_sweep` selecciona
`generation_status` y descarta lo que no esté en `_PLAN_FREEZE_ACTIVE_STATUSES`
(que no incluye `paused_by_user`). El hook la había perdido. Dos caminos hacia la
misma mutación, uno defendido y otro no — la lección del repo otra vez: *una
defensa que vive en un CAMINO y no en el DATO desaparece al abrir un camino nuevo*.

POR QUÉ REUSAR `_PLAN_FREEZE_ACTIVE_STATUSES` y no escribir un `!= 'paused_by_user'`
en el hook: una segunda lista de estados driftearía de la primera (ya pasó con
`canonicalize_diet_type`, que llegó a tener 3 tablas y la del filtro olvidó
'vegetariana'). Un solo criterio, dos consumidores.

NO GASTABA LLM: el gate SQL del pickup (`plan_mode.PICKUP_GATE_SQL`) sigue
cerrado, así que la cola resucitada no llega a generar. El daño es la mutación de
las anclas y la mentira del push.

Tooltip-anchor: P1-UNFREEZE-RESPECTS-PAUSE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"


def _fuente() -> str:
    return _CRON.read_text(encoding="utf-8")


def _cuerpo(nombre: str) -> str:
    """El cuerpo de una función top-level, hasta el siguiente `def` de nivel 0."""
    src = _fuente()
    i = src.find(f"\ndef {nombre}(")
    assert i >= 0, f"[P1-UNFREEZE-RESPECTS-PAUSE] No existe la función {nombre}"
    j = src.find("\ndef ", i + 1)
    cuerpo = src[i:j if j > 0 else len(src)]
    # Fuera comentarios: la prosa explicativa no es una defensa.
    return re.sub(r"^\s*#.*$", "", cuerpo, flags=re.MULTILINE)


def test_el_hook_lee_el_estado_de_generacion():
    cuerpo = _cuerpo("try_unfreeze_plan_for_user")
    assert "generation_status" in cuerpo, (
        "[P1-UNFREEZE-RESPECTS-PAUSE] `try_unfreeze_plan_for_user` no consulta "
        "`generation_status`.\n"
        "Su SELECT pedía solo `_frozen_at`, y la pausa NO limpia ese flag. Un plan "
        "en modo contador seguía pareciendo 'solo congelado', así que «Ya compré la "
        "lista» le corría las fechas, le revivía la cola y mandaba un push «¡Tu plan "
        "está de vuelta!» sobre un plan que el usuario apagó."
    )


def test_el_hook_descarta_los_estados_que_el_sweep_ya_descartaba():
    cuerpo = _cuerpo("try_unfreeze_plan_for_user")
    assert "_PLAN_FREEZE_ACTIVE_STATUSES" in cuerpo, (
        "[P1-UNFREEZE-RESPECTS-PAUSE] El hook no usa `_PLAN_FREEZE_ACTIVE_STATUSES`.\n"
        "El sweep ya filtraba por esa tupla; el hook debe compartir EL MISMO "
        "criterio, no escribir el suyo. Dos listas de estados driftean —ya pasó con "
        "`canonicalize_diet_type`, que llegó a tener 3 tablas y la del filtro olvidó "
        "'vegetariana' y servía Pollo a vegetarianas."
    )


def test_paused_by_user_no_esta_entre_los_estados_activos():
    """El criterio compartido tiene que EXCLUIR de verdad al plan pausado."""
    m = re.search(r"_PLAN_FREEZE_ACTIVE_STATUSES\s*=\s*\(([^)]*)\)", _fuente())
    assert m, "[P1-UNFREEZE-RESPECTS-PAUSE] No se encontró _PLAN_FREEZE_ACTIVE_STATUSES"
    estados = {s.strip().strip("\"'") for s in m.group(1).split(",") if s.strip()}
    assert "paused_by_user" not in estados, (
        "[P1-UNFREEZE-RESPECTS-PAUSE] `paused_by_user` entró en los estados "
        "'activos' del freeze. Si está ahí, tanto el sweep COMO el hook volverán a "
        "descongelar planes pausados — y este arreglo queda anulado desde el otro "
        "extremo."
    )
    assert "complete" in estados, "Sanity: la tupla debe seguir cubriendo los planes vivos"


def test_la_decision_ocurre_antes_de_mutar_nada():
    """Orden: primero descartar, después tocar. Un guard después del UPDATE no guarda."""
    cuerpo = _cuerpo("try_unfreeze_plan_for_user")
    pos_guard = cuerpo.find("_PLAN_FREEZE_ACTIVE_STATUSES")
    pos_mutacion = cuerpo.find("_resume_frozen_plan")
    assert pos_guard >= 0 and pos_mutacion >= 0, "Faltan el guard o la llamada al resume"
    assert pos_guard < pos_mutacion, (
        "[P1-UNFREEZE-RESPECTS-PAUSE] El filtro de estado aparece DESPUÉS de "
        "`_resume_frozen_plan`. Para entonces las anclas ya se corrieron, la cola ya "
        "revivió y el push ya salió: un guard que corre después de la mutación no "
        "es un guard, es un comentario."
    )


def test_el_sweep_conserva_su_propio_filtro():
    """El arreglo del hook no puede desarmar al hermano que ya estaba bien."""
    cuerpo = _cuerpo("_plan_freeze_sweep")
    assert "_PLAN_FREEZE_ACTIVE_STATUSES" in cuerpo, (
        "[P1-UNFREEZE-RESPECTS-PAUSE] El sweep perdió su filtro por estado. Era la "
        "mitad que SÍ estaba defendida; si se va, el cron horario hace exactamente "
        "lo que este P-fix impide en el hook."
    )
