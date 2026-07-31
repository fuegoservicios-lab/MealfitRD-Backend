"""[P1-ALERT-AFFECTED-UUID · 2026-07-30] Una alerta que nunca llegó a existir.

Investigando por qué el chunk de la semana 2 del owner quedó pausado
(P1-SYNTH-LESSON-NOT-STUB), la tabla `system_alerts` no tenía NI UNA fila de
`chunk_synthesis_overload_per_user` — en toda su vida — pese a que el chunk sí se
pausó y el log `[P0-B/PAUSED]` sí salió, y ese log va DESPUÉS del insert.

El journal de producción dio la causa exacta:

    [P0-B/ALERT] No se pudo persistir alert per-user:
    Object of type UUID is not JSON serializable

`json.dumps([user_id])` con `user_id` siendo un `uuid.UUID` (psycopg3 devuelve UUID
para columnas uuid). El `except` de alrededor lo tragaba como warning. Resultado: el
chunk se pausaba, el usuario recibía su push… y el operador no se enteraba nunca.

Nota de método: reproducir el INSERT a mano NO encontró el bug — pasé un `user_id`
de tipo `str` y funcionó. La diferencia entre mi valor inventado y el real ERA el
bug. De ahí que este test invoque la función de verdad con los tipos del callsite
real en vez de afirmar sobre un payload forjado.
"""
import json
import uuid
from pathlib import Path

import pytest

CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"


@pytest.fixture
def guard_sin_db(monkeypatch):
    """Aísla `_pause_chunk_for_synthesis_overload` de la DB y del push real.

    Devuelve la lista de (sql, params) que el guard intentó escribir.
    """
    import cron_tasks as ct

    escrituras = []

    monkeypatch.setattr(ct, "execute_sql_query", lambda *a, **k: None)      # sin cooldown
    monkeypatch.setattr(ct, "execute_sql_write",
                        lambda sql, params=None, *a, **k: escrituras.append((sql, params)))
    monkeypatch.setattr(ct, "_cas_pause_chunk_to_pending_user_action",
                        lambda *a, **k: True)

    # El guard lanza un hilo de push. Lo neutralizamos: un test NO manda notificaciones.
    import utils_push
    enviados = []
    monkeypatch.setattr(utils_push, "send_push_notification",
                        lambda **kw: enviados.append(kw))

    return escrituras


def test_alerta_se_persiste_con_uuids_reales(guard_sin_db):
    """Con los tipos del callsite real (UUID, no str), la alerta DEBE intentarse.

    Sin el `str(...)`, `json.dumps` lanza ANTES de llamar a `execute_sql_write`, así
    que el insert ni se intenta y esta aserción cae — que es exactamente lo que
    pasaba en producción.
    """
    from cron_tasks import _pause_chunk_for_synthesis_overload

    task_id = uuid.uuid4()
    user_id = uuid.uuid4()
    meal_plan_id = uuid.uuid4()

    ok = _pause_chunk_for_synthesis_overload(
        task_id=task_id,
        snap={},
        user_id=user_id,
        meal_plan_id=meal_plan_id,
        week_number=2,
        ratio_info={"synth": 26, "total": 4, "ratio": 6.5, "exceeded": True},
        source="last_chunk_learning_synth",
    )
    assert ok is True, "el guard debería reportar que pausó"

    inserts = [(s, p) for s, p in guard_sin_db if "system_alerts" in s]
    assert inserts, (
        "P1-ALERT-AFFECTED-UUID regresión: el guard pausó el chunk pero NO intentó "
        "escribir la alerta en system_alerts. Con user_id/task_id de tipo UUID, "
        "`json.dumps` lanza y el `except` se lo traga: el operador queda ciego ante "
        "una pausa que sí ocurrió."
    )

    _sql, params = inserts[0]
    # Los dos payloads jsonb del insert deben ser JSON válido y sin objetos UUID.
    payloads = [p for p in params if isinstance(p, str) and p.startswith(("{", "["))]
    assert len(payloads) >= 2, f"esperaba metadata + affected_user_ids, vi {payloads!r}"
    for crudo in payloads:
        json.loads(crudo)  # lanza si no es JSON válido

    afectados = json.loads([p for p in payloads if p.startswith("[")][0])
    assert afectados == [str(user_id)], (
        f"affected_user_ids debe llevar el user_id como string; vi {afectados!r}"
    )

    metadata = json.loads([p for p in payloads if p.startswith("{")][0])
    assert metadata["task_id"] == str(task_id), (
        "el task_id de metadata también es UUID en el callsite real y también hay "
        "que coercionarlo — es el segundo campo que hacía lanzar al dumps"
    )


def test_ningun_affected_user_ids_pasa_un_uuid_crudo():
    """Blanket sobre la clase, no sobre el incidente.

    7 de los 14 sitios de `json.dumps([user_id])` en cron_tasks ya hacían `str()` y 7
    no: la misma asimetría "el fix aterrizó en unas superficies y no en sus hermanas"
    que produjo el bug. 5 de esos 7 estaban vivos solo porque su `user_id` resultaba
    ser un str — dependencia invisible del tipo que llegue por el callsite.
    """
    src = CRON.read_text(encoding="utf-8")
    crudos = [
        (n, linea.strip())
        for n, linea in enumerate(src.split("\n"), start=1)
        if "json.dumps([user_id]" in linea and "str(user_id)" not in linea
    ]
    assert not crudos, (
        "P1-ALERT-AFFECTED-UUID regresión: estos `affected_user_ids` pasan `user_id` "
        "sin `str()`. Si el callsite lo saca de una fila (psycopg3 devuelve `uuid.UUID` "
        "para columnas uuid), el insert lanza y la alerta se pierde en silencio:\n  "
        + "\n  ".join(f"linea {n}: {t}" for n, t in crudos)
    )
