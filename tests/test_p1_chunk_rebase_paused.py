"""[P1-CHUNK-REBASE-PAUSED · 2026-08-08] El reanclaje de offsets era CIEGO a los
chunks pausados y reasignaba sus días a los siguientes de la cola.

Incidente vivo (plan f380821a del dueño, medido en producción 2026-08-08):

    A las 04:00 el shift archivó el último día vivo (days=[]) y la semana 2
    disparó a tiempo (el fix P1-CHUNK-OFFSET-REBASE funcionó). Pero el LLM no
    convergió contra la nevera (25 ítems) tras los reintentos y w2/w3 quedaron
    en `pending_user_action:pantry_violation_after_retries` (TTL 12h).

    A las 06:32 otro shift corrió la rebase. El SELECT de la cadena filtraba
    `status IN ('pending','stale')` y el `en_vuelo` solo contaba `processing`:
    los pausados eran INVISIBLES. Resultado medido en la cola real:

        w2 pausado  offset 0   ┐ mismos días
        w4 pending  offset 0   ┘ (w4 dispararía el 08-09 a las 04:00)
        w3 pausado  offset 4   ┐ mismos días
        w5 pending  offset 4   ┘

    Un chunk pausado NO está muerto: el recovery cron lo resucita (TTL →
    flexible_mode) y entrega sus días. Dos chunks con el mismo offset son dos
    generaciones escribiendo el MISMO tramo de `plan_data` — días duplicados y
    doble costo LLM. Es la clase hermana del caso `processing` que el propio
    docstring del ejecutor ya documentaba.

La regla: la cadena de la rebase se compone de TODO chunk que aún va a entregar
días — `pending`, `stale` y `pending_user_action` — en orden de `week_number`.
Los `processing` siguen contándose como ventana (`en_vuelo`): su generación ya
arrancó con su snapshot y mover su offset a mitad de vuelo sería otra clase de
bug. Los `dead_letter`/`cancelled`/`completed` no entregan nada: fuera.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cursor fake que respeta los filtros de status del SQL real.
#    Si el código SELECTea solo pending/stale, el fake devuelve solo esos —
#    igual que Postgres. Así el test falla por el FILTRO, no por el mock.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, rows):
        # rows: dicts con id, status, week_number, days_offset, days_count,
        # execute_after_dias (relativo, para verificar el movimiento).
        self.rows = {r["id"]: dict(r) for r in rows}
        self._result = None
        self.updates = []

    @staticmethod
    def _statuses_in(sql: str) -> list[str]:
        m = re.search(r"status\s+IN\s*\(([^)]*)\)", sql)
        if not m:
            return []
        return re.findall(r"'([a-z_]+)'", m.group(1))

    def execute(self, sql, params=None):
        params = params or ()
        if "SUM(days_count)" in sql:
            en_vuelo = sum(
                r["days_count"] for r in self.rows.values() if r["status"] == "processing"
            )
            self._result = [{"en_vuelo": en_vuelo}]
        elif sql.strip().upper().startswith("SELECT"):
            estados = self._statuses_in(sql)
            filas = [
                {"id": r["id"], "days_offset": r["days_offset"], "days_count": r["days_count"]}
                for r in sorted(self.rows.values(), key=lambda x: x["week_number"])
                if r["status"] in estados
            ]
            self._result = filas
        elif sql.strip().upper().startswith("UPDATE"):
            nuevo_offset, delta, _dias_turno, chunk_id = params
            estados = self._statuses_in(sql)
            fila = self.rows.get(chunk_id)
            if fila and fila["status"] in estados:
                fila["days_offset"] = nuevo_offset
                fila["execute_after_dias"] = fila["execute_after_dias"] - delta
                self.updates.append((chunk_id, nuevo_offset, delta))
            self._result = []
        else:  # pragma: no cover - una query nueva debe fallar visible
            raise AssertionError(f"SQL inesperado en el fake: {sql[:120]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


def _cola_del_incidente():
    """La cola REAL del plan f380821a a las 06:32 del 2026-08-08 (post-colisión)."""
    return [
        {"id": "w2", "status": "pending_user_action", "week_number": 2, "days_offset": 0, "days_count": 4, "execute_after_dias": 0},
        {"id": "w3", "status": "pending_user_action", "week_number": 3, "days_offset": 4, "days_count": 4, "execute_after_dias": 0},
        {"id": "w4", "status": "pending", "week_number": 4, "days_offset": 0, "days_count": 4, "execute_after_dias": 1},
        {"id": "w5", "status": "pending", "week_number": 5, "days_offset": 4, "days_count": 4, "execute_after_dias": 5},
        {"id": "w6", "status": "pending", "week_number": 6, "days_offset": 8, "days_count": 4, "execute_after_dias": 9},
        {"id": "w7", "status": "pending", "week_number": 7, "days_offset": 12, "days_count": 4, "execute_after_dias": 13},
        {"id": "w8", "status": "pending", "week_number": 8, "days_offset": 16, "days_count": 3, "execute_after_dias": 17},
    ]


def test_los_pausados_reservan_su_tramo_y_los_pendientes_encadenan_detras():
    """El incidente: con w2/w3 pausados en 0/4, los pendientes deben quedar en
    8/12/16/20/24 — no repartirse los tramos de los pausados."""
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    cur = _FakeCursor(_cola_del_incidente())
    _rebase_pending_chunk_offsets_sql(cur, "f380821a", live_days_count=0)

    offsets = {rid: r["days_offset"] for rid, r in cur.rows.items()}
    assert offsets == {"w2": 0, "w3": 4, "w4": 8, "w5": 12, "w6": 16, "w7": 20, "w8": 24}, (
        "La cadena no reservó el tramo de los chunks pausados: dos chunks "
        "comparten offset y generarán el MISMO tramo de plan_data."
    )
    # Sin duplicados jamás — la aserción que producción violó.
    valores = list(offsets.values())
    assert len(set(valores)) == len(valores)


def test_un_pausado_desanclado_tambien_se_reancla():
    """El caso original de P1-CHUNK-OFFSET-REBASE, pero con la fila pausada:
    si el ancla se movió mientras w2 dormía, su offset también se corrige —
    el recovery lo resucita con `execute_after=NOW()` y NO recalcula offsets."""
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    filas = [
        {"id": "w2", "status": "pending_user_action", "week_number": 2, "days_offset": 3, "days_count": 4, "execute_after_dias": 2},
        {"id": "w3", "status": "pending", "week_number": 3, "days_offset": 7, "days_count": 4, "execute_after_dias": 6},
    ]
    cur = _FakeCursor(filas)
    _rebase_pending_chunk_offsets_sql(cur, "plan-x", live_days_count=1)

    assert cur.rows["w2"]["days_offset"] == 1, (
        "El pausado quedó desanclado: al resucitar generará días corridos "
        "por exactamente los días archivados mientras dormía."
    )
    assert cur.rows["w3"]["days_offset"] == 5


def test_los_processing_siguen_contando_como_ventana_no_como_cadena():
    """Un chunk en vuelo NO se mueve (su generación ya corre con su snapshot),
    pero sus días empujan a los demás — comportamiento P1-CHUNK-OFFSET-REBASE
    original que este fix NO debe romper."""
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    filas = [
        {"id": "w2", "status": "processing", "week_number": 2, "days_offset": 0, "days_count": 4, "execute_after_dias": 0},
        {"id": "w3", "status": "pending_user_action", "week_number": 3, "days_offset": 9, "days_count": 4, "execute_after_dias": 0},
        {"id": "w4", "status": "pending", "week_number": 4, "days_offset": 13, "days_count": 4, "execute_after_dias": 9},
    ]
    cur = _FakeCursor(filas)
    _rebase_pending_chunk_offsets_sql(cur, "plan-y", live_days_count=0)

    assert cur.rows["w2"]["days_offset"] == 0, "un chunk en vuelo no se toca"
    assert cur.rows["w3"]["days_offset"] == 4, "el pausado arranca tras la ventana (0 vivos + 4 en vuelo)"
    assert cur.rows["w4"]["days_offset"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parser-based: los DOS sitios del SQL (SELECT de la cadena y guard del
#    UPDATE) incluyen 'pending_user_action'. Si solo uno lo lleva, el UPDATE
#    hace no-op silencioso sobre los pausados (rowcount 0) — el fix a medias.
# ─────────────────────────────────────────────────────────────────────────────

def _cuerpo_del_ejecutor() -> str:
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    ini = src.index("def _rebase_pending_chunk_offsets_sql(")
    return src[ini: src.index("\ndef ", ini + 10)]


def test_el_select_y_el_update_incluyen_a_los_pausados():
    cuerpo = _cuerpo_del_ejecutor()
    ocurrencias = cuerpo.count("'pending_user_action'")
    assert ocurrencias >= 2, (
        "P1-CHUNK-REBASE-PAUSED: la cadena de la rebase debe incluir "
        "'pending_user_action' en el SELECT Y en el guard del UPDATE "
        f"(encontradas {ocurrencias}/2). Un chunk pausado sigue siendo un "
        "chunk que ENTREGARÁ días: dejarlo fuera reparte su tramo al "
        "siguiente pendiente y ambos generan los mismos días."
    )


def test_en_vuelo_sigue_contando_solo_processing():
    """`en_vuelo` NO debe absorber a los pausados: sus filas van en la CADENA
    (con offset propio), no en la ventana — meterlos en ambos lados los
    contaría doble."""
    cuerpo = _cuerpo_del_ejecutor()
    m = re.search(r"SUM\(days_count\).*?status\s*(?:=|IN)\s*\(?\s*'([^']+)'", cuerpo, re.S)
    assert m and m.group(1) == "processing", (
        "La ventana en_vuelo debe seguir contando SOLO 'processing'."
    )
