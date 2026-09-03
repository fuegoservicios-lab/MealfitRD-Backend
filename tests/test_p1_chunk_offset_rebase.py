"""[P1-CHUNK-OFFSET-REBASE · 2026-08-07] El shift movía el ancla y dejaba los
`days_offset` quietos, así que el relleno llegaba tarde por exactamente los días
archivados.

Incidente vivo (plan f380821a del dueño, medido en producción 2026-08-07):

    El plan nació el 08-06 con 3 días generados (08-05, 08-06, 08-07) y 7 chunks
    pendientes en offsets 3, 7, 11, 15, 19, 23, 27. El shift rolling archivó los
    dos días ya vividos hacia `_archived_days`, renumeró `days` y reescribió el
    ancla (`grocery_start_date` → hoy, propagada al snapshot del chunk como
    `_plan_start_date`). Los offsets NO se tocaron.

    `execute_after` se calcula `ancla + days_offset` (modo `strict`,
    CHUNK_PROACTIVE_MARGIN_DAYS=0). Con el ancla vieja daba 08-05 + 3 = **08-08**,
    justo el día en que se acababa la ventana viva. Con el ancla nueva da
    08-07 + 3 = **08-10**: dos días de sequía, exactamente los dos archivados.

    Consecuencia visible: el usuario se queda sin plan mañana; y como la lista de
    compras se calcula sobre los días VIVOS, encoge con la ventana (34 → 25 ítems
    medidos) y con ella la Nevera, que es su espejo.

La regla que el propio código ya enuncia para los chunks que ENCOLA en el shift
(`days_offset = len(shifted_days)`, cron_tasks.py y routers/plans.py) no se
aplicaba a los que YA existían. Esta rebase la aplica a la cola entera.

Invariante: **el primer chunk pendiente cubre el día siguiente al último día
vivo** — `primer_offset == len(days)` — y los demás encadenan sumando su
`days_count`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_SQL = Path(__file__).resolve().parents[1]

from constants import (
    chunk_refill_arrives_in_time,
    plan_chunk_offset_moves,
    rebase_pending_chunk_offsets,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. La aritmética (SSOT puro)
# ─────────────────────────────────────────────────────────────────────────────

def test_el_incidente_real_del_dueno():
    """1 día vivo + los 7 chunks de f380821a → el primero pasa a offset 1."""
    chunks = [("wk2", 4), ("wk3", 4), ("wk4", 4), ("wk5", 4),
              ("wk6", 4), ("wk7", 4), ("wk8", 3)]
    out = rebase_pending_chunk_offsets(live_days_count=1, chunks=chunks)
    assert [o for _, o in out] == [1, 5, 9, 13, 17, 21, 25]
    # El primero cubre el día siguiente al último vivo: sin sequía.
    assert out[0][1] == 1


def test_sin_dias_vivos_el_relleno_dispara_ya():
    out = rebase_pending_chunk_offsets(live_days_count=0, chunks=[("a", 4), ("b", 4)])
    assert [o for _, o in out] == [0, 4]


def test_encadena_por_days_count_no_por_un_paso_fijo():
    """Los tamaños son heterogéneos (split_with_absorb): 3+4+4 ≠ 3 pasos iguales."""
    out = rebase_pending_chunk_offsets(live_days_count=2, chunks=[("a", 3), ("b", 4), ("c", 2)])
    assert [o for _, o in out] == [2, 5, 9]


def test_es_idempotente():
    """Correrla dos veces sobre su propia salida no mueve nada."""
    chunks = [("a", 4), ("b", 4)]
    primera = rebase_pending_chunk_offsets(3, chunks)
    segunda = rebase_pending_chunk_offsets(3, chunks)
    assert primera == segunda == [("a", 3), ("b", 7)]


def test_cola_vacia_no_revienta():
    assert rebase_pending_chunk_offsets(5, []) == []


@pytest.mark.parametrize("live", [-1, None, "x"])
def test_entradas_basura_degradan_a_cero_no_a_excepcion(live):
    """El shift corre dentro de una transacción con lock: reventar aquí abortaría
    el shift entero y dejaría al usuario sin renovación."""
    out = rebase_pending_chunk_offsets(live, [("a", 4)])
    assert out == [("a", 0)]


def test_days_count_invalido_no_colapsa_la_cadena():
    """Un `days_count` corrupto no debe hacer que dos chunks compartan offset."""
    out = rebase_pending_chunk_offsets(1, [("a", 0), ("b", 4)])
    assert out[0][1] == 1 and out[1][1] > out[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Los movimientos que van al UPDATE — el SIGNO del delta es el fix
# ─────────────────────────────────────────────────────────────────────────────

def test_el_delta_adelanta_el_chunk_del_dueno_dos_dias():
    """`execute_after -= delta`, así que delta POSITIVO adelanta. El plan real:
    offset 3 con 1 día vivo ⇒ nuevo 1, delta +2 ⇒ 08-10 vuelve a 08-08."""
    filas = [("wk2", 3, 4), ("wk3", 7, 4), ("wk4", 11, 4)]
    assert plan_chunk_offset_moves(1, filas) == [
        ("wk2", 1, 2, 0), ("wk3", 5, 2, 4), ("wk4", 9, 2, 8),
    ]


def test_no_toca_los_que_ya_estan_en_su_sitio():
    """Un UPDATE inocuo igual escribe `updated_at` y ensucia el rastro de quién
    movió el chunk — el mismo rastro con el que encontré este bug."""
    assert plan_chunk_offset_moves(1, [("wk2", 1, 4), ("wk3", 5, 4)]) == []


def test_delta_negativo_cuando_toca_retrasar():
    """Si la ventana creció, el chunk se retrasa: delta negativo ⇒ el UPDATE
    SUMA días. Sin este caso, invertir el signo pasaría desapercibido."""
    movimientos = plan_chunk_offset_moves(5, [("wk2", 1, 4)])
    assert movimientos == [("wk2", 5, -4, 0)]


def test_los_chunks_vencidos_no_colapsan_todos_a_la_misma_hora():
    """Caso real medido en el ensayo (plan 9cf5e313, 3 días vivos, offsets
    9/12/15…): sin suelo por chunk, TODOS los vencidos caen a `NOW()` y dos
    generaciones concurrentes escriben el mismo `plan_data`. El suelo es
    `nuevo_offset - vivos`: el primero sale ya, el siguiente en su turno."""
    movimientos = plan_chunk_offset_moves(3, [("wk4", 9, 3), ("wk5", 12, 3), ("wk6", 15, 3)])
    suelos = [m[3] for m in movimientos]
    assert suelos == [0, 3, 6]
    assert len(set(suelos)) == len(suelos), "dos chunks compartirían instante de disparo"


def test_sin_chunks_no_hay_movimientos():
    assert plan_chunk_offset_moves(3, []) == []


def test_los_dias_que_ya_vienen_en_camino_cuentan_como_ventana():
    """Un chunk EN VUELO ya está fabricando días que el plan aún no tiene. El
    ejecutor SQL le suma su `days_count` a la ventana antes de llamar aquí; sin
    eso el siguiente pendiente se reancla ENCIMA del que está corriendo y los dos
    generan el mismo tramo. Observado en vivo el 2026-08-07 (plan 76a6836d,
    semana 3 en `processing` y la 4 aterrizando sobre ella).

    Aquí se comprueba la consecuencia: con la ventana ya sumada (0 vivos + 4 en
    vuelo), el pendiente arranca DESPUÉS, no encima."""
    assert plan_chunk_offset_moves(0 + 4, [("wk4", 11, 4)]) == [("wk4", 4, 7, 0)]
    # ...y sin sumar el vuelo aterrizaría en 0, pisando al que corre.
    assert plan_chunk_offset_moves(0, [("wk4", 11, 4)])[0][1] == 0


def test_el_ejecutor_sql_suma_los_chunks_en_vuelo():
    """Parser-based: el SELECT de `processing` no puede desaparecer en un
    refactor sin que esto lo cante."""
    src = (_BACKEND_SQL / "cron_tasks.py").read_text(encoding="utf-8")
    ini = src.index("def _rebase_pending_chunk_offsets_sql(")
    cuerpo = src[ini: src.index("\ndef ", ini + 10)]
    assert "status = 'processing'" in cuerpo and "en_vuelo" in cuerpo, (
        "El reanclaje dejó de contar los chunks en vuelo: el siguiente pendiente "
        "volverá a aterrizar encima del que está generando."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. El predicado de sequía (lo que la telemetría podrá vigilar)
# ─────────────────────────────────────────────────────────────────────────────

def test_el_estado_real_del_plan_del_dueno_es_tarde():
    # offset 3 contra 1 día vivo = 2 días de sequía (los 2 archivados).
    assert chunk_refill_arrives_in_time(live_days_count=1, next_offset=3) is False


def test_tras_la_rebase_llega_a_tiempo():
    assert chunk_refill_arrives_in_time(live_days_count=1, next_offset=1) is True


def test_adelantado_tambien_esta_a_tiempo():
    """`safety_margin` adelanta chunks: offset < vivos NO es una anomalía."""
    assert chunk_refill_arrives_in_time(live_days_count=4, next_offset=2) is True


def test_sin_chunk_pendiente_no_opina():
    assert chunk_refill_arrives_in_time(live_days_count=1, next_offset=None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Las DOS ramas del shift la aplican (parser-based)
#
# Lección de 2026-08-05: «un P-fix aplicado a una rama de dos reproduce su bug
# en la otra». El shift vive duplicado en `api_shift_plan` (HTTP) y
# `_background_shift_plan_for_user` (cron). Si mañana alguien arregla una sola,
# este test lo caza antes que producción.
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND = _BACKEND_SQL


def _cuerpo(path: Path, nombre: str) -> str:
    src = path.read_text(encoding="utf-8")
    ini = src.index(f"def {nombre}(")
    sig = re.search(r"\ndef [a-zA-Z_]", src[ini + 10:])
    return src[ini: ini + 10 + (sig.start() if sig else len(src))]


@pytest.mark.parametrize(
    "fichero,funcion",
    [
        ("cron_tasks.py", "_background_shift_plan_for_user"),
        ("routers/plans.py", "api_shift_plan"),
    ],
)
def test_ambas_ramas_del_shift_rebasan_la_cola(fichero, funcion):
    cuerpo = _cuerpo(_BACKEND / fichero, funcion)
    assert "rebase_pending_chunk_offsets" in cuerpo, (
        f"{fichero}::{funcion} no rebasa los offsets de los chunks pendientes. "
        f"El ancla se reescribe a hoy en las dos ramas; dejar los offsets quietos "
        f"reintroduce P1-CHUNK-OFFSET-REBASE (relleno tarde por los días archivados)."
    )
    assert "P1-CHUNK-OFFSET-REBASE" in cuerpo, (
        f"{fichero}::{funcion} aplica la rebase sin el marker que la explica."
    )


def test_el_knob_de_rollback_existe():
    """Convención del repo: cambios de comportamiento revertibles sin redeploy."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "MEALFIT_CHUNK_OFFSET_REBASE" in src
