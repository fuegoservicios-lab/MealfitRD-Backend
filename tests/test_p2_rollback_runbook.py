"""[P2-ROLLBACK-RUNBOOK · 2026-08-21] El runbook prometía un rollback limpio que no lo es.

§5 dice que apagar `MEALFIT_COUNTRY_SYSTEM` «vuelve el motor a byte-identidad DO en segundos». Es
cierto para los planes **nuevos** y falso para los que ya existen.

Medido en la cola viva el 2026-08-21: **7 chunks** `pending`/`pending_user_action` llevan
`country='US'` en su `pipeline_snapshot` y despiertan por su `execute_after`. Con el knob apagado el
worker sigue leyendo ese snapshot, pero `country_for_form_data` devuelve `'DO'` incondicional. Las
semanas 2-8 se generarían **criollas dentro de un plan estadounidense** que además sigue marcado
`beta_no_prices`: un híbrido que no es ni el estado nuevo ni el viejo.

Un runbook que promete más de lo que entrega es peor que uno incompleto: el operador lo sigue
CREYENDO que terminó, y el daño aparece días después cuando despierta el primer chunk.

POR QUÉ EL PASO ES *DECIDIR* Y NO *EJECUTAR*. Las dos salidas son defendibles y ninguna es
obviamente correcta: cancelar deja el plan incompleto; reescribir el snapshot a 'DO' le cambia la
cocina a alguien que eligió otra. Por eso el runbook pone las dos consultas con su consecuencia
escrita al lado, en vez de un comando único que decida por el operador. Lo que no es defendible es
apagar el knob y dejar que el worker resuelva la ambigüedad solo.

Este fichero no cambia código: ancla que el aviso siga ahí y que siga siendo accionable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_DOC = Path(__file__).resolve().parent.parent / "docs" / "country_system_f1.md"


@pytest.fixture(scope="module")
def doc() -> str:
    if not _DOC.is_file():
        pytest.skip("country_system_f1.md no está en este árbol")
    return _DOC.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def seccion_rollback(doc) -> str:
    i = doc.index("### 5. Rollback")
    j = doc.index("### 6.", i)
    return doc[i:j]


def test_el_rollback_avisa_de_la_cola_antes_de_prometer_segundos(seccion_rollback):
    """El aviso va ANTES de la promesa: un operador que lee de arriba abajo tiene que tropezarse
    con él antes de leer «byte-identidad DO en segundos» y darse por satisfecho."""
    i_aviso = seccion_rollback.find("PASO PREVIO OBLIGATORIO")
    i_promesa = seccion_rollback.find("byte-identidad DO en segundos")
    assert i_aviso > 0, "el runbook volvió a prometer un rollback limpio sin avisar de la cola"
    assert i_aviso < i_promesa, (
        "el aviso quedó DESPUÉS de la promesa: quien lee de arriba abajo se da por satisfecho antes"
    )


def test_el_aviso_nombra_el_mecanismo_exacto(seccion_rollback):
    """No basta con «ojo con la cola»: el operador necesita saber POR QUÉ, o no puede juzgar si su
    caso aplica. El mecanismo es que el worker sigue leyendo el snapshot mientras la derivación
    devuelve 'DO'."""
    for pieza in ("pipeline_snapshot", "country_for_form_data", "plan_chunk_queue"):
        assert pieza in seccion_rollback, f"el aviso no nombra {pieza!r}"


def test_el_aviso_trae_la_consulta_para_saber_si_aplica(seccion_rollback):
    """Primero contar. Si salen 0 chunks, el rollback ES limpio y el operador puede seguir sin
    hacer nada — decírselo evita que trate un caso que no tiene."""
    assert "SELECT status" in seccion_rollback and "count(*)" in seccion_rollback
    assert "si sale 0" in seccion_rollback.lower(), (
        "el runbook no dice qué significa que la consulta salga vacía"
    )


def test_ofrece_LAS_DOS_salidas_con_su_consecuencia(seccion_rollback):
    """Las dos son defendibles y ninguna es obviamente correcta: cancelar deja el plan incompleto,
    reescribir le cambia la cocina a quien eligió otra. Un runbook que ofrece una sola decide por
    el operador sin decírselo."""
    assert "cancelled" in seccion_rollback, "falta la opción de congelar la cola"
    assert "jsonb_set" in seccion_rollback, "falta la opción de convertir el snapshot a 'DO'"
    assert "OPCIÓN A" in seccion_rollback and "OPCIÓN B" in seccion_rollback


def test_las_dos_consultas_filtran_por_estado_vivo(seccion_rollback):
    """El filtro importa tanto como el UPDATE: sin `status NOT IN ('completed','cancelled',
    'failed')` el operador reescribiría el snapshot de chunks YA COMPLETADOS, reescribiendo
    historia que nadie va a volver a ejecutar y ensuciando la telemetría."""
    assert seccion_rollback.count("status NOT IN ('completed','cancelled','failed')") >= 3, (
        "alguna de las consultas del runbook dejó de acotar a los chunks vivos"
    )


def test_el_runbook_dice_que_el_paso_es_decidir(seccion_rollback):
    """La diferencia entre un runbook y un script. Si alguien lo convierte en «ejecuta esto», el
    operador aplicará una de las dos opciones sin saber que había otra."""
    assert "decidir" in seccion_rollback.lower(), (
        "el runbook dejó de decir que el paso es una DECISIÓN, no un comando"
    )
