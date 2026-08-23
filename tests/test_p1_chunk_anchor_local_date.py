"""[P1-CHUNK-ANCHOR-LOCAL-DATE · 2026-08-21] Los bloques del plan de un español se programaban
23,5 h ANTES de tiempo, y el techo de P1-CHUNK-EXECUTE-CEILING calculaba el mismo valor equivocado.

EL MECANISMO. `/analyze` escribe `_plan_start_date` como el INSTANTE UTC de la medianoche local
del usuario. Para alguien al ESTE de UTC ese instante cae en el día UTC ANTERIOR:

    España (verano, tzOffset=-120)   medianoche local del 21-ago  =  2026-08-20T22:00Z
    RD     (tzOffset=+240)           medianoche local del 21-ago  =  2026-08-21T04:00Z

Los tres sitios que programan `execute_after` hacen
`datetime.combine(ancla.date(), 00:00, UTC) + days + tzOffset + 30min`. Ese `ancla.date()` es la
fecha **UTC**, no la local — así que para España sale 20-ago, y luego el `+ tzOffset` (negativo)
vuelve a restar. El día se descuenta DOS VECES.

Medido con las funciones reales:

    DO     tz= 240   delta = +0,5 h        MX      tz=360   delta = +0,5 h
    ES(v)  tz=-120   delta = −23,5 h  ⚠    CO      tz=300   delta = +0,5 h
    ES(i)  tz= -60   delta = −23,5 h  ⚠    PR      tz=240   delta = +0,5 h
                                           US-Pac  tz=420   delta = +0,5 h

España es el único país beta al este de UTC, así que es el único que lo sufre — y por eso nadie lo
vio: los cinco restantes están al oeste y ahí la medianoche local SIEMPRE cae en la misma fecha UTC.

LO QUE LE PASA AL USUARIO. El bloque se arma sin que haya vivido el último día del anterior, así
que el «aprendizaje continuo» que justifica el chunking evalúa un día que aún no ha ocurrido —
`_check_chunk_learning_ready` puede pausar el chunk con `learning_zero_logs` y sacar el banner
«plan incompleto»—; se gasta LLM un día antes; y como el TECHO de P1-CHUNK-EXECUTE-CEILING calcula
el mismo valor equivocado, el `LEAST` lo fija y el rebase lo conserva **para siempre**. Es
exactamente el modo de fallo que ese P-fix documenta: nadie compara el par contra el ancla.

POR QUÉ EL GUARD EXISTENTE NO PODÍA VERLO. `test_p1_chunk_execute_ceiling.py` sólo instancia
`tzOffset ∈ {0, 240}`: **cero casos con offset negativo**. Un guard que sólo prueba el hemisferio
en el que el bug no existe no es un guard, es una coincidencia.

EL ARREGLO. Un SSOT —`constants.chunk_anchor_local_midnight_utc`— que calcula la fecha LOCAL del
ancla antes de reconstruir su medianoche, y los tres sitios lo llaman. La aritmética vivía copiada
tres veces; ésa es la razón de que el defecto sobreviviera a dos P-fixes de esta misma familia.

Cubre:
  A. El helper: medianoche local correcta para los 6 países, incluidos los offsets negativos.
  B. Byte-identidad para los offsets al oeste de UTC (lo que hoy funciona no se mueve).
  C. El delta contra la medianoche local real cae en [0, 1h] para los 6.
  D. El techo del ceiling usa el helper.
  E. Parser-based: los tres sitios comparten la aritmética.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CONSTANTS = _BACKEND_ROOT / "constants.py"
_CRON = _BACKEND_ROOT / "cron_tasks.py"

# Los 6 países del sistema + las dos mitades del año de España (DST).
_OFFSETS = {
    "DO": 240, "PR": 240, "CO": 300, "MX": 360, "US-Pac": 420,
    "ES-verano": -120, "ES-invierno": -60,
}


def _ancla_utc(fecha_local: str, tz_min: int) -> datetime:
    """El instante UTC de la medianoche local — exactamente lo que `/analyze` persiste en
    `_plan_start_date`."""
    y, m, d = (int(x) for x in fecha_local.split("-"))
    return datetime(y, m, d, tzinfo=timezone.utc) + timedelta(minutes=tz_min)


# ── A/B/C. El helper ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("etiqueta,tz_min", sorted(_OFFSETS.items()))
def test_la_medianoche_local_del_ancla_es_la_del_dia_correcto(etiqueta, tz_min):
    """RED pre-fix para ES: devolvía la medianoche del día ANTERIOR. El ancla se construye desde
    una fecha local conocida, así que la respuesta correcta es esa misma fecha."""
    from constants import chunk_anchor_local_midnight_utc

    ancla = _ancla_utc("2026-08-21", tz_min)
    got = chunk_anchor_local_midnight_utc(ancla, tz_min)
    assert got == _ancla_utc("2026-08-21", tz_min), (
        f"{etiqueta}: el ancla del 21-ago se resolvió a {got.isoformat()}"
    )


@pytest.mark.parametrize("etiqueta,tz_min", sorted(_OFFSETS.items()))
def test_el_bloque_no_se_adelanta_a_la_medianoche_del_dia_que_cubre(etiqueta, tz_min):
    """La invariante de P1-CHUNK-EXECUTE-CEILING, ahora medida en los 6 offsets: el `execute_after`
    de un chunk debe caer DENTRO de la primera hora del día local que cubre. Antes del fix, España
    daba −23,5 h — el bloque corría el día antes de empezar el tramo."""
    from constants import chunk_anchor_local_midnight_utc

    ancla = _ancla_utc("2026-08-21", tz_min)
    offset_dias = 4
    programado = chunk_anchor_local_midnight_utc(ancla, tz_min) + timedelta(
        days=offset_dias, minutes=30)
    medianoche_real = _ancla_utc("2026-08-25", tz_min)
    delta_h = (programado - medianoche_real).total_seconds() / 3600.0
    assert 0 <= delta_h <= 1, (
        f"{etiqueta}: el bloque se programa {delta_h:+.1f} h respecto a la medianoche local "
        f"del primer día que cubre"
    )


def test_los_offsets_al_oeste_de_utc_no_se_mueven():
    """Byte-identidad de lo que hoy funciona: para tz_min >= 0 la fecha UTC del ancla YA es la
    local, así que el helper debe devolver exactamente lo que devolvía la aritmética vieja."""
    from constants import chunk_anchor_local_midnight_utc

    for tz_min in (0, 240, 300, 360, 420):
        ancla = _ancla_utc("2026-08-21", tz_min)
        viejo = datetime.combine(ancla.date(), datetime.min.time()).replace(
            tzinfo=timezone.utc) + timedelta(minutes=tz_min)
        assert chunk_anchor_local_midnight_utc(ancla, tz_min) == viejo


def test_el_helper_tolera_basura_sin_reventar():
    """Corre dentro de la transacción que sostiene el advisory lock del shift: una excepción aquí
    retiene el lock. Ante entrada inválida devuelve None y el caller conserva su conducta previa —
    el mismo contrato de «no opino» que ya tiene `chunk_execute_after_ceiling`."""
    from constants import chunk_anchor_local_midnight_utc

    assert chunk_anchor_local_midnight_utc(None, 240) is None
    assert chunk_anchor_local_midnight_utc("no soy una fecha", 240) is None
    ancla = _ancla_utc("2026-08-21", 240)
    assert chunk_anchor_local_midnight_utc(ancla, "no soy un offset") is not None


def test_un_ancla_naive_se_interpreta_utc():
    """`_plan_start_date` viaja como ISO y no siempre trae tzinfo. Interpretarla como local sería
    peor que interpretarla como UTC: el resto del pipeline la trata como UTC."""
    from constants import chunk_anchor_local_midnight_utc

    naive = datetime(2026, 8, 21, 4, 0, 0)
    assert chunk_anchor_local_midnight_utc(naive, 240) == _ancla_utc("2026-08-21", 240)


# ── D. El techo lo usa ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("etiqueta,tz_min", sorted(_OFFSETS.items()))
def test_el_techo_del_ceiling_usa_la_fecha_local(etiqueta, tz_min):
    """RED pre-fix para ES: el techo salía un día antes, y como el `LEAST` lo fija y el rebase lo
    conserva, el error se volvía permanente."""
    from constants import chunk_execute_after_ceiling

    snapshot = {"form_data": {
        "_plan_start_date": _ancla_utc("2026-08-21", tz_min).isoformat(),
        "tzOffset": tz_min,
    }}
    techo = chunk_execute_after_ceiling(snapshot, 4)
    assert techo is not None
    delta_h = (techo - _ancla_utc("2026-08-25", tz_min)).total_seconds() / 3600.0
    assert 0 <= delta_h <= 1, f"{etiqueta}: el techo cae {delta_h:+.1f} h de la medianoche local"


def test_el_ceiling_sigue_diciendo_no_opino_sin_snapshot():
    """Contrato preservado: `None` significa «no opino» y el caller conserva su conducta. Tratarlo
    como techo cero mandaría todos los chunks vencidos a NOW() a la vez."""
    from constants import chunk_execute_after_ceiling
    assert chunk_execute_after_ceiling({}, 3) is None
    assert chunk_execute_after_ceiling({"form_data": {}}, 3) is None


# ── E. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_los_tres_sitios_comparten_la_aritmetica():
    """El SSOT gobierna encolado, recovery y ahora también el gate temporal; el techo lo
    consume dentro de constants. AST cuenta las derivaciones reales de fecha y no sólo dos
    grafías literales: la cuarta forma que originó G15 ya no puede quedar invisible."""
    cron = _CRON.read_text(encoding="utf-8", errors="replace")
    const = _CONSTANTS.read_text(encoding="utf-8", errors="replace")
    assert "P1-CHUNK-ANCHOR-LOCAL-DATE" in const
    assert "def chunk_anchor_local_midnight_utc" in const
    tree = ast.parse(cron)

    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "constants"
        for alias in node.names
        if alias.name == "chunk_anchor_local_midnight_utc"
    }
    # Las tres derivaciones con fallback importan aliases `_calmu_*`. G16 añadió una
    # cuarta llamada directa dentro del normalizador de fuentes 2/3; no debe compensar
    # una derivación sin SSOT ni convertir este conteo en un exact-marker histórico.
    operational_aliases = {name for name in aliases if name.startswith("_calmu_")}
    ssot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in operational_aliases
    ]
    anchor_date_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "date"
        and any(
            isinstance(child, ast.Name)
            and child.id in {"plan_start_dt", "anchor_start_dt"}
            for child in ast.walk(node.func.value)
        )
    ]

    assert len(anchor_date_calls) == 3, (
        "cambió el número de fallbacks que derivan fecha del ancla; auditar cada sitio"
    )
    assert len(ssot_calls) == len(anchor_date_calls), (
        "cada derivación de fecha del ancla en cron debe tener una llamada al SSOT; "
        f"date_calls={len(anchor_date_calls)}, ssot_calls={len(ssot_calls)}"
    )


def test_el_guard_del_techo_ya_prueba_offsets_negativos():
    """La razón por la que este bug vivió: `test_p1_chunk_execute_ceiling.py` sólo instanciaba
    offsets 0 y 240. Un guard que sólo prueba el hemisferio donde el bug no existe no es un guard.
    Este assert obliga a que ese fichero cubra al menos un offset negativo."""
    src = (_BACKEND_ROOT / "tests" / "test_p1_chunk_execute_ceiling.py").read_text(
        encoding="utf-8", errors="replace")
    assert "-120" in src or "-60" in src, (
        "el guard del techo sigue sin probar ningún país al este de UTC"
    )
