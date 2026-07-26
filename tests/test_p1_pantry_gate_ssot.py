"""[P1-PANTRY-GATE-SSOT · 2026-07-26] Las dos guardas de nevera del chunk worker deben
responder LO MISMO. Cuando discreparon, un plan quedó oscilando 14 días.

## El lazo, medido en producción

Plan `69f9e03d` (creado 2026-07-12, 4 días, nevera del dueño = 0 filas), semana 5,
`chunk_kind='rolling_refill'`, `_pantry_flexible_mode=True`:

    14:37:21  el cron de TTL resuelve la pausa  (_pantry_pause_resolution='degraded_flexible_meal')
    14:43:30  el chunk arranca el pipeline completo
    14:50:29  genera la semana, la mergea, agrega lecciones        ← ~7 min de LLM gastados
    14:50:37  [P0-5/PARTIAL] reservas 0/87 (min=43)
    14:50:50  [P1-CHUNKS-2/RECONCILE-EXHAUSTED] → pending_user_action
              ↑ y vuelta a empezar 12h después

Seis ejecuciones completas solo el 2026-07-26 (01:07, 01:23, 01:39, 02:09, 02:18, 14:43).

## Por qué

`_should_pause_for_empty_pantry` (PRE-pipeline, gratis) respetaba `_pantry_flexible_mode`.
El gate de reservas (POST-pipeline, después de pagar) solo eximía `chunk_kind ==
"initial_plan"`. El modo flexible existe justamente para decir "esta nevera está vacía,
deja de esperar y genera igual" — y la guarda cara lo ignoraba, pausando por la condición
que el modo flexible ya había perdonado. Ni converge ni falla: oscila.

El `guest` tenía el mismo agujero por otra puerta (pasa la guarda barata porque un anónimo
no tiene perfil que refrescar, moría en la cara).

## Qué ancla este archivo

La CLASE, no la instancia: **paridad**. Con la nevera vacía, si la guarda barata deja
pasar el chunk, la cara también tiene que dejarlo pasar. Da igual por qué motivo.
Un tercer flag futuro (`_pantry_lo_que_sea`) que solo una de las dos consulte rompe este
test sin que nadie tenga que acordarse de este incidente.

tooltip-anchor: P1-PANTRY-GATE-SSOT
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CRON_PATH = _BACKEND / "cron_tasks.py"
_CRON = _CRON_PATH.read_text(encoding="utf-8")

VACIA: list = []          # nevera sin items → reservas 0/N
LLENA = [{"ingredient_name": f"alimento {i}", "quantity": 5} for i in range(40)]

KINDS = ("initial_plan", "rolling_refill", "catchup", None)
SNAPS = (
    {},
    {"_pantry_flexible_mode": True},
    {"_pantry_advisory_only": True},
)
SOURCES = ("live", "snapshot", "guest", None)


def _fns():
    from cron_tasks import _pantry_gate_waiver_reason, _should_pause_for_empty_pantry
    return _pantry_gate_waiver_reason, _should_pause_for_empty_pantry


def _pasa_guarda_barata(kind, snap, fd, src, inventario):
    """Reproduce el call site PRE-pipeline: `if should_pause: if waiver: sigue else: pausa`."""
    waiver_fn, should_pause = _fns()
    if not should_pause(src, inventario, snap, fd):
        return True
    return bool(waiver_fn(chunk_kind=kind, snapshot=snap, form_data=fd))


def _pasa_guarda_cara(kind, snap, fd, src):
    """Reproduce el gate de reservas: `elif _res_waiver:` → best_effort sin pausa."""
    waiver_fn, _ = _fns()
    return bool(waiver_fn(
        chunk_kind=kind, snapshot=snap, form_data=fd, fresh_inventory_source=src,
    ))


# ───────────── 1. LA INVARIANTE (el ancla de la clase) ─────────────

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("snap", SNAPS)
@pytest.mark.parametrize("src", SOURCES)
def test_paridad_con_nevera_vacia(kind, snap, src):
    """Si la guarda gratis deja pasar, la guarda cara TIENE que dejar pasar.

    Lo contrario es, por definición, gastar el pipeline entero para después bloquear por
    algo que ya se sabía antes de gastarlo.
    """
    barata = _pasa_guarda_barata(kind, snap, {}, src, VACIA)
    cara = _pasa_guarda_cara(kind, snap, {}, src)
    assert barata == cara, (
        f"DISCREPAN con kind={kind!r} snap={snap} source={src!r}: "
        f"pre-pipeline {'deja pasar' if barata else 'bloquea'} pero el gate de reservas "
        f"{'deja pasar' if cara else 'bloquea'}. Este desacuerdo es el lazo del plan 69f9e03d."
    )


@pytest.mark.parametrize("snap", SNAPS)
@pytest.mark.parametrize("src", SOURCES)
def test_paridad_tambien_cuando_el_flag_viaja_en_form_data(snap, src):
    """Los flags llegan por `snapshot` O por `form_data` — ambos caminos cuentan."""
    barata = _pasa_guarda_barata("rolling_refill", {}, dict(snap), src, VACIA)
    cara = _pasa_guarda_cara("rolling_refill", {}, dict(snap), src)
    assert barata == cara, f"DISCREPAN vía form_data: snap={snap} source={src!r}"


# ───────────── 2. el caso vivo, explícito ─────────────

def test_el_caso_del_plan_69f9e03d():
    """rolling_refill + flexible_mode + nevera vacía = pasa. Antes: gastaba y era pausado."""
    waiver_fn, _ = _fns()
    assert waiver_fn(
        chunk_kind="rolling_refill", snapshot={"_pantry_flexible_mode": True},
    ) == "flexible_mode"


def test_rolling_refill_sin_waiver_sigue_gateado():
    """La corrección NO desactiva el gate: a mitad de plan, sin modo flexible, sigue vivo.

    Si esto se vuelve permisivo, un chunk reserva contra inventario que no existe y el
    siguiente ve sobre-reserva (overbooking silente).
    """
    waiver_fn, _ = _fns()
    assert waiver_fn(chunk_kind="rolling_refill") is None
    assert waiver_fn(chunk_kind="catchup") is None


def test_los_motivos_son_distinguibles_en_el_log():
    """El waiver se loguea; motivos colapsados a un bool esconderían por qué no se pausó."""
    waiver_fn, _ = _fns()
    motivos = {
        waiver_fn(chunk_kind="initial_plan"),
        waiver_fn(chunk_kind="rolling_refill", snapshot={"_pantry_flexible_mode": True}),
        waiver_fn(chunk_kind="rolling_refill", snapshot={"_pantry_advisory_only": True}),
        waiver_fn(chunk_kind="rolling_refill", fresh_inventory_source="guest"),
    }
    assert motivos == {"initial_plan_autonomy", "flexible_mode", "advisory_only", "guest"}


# ───────────── 3. estructura: que el except vea el waiver ─────────────

def test_res_waiver_se_calcula_FUERA_del_try():
    """El branch del `except` consulta `_res_waiver`. Si la asignación viviera dentro del
    `try` y `reserve_plan_ingredients` fallara antes, el handler leería una variable no
    bound → NameError tragado por el `except Exception` de más afuera, y el chunk volvería
    a pausarse en silencio. Es el mismo modo de fallo que dejó inerte P1-CAPPED-STAPLE-
    HONESTY: código presente, efecto nulo, test en verde."""
    tree = ast.parse(_CRON)
    worker = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_chunk_worker"
    )
    asignaciones = [
        n.lineno for n in ast.walk(worker)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_res_waiver" for t in n.targets)
    ]
    assert len(asignaciones) == 1, f"esperaba 1 asignación de _res_waiver, hay {len(asignaciones)}"

    trys = [
        t for t in ast.walk(worker)
        if isinstance(t, ast.Try)
        and any(h.name == "reserve_err" for h in t.handlers)
    ]
    assert len(trys) == 1, "no encontré el try/except de reserve_plan_ingredients"
    assert asignaciones[0] < trys[0].lineno, (
        f"_res_waiver se asigna en la línea {asignaciones[0]}, DENTRO del try que empieza "
        f"en {trys[0].lineno}. El handler no lo vería bound."
    )


def test_ambos_branches_del_gate_usan_el_mismo_waiver():
    assert "elif _res_waiver:" in _CRON, "el branch normal del gate no usa la SSOT"
    assert "if _res_waiver:" in _CRON, "el branch del except no usa la SSOT"


def test_ninguna_guarda_lee_el_flag_por_su_cuenta():
    """La SSOT es la ÚNICA que decide con `_pantry_flexible_mode`/`_pantry_advisory_only`.

    Otros sitios pueden LEER el flag para telemetría o para el prompt (`_learning_flexible_
    mode`, etc.) — lo que este test prohíbe es que una decisión de *pausar/gatear* se tome
    fuera de la SSOT, que es exactamente como las dos guardas se desincronizaron.
    """
    tree = ast.parse(_CRON)
    ssot = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_pantry_gate_waiver_reason"
    )
    for fn_name, marcador in (
        ("_should_pause_for_empty_pantry", "guarda pre-pipeline"),
    ):
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == fn_name
        )
        cuerpo = ast.get_source_segment(_CRON, fn) or ""
        assert "_pantry_flexible_mode" not in cuerpo, (
            f"{marcador} ({fn_name}) volvió a decidir por su cuenta con el flag; "
            "debe delegar en _pantry_gate_waiver_reason"
        )
        assert "_pantry_gate_waiver_reason" in cuerpo, f"{marcador} no llama a la SSOT"

    ssot_src = ast.get_source_segment(_CRON, ssot) or ""
    assert "_pantry_flexible_mode" in ssot_src, "la SSOT dejó de mirar el flag"


def test_marker_anclado_en_el_fuente():
    assert _CRON.count("P1-PANTRY-GATE-SSOT") >= 4, (
        "los anchors de la SSOT desaparecieron de cron_tasks"
    )
