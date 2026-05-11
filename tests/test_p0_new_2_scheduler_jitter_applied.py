"""[P0-NEW-2 · 2026-05-10] Todo `scheduler.add_job` registrado por el módulo
debe pasar por `_add_job_jittered` (SSOT que aplica `jitter` por default).

Bug original (audit 2026-05-10):
    Tras P0-2 (bump del pool APScheduler a 32 workers + 180s grace), producción
    seguía emitiendo `scheduler_missed_*` en cascada: 25 unique unresolved
    en `system_alerts` durante las últimas 24h. Causa raíz: todos los crons
    usan `IntervalTrigger(minutes=N)` con baseline común desde el restart del
    worker, así que tras un restart se alinean y disparan en bursts. El pool
    de 32 workers absorbe la cola normal pero satura si varios jobs caen en
    la misma ventana de pickup → MISSED tras 180s de grace.

    Downstream: `proactive_refresh_pantry_snapshots` entre los missed →
    `chunk_pantry_snapshots_stale` con chunks de hasta 52h de edad → forzaba
    flexible_mode al cruzar las 24h (UX degradada).

Fix:
    `cron_tasks._add_job_jittered(scheduler, ...)` aplica
    `jitter=_SCHEDULER_JITTER_S` por default a TODOS los `add_job`.
    APScheduler propaga `jitter` al Trigger subyacente (Interval/Cron)
    automáticamente. Knob `MEALFIT_SCHEDULER_JITTER_S` (default 45s, clamp
    [0, 600]) — 0 desactiva (rollback en caliente).

Estrategia del test (parser estático, mismo patrón que
`test_p1_a_no_runtime_ddl_in_active_code.py`):
    1. Parsear `cron_tasks.py` línea por línea.
    2. Verificar que NO existe ningún `scheduler.add_job(` directo en
       contexto de "callsite" (excluye la definición del wrapper y
       comentarios documentando el patrón).
    3. Verificar que `_add_job_jittered` está definido módulo-level.
    4. Verificar que el knob `MEALFIT_SCHEDULER_JITTER_S` aparece declarado
       vía `_env_int` (auto-registra en `_KNOBS_REGISTRY`).
    5. Verificar que `app.py` también usa `_add_job_jittered` para el único
       `add_job` que registra (`run_proactive_checks`).

Drift detection:
    - Un futuro callsite que olvide el wrapper y haga `scheduler.add_job(`
      directo → falla `test_no_direct_add_job_outside_wrapper`.
    - Quitar el knob → falla `test_jitter_knob_declared_via_env_int`.
    - Quitar el wrapper → falla `test_jittered_wrapper_defined`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CRON_TASKS_PY = _REPO_ROOT / "backend" / "cron_tasks.py"
_APP_PY = _REPO_ROOT / "backend" / "app.py"


@pytest.fixture(scope="module")
def cron_tasks_src() -> str:
    return _CRON_TASKS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


def test_jittered_wrapper_defined(cron_tasks_src: str):
    """`_add_job_jittered` debe existir módulo-level en cron_tasks.py.
    Sin esto, el resto del test pierde sentido y el knob queda huérfano.
    """
    pattern = re.compile(
        r"^def\s+_add_job_jittered\s*\(\s*scheduler\s*,\s*\*args\s*,\s*\*\*kwargs\s*\)\s*:",
        re.MULTILINE,
    )
    assert pattern.search(cron_tasks_src), (
        "P0-NEW-2 regresión: `def _add_job_jittered(scheduler, *args, **kwargs):` "
        "ya no aparece en cron_tasks.py. Si alguien refactorizó el SSOT del "
        "jitter sin propagar a todos los callsites, restaurar el wrapper "
        "para que `MEALFIT_SCHEDULER_JITTER_S` siga aplicándose a cada job."
    )


def test_jitter_knob_declared_via_env_int(cron_tasks_src: str):
    """El knob `MEALFIT_SCHEDULER_JITTER_S` debe auto-registrarse en
    `_KNOBS_REGISTRY` vía `_env_int` (consistente con el patrón P3-NEW-D
    para que `/admin/knobs` lo exponga).
    """
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_JITTER_S["\']\s*,\s*\d+',
    )
    assert pattern.search(cron_tasks_src), (
        "P0-NEW-2 regresión: el knob `MEALFIT_SCHEDULER_JITTER_S` ya no se "
        "declara vía `_env_int`. Si alguien lo migró a `os.environ.get` "
        "directo, perdemos el auto-registro en `_KNOBS_REGISTRY` y "
        "`/admin/knobs` deja de exponerlo."
    )


def test_jittered_wrapper_uses_setdefault(cron_tasks_src: str):
    """El wrapper debe usar `kwargs.setdefault('jitter', ...)` (no `kwargs[...]= ...`)
    para que los callsites puedan sobre-escribir el valor explícitamente
    (ej. jitter=0 por motivo operacional puntual). Sin setdefault, el wrapper
    pisaría overrides legítimos.
    """
    pattern = re.compile(
        r'kwargs\.setdefault\(\s*["\']jitter["\']\s*,\s*_SCHEDULER_JITTER_S\s*\)',
    )
    assert pattern.search(cron_tasks_src), (
        "P0-NEW-2 regresión: el wrapper `_add_job_jittered` ya no usa "
        "`kwargs.setdefault('jitter', _SCHEDULER_JITTER_S)`. Si alguien lo "
        "cambió a asignación directa (`kwargs['jitter'] = ...`), pisamos "
        "overrides explícitos del callsite — restaurar setdefault."
    )


def test_no_direct_add_job_outside_wrapper(cron_tasks_src: str):
    """Después del fix, ningún `scheduler.add_job(` directo debe quedar
    como callsite real en cron_tasks.py. Solo:
      - El cuerpo del wrapper (`return scheduler.add_job(*args, **kwargs)`).
      - Comentarios documentando el patrón (entre backticks).

    Estrategia: contar todas las ocurrencias y verificar que cada una cae
    en uno de esos dos contextos.
    """
    direct_pattern = re.compile(r"scheduler\.add_job\(")
    matches = list(direct_pattern.finditer(cron_tasks_src))
    assert matches, (
        "P0-NEW-2 sanity: no se encontró NINGÚN `scheduler.add_job(` en "
        "cron_tasks.py. Si la entera arquitectura de scheduler se reemplazó, "
        "este test pierde sentido — actualizar/eliminar."
    )

    offenders = []
    for m in matches:
        # Línea completa para contexto.
        line_start = cron_tasks_src.rfind("\n", 0, m.start()) + 1
        line_end = cron_tasks_src.find("\n", m.end())
        if line_end == -1:
            line_end = len(cron_tasks_src)
        line = cron_tasks_src[line_start:line_end]
        line_no = cron_tasks_src.count("\n", 0, m.start()) + 1

        # Contextos aceptados:
        # (a) Cuerpo del wrapper: `return scheduler.add_job(*args, **kwargs)`
        if "return scheduler.add_job(*args" in line:
            continue
        # (b) Comentario/docstring documentando el patrón — la línea
        #     empieza con `#` o contiene backtick-quoted reference.
        stripped = line.lstrip()
        if stripped.startswith("#") or "`scheduler.add_job" in line:
            continue

        offenders.append(f"  L{line_no}: {line.strip()}")

    assert not offenders, (
        "P0-NEW-2 regresión: encontrados callsites directos de "
        "`scheduler.add_job(` que NO pasan por `_add_job_jittered`:\n"
        + "\n".join(offenders)
        + "\n\nSin el wrapper, estos jobs no reciben jitter y vuelven a "
        "alinearse post-restart → cascada `scheduler_missed_*` de P0-NEW-2 "
        "reaparece. Reemplazar con `_add_job_jittered(scheduler, ...)`."
    )


def test_app_py_run_proactive_checks_uses_wrapper(app_src: str):
    """`app.py` registra `run_proactive_checks` con CronTrigger (minute=30).
    Como es un único job, no participa en bursts post-restart por sí solo,
    pero sí compite por el pool de workers con los crons interval. Mismo
    wrapper para consistencia y para que el knob `JITTER_S=0` apague TODOS
    los jitters de un solo flip.
    """
    pattern = re.compile(
        r"_add_job_jittered\(\s*scheduler\s*,\s*run_proactive_checks\b",
    )
    assert pattern.search(app_src), (
        "P0-NEW-2 regresión: `run_proactive_checks` en app.py ya no se "
        "registra vía `_add_job_jittered`. Si alguien volvió a "
        "`scheduler.add_job(run_proactive_checks, ...)`, el knob "
        "`MEALFIT_SCHEDULER_JITTER_S=0` deja de apagar el jitter "
        "globalmente (este job mantendría el spread pero los crons interval "
        "no, o viceversa). Mantener todos los add_job tras el mismo SSOT."
    )


def test_knob_clamp_safe(cron_tasks_src: str):
    """El knob debe clampear a [0, 600] para evitar:
      - Negativos (APScheduler crashea con jitter<0).
      - Valores absurdos (>600s deja un job interval=5min con drift de 10min).
    """
    pattern = re.compile(
        r"_SCHEDULER_JITTER_S\s*=\s*max\(\s*0\s*,\s*min\(\s*_env_int\(\s*"
        r"['\"]MEALFIT_SCHEDULER_JITTER_S['\"]\s*,\s*\d+\s*\)\s*,\s*600\s*\)\s*\)",
        re.DOTALL,
    )
    assert pattern.search(cron_tasks_src), (
        "P0-NEW-2 regresión: el clamp `max(0, min(_env_int(...), 600))` "
        "del knob `MEALFIT_SCHEDULER_JITTER_S` desapareció. Sin clamp, un "
        "operador que setee `=-1` o `=99999` rompe el scheduler. Restaurar."
    )
