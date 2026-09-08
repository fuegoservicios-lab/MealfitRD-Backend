# -*- coding: utf-8 -*-
"""[P1-I18N-RECONCILE · 2026-09-08] Un plan traducido a medias no se queda así para siempre.

## El defecto, medido en producción

`meal_plans.revision` la sube un **TRIGGER de base de datos** en cada `UPDATE OF plan_data`
(`meal_plans_bump_revision_trg`, migración ARQ25-F1) — decenas de caminos de escritura. El trabajo
`display_i18n` se encolaba desde **tres** sitios de aplicación. El desajuste es estructural: el
resto de escritores sube la revisión y nadie pide la traducción de la nueva.

Medido el 08-sep sobre el único usuario real de producción: su plan iba por la revisión **26** y el
último `display_i18n` era de la **24**. Veía la app en francés y las **12 comidas de su plan en
español**, desde el 5-sep, sin que nada lo reintentara.

La cadena de re-encolado por `revision_changed` YA funcionaba —se la vio encadenar 18→22→23→24—.
Lo que faltaba era el encolado en los caminos que no son esos tres. Por eso el arreglo **no añade
un cuarto call site**: barre contra la misma columna que el trigger mantiene. *Una defensa que
depende de que alguien la invoque es una costumbre, no una defensa.*

## El segundo defecto, que apareció al ejercitar el arreglo

Al encolar el trabajo que faltaba, producción lo procesó y lo marcó **`failed`** con
`already_enriched` — «no quedaba nada por traducir», el estado más terminado que existe. Habría
reintentado con backoff hasta morir en dead-letter, **por haber hecho el trabajo**.

Ese mismo día `plan_display_i18n._RAZONES_BENIGNAS` había aprendido esa palabra
(`P2-I18N-YA-TRADUCIDO-NO-ES-DEGRADACION`) y `plan_jobs._DONE_SKIPS` no: dos vocabularios para el
mismo hecho y sólo uno actualizado.
"""
import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- el veredicto
def test_ya_traducido_es_TERMINADO_no_un_fallo():
    """`already_enriched` ⇒ `done`. Reintentar «ya está todo hecho» es reintentar el éxito."""
    from plan_jobs import verdict_for_display_result

    assert verdict_for_display_result({"enriched_meals": 0, "skipped": "already_enriched"}) == ("done", None), (
        "un plan enteramente traducido se marcaba `failed` y reintentaba con backoff hasta el "
        "dead-letter — castigando al job por haber terminado su trabajo")


def test_lo_transitorio_SIGUE_reintentando():
    """El arreglo de arriba no puede convertir un fallo real en un `done` silencioso.

    `invocation_budget_exhausted` es justo lo que dejó al usuario real en español: si se clasifica
    como terminado, el plan se queda a medias y nadie vuelve.
    """
    from plan_jobs import verdict_for_display_result

    for transitorio in ("invocation_budget_exhausted", "json_parse_error", "llm_exception",
                        "partial_loss", "circuit_breaker_open", "dedupe_locked"):
        estado, code = verdict_for_display_result({"skipped": transitorio})
        assert estado == "failed" and code == transitorio, (
            f"{transitorio!r} dejó de reintentar: un plan a medias se quedaría así")


def test_los_dos_vocabularios_NO_se_fusionan():
    """`_DONE_SKIPS` (¿reintentar?) y `_RAZONES_BENIGNAS` (¿alertar?) comparten literales pero NO
    son el mismo conjunto, y fusionarlos rompería uno de los dos.

    `dedupe_locked` es la prueba viva: benigno para la alerta (otro proceso lo tiene tomado, no hay
    degradación que avisar) y **reintentable** como job (nadie ha hecho el trabajo todavía).
    """
    from plan_jobs import _DONE_SKIPS, _RETRY_SKIPS
    from plan_display_i18n import _RAZONES_BENIGNAS

    assert "already_enriched" in _DONE_SKIPS and "already_enriched" in _RAZONES_BENIGNAS, (
        "el literal que ambos lados comparten debe estar en los dos")
    assert "dedupe_locked" in _RAZONES_BENIGNAS and "dedupe_locked" in _RETRY_SKIPS, (
        "si esto deja de cumplirse, alguien fusionó los conjuntos: benigno-para-alertar y "
        "terminado-como-trabajo no son lo mismo")
    assert not (_DONE_SKIPS & _RETRY_SKIPS), "un `skipped` no puede ser a la vez terminal y reintentable"


# --------------------------------------------------------------------- el barrido
def test_el_barrido_existe_y_no_es_un_cuarto_call_site():
    """El arreglo barre contra `meal_plans.revision` —la columna que el TRIGGER mantiene— en vez de
    añadir un cuarto sitio que hay que acordarse de invocar."""
    src = (_BACKEND / "plan_jobs.py").read_text(encoding="utf-8")
    assert "def reconcile_missing_display_i18n" in src
    assert "m.revision" in src and "j.plan_revision = m.revision" in src, (
        "el barrido no compara la revisión VIGENTE del plan contra la del trabajo: entonces no "
        "detecta el desajuste que motivó todo esto")
    assert "'pending', 'processing', 'done'" in src, (
        "los estados que cuentan como «ya cubierta» deben ser explícitos; incluir `failed` haría "
        "que un fallo permanente pasara por cobertura, y excluir `done` re-encolaría para siempre")


def test_el_barrido_pregunta_al_SSOT_que_locale_se_traduce():
    """No escribe un cuarto literal `!= "es-DO"`: usa `should_enrich_locale`.

    Es la lección de `P1-DIET-CANON-SSOT` — tres tablas a mano de `dietType` drifearon y la del
    filtro olvidó `vegetariana`, sirviendo pollo a vegetarianas.

    Se mira el **AST**, no el texto: la primera versión de este test cazaba el COMENTARIO que
    explica por qué no hay tal literal — el mismo error de instrumento que el blanket anti-Gemini
    ya resolvió distinguiendo «usar un API» de «contar por qué no se usa».
    """
    src = (_BACKEND / "plan_jobs.py").read_text(encoding="utf-8")
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "reconcile_missing_display_i18n"), None)
    assert fn is not None
    nombres = {n.name for i in ast.walk(fn) if isinstance(i, ast.ImportFrom) for n in i.names}
    assert "should_enrich_locale" in nombres, "reimplementa el gate de locale en vez de preguntarle al SSOT"
    # El docstring TAMBIÉN es una `ast.Constant`, y explica la regla nombrándola. Excluirlo es la
    # misma distinción de una vuelta más arriba: prosa que documenta ≠ código que decide.
    doc = ast.get_docstring(fn, clean=False)
    literales = {n.value for n in ast.walk(fn)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc}
    culpables = sorted(s for s in literales if "es-DO" in s)
    assert not culpables, (
        f"un literal del idioma base vive en el CÓDIGO del barrido: eso es la cuarta tabla. {culpables}")


def test_el_barrido_esta_registrado_como_cron():
    """Una función correcta con cero call sites no arregla nada — hoy mismo encontramos dos."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "def _reconcile_display_i18n_job" in src
    assert "reconcile_missing_display_i18n" in src, "el cron no llama al motor"
    assert 'id="reconcile_display_i18n"' in src, "no se registra en el scheduler: nace inerte"
    arbol = ast.parse(src)
    registrador = next((n for n in arbol.body
                        if isinstance(n, ast.FunctionDef) and n.name == "register_plan_chunk_scheduler"), None)
    assert registrador is not None
    ids = {n.value for n in ast.walk(registrador) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "reconcile_display_i18n" in ids, (
        "el job no se registra DENTRO de `register_plan_chunk_scheduler`, que es el SSOT de crons")


@pytest.mark.parametrize("knob", ["MEALFIT_PLAN_JOBS_ENABLED", "MEALFIT_I18N_RECONCILE"])
def test_apagado_por_cualquiera_de_los_dos_knobs(monkeypatch, knob):
    """Encolar traducciones cuesta dinero: tiene que poder apagarse sin redeploy, y el motivo del
    no-op tiene que distinguirse (`knob_off` ≠ `reconcile_off`) o el operador no sabe cuál tocó."""
    import plan_jobs

    monkeypatch.setenv("MEALFIT_PLAN_JOBS_ENABLED", "1")
    monkeypatch.setenv("MEALFIT_I18N_RECONCILE", "1")
    monkeypatch.setenv(knob, "0")
    r = plan_jobs.reconcile_missing_display_i18n()
    esperado = "knob_off" if knob == "MEALFIT_PLAN_JOBS_ENABLED" else "reconcile_off"
    assert r["skipped"] == esperado and r["encolados"] == 0, (
        f"con {knob}=0 debía no-opear con motivo {esperado!r}, dio {r!r}")


def test_los_knobs_de_coste_tienen_topes():
    """El batch y la ventana acotan lo que un barrido puede llegar a gastar en un tick."""
    import plan_jobs

    for nombre, fn, lo, hi in (
        ("MEALFIT_I18N_RECONCILE_BATCH", plan_jobs.i18n_reconcile_batch, 1, 200),
        ("MEALFIT_I18N_RECONCILE_MAX_AGE_DAYS", plan_jobs.i18n_reconcile_max_age_days, 1, 90),
        ("MEALFIT_I18N_RECONCILE_INTERVAL_MIN", plan_jobs.i18n_reconcile_interval_min, 5, 720),
    ):
        import os
        previo = os.environ.get(nombre)
        try:
            os.environ[nombre] = "999999"
            assert fn() == hi, f"{nombre} sin tope superior: un valor absurdo pasaría tal cual"
            os.environ[nombre] = "-5"
            assert fn() == lo, f"{nombre} sin suelo"
        finally:
            os.environ.pop(nombre, None)
            if previo is not None:
                os.environ[nombre] = previo
