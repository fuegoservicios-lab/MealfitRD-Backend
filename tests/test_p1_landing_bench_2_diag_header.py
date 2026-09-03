"""[P1-LANDING-BENCH-2 · 2026-08-07] El 422 crítico lleva su diagnóstico en un header.

QUÉ PASÓ. La corrida n=20 del benchmark del landing (2026-08-07, issue #9) terminó
13/20 en rechazo crítico y NO se pudo saber por qué desde fuera: las razones
(`_review_issues`) se logueaban en el VPS y se descartaban del response. El detail
del 422 no puede crecer a dict porque el frontend identifica critical_restriction
por `typeof detail === 'string'` (P2-CRITICAL-REJECTION-CODE, Plan.jsx) — así que
el diagnóstico viaja en el header `X-Bioboros-Review-Diag` (ASCII-safe, truncado,
datos del PROPIO plan del solicitante).

QUÉ ANCLA:
  1. El sitio del 422 envía el header Y el detail sigue siendo el string del
     disclaimer (romper cualquiera de las dos mitades rompe frontend o diagnóstico).
  2. ASCII-safe (`ensure_ascii=True`): los headers HTTP son latin-1; una tilde
     cruda en un ingrediente rompería el response entero.
  3. El runner del benchmark lee el header y tiene `--ids` (re-correr SOLO los
     perfiles clínicos tras un deploy, sin pagar la matriz completa).
  4. El gate del frontend que motiva el diseño sigue existiendo — si alguien lo
     migra a dict, este test le recuerda coordinar el header.

tooltip-anchor: P1-LANDING-BENCH-2
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND_SRC = _BACKEND.parent / "frontend" / "src"

_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_RUNNER = (_BACKEND / "scripts" / "landing_benchmark.py").read_text(encoding="utf-8")
_WORKFLOW = (_BACKEND / ".github" / "workflows" /
             "landing-benchmark-remote-guest.yml").read_text(encoding="utf-8")


def _critical_422_block() -> str:
    m = re.search(r"if result\.get\(\"_critical_rejection\"\):(.*?)raise HTTPException",
                  _PLANS, re.DOTALL)
    assert m, "P1-LANDING-BENCH-2: no encuentro el bloque del 422 crítico en routers/plans.py."
    return m.group(0)


def test_422_sends_diag_header_and_keeps_string_detail():
    block = _critical_422_block()
    assert "X-Bioboros-Review-Diag" in block, (
        "P1-LANDING-BENCH-2: el 422 crítico perdió el header X-Bioboros-Review-Diag — "
        "el benchmark vuelve a quedar ciego ante rechazos críticos (como la corrida "
        "n=20 del 2026-08-07)."
    )
    assert "_review_issues" in block, "el diagnóstico debe salir de `_review_issues` del fallback."
    # El detail DEBE seguir siendo el string del disclaimer: el frontend gatea
    # critical_restriction por `typeof detail === 'string'`.
    assert re.search(r"raise HTTPException\(status_code=422, detail=_crit_msg, headers=", _PLANS), (
        "P1-LANDING-BENCH-2: el detail del 422 crítico cambió de forma. Plan.jsx lo "
        "identifica por typeof string (P2-CRITICAL-REJECTION-CODE) — coordinar ambos "
        "lados antes de tocar esto."
    )


def test_diag_header_is_ascii_safe_and_truncated():
    block = _critical_422_block()
    assert "ensure_ascii=True" in block, (
        "P1-LANDING-BENCH-2: el diag del header debe serializarse ensure_ascii=True — "
        "los headers HTTP son latin-1 y las razones traen tildes/ñ."
    )
    assert re.search(r"\[:160\]", block) and re.search(r"\[:3000\]", block), (
        "P1-LANDING-BENCH-2: el header debe truncar (160/issue, 3000 total) — un header "
        "gigante lo rechazan proxies intermedios."
    )


def test_runner_reads_diag_header_and_supports_ids():
    assert "x-bioboros-review-diag" in _RUNNER.lower(), (
        "P1-LANDING-BENCH-2: el runner ya no captura el header de diagnóstico."
    )
    assert "--ids" in _RUNNER, (
        "P1-LANDING-BENCH-2: el runner perdió --ids (re-correr solo perfiles clínicos)."
    )


def test_workflow_supports_ids_and_cheap_push_smoke():
    assert "ids:" in _WORKFLOW and "PUSH_RUN_IDS" in _WORKFLOW, (
        "P1-LANDING-BENCH-2: el workflow perdió el input/env de ids."
    )
    assert 'PUSH_RUN_N: "2"' in _WORKFLOW, (
        "P1-LANDING-BENCH-2: el push-trigger debe quedar en smoke barato (n=2) — el "
        "fallback n=0 disparaba la matriz completa (2h de LLM de prod) en cada push."
    )


def test_frontend_gate_that_motivates_the_header_still_exists():
    plan_jsx = (_FRONTEND_SRC / "pages" / "Plan.jsx").read_text(encoding="utf-8")
    assert "typeof body?.detail === 'string'" in plan_jsx, (
        "P1-LANDING-BENCH-2: Plan.jsx ya no gatea critical_restriction por detail-string. "
        "Si ese contrato migró a dict {code,...}, el diagnóstico puede mudarse del header "
        "al detail — actualizar ambos lados y este test juntos."
    )
