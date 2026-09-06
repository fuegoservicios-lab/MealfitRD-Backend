# -*- coding: utf-8 -*-
"""[P1-PIPELINE-TIMEOUT-DEGRADED · 2026-09-06] La alerta afirmaba el fallback sin comprobar si ocurrió.

`_persist_pipeline_crash_alert` tenía el título y el mensaje **hardcodeados**:

> «El pipeline de generacion de plan no pudo entregar un plan del LLM y cayo al fallback matematico
> de emergencia. Causa tipica: outage/timeout del proveedor LLM.»

En el único evento vivo de esta alerta —2026-09-02 14:45, el usuario `f47126cb`— **las dos afirmaciones
eran falsas**, y se puede demostrar con la telemetría de esa misma corrida:

- `delivered_was_fallback: False` en `pipeline_holistic`, `resolution_coverage` **y**
  `solver_convergence`. El plan `2b692ef8` se entregó: 3 días, 2500 kcal, desviación calórica del
  2,9 %, cero días con 0 kcal.
- No fue el proveedor: `pipeline_holistic = 900.320 ms`, clavado en
  `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S=900`. El pipeline se comió su propio presupuesto —
  `self_critique` 190,7 s y `assemble_plan` 256,4 s, con el solver sin converger en 4 de 12 comidas.
  Las llamadas LLM de la corrida siguiente tardaron entre 8 y 62 s sin problema.

Y `detail` llegaba vacío: `"TimeoutError: "`. `asyncio.TimeoutError` no lleva mensaje, así que el
operador no podía distinguir «se cayó el proveedor» de «me pasé de mi propio techo» — que es
justamente la primera bifurcación de cualquier diagnóstico.

## Lo que cambia

El `except` del pipeline tiene **dos ramas** y la alerta no las distinguía:

| rama | qué recibe el usuario | alerta |
|---|---|---|
| `plan_partial` vacío → `_get_extreme_fallback_plan` | el fallback matemático | `pipeline_crash_fallback` (texto de siempre) |
| `plan_partial` existe → `_repair_partial_plan` | **el plan del LLM**, reparado o no | `pipeline_timeout_degraded` (texto honesto) |

Es la misma familia que `P1-PERSIST-DECLINED-NOT-FAILED`, cerrado el mismo día: **una alerta que
describe el peor caso sin comprobar si ocurrió.** Y el mismo coste: es de modelo Manual, así que
obliga a un humano a investigar un desastre que no pasó.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DOC = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")


def _bloque_emision() -> str:
    i = _SRC.index("[P1-PIPELINE-TIMEOUT-DEGRADED · 2026-09-06] Dos ramas, dos alertas")
    return _SRC[i:i + 3200]


# ── las dos ramas ─────────────────────────────────────────────────────────────────────────────
def test_hay_dos_alertas_distintas():
    b = _bloque_emision()
    assert '"pipeline_crash_fallback"' in b
    assert '"pipeline_timeout_degraded"' in b


def test_la_rama_del_fallback_total_conserva_su_texto():
    """El texto de siempre es CORRECTO cuando el fallback matemático sí ocurrió. Cambiarlo allí
    habría sido sustituir una mentira por otra."""
    b = _bloque_emision()
    i_fb = b.index('"pipeline_crash_fallback"')
    i_deg = b.index('"pipeline_timeout_degraded"')
    assert i_fb < i_deg
    # la rama del fallback total NO pasa title/message: usa los defaults del helper
    assert "title=" not in b[i_fb:i_deg]


def test_la_rama_degradada_dice_que_el_plan_SI_se_entrego():
    b = _bloque_emision()
    seg = b[b.index('"pipeline_timeout_degraded"'):]
    assert "SI se entrego" in seg
    assert "NO es el fallback matematico" in seg


def test_el_flag_lo_pone_cada_rama_del_except():
    """Sin esto la alerta no sabría cuál de las dos ocurrió — y adivinarlo es como llegamos aquí."""
    assert _SRC.count("_entrego_fallback_total = True") == 2   # el default fail-safe + la rama real
    assert _SRC.count("_entrego_fallback_total = False") == 1


def test_el_default_falla_hacia_la_alerta_ruidosa():
    """Si algún día aparece un camino que no pasa por el if/else, es mejor gritar de más que
    tranquilizar de menos: el default es el fallback total, que es el mensaje alarmante."""
    i_def = _SRC.index("Fail-safe hacia la alerta RUIDOSA")
    i_partial = _SRC.index('plan_partial = final_state.get("plan_result")', i_def - 400)
    assert i_def < i_partial, "el default tiene que asignarse ANTES del if/else"
    assert "_entrego_fallback_total = True" in _SRC[i_def:i_partial]


# ── el detail deja de llegar vacío ────────────────────────────────────────────────────────────
def test_el_detail_completa_el_timeout_sin_mensaje():
    """`TimeoutError: ` a secas no dice QUÉ timeout. Con elapsed y el techo, el operador distingue
    «me pasé de mi presupuesto» de «se cayó el proveedor» sin abrir un log."""
    b = _bloque_emision()
    assert "if not _msg_e:" in b
    assert "MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S=" in b
    assert "elapsed=" in b


def test_el_calculo_del_elapsed_es_fail_safe():
    """Una alerta que revienta calculando su propio detalle no se emite, y entonces no hay alerta."""
    b = _bloque_emision()
    seg = b[b.index("_msg_e = str(e)"):b.index("if _entrego_fallback_total")]
    assert "try:" in seg and "except Exception:" in seg


# ── el helper ─────────────────────────────────────────────────────────────────────────────────
def test_el_helper_acepta_su_texto_y_conserva_los_defaults():
    i = _SRC.index("def _persist_pipeline_crash_alert(")
    firma = _SRC[i:i + 300]
    assert "title: Optional[str] = None" in firma and "message: Optional[str] = None" in firma
    cuerpo = _SRC[i:i + 4000]
    assert 'title or "Pipeline de generacion cayo a fallback de emergencia"' in cuerpo
    assert "message or" in cuerpo


def test_los_dos_callers_historicos_siguen_sin_pasar_texto():
    """`pipeline_emergency_fallback_p1_5` no se toca: allí el fallback de emergencia SÍ ocurrió."""
    i = _SRC.index('"pipeline_emergency_fallback_p1_5"')
    assert "title=" not in _SRC[i:i + 260]


# ── la clave nueva está documentada ───────────────────────────────────────────────────────────
def test_la_clave_nueva_tiene_su_fila():
    """El test de deriva bidireccional (`test_p2_audit_4_alert_keys_documented`) ya lo exige; esto lo
    dice explícito y comprueba que la fila lleva lo que un operador necesita leer primero."""
    assert "`pipeline_timeout_degraded`" in _DOC
    fila = _DOC[_DOC.index("| `pipeline_timeout_degraded`"):]
    fila = fila[:fila.index("\n")]
    assert "sí recibió su menú" in fila
    assert "MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S" in fila
    assert fila.rstrip().endswith("| Manual |"), "el modelo de resolution tiene que estar declarado"


@pytest.mark.parametrize("evidencia", ["delivered_was_fallback", "900", "2b692ef8"])
def test_el_docstring_conserva_la_evidencia(evidencia):
    """La medición que justificó el cambio vive con el test: dentro de seis meses, «la alerta mentía»
    sin las cifras es indistinguible de una opinión."""
    assert evidencia in __doc__
