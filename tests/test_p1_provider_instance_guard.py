"""[P1-PROVIDER-INSTANCE-GUARD · 2026-07-28] Fix del guard `_is_glm_provider`
en `ChatGLM.with_structured_output` (llm_provider.py).

Bug (medido en vivo 2026-07-28): `ChatGLM` es una subclase de `ChatOpenAI`
usada cada vez más contra back-ends OpenAI-compatibles DISTINTOS de GLM
(el meal-photo scanner apuntado a OpenAI, sidesteado en commit `b98efbb`
dispatcheando a `ChatOpenAI` en vez de arreglar la causa raíz — ver
P1-VISION-LUNA). El constructor (`__init__`, línea ~297) ya llamaba
`_is_glm_provider(base_url)` correctamente — pasando el `base_url` que
ESA instancia recibió. Pero `with_structured_output` (línea ~328) llamaba
`_is_glm_provider()` SIN argumento: sin argumento el guard cae al knob
global `MEALFIT_ZAI_BASE_URL`, que SIEMPRE contiene "glm", así que
retornaba `True` para TODA instancia — incluida una `ChatGLM` apuntada a
`https://api.openai.com/v1` — inyectando
`extra_body={"thinking": {"type": "disabled"}}` (parámetro GLM-only) →
`400 Unknown parameter: 'thinking'` contra OpenAI.

Fix: resolver el provider de la INSTANCIA (`self.openai_api_base`, el
atributo real que `ChatOpenAI.__init__` persiste desde el kwarg `base_url` —
verificado empíricamente que `base_url` NO es atributo de instancia), con
fallback defensivo al knob global si el atributo faltara (futura versión de
langchain-openai que renombre el campo) — preserva el comportamiento
GLM-only de los ~15 callsites productivos.

Qué ancla este test (BEHAVIORAL, no solo parsing de source):
  A. Instancia apuntada a OpenAI → `with_structured_output` NO inyecta `thinking`.
  B. Instancia apuntada a GLM (default) → SIGUE inyectando `thinking`
     (anti-regresión de los ~15 callsites de generación).
  C. Fallback defensivo: si `openai_api_base` falta en la instancia, el guard
     cae al knob global (`MEALFIT_ZAI_BASE_URL`) — mismo comportamiento
     que existía ANTES de este fix, en ambas direcciones (knob apunta a
     glm / knob apunta a otro host).
  D. Supersession-proof marker: `_LAST_KNOWN_PFIX` en app.py ≥ este P-fix.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

import llm_provider as lp

_BACKEND = Path(__file__).resolve().parent.parent


class _TrivialSchema(BaseModel):
    """Schema mínimo, solo para ejercer `with_structured_output` sin red."""

    x: int


def _bound_extra_body(runnable):
    """`with_structured_output` retorna un `RunnableSequence` cuyo `.first`
    es un binding sobre el chat model (posiblemente un `model_copy` con
    `extra_body` mergeado — ver `ChatGLM.with_structured_output:331`).
    El `extra_body` vive en el modelo bound, no en los kwargs del binding
    (verificado empíricamente inspeccionando el objeto retornado)."""
    return runnable.first.bound.extra_body


# ---------------------------------------------------------------------------
# A. Instancia OpenAI: NO debe inyectar `thinking`. Este es el bug real.
# ---------------------------------------------------------------------------


def test_openai_pointed_instance_does_not_inject_thinking():
    """Una ChatGLM construida contra OpenAI NO debe recibir el
    `extra_body.thinking` GLM-only — de lo contrario OpenAI responde
    400 'Unknown parameter: thinking' (caso real: meal-photo scanner)."""
    instance = lp.ChatGLM(
        model="glm-5.3-flash",
        api_key="sk-test-fake",
        base_url="https://api.openai.com/v1",
    )
    runnable = instance.with_structured_output(_TrivialSchema)
    extra_body = _bound_extra_body(runnable)
    assert extra_body is None or "thinking" not in (extra_body or {}), (
        f"con base_url OpenAI, with_structured_output NO debe inyectar "
        f"'thinking' (parámetro GLM-only) — extra_body={extra_body!r}"
    )
    # Además: como el guard es False, no debe ocurrir model_copy — el
    # runnable debe estar bound sobre la MISMA instancia (evita una
    # asignación superflua, y confirma que la rama `base = self` corrió).
    assert runnable.first.bound is instance


# ---------------------------------------------------------------------------
# B. Instancia GLM real: SIGUE inyectando `thinking` (anti-regresión).
# ---------------------------------------------------------------------------


def test_glm_pointed_instance_still_injects_thinking():
    """Anti-regresión de los ~15 callsites productivos: una ChatGLM
    apuntada al GLM real (base_url default) sigue recibiendo el
    `extra_body.thinking` del que depende el pipeline de generación."""
    instance = lp.ChatGLM(model="glm-5.3-flash", api_key="sk-test-fake")
    assert "z.ai" in (instance.openai_api_base or "").lower()  # sanity del fixture

    runnable = instance.with_structured_output(_TrivialSchema)
    extra_body = _bound_extra_body(runnable)
    assert extra_body == {"thinking": {"type": "enabled"}}, (
        f"instancia GLM real debe seguir inyectando 'thinking' — "
        f"extra_body={extra_body!r} (regresión de los ~15 callsites de "
        f"generación si esto falla)"
    )


# ---------------------------------------------------------------------------
# C. Fallback defensivo: atributo `openai_api_base` ausente en la instancia.
# ---------------------------------------------------------------------------


def test_missing_openai_api_base_falls_back_to_global_knob_glm_default(monkeypatch):
    """Si `openai_api_base` desapareciera de la instancia (rename futuro en
    langchain-openai), el guard debe caer al knob global — mismo comportamiento
    que existía ANTES de este fix. [P0-GLM-MIGRATION · 2026-09-02] Lo que el guard
    decide ahora en `with_structured_output` no es inyectar `thinking` (eso vive en
    el constructor) sino reencaminar `json_mode` → `function_calling`, porque en Z.ai
    `json_mode` ignora el esquema. Con el atributo ausente y el knob global apuntando
    a z.ai, un `json_mode` pedido debe salir como function_calling (tools bound)."""
    monkeypatch.delenv("MEALFIT_ZAI_BASE_URL", raising=False)  # default contiene "z.ai"
    instance = lp.ChatGLM(
        model="glm-5.3-flash",
        api_key="sk-test-fake",
        base_url="https://api.openai.com/v1",  # instancia real apunta a OpenAI...
    )
    delattr(instance, "openai_api_base")  # ...pero simulamos el atributo AUSENTE
    assert not hasattr(instance, "openai_api_base")
    runnable = instance.with_structured_output(_TrivialSchema, method="json_mode")
    bound_kwargs = getattr(runnable.first, "kwargs", {}) or {}
    assert "tools" in bound_kwargs, (
        f"con el atributo ausente, el guard debe caer al knob global (z.ai) y "
        f"reencaminar json_mode a function_calling — kwargs={list(bound_kwargs)}"
    )
def test_missing_openai_api_base_falls_back_to_global_knob_non_glm(monkeypatch):
    """Mismo fallback, dirección opuesta: si el knob global apunta a un host
    SIN 'glm', el fallback tampoco inyecta — confirma que el fallback
    replica fielmente `_is_glm_provider()` sin argumento (el
    comportamiento de hoy), no un hardcode a True."""
    monkeypatch.setenv("MEALFIT_ZAI_BASE_URL", "https://proxy.example.com")

    instance = lp.ChatGLM(
        model="glm-5.3-flash",
        api_key="sk-test-fake",
        base_url="https://api.openai.com/v1",
    )
    delattr(instance, "openai_api_base")
    assert not hasattr(instance, "openai_api_base")

    runnable = instance.with_structured_output(_TrivialSchema)
    extra_body = _bound_extra_body(runnable)
    assert extra_body is None or "thinking" not in (extra_body or {}), (
        f"con el atributo ausente Y el knob global apuntado a un host "
        f"no-GLM, el fallback NO debe inyectar 'thinking' — "
        f"extra_body={extra_body!r}"
    )


# ---------------------------------------------------------------------------
# D. Marker supersession-proof.
# ---------------------------------------------------------------------------


def test_marker_bumped():
    """Supersession-proof: este P-fix o uno posterior (fecha ≥) — mirror de
    `test_p2_audit_v5_batch.py::test_marker_bumped`. Un `startswith` es
    landmine (rompió dos veces esta semana en el repo) porque el owner
    landea P-fixes en paralelo y puede bumpear el marker de nuevo el mismo
    día."""
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-PROVIDER-INSTANCE-GUARD" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-28", (
        f"marker {m.group(1)!r} anterior a P1-PROVIDER-INSTANCE-GUARD"
    )
