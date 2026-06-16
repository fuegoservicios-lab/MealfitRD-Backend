"""[P1-PROMPT-CACHE-STAGGER · 2026-05-16] Optimización del implicit prompt
caching de Gemini.

Audit 2026-05-16 reveló cache hit de solo 16% en gemini-3-flash-preview
(vs 50%+ esperado tras P1-PROMPT-CACHE-SYSTEMMSG). Análisis identificó dos
causas:

  1. **Schema regenerado per-call**: `SingleDayPlanModel.model_json_schema()`
     se invocaba dentro de `generate_single_day` en cada llamada. Pydantic V2
     es generalmente determinístico, pero `json.dumps` sin `sort_keys=True`
     dependía del orden de inserción del dict — fragile. Cualquier variance
     byte = cache miss garantizado.

  2. **Day_gen paralelos arrancan simultáneamente** sin compartir cache.
     Los 3 days disparan en `asyncio.gather`, todos cold-start, ninguno
     hereda el cache de los otros.

Fix:
  - `_DAY_SCHEMA_INSTRUCTION` + `_DAY_SYSTEM_INSTRUCTION_CACHED` pre-computados
    a nivel módulo con `json.dumps(..., sort_keys=True)`. Byte-equivalence
    garantizada.
  - Knob `MEALFIT_DAY_GEN_CACHE_STAGGER_MS` (default 0 = legacy). Cuando >0,
    days 2..N esperan `i * stagger_ms` antes de disparar → day 1 popula cache,
    days 2..N hitan warm cache.

Este test ancla las invariantes del fix.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PY = _BACKEND_ROOT / "graph_orchestrator.py"


def test_knob_registered():
    """Knob `MEALFIT_DAY_GEN_CACHE_STAGGER_MS` debe estar registrado vía
    `_env_int` (auto-registro en `_KNOBS_REGISTRY`, P3-NEW-D)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert 'DAY_GEN_CACHE_STAGGER_MS' in src
    assert '_env_int  ("MEALFIT_DAY_GEN_CACHE_STAGGER_MS"' in src, (
        "Knob debe usar `_env_int(...)` para auto-registrarse en "
        "_KNOBS_REGISTRY (convención P3-NEW-D)."
    )


def test_module_level_constants_present():
    """Constantes pre-computadas a nivel módulo eliminan re-cómputo por
    call + garantizan byte-equivalence del SystemMessage.

    [stale-parser fix] `_DAY_SYSTEM_INSTRUCTION_CACHED` evolucionó de un
    single-line `= DAY_GENERATOR_SYSTEM_PROMPT` a una concatenación
    multi-línea `(DAY_GENERATOR_SYSTEM_PROMPT + _DAY_SCHEMA_INSTRUCTION +
    _NUTRITION_LOOKUP_INSTRUCTION)` — la instrucción cacheable ahora
    embebe schema + nutrition lookup (más superficie cacheable estable).
    Sigue siendo una constante módulo-level construida desde el prompt
    base. El regex acepta el `=` seguido de `DAY_GENERATOR_SYSTEM_PROMPT`
    aunque medie un `(` y salto de línea."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "_DAY_SCHEMA_INSTRUCTION = _build_day_schema_instruction()" in src
    assert re.search(
        r"_DAY_SYSTEM_INSTRUCTION_CACHED\s*=\s*\(?\s*DAY_GENERATOR_SYSTEM_PROMPT",
        src,
    ), (
        "`_DAY_SYSTEM_INSTRUCTION_CACHED` debe ser una constante módulo-level "
        "construida desde `DAY_GENERATOR_SYSTEM_PROMPT` (regression de "
        "P1-PROMPT-CACHE-STAGGER si se re-genera per-call)."
    )


def test_schema_instruction_uses_sort_keys():
    """`json.dumps(schema_dict, sort_keys=True)` garantiza orden
    determinístico independiente de Pydantic. Sin sort_keys, cualquier
    variance de orden = cache miss."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Buscar el builder y validar sort_keys=True.
    builder = re.search(
        r"def _build_day_schema_instruction.*?(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert builder, "Helper `_build_day_schema_instruction` no encontrado."
    body = builder.group(0)
    assert "json.dumps(schema_dict, sort_keys=True)" in body, (
        "`_build_day_schema_instruction` debe usar `sort_keys=True` para "
        "garantizar byte-equivalence determinística del schema serializado."
    )


def test_generate_single_day_uses_module_constants():
    """`generate_single_day` debe usar `_DAY_SYSTEM_INSTRUCTION_CACHED` y
    `_DAY_SCHEMA_INSTRUCTION` en lugar de generar el schema per-call."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Aislar el cuerpo de generate_single_day.
    fn = re.search(
        r"async def generate_single_day\b.*?(?=\n    async def |\n    def |\Z)",
        src,
        re.DOTALL,
    )
    assert fn, "generate_single_day no encontrada."
    body = fn.group(0)
    # Debe usar los constants.
    assert "_DAY_SYSTEM_INSTRUCTION_CACHED" in body, (
        "generate_single_day no usa la constante pre-computada — re-genera "
        "el schema en cada call (regression de P1-PROMPT-CACHE-STAGGER)."
    )
    # No debe regenerar schema_dict inline.
    assert "schema_dict = SingleDayPlanModel.model_json_schema()" not in body, (
        "generate_single_day NO debe invocar `model_json_schema()` inline — "
        "el schema ya está pre-computado en `_DAY_SCHEMA_INSTRUCTION`."
    )


def test_stagger_logic_present_in_gather():
    """El despacho de los N day_coros debe respetar el knob: cuando
    `DAY_GEN_CACHE_STAGGER_MS > 0`, days 2..N esperan antes de disparar."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Buscar la sección con `asyncio.gather(*day_coros)`.
    assert "stagger_ms = DAY_GEN_CACHE_STAGGER_MS" in src, (
        "El gather de day_coros no consulta `DAY_GEN_CACHE_STAGGER_MS` — "
        "el knob no surte efecto."
    )
    assert "async def _staggered(coro, delay_s):" in src, (
        "Wrapper `_staggered` ausente — sin él no hay forma de retrasar "
        "days 2..N."
    )
    # Default 0 preserva comportamiento legacy (paralelismo puro).
    assert 'stagger_ms > 0' in src, (
        "Falta gate `if stagger_ms > 0` — el knob default 0 debe NO "
        "alterar el comportamiento legacy."
    )


def test_default_stagger_is_1500():
    """[P2-ORCH-1 · 2026-05-28] Default del knob cambió de 0 → 1500ms
    (activo). day_gen es el nodo más caro (>50% del gasto) y con stagger=0
    los N días disparaban simultáneos → 16% cache hit medido vs ~50-60%
    esperado ($0.20-0.30/plan). Worst-case latencia añadida = (N-1)*stagger
    (~3s con PLAN_CHUNK_SIZE=3), despreciable vs el timeout global. Clamp
    [0,10000]. Revertir sin redeploy: MEALFIT_DAY_GEN_CACHE_STAGGER_MS=0.

    El default vive en el llamado a `_env_int(..., 1500, validator=...)`."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert re.search(
        r'_env_int\s*\(\s*"MEALFIT_DAY_GEN_CACHE_STAGGER_MS"\s*,\s*1500\s*,',
        src,
    ), (
        "Default del knob debe ser 1500 (P2-ORCH-1: stagger activo por "
        "default para subir cache hit del nodo day-gen)."
    )


def test_no_per_call_json_dumps_of_schema_in_generate_single_day():
    """Defense in depth: `generate_single_day` no debe contener
    `json.dumps(schema_dict)` en ninguna forma — el schema vive ahora
    en la constante módulo-level."""
    src = _GO_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"async def generate_single_day\b.*?(?=\n    async def |\n    def |\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    assert "json.dumps(schema_dict" not in body, (
        "Regresión: schema vuelve a serializarse per-call. Esto rompe la "
        "byte-equivalence garantizada y vuelve a romper el cache."
    )
