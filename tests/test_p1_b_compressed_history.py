"""[P1-B] Tests para awareness de chunks históricos comprimidos en el prompt.

Cubre:
  1. `build_chunk_lessons_context`:
       - compressed_history_chunks_count = 0 → no aparece bullet de historial.
       - count > 0 con avg/max bajo → tampoco aparece (no hay señal accionable).
       - count > 0 con avg_repeat_pct alto → bullet con N y avg.
       - count > 0 con max_violations > 0 → bullet incluye conteo de violaciones.
       - count > 0 con max_repeat_pct alto → bullet aparece.
  2. Sanity: el bullet P1-B no rompe los bullets pre-existentes (URGENTE,
     RECHAZADOS, alergia, repeated_meal_names) cuando coexisten.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_b_compressed_history.py -v
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs minimal para que `from prompts.plan_generator import ...` cargue sin
# requerir DB / Google clients reales.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )

import pytest
from prompts.plan_generator import build_chunk_lessons_context


def _base_lessons(**overrides):
    """Lessons mínimas que NO disparan los bullets pre-existentes (sin
    `repeated_bases` con pct alto, sin rechazados, sin alergias). Esto deja al
    bullet P1-B como único candidato salvo que el test añada más señales.
    """
    base = {
        "chunk_numbers": [4, 5],
        "ingredient_base_repeat_pct": 10.0,    # < 30, no dispara URGENTE
        "repeated_bases": [],
        "repeat_pct": 5.0,                     # < 15, no dispara repeated_names
        "repeated_meal_names": [],
        "rejection_violations": 0,
        "rejected_meals_that_reappeared": [],
        "allergy_violations": 0,
        "allergy_hits": [],
        "is_lifetime_aggregated": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Sin chunks comprimidos → no aparece bullet
# ---------------------------------------------------------------------------
def test_no_compressed_history_no_bullet():
    lessons = _base_lessons(compressed_history_chunks_count=0)
    # Necesitamos AL MENOS UN bullet para que ctx no sea string vacío.
    # Forzamos uno añadiendo repeated_bases con pct alto.
    lessons.update({
        "ingredient_base_repeat_pct": 70.0,
        "repeated_bases": [{"chunk": 1, "bases": ["pollo"]}],
    })
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" not in ctx
    assert "URGENTE" in ctx  # bullet pre-existente sigue ahí


def test_compressed_history_below_threshold_no_bullet():
    """count>0 pero avg/max/violaciones bajo → no emite bullet (sin señal accionable)."""
    lessons = _base_lessons(
        compressed_history_chunks_count=3,
        compressed_history_avg_repeat_pct=10.0,   # < 15 threshold
        compressed_history_max_repeat_pct=20.0,   # < 30 threshold
        compressed_history_max_violations=0,
        # Forzar otro bullet para que ctx no sea vacío.
        ingredient_base_repeat_pct=70.0,
        repeated_bases=[{"chunk": 1, "bases": ["pollo"]}],
    )
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" not in ctx


# ---------------------------------------------------------------------------
# 2. Con compressed_history accionable → aparece bullet con métricas
# ---------------------------------------------------------------------------
def test_high_avg_repeat_triggers_bullet():
    lessons = _base_lessons(
        compressed_history_chunks_count=4,
        compressed_history_avg_repeat_pct=22.0,   # > 15 threshold
        compressed_history_max_repeat_pct=35.0,
        compressed_history_max_violations=0,
        # Forzar otro bullet para que ctx no sea vacío.
        ingredient_base_repeat_pct=70.0,
        repeated_bases=[{"chunk": 1, "bases": ["pollo"]}],
    )
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" in ctx
    assert "4 chunk(s)" in ctx
    assert "22%" in ctx  # avg
    assert "35%" in ctx  # max


def test_high_max_repeat_alone_triggers_bullet():
    """max alto aún si avg es bajo → emite bullet."""
    lessons = _base_lessons(
        compressed_history_chunks_count=2,
        compressed_history_avg_repeat_pct=8.0,    # < 15
        compressed_history_max_repeat_pct=45.0,   # > 30 threshold
        compressed_history_max_violations=0,
        ingredient_base_repeat_pct=70.0,
        repeated_bases=[{"chunk": 1, "bases": ["pollo"]}],
    )
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" in ctx
    assert "2 chunk(s)" in ctx


def test_violations_alone_trigger_bullet():
    """violations > 0 dispara bullet aunque repeat_pct sean bajos."""
    lessons = _base_lessons(
        compressed_history_chunks_count=5,
        compressed_history_avg_repeat_pct=5.0,
        compressed_history_max_repeat_pct=10.0,
        compressed_history_max_violations=3,
        ingredient_base_repeat_pct=70.0,
        repeated_bases=[{"chunk": 1, "bases": ["pollo"]}],
    )
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" in ctx
    assert "5 chunk(s)" in ctx
    assert "máx violaciones por chunk=3" in ctx


def test_zero_violations_omits_violations_substring():
    """Cuando max_violations=0 el bullet no incluye la sub-frase de violaciones."""
    lessons = _base_lessons(
        compressed_history_chunks_count=3,
        compressed_history_avg_repeat_pct=20.0,
        compressed_history_max_repeat_pct=30.0,
        compressed_history_max_violations=0,
        ingredient_base_repeat_pct=70.0,
        repeated_bases=[{"chunk": 1, "bases": ["pollo"]}],
    )
    ctx = build_chunk_lessons_context(lessons)
    assert "HISTORIAL COMPRIMIDO" in ctx
    assert "violaciones" not in ctx.lower(), (
        "no debe mencionar violaciones cuando max=0"
    )


# ---------------------------------------------------------------------------
# 3. Coexistencia con bullets pre-existentes (regresión)
# ---------------------------------------------------------------------------
def test_p1b_bullet_does_not_break_existing_bullets():
    """Cuando hay señales pre-existentes Y compressed_history, todos los bullets
    coexisten en orden."""
    lessons = {
        "chunk_numbers": [4],
        "ingredient_base_repeat_pct": 65.0,        # URGENTE
        "repeated_bases": [{"chunk": 4, "bases": ["pollo", "arroz blanco"]}],
        "repeat_pct": 22.0,
        "repeated_meal_names": ["Pollo a la plancha"],
        "rejection_violations": 2,
        "rejected_meals_that_reappeared": ["pollo a la plancha"],
        "allergy_violations": 1,
        "allergy_hits": ["mani"],
        "is_lifetime_aggregated": False,
        # P1-B
        "compressed_history_chunks_count": 6,
        "compressed_history_avg_repeat_pct": 18.0,
        "compressed_history_max_repeat_pct": 33.0,
        "compressed_history_max_violations": 2,
    }
    ctx = build_chunk_lessons_context(lessons)
    # Pre-existentes
    assert "URGENTE" in ctx
    assert "RECHAZADOS" in ctx
    assert "alergia" in ctx.lower()
    assert "Pollo a la plancha" in ctx
    # P1-B
    assert "HISTORIAL COMPRIMIDO" in ctx
    assert "6 chunk(s)" in ctx


def test_empty_lessons_dict_returns_empty():
    """Sanity preservado: dict vacío → string vacío (no rompe el path)."""
    assert build_chunk_lessons_context({}) == ""
    assert build_chunk_lessons_context(None) == ""


def test_only_compressed_history_no_other_bullets_returns_empty():
    """Si SOLO hay compressed_history pero NINGÚN otro bullet califica, ctx
    debería ser vacío. Esto es una decisión de diseño: el bullet P1-B es
    AUMENTATIVO sobre otros bullets, no un bullet primario.

    Razón: el LLM no necesita el contexto histórico si no hay nada concreto
    que reprochar en el chunk anterior; emite ruido sin valor accionable.
    """
    lessons = _base_lessons(
        compressed_history_chunks_count=4,
        compressed_history_avg_repeat_pct=25.0,
        compressed_history_max_repeat_pct=40.0,
        compressed_history_max_violations=2,
    )
    # NO añadimos otros bullets primarios.
    ctx = build_chunk_lessons_context(lessons)
    assert ctx == "", (
        f"Sin bullets primarios, P1-B no debe emitir solo (evitar ruido). Got: {ctx!r}"
    )
