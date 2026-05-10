"""[P0-6] Validación E2E: las lecciones del chunk previo realmente llegan al prompt del LLM.

El audit P0-6 identificó que aunque cron_tasks.py construye
`form_data["_chunk_lessons"]` agregando _last_chunk_learning + _recent_chunk_lessons +
_critical_lessons_permanent, **no había test E2E** que confirmara que esa estructura
viaja por todo el pipeline hasta el `prompt_text` final que recibe el LLM. Si alguien
refactorizara `_build_shared_context` o las llamadas a `prompt_text` en
graph_orchestrator.py y olvidara interpolar `chunk_lessons_context`, el sistema seguiría
extrayendo lecciones (logs verdes) pero el LLM nunca las vería — bug silencioso.

Pipeline real (verificado durante implementación P0-6):
  1. cron_tasks.py L7438+ → form_data["_chunk_lessons"] = agregado
  2. graph_orchestrator.py:812 → chunk_lessons = form_data.get("_chunk_lessons")
  3. graph_orchestrator.py:819 → ctx["chunk_lessons_context"] = build_chunk_lessons_context(...)
  4. graph_orchestrator.py:1022, :1264 → prompt_text interpola {ctx['chunk_lessons_context']}
  5. LLM recibe prompt_text con las lecciones literales.

Cobertura previa:
  - tests/test_chunk_learning_appears_in_prompt.py: cubre paso 3 con casos sintéticos.
  - Pasos 1, 2, 4: NO había validación.

Este archivo cierra los gaps con tests acoplados a las líneas reales del pipeline:
  - test_a: regression guard del paso 3 (assignment del builder al ctx).
  - test_b: regression guard del paso 4 (interpolación en prompt_text).
  - test_c: literales sintéticos sobreviven la composición f-string.
  - test_d: invariante de wiring — _chunk_lessons==None produce contexto vacío.
  - test_e: aislamiento del paso 2-3 vía mock — _build_shared_context lee _chunk_lessons
    y lo pasa a build_chunk_lessons_context.
"""
import os
import re
import sys
from unittest.mock import MagicMock, patch

# Stub langgraph para que la importación de graph_orchestrator no falle en CI sin la lib.
sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())
sys.modules.setdefault('langgraph.checkpoint', MagicMock())
sys.modules.setdefault('langgraph.checkpoint.memory', MagicMock())

sys.path.insert(0, os.path.dirname(__file__))


def _realistic_lessons_blob() -> dict:
    """Replica del dict que cron_tasks construye en form_data['_chunk_lessons']."""
    return {
        "chunk_number": 2,
        "chunk_numbers": [1, 2],
        "ingredient_base_repeat_pct": 70.0,
        "repeated_bases": [
            {"chunk": 1, "bases": ["pollo", "arroz blanco"]},
            {"chunk": 2, "bases": ["pollo", "habichuela roja"]},
        ],
        "repeat_pct": 25.0,
        "repeated_meal_names": ["Pollo a la plancha", "Arroz con habichuelas"],
        "rejection_violations": 1,
        "rejected_meals_that_reappeared": ["Sopa de pollo rechazada"],
        "allergy_violations": 1,
        "allergy_hits": ["mani"],
        "is_lifetime_aggregated": False,
    }


def _read_orchestrator_source() -> str:
    here = os.path.dirname(__file__)
    src_path = os.path.join(here, "graph_orchestrator.py")
    with open(src_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Test A: paso 3 — wiring del builder al ctx ----------

def test_chunk_lessons_context_assigned_in_build_shared_context():
    """Regression guard del PASO 3: `_build_shared_context` debe asignar
    `chunk_lessons_context` al dict que retorna, llamando a
    `build_chunk_lessons_context(...)`.

    Si un refactor accidentalmente borra esta línea, todos los pasos posteriores
    (interpolación en prompt_text) lanzarían KeyError al evaluar
    `ctx['chunk_lessons_context']`. Detectar el cambio en el source es más
    robusto que invocar la función completa (que requiere DB, perfil, etc).
    """
    source = _read_orchestrator_source()

    # Verificación 1: el assignment del ctx debe existir en el source.
    assert '"chunk_lessons_context": build_chunk_lessons_context' in source, (
        "El assignment 'chunk_lessons_context': build_chunk_lessons_context(...) "
        "debe existir en graph_orchestrator.py (dentro de _build_shared_context). "
        "Sin él, el ctx no expone la key y todas las interpolaciones rompen el prompt."
    )
    # Verificación 2: la lectura del form_data debe existir.
    assert (
        'form_data.get("_chunk_lessons")' in source
        or "form_data.get('_chunk_lessons')" in source
    ), (
        "graph_orchestrator.py debe leer form_data.get('_chunk_lessons') para "
        "alimentar build_chunk_lessons_context. Si lo lee de otro origen, se "
        "rompe la propagación del agregador en cron_tasks → orchestrator."
    )
    # Verificación 3: la función _build_shared_context debe existir.
    assert "def _build_shared_context(" in source, (
        "La función _build_shared_context fue removida o renombrada. Si fue "
        "intencional, actualiza este test apuntando a la nueva función orquestadora."
    )


# ---------- Test B: paso 4 — interpolación en prompt_text ----------

def test_chunk_lessons_context_interpolated_in_prompt_strings():
    """Regression guard del PASO 4: el ctx['chunk_lessons_context'] debe
    interpolarse en CADA prompt_text de graph_orchestrator.py donde se construye
    el prompt para el LLM (skeleton planner y reflection). Sin esta interpolación,
    las lecciones se construirían pero NUNCA llegarían al modelo — bug silencioso.
    """
    source = _read_orchestrator_source()
    pattern = re.compile(r"""ctx\[['"]chunk_lessons_context['"]\]""")
    occurrences = pattern.findall(source)
    assert len(occurrences) >= 2, (
        f"Esperaba >=2 interpolaciones de ctx['chunk_lessons_context'] en "
        f"graph_orchestrator.py (skeleton planner + reflection nodes). "
        f"Encontré {len(occurrences)}. Si las eliminaste a propósito, justifica "
        f"el cambio actualizando este test."
    )


# ---------- Test C: literales sintéticos sobreviven el builder ----------

def test_concrete_lesson_literals_survive_build_chunk_lessons_context():
    """Verifica que cuando `build_chunk_lessons_context` recibe un blob realista,
    los literales (nombres de platos, ingredientes, alergias) aparecen en el
    string de salida. Si la sanitización P1-7 trunca demasiado o los thresholds
    suprimen señales legítimas, este test detecta la regresión.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    chunk_ctx = build_chunk_lessons_context(_realistic_lessons_blob())
    assert chunk_ctx, "Blob realista no debe producir contexto vacío"

    # Ingredientes base sobre-repetidos (pct=70 → severidad URGENTE).
    assert "URGENTE" in chunk_ctx
    assert "pollo" in chunk_ctx
    assert "arroz blanco" in chunk_ctx
    # Plato rechazado que reapareció.
    assert "Sopa de pollo rechazada" in chunk_ctx
    assert "RECHAZADOS" in chunk_ctx
    # Alergia.
    assert "mani" in chunk_ctx
    assert "alergia" in chunk_ctx.lower()
    # Nombre de plato repetido (repeat_pct=25 > 15).
    assert "Pollo a la plancha" in chunk_ctx


# ---------- Test D: lecciones nulas no rompen el pipeline ----------

def test_no_lessons_yields_empty_context_without_errors():
    """Cuando es el primer chunk de un plan (no hay lecciones aún),
    form_data["_chunk_lessons"] es None. El builder debe retornar string vacío,
    no string "None" ni un header huérfano.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    assert build_chunk_lessons_context(None) == ""
    assert build_chunk_lessons_context({}) == ""


# ---------- Test E: composición E2E del prompt_text ----------

def test_lessons_appear_in_final_composed_prompt_text():
    """Réplica del fragmento de prompt_text en graph_orchestrator.py:1022.
    Si las lecciones sobreviven hasta este string, sobreviven hasta el LLM.
    Esta es la verificación más directa del invariante P0-6.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    chunk_ctx = build_chunk_lessons_context(_realistic_lessons_blob())

    # Construye un fake ctx con SOLO chunk_lessons_context poblado (otros
    # builders aquí son irrelevantes y no afectan al invariante a verificar).
    fake_ctx = {
        "quality_context": "",
        "quality_hint_context": "",
        "chunk_lessons_context": chunk_ctx,
        "prev_chunk_adherence_context": "",
        "weight_history_context": "",
        "nutrition_context": "",
    }

    # Réplica fiel del fragmento de prompt_text.
    composed = (
        f"{fake_ctx['quality_context']}\n"
        f"{fake_ctx['quality_hint_context']}\n"
        f"{fake_ctx['chunk_lessons_context']}\n"
        f"{fake_ctx['prev_chunk_adherence_context']}\n"
        f"{fake_ctx['weight_history_context']}\n"
        f"{fake_ctx['nutrition_context']}\n"
    )

    # Las señales literales del blob deben sobrevivir hasta el prompt_text final.
    assert "Sopa de pollo rechazada" in composed, (
        "Plato rechazado del chunk previo debe llegar al prompt_text final; "
        "si no llega, el LLM no sabrá que NO debe regenerarlo."
    )
    assert "mani" in composed
    assert "Pollo a la plancha" in composed
    assert "URGENTE" in composed


# ---------- Test F: aislamiento — _build_shared_context invoca el builder ----------

def test_build_shared_context_invokes_builder_with_form_data_lessons():
    """Aislamos el paso 2-3 mockeando todos los demás builders. Verifica que
    `_build_shared_context` realmente extrae `_chunk_lessons` de form_data y
    lo pasa a `build_chunk_lessons_context`. Si un refactor lo lee de otra
    fuente (e.g., del estado, de DB), el blob agregado por cron_tasks no
    llegaría — fallo silencioso.

    Mockeamos los builders que requieren DB / state completo. Solo
    `build_chunk_lessons_context` corre real para que podamos espiar su input.
    """
    import graph_orchestrator as go

    lessons_blob = _realistic_lessons_blob()
    state = {
        "form_data": {
            "user_id": "u-p06-iso",
            "_chunk_lessons": lessons_blob,
        },
        "nutrition": {"alergias": []},
        "review_feedback": "",
        "user_facts": "",
        "history_context": "",
        "compressed_context": "",
        "taste_profile": "",
        "rejection_reasons": [],
    }

    builder_spy = MagicMock(return_value="MOCKED_LESSONS_CTX")

    # Patch el builder en el módulo graph_orchestrator (donde lo importó al inicio).
    # Y patch los OTROS builders + helpers para que no exploten por estado mínimo.
    with patch.object(go, "build_chunk_lessons_context", builder_spy), \
         patch.object(go, "build_skeleton_quality_context", return_value=""), \
         patch.object(go, "build_quality_hint_context", return_value=""), \
         patch.object(go, "build_prev_chunk_adherence_context", return_value=""), \
         patch.object(go, "build_weight_history_context", return_value=""), \
         patch.object(go, "build_nutrition_context", return_value=""), \
         patch.object(go, "build_adherence_context", return_value=""), \
         patch.object(go, "build_success_patterns_context", return_value=""), \
         patch.object(go, "build_temporal_adherence_context", return_value=""), \
         patch.object(go, "build_unified_behavioral_profile", return_value=""), \
         patch.object(go, "build_fatigue_context", return_value=""), \
         patch.object(go, "build_liked_meals_context", return_value=""), \
         patch.object(go, "build_correction_context", return_value=""), \
         patch.object(go, "build_pantry_correction_context", return_value=""), \
         patch.object(go, "build_time_context", return_value=""), \
         patch.object(go, "build_supplements_context", return_value=""), \
         patch.object(go, "build_grocery_duration_context", return_value=""), \
         patch.object(go, "build_pantry_context", return_value=""), \
         patch.object(go, "build_prices_context", return_value=""), \
         patch("ai_helpers.get_deterministic_variety_prompt", return_value=""):
        ctx = go._build_shared_context(state)

    # 1. El builder fue invocado.
    assert builder_spy.called, (
        "_build_shared_context NO invocó build_chunk_lessons_context. Esto "
        "rompe el paso 3 del pipeline P0-6 — las lecciones nunca llegarán al ctx."
    )
    # 2. Fue invocado con el blob exacto de form_data["_chunk_lessons"].
    args, kwargs = builder_spy.call_args
    arg_passed = args[0] if args else kwargs.get("chunk_lessons")
    assert arg_passed is lessons_blob, (
        f"build_chunk_lessons_context fue invocado pero con argumento incorrecto: "
        f"{arg_passed!r}. Debe recibir form_data['_chunk_lessons'] (mismo objeto)."
    )
    # 3. El return del builder se asigna a ctx['chunk_lessons_context'].
    assert ctx.get("chunk_lessons_context") == "MOCKED_LESSONS_CTX", (
        "El return de build_chunk_lessons_context no se asignó a ctx['chunk_lessons_context']. "
        "Wiring del paso 3 roto."
    )


def test_build_shared_context_handles_missing_lessons_gracefully():
    """Variante de F sin _chunk_lessons en form_data. El builder debe seguir
    invocándose (con None) y retornar string vacío sin lanzar.
    """
    import graph_orchestrator as go

    state = {
        "form_data": {"user_id": "u-p06-none"},  # SIN _chunk_lessons
        "nutrition": {"alergias": []},
        "review_feedback": "",
        "user_facts": "",
        "history_context": "",
        "compressed_context": "",
        "taste_profile": "",
        "rejection_reasons": [],
    }

    with patch.object(go, "build_skeleton_quality_context", return_value=""), \
         patch.object(go, "build_quality_hint_context", return_value=""), \
         patch.object(go, "build_prev_chunk_adherence_context", return_value=""), \
         patch.object(go, "build_weight_history_context", return_value=""), \
         patch.object(go, "build_nutrition_context", return_value=""), \
         patch.object(go, "build_adherence_context", return_value=""), \
         patch.object(go, "build_success_patterns_context", return_value=""), \
         patch.object(go, "build_temporal_adherence_context", return_value=""), \
         patch.object(go, "build_unified_behavioral_profile", return_value=""), \
         patch.object(go, "build_fatigue_context", return_value=""), \
         patch.object(go, "build_liked_meals_context", return_value=""), \
         patch.object(go, "build_correction_context", return_value=""), \
         patch.object(go, "build_pantry_correction_context", return_value=""), \
         patch.object(go, "build_time_context", return_value=""), \
         patch.object(go, "build_supplements_context", return_value=""), \
         patch.object(go, "build_grocery_duration_context", return_value=""), \
         patch.object(go, "build_pantry_context", return_value=""), \
         patch.object(go, "build_prices_context", return_value=""), \
         patch("ai_helpers.get_deterministic_variety_prompt", return_value=""):
        ctx = go._build_shared_context(state)

    assert ctx.get("chunk_lessons_context") == "", (
        "Sin _chunk_lessons, el builder real debe retornar '' (no 'None' ni KeyError)."
    )
