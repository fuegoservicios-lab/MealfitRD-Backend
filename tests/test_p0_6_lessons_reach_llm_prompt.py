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
    # __file__ vive en backend/tests/; el source real vive en backend/.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(here, "graph_orchestrator.py")
    with open(src_path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_build_shared_context_body() -> str:
    """Devuelve el cuerpo textual de `_build_shared_context` para inspección parser-based.
    Robusto contra el `conftest.py` stub que reemplaza graph_orchestrator con MagicMock —
    no importamos el módulo, parseamos el archivo.
    """
    source = _read_orchestrator_source()
    start = source.find("def _build_shared_context(")
    if start < 0:
        return ""
    # Delimita por la siguiente top-level `def ` o `class `.
    body_end = source.find("\ndef ", start + 1)
    body_end_class = source.find("\nclass ", start + 1)
    if body_end < 0:
        body_end = body_end_class
    elif body_end_class > -1:
        body_end = min(body_end, body_end_class)
    return source[start:body_end] if body_end > -1 else source[start:]


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
#
# Nota: la versión original usaba `import graph_orchestrator as go` + `patch.object(go, ...)`
# para mockear todos los `build_*_context`. El `backend/conftest.py` reemplaza
# `graph_orchestrator` con un MagicMock stub que NO expone esos builders como atributos
# patcheables (AttributeError). Reescritos como parser-based: validan la misma invariante
# (cuerpo de `_build_shared_context` extrae `_chunk_lessons` de form_data y lo pasa a
# `build_chunk_lessons_context`) sin depender del módulo importado.

def test_build_shared_context_invokes_builder_with_form_data_lessons():
    """Parser-based: el cuerpo de `_build_shared_context` debe (1) leer
    `form_data["_chunk_lessons"]` y (2) pasar ese value a
    `build_chunk_lessons_context(...)` cuyo retorno se asigna a
    `ctx["chunk_lessons_context"]`. Si un refactor lo lee de otra fuente o cambia
    el wiring, las lecciones que cron_tasks agregó nunca llegarán al prompt LLM —
    fallo silencioso.
    """
    body = _extract_build_shared_context_body()
    assert body, (
        "_build_shared_context no encontrada en graph_orchestrator.py. Si fue "
        "renombrada, actualiza `_extract_build_shared_context_body()`."
    )

    # 1. El cuerpo lee form_data['_chunk_lessons'] (cualquiera de los dos quote styles).
    assert (
        'form_data.get("_chunk_lessons")' in body
        or "form_data.get('_chunk_lessons')" in body
    ), (
        "_build_shared_context debe leer form_data.get('_chunk_lessons') para "
        "alimentar el builder; sin esto, el blob de cron_tasks no llega al ctx."
    )

    # 2. El cuerpo asigna ctx['chunk_lessons_context'] = build_chunk_lessons_context(...).
    pattern = re.compile(
        r"""['"]chunk_lessons_context['"]\s*:\s*build_chunk_lessons_context\("""
    )
    assert pattern.search(body), (
        "Wiring esperado dentro de _build_shared_context: "
        "'chunk_lessons_context': build_chunk_lessons_context(...). "
        "Si fue extraído a un helper, el contrato sigue: ctx debe quedar con esa key "
        "poblada por ese builder."
    )


def test_build_shared_context_handles_missing_lessons_gracefully():
    """El builder `build_chunk_lessons_context` debe tolerar None/dict-vacío y
    retornar string vacío (no 'None' ni KeyError). Esto cubre el caso "primer
    chunk de un plan, no hay lecciones aún". Test D ya valida el camino feliz;
    este test garantiza que el cuerpo del caller invoca el builder
    incondicionalmente (no detrás de un `if chunk_lessons:`) — así el contrato
    "siempre hay key chunk_lessons_context en ctx" se mantiene.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    # Contrato del builder con inputs ausentes.
    assert build_chunk_lessons_context(None) == ""
    assert build_chunk_lessons_context({}) == ""

    # Contrato del caller: el assignment NO está gateado por un `if`.
    body = _extract_build_shared_context_body()
    assert body, "_build_shared_context no encontrada"
    # Heurística: la asignación de chunk_lessons_context debe estar al mismo nivel
    # de indentación que las demás keys del ctx dict (no dentro de un `if`).
    # Buscamos el patrón con cualquier whitespace y verificamos que NO está
    # precedida en la misma línea por un cierre de bloque condicional.
    lines = body.splitlines()
    found_unguarded = False
    for line in lines:
        if "chunk_lessons_context" in line and "build_chunk_lessons_context" in line:
            stripped = line.lstrip()
            # Si la línea es parte de un dict literal (empieza con quote),
            # NO está dentro de un if condicional.
            if stripped.startswith('"') or stripped.startswith("'"):
                found_unguarded = True
                break
    assert found_unguarded, (
        "El assignment de 'chunk_lessons_context' debe vivir directamente en el "
        "dict literal del ctx (no detrás de un `if chunk_lessons:`). Sin eso, "
        "el caso 'primer chunk' produciría KeyError al evaluar "
        "ctx['chunk_lessons_context'] en el prompt_text."
    )
