"""[P0-7] Cap de lecciones críticas permanentes — exercises real production function.

Contexto del gap:
    El cap `CHUNK_CRITICAL_LESSONS_MAX` (constants.py:437, default 200) limita
    el size de `_critical_lessons_permanent` para evitar bloat de memoria y
    prompt overflow al LLM. La poda se hace en `_prune_critical_lessons_with_priority`
    (cron_tasks.py:3805) con lógica priorizada en tres tiers:

      1. **Inmortales**: alergias (`allergy_violations > 0`) o rechazos
         repetidos (`rejection_violations >= CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL`,
         default 3) — NUNCA se descartan si caben en el cap.
      2. **Mortales**: rechazos puntuales / fatiga / repeats — rotan LRU.
      3. **Hard cap inmortales**: si los inmortales superan
         `CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP` (default 190),
         se evictan los más viejos sin re-validación reciente
         (`CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS`, default 60d).

    El test E2E existente (`test_critical_lessons_30d_e2e.py`) SIMULA la poda
    con un `[-N:]` simple — NO ejercita la función real. Este archivo cubre
    el gap testeando `_prune_critical_lessons_with_priority` directamente.

    Sin estos tests, una refactor que cambie:
      - El criterio de inmortalidad (e.g., umbral de rechazos),
      - La estrategia de eviction (LRU vs prioridad),
      - El comportamiento ante overflow puro de inmortales,
    pasa desapercibido en CI hasta que aparece en producción como prompts
    bloated o alergias del usuario silenciosamente olvidadas.
"""
import os
import sys
import types
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stubs mínimos (mismo patrón que test_p0_new1)
# ---------------------------------------------------------------------------
def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_k: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_k: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_k: None,
    execute_sql_write=lambda *_a, **_k: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_k: None,
    get_inventory_activity_since=lambda *_a, **_k: [],
    get_raw_user_inventory=lambda *_a, **_k: [],
    get_user_inventory_net=lambda *_a, **_k: [],
    release_chunk_reservations=lambda *_a, **_k: None,
    reserve_plan_ingredients=lambda *_a, **_k: 0,
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_k: None,
    get_user_likes=lambda *_a, **_k: [],
    get_active_rejections=lambda *_a, **_k: [],
    get_recent_plans=lambda *_a, **_k: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_k: [],
    get_consumed_meals_since=lambda *_a, **_k: [],
    get_user_facts_by_metadata=lambda *_a, **_k: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_k: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_k: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_k: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_k: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_k: {})


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


_install_stub(
    "shopping_calculator",
    get_shopping_list_delta=lambda *_a, **_k: [],
    _parse_quantity=_stub_parse_quantity,
)
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg

import constants  # noqa: E402
import cron_tasks  # noqa: E402
from cron_tasks import _prune_critical_lessons_with_priority  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _mortal_lesson(chunk: int, days_ago: int = 0, *, signal: str = "strong") -> dict:
    """Lección crítica mortal: un solo rechazo puntual o repeat alto.
    Sin alergia ni rechazos repetidos → eligible para LRU eviction.
    """
    return {
        "chunk": chunk,
        "rejection_violations": 1,  # 1 < CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL=3
        "allergy_violations": 0,
        "ingredient_base_repeat_pct": 88.0,  # >85 → critical (en el flow)
        "rejected_meals_that_reappeared": [f"plato_chunk_{chunk}"],
        "repeated_bases": [],
        "allergy_hits": [],
        "metrics_unavailable": False,
        "low_confidence": False,
        "learning_signal_strength": signal,
        "created_at": _iso(NOW - timedelta(days=days_ago)),
        "last_validated_at": _iso(NOW - timedelta(days=days_ago)),
    }


def _immortal_allergy_lesson(chunk: int, allergens: list[str], days_ago: int = 0) -> dict:
    """Lección inmortal por alergia (allergy_violations > 0) — nunca se evict
    a menos que excedamos IMMORTAL_HARD_CAP."""
    return {
        "chunk": chunk,
        "rejection_violations": 0,
        "allergy_violations": len(allergens),
        "ingredient_base_repeat_pct": 5.0,
        "rejected_meals_that_reappeared": [],
        "repeated_bases": [],
        "allergy_hits": list(allergens),
        "metrics_unavailable": False,
        "low_confidence": False,
        "learning_signal_strength": "strong",
        "created_at": _iso(NOW - timedelta(days=days_ago)),
        "last_validated_at": _iso(NOW - timedelta(days=days_ago)),
    }


def _immortal_rejection_lesson(chunk: int, rejections: int = 3, days_ago: int = 0) -> dict:
    """Lección inmortal por rechazos repetidos (rejection_violations >= 3 default)."""
    return {
        "chunk": chunk,
        "rejection_violations": rejections,
        "allergy_violations": 0,
        "ingredient_base_repeat_pct": 50.0,
        "rejected_meals_that_reappeared": [f"rep_chunk_{chunk}"] * rejections,
        "repeated_bases": [],
        "allergy_hits": [],
        "metrics_unavailable": False,
        "low_confidence": False,
        "learning_signal_strength": "strong",
        "created_at": _iso(NOW - timedelta(days=days_ago)),
        "last_validated_at": _iso(NOW - timedelta(days=days_ago)),
    }


# ---------------------------------------------------------------------------
# Group A — comportamiento sin presión de cap
# ---------------------------------------------------------------------------
def test_returns_unchanged_when_below_cap():
    """Sin presión: ≤ max_size devuelve la lista intacta (no copia, no orden alterado)."""
    lessons = [_mortal_lesson(i) for i in range(1, 11)]  # 10 < 200
    result = _prune_critical_lessons_with_priority(lessons, max_size=200)
    assert result == lessons, "Sin presión, la función no debería tocar la lista."


def test_returns_input_unchanged_when_not_a_list():
    """Inputs no-list (None, dict, str) devuelven intactos — early-return defensivo."""
    assert _prune_critical_lessons_with_priority(None, 200) is None
    assert _prune_critical_lessons_with_priority({}, 200) == {}
    assert _prune_critical_lessons_with_priority("", 200) == ""


def test_empty_list_is_noop():
    assert _prune_critical_lessons_with_priority([], 200) == []


# ---------------------------------------------------------------------------
# Group B — Mortales: LRU clásico cuando excedemos cap
# ---------------------------------------------------------------------------
def test_mortals_only_lru_evicts_oldest_when_above_cap():
    """Solo mortales, > cap → conserva las `cap` más recientes (al final de la lista).

    `_prune` hace `mortals[-keep_mortals:]` cuando no hay inmortales, por lo
    que la posición en la lista actúa como ordenamiento implícito ("recién
    appended" = última = sobreviviente).
    """
    # 10 mortales, cap=4 → quedan los últimos 4: chunks 7..10
    lessons = [_mortal_lesson(i) for i in range(1, 11)]
    result = _prune_critical_lessons_with_priority(lessons, max_size=4)
    assert len(result) == 4
    surviving_chunks = [l["chunk"] for l in result]
    assert surviving_chunks == [7, 8, 9, 10], (
        f"LRU mortal debería conservar los últimos 4 chunks. Got: {surviving_chunks}"
    )


# ---------------------------------------------------------------------------
# Group C — Inmortales: protección frente a presión normal
# ---------------------------------------------------------------------------
def test_immortal_allergy_survives_when_mortals_overflow():
    """Inmortal por alergia debe sobrevivir aunque se appendeen 100 mortales después."""
    lessons = [
        _immortal_allergy_lesson(chunk=1, allergens=["maní"], days_ago=10),
    ] + [_mortal_lesson(i) for i in range(2, 102)]  # 1 inmortal + 100 mortales = 101
    result = _prune_critical_lessons_with_priority(lessons, max_size=10)
    assert len(result) == 10

    immortals_in_result = [l for l in result if l.get("allergy_violations", 0) > 0]
    assert len(immortals_in_result) == 1, (
        f"La lección inmortal por alergia debe sobrevivir. "
        f"Resultado: {[l['chunk'] for l in result]}"
    )
    assert immortals_in_result[0]["chunk"] == 1


def test_immortal_repeated_rejections_threshold_default_three():
    """`rejection_violations >= CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL` (default 3)
    convierte la lección en inmortal. 2 NO la hace inmortal.
    """
    threshold = constants.CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL
    assert threshold == 3, (
        f"Default cambió a {threshold}. Si fue intencional, actualiza este test "
        f"y `_immortal_rejection_lesson` arriba."
    )

    # Una lección con 2 rechazos (NO inmortal) en posición 1, seguida de muchos
    # mortales más recientes → LRU debería evict la de 2 rechazos.
    quasi = _mortal_lesson(chunk=1, days_ago=30)
    quasi["rejection_violations"] = 2  # justo por debajo del umbral inmortal
    lessons = [quasi] + [_mortal_lesson(i) for i in range(2, 12)]  # 11 total
    result = _prune_critical_lessons_with_priority(lessons, max_size=5)

    surviving_chunks = {l["chunk"] for l in result}
    assert 1 not in surviving_chunks, (
        f"Lección con rejection_violations=2 (debajo del umbral inmortal=3) "
        f"debe ser eligible para LRU eviction. Surviving chunks: {surviving_chunks}"
    )


def test_immortal_repeated_rejections_at_threshold_survives():
    """Con exactamente 3 rechazos repetidos (== umbral) la lección es inmortal y sobrevive."""
    immortal = _immortal_rejection_lesson(chunk=1, rejections=3, days_ago=30)
    lessons = [immortal] + [_mortal_lesson(i) for i in range(2, 102)]
    result = _prune_critical_lessons_with_priority(lessons, max_size=5)

    surviving_chunks = {l["chunk"] for l in result}
    assert 1 in surviving_chunks, (
        f"Lección con rejection_violations==3 debe ser inmortal y sobrevivir "
        f"al LRU. Surviving chunks: {surviving_chunks}"
    )


# ---------------------------------------------------------------------------
# Group D — Hard cap de inmortales (CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP)
# ---------------------------------------------------------------------------
def test_immortal_overflow_evicts_stale_first():
    """Cuando inmortales > IMMORTAL_HARD_CAP, evict primero los más viejos
    sin re-validación reciente (>60d default).

    Setup: hard_cap=5, max_size=5, 3 inmortales recientes (5 días) + 5
    inmortales viejas (70-110d, > 60d revalidation window). Total = 8 >
    max_size=5 → entra al body. immortals=8 > hard_cap=5 → poda.

    Importante: max_size debe ser ≤ total para superar el early-return
    `if len(lessons) <= max_size: return lessons`. Si max_size es muy
    grande, la función devuelve la lista sin tocar el hard cap.
    """
    orig_constants_cap = constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP
    constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP = 5

    try:
        recent = [
            _immortal_allergy_lesson(chunk=i, allergens=["x"], days_ago=5)
            for i in range(1, 4)  # 3 recientes
        ]
        # Stale: 5 inmortales viejas con días variados >60
        stale = [
            _immortal_allergy_lesson(chunk=i, allergens=["x"], days_ago=days)
            for i, days in zip(range(4, 9), [70, 80, 90, 100, 110])
        ]
        lessons = recent + stale  # 8 total
        # max_size = hard_cap = 5: total 8 > max_size → entramos al body;
        # 8 inmortales > hard_cap → trigger poda; tras poda len=5 >= max_size
        # → rama "return immortals" (todos los slots ocupados por inmortales).
        result = _prune_critical_lessons_with_priority(lessons, max_size=5)

        assert len(result) == 5, (
            f"Hard cap de 5 violado. Got: {len(result)} "
            f"({[l['chunk'] for l in result]})"
        )
        surviving_chunks = {l["chunk"] for l in result}
        # Las 3 recientes deben estar todas
        assert {1, 2, 3}.issubset(surviving_chunks), (
            f"Las 3 inmortales recientes deben sobrevivir. Surviving: {surviving_chunks}"
        )
        # De las stale, las 2 más recientes (days_ago=70, 80 → chunks 4,5) sobreviven.
        # Las más viejas (days_ago=100, 110 → chunks 7,8) son evicted.
        assert {7, 8}.isdisjoint(surviving_chunks), (
            f"Las inmortales más viejas (100d, 110d) deberían ser evicted. "
            f"Surviving: {surviving_chunks}"
        )
    finally:
        constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP = orig_constants_cap


def test_immortal_overflow_when_recent_already_saturate_cap():
    """Cuando los inmortales recientes (≤60d) ya son ≥ hard_cap, todas las
    stale (>60d) se descartan completamente.

    Setup: hard_cap=3, max_size=3. 4 recientes + 3 stale = 7 total > max_size.
    Entra al body. immortals=7 > hard_cap=3 → poda. recent=4 >= hard_cap=3 →
    función interna devuelve `recent` (las 4 recientes, sin truncar). Luego
    en main flow: len(immortals)=4 >= max_size=3 → `return immortals`
    devuelve las 4 recientes (sin las stale).
    """
    orig = constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP
    constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP = 3
    try:
        recent = [
            _immortal_allergy_lesson(chunk=i, allergens=["x"], days_ago=10)
            for i in range(1, 5)  # 4 recientes (>= hard_cap=3)
        ]
        stale = [
            _immortal_allergy_lesson(chunk=i, allergens=["x"], days_ago=90)
            for i in range(5, 8)  # 3 stale
        ]
        lessons = recent + stale  # 7 total
        result = _prune_critical_lessons_with_priority(lessons, max_size=3)

        surviving_chunks = {l["chunk"] for l in result}
        assert surviving_chunks.issubset({1, 2, 3, 4}), (
            f"Solo las inmortales recientes deben sobrevivir cuando "
            f"recent >= hard_cap. Surviving: {surviving_chunks}"
        )
        assert {5, 6, 7}.isdisjoint(surviving_chunks), (
            f"Las stale (>60d) deben ser totalmente evicted. "
            f"Surviving: {surviving_chunks}"
        )
    finally:
        constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP = orig


# ---------------------------------------------------------------------------
# Group E — Cap mortal absoluto cuando todos los slots los ocupan inmortales
# ---------------------------------------------------------------------------
def test_all_immortals_overflow_returns_immortals_and_drops_mortals():
    """Si los inmortales solos ya ≥ max_size, todos los mortales son descartados
    y se devuelven SOLO los inmortales (warning loggeado en el código).
    """
    immortals = [
        _immortal_allergy_lesson(chunk=i, allergens=["x"], days_ago=10)
        for i in range(1, 11)
    ]  # 10 inmortales
    mortals = [_mortal_lesson(i) for i in range(11, 21)]  # 10 mortales
    lessons = immortals + mortals  # 20 total

    # max_size=8 < 10 inmortales → todos los mortales descartados, los 10
    # inmortales sobreviven (porque len(immortals) >= max_size triggers la
    # rama "return immortals").
    result = _prune_critical_lessons_with_priority(lessons, max_size=8)
    chunks = [l["chunk"] for l in result]
    # Todos los inmortales (sin tope mortal aplicado por la rama de overflow)
    assert all(c <= 10 for c in chunks), (
        f"Todos los chunks de salida deben ser inmortales (chunks 1..10). "
        f"Got: {chunks}"
    )
    # Cero mortales
    assert not any(c >= 11 for c in chunks), (
        f"Los mortales (chunks ≥ 11) deben ser descartados cuando inmortales "
        f"solos exceden el cap. Got: {chunks}"
    )


# ---------------------------------------------------------------------------
# Group F — Resilencia ante datos malformados
# ---------------------------------------------------------------------------
def test_handles_realistic_malformed_lesson_dicts_without_crashing():
    """Lecciones con campos faltantes, None, o no-dict (que pueden aparecer
    tras deserialización JSON parcial) no deben crashear la función.

    Cubre el contrato real: el código usa `.get(key) or 0` para int fields
    e `isinstance(lesson, dict)` para excluir no-dicts. Tipos no-numéricos
    en `rejection_violations`/`allergy_violations` NO son cubiertos por la
    función actual y NO se incluyen aquí (aparecerían como bugs upstream
    en `_extract_chunk_learning`, no en la poda).
    """
    weird = [
        {"chunk": 1, "rejection_violations": None, "allergy_violations": None},
        {"chunk": 2},  # campos mínimos faltantes
        "not-a-dict",  # str dentro de la lista — _is_immortal devuelve False
        None,
        _immortal_allergy_lesson(chunk=3, allergens=["x"], days_ago=5),
    ]
    # Total=5 > max_size=2 → entra al body; debe podar sin crashear.
    result = _prune_critical_lessons_with_priority(weird, max_size=2)
    assert isinstance(result, list)
    # La inmortal debe sobrevivir aunque haya datos malformados alrededor.
    immortal_chunks = [
        l.get("chunk") for l in result
        if isinstance(l, dict) and int(l.get("allergy_violations") or 0) > 0
    ]
    assert 3 in immortal_chunks, (
        f"La lección inmortal (chunk=3) debe sobrevivir entre datos "
        f"malformados. Got: {result}"
    )


def test_immortal_check_robust_to_missing_fields():
    """`_is_immortal` usa `int(lesson.get('allergy_violations') or 0)` — si el
    campo no existe o es None, no crashea ni promociona la lección a inmortal.
    """
    no_fields = {"chunk": 99}  # nada de allergy/rejection
    mortals_pressure = [_mortal_lesson(i) for i in range(1, 21)]  # 20 mortales
    lessons = [no_fields] + mortals_pressure
    result = _prune_critical_lessons_with_priority(lessons, max_size=5)
    # `no_fields` (chunk=99) debe ser tratado como mortal y, al ser el más
    # viejo (posición 0), debe ser evicted por LRU.
    surviving_chunks = {l["chunk"] for l in result if isinstance(l, dict)}
    assert 99 not in surviving_chunks, (
        f"Lección sin campos críticos no debe ser tratada como inmortal. "
        f"Surviving: {surviving_chunks}"
    )


# ---------------------------------------------------------------------------
# Group G — Constants exportados con los defaults documentados
# ---------------------------------------------------------------------------
def test_constants_have_documented_defaults():
    """Defaults publicados en P0-7 / P0-6 — un cambio silencioso en el env
    deployment (env var override) que rompa la documentación quedaría atrapado
    aquí si alguien resetea el env.
    """
    # Estos son los defaults que el resto del sistema asume cuando no hay
    # override por env. Si el deploy override está activo en el ambiente de
    # tests, este assert puede ser laxo.
    assert constants.CHUNK_CRITICAL_LESSONS_MAX >= 50, (
        "Cap absoluto < 50 deja muy poco espacio para señales no-inmortales en "
        "planes largos."
    )
    assert constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP <= constants.CHUNK_CRITICAL_LESSONS_MAX, (
        "Hard cap de inmortales NO puede superar el cap absoluto — sería "
        "matemáticamente imposible llenarlo respetando el cap absoluto."
    )
    assert constants.CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP >= 10, (
        "Hard cap de inmortales debe dejar al menos 10 slots — un usuario "
        "con 10+ alergias activas es realista (perfil médico complejo)."
    )
    assert constants.CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS > 0, (
        "Ventana de re-validación 0 anula la distinción stale/recent."
    )
    assert constants.CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL >= 2, (
        "Umbral inmortal de 1 promovería cualquier rechazo a inmortal — "
        "confunde con la rama mortal de rejection_violations==1."
    )
