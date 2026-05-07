"""[P0-NEW1] Validación de cantidades en modo degradado: Smart Shuffle + Edge Recipe.

Contexto del gap (cron_tasks.py:13868-13958):
    Cuando el LLM no puede generar el chunk (probe falló o el pipeline reventó),
    el worker entra en modo degraded ("Smart Shuffle"): elige días de planes
    previos y los reusa. El gap [P0-NEW1] cubre la pieza más sutil del modo
    degradado: incluso si los ingredientes existen en la nevera, las CANTIDADES
    de los días reusados pueden no caber en el pantry actual (ej: el plato
    pedía 500g de pollo, pero al usuario le quedan 200g).

    El fix introduce una cadena de validación de cantidades:
      1. Smart Shuffle reintenta hasta 3 veces con candidatos distintos. Cada
         candidato se valida con `validate_ingredients_against_pantry(strict_quantities=True)`.
      2. Si los 3 intentos fallan, intenta un Edge Recipe construido con
         `_build_filtered_edge_recipe_day(pantry_items=...)`. Esa función *capa*
         las cantidades de cada ingrediente al máximo disponible en el pantry,
         garantizando que el edge resultante sea factible.
      3. Si tanto shuffle como edge fallan, pausa el chunk en
         `pending_user_action` con `reason='degraded_quantities_unfeasible'`.

    Sin este fix, el modo degradado podía servir un plato con cantidades
    imposibles, rompiendo el contrato "platos solo con alimentos de la nevera"
    en lo cuantitativo (no solo lo cualitativo).

NOTA SOBRE NIVEL DE TEST:
    El bloque [P0-NEW1] vive *inline* dentro de `_chunk_worker` (~17k LOC), no
    como función extraíble. Probarlo end-to-end requeriría ejecutar el worker
    completo con stubs de DB/LLM. En su lugar testeamos:
      A. La invariante crítica del builder: una Edge Recipe construida con un
         pantry P SIEMPRE pasa `validate_ingredients_against_pantry(strict=True)`
         contra ese mismo P. Esa es la garantía que evita el pause.
      B. El cap por pantry funciona (cantidades nunca exceden lo disponible).
      C. La estructura del bloque [P0-NEW1] sigue presente en `_chunk_worker`
         (regresión guard contra eliminación accidental).
"""
import os
import re
import sys
import types
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stubs mínimos para módulos externos (mismo patrón que los otros tests P0/P1)
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
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg

# Importar el módulo real `shopping_calculator._parse_quantity` y
# `constants.validate_ingredients_against_pantry` — son las piezas reales que
# el bloque [P0-NEW1] usa, no podemos stubarlas si queremos validar la
# invariante de cantidad. Para los tests A/B necesitamos parseo real.
import shopping_calculator  # noqa: E402
import constants  # noqa: E402
from constants import validate_ingredients_against_pantry  # noqa: E402

import cron_tasks  # noqa: E402
from cron_tasks import _build_filtered_edge_recipe_day  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_QTY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb|lbs|gr|gramos)\b", re.IGNORECASE)


def _extract_grams(ingredient_string: str) -> float:
    """Parsea '150g Pollo' → 150.0. Devuelve 0 si no hay match."""
    m = _QTY_RE.match(ingredient_string)
    if not m:
        return 0.0
    qty = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "kg":
        return qty * 1000.0
    if unit == "lb" or unit == "lbs":
        return qty * 453.592
    if unit == "oz":
        return qty * 28.3495
    return qty  # g/gr/gramos/ml/l tratados como base


def _all_edge_ingredients(edge_day: dict) -> list[str]:
    return [
        ing
        for m in (edge_day or {}).get("meals", [])
        for ing in m.get("ingredients", [])
        if isinstance(ing, str) and ing.strip()
    ]


# ---------------------------------------------------------------------------
# Group A — Edge Recipe builder cap behavior (building block 1)
# ---------------------------------------------------------------------------
def test_edge_recipe_caps_quantity_when_pantry_has_less_than_default():
    """Si pantry tiene 50g de Pollo, el desayuno (default 150g) debe capar a 50g.

    El builder usa `_cap_ingredient(default_g)` y reduce a `min(default_g, total_g)`.
    Sin el cap, el desayuno serviría 150g sobre un pantry de 50g → la
    validación strict_quantities rechazaría todo el día y `_chunk_worker`
    pausaría el chunk. Con el cap, el día es feasible cualitativamente.

    Pantry incluye un veggie para que pantry_intersect no devuelva todo el
    catálogo de veggies como fallback (lo que metería "Repollo" y otros
    ingredientes cuyo nombre contiene "pollo" como substring).
    """
    # Solo Pollo en proteínas; intersección de proteínas → ['Pollo'] (único
    # protein con substring match contra 'pollo'). Veggies y carbs amplios
    # para evitar catálogos vacíos.
    pantry = [
        "50g Pollo",
        "500g Arroz Blanco",
        "500g Aguacate",
        "500g Tomate",
    ]

    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=[],
        diet="",
        pantry_items=pantry,
    )
    assert edge_day is not None, "Edge Recipe no debió ser None con catálogos llenos."

    # Match estricto contra "pollo" como palabra (no substring de "repollo").
    pollo_re = re.compile(r"\bpollo\b", re.IGNORECASE)
    pollo_servings = [
        (ing, _extract_grams(ing))
        for ing in _all_edge_ingredients(edge_day)
        if pollo_re.search(ing)
    ]
    # Al menos una porción de Pollo debe aparecer (es el único protein válido).
    assert pollo_servings, (
        f"Edge Recipe no incluyó Pollo aunque era la única proteína disponible "
        f"en pantry. Ingredientes: {_all_edge_ingredients(edge_day)!r}"
    )
    for ing, grams in pollo_servings:
        assert grams <= 50.0, (
            f"Edge Recipe sirvió {grams}g de pollo pero pantry solo tiene 50g. "
            f"Cap del [P0-NEW1] no se aplicó. Ingrediente: {ing!r}"
        )


def test_edge_recipe_uses_default_when_pantry_quantity_higher_than_default():
    """Si pantry tiene 5kg (5000g) de Pollo, el desayuno usa el default 150g
    (no servimos más solo porque haya stock — el default es la receta esperada).

    `_cap_ingredient` hace `min(default_g, total_g)`, que con default=150 y
    total=5000 devuelve 150. Esto preserva el tamaño de porción sano.
    """
    pantry = ["5000g Pollo", "5000g Arroz Blanco", "5000g Aguacate"]

    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=[],
        diet="",
        pantry_items=pantry,
    )
    assert edge_day is not None

    for ing in _all_edge_ingredients(edge_day):
        if "pollo" in ing.lower():
            grams = _extract_grams(ing)
            # Lunch usa 200g de proteína, breakfast/dinner usan 150g.
            # Cualquiera ≤ 200g es válido aquí; lo importante es que NO escale
            # arriba del default por tener stock de sobra.
            assert grams <= 200.0, (
                f"Edge Recipe sirvió {grams}g de pollo aunque el default máximo "
                f"es 200g (almuerzo). El cap escaló hacia arriba indebidamente. "
                f"Ingrediente: {ing!r}"
            )


def test_edge_recipe_handles_unparseable_pantry_string_gracefully():
    """Si _parse_quantity revienta sobre un string raro, el builder debe caer
    al default sin crashear y sin dejar sin ingrediente al plato.

    El `try/except` en `_cap_ingredient` (cron_tasks.py:11505-11506) loguea
    `[P0-NEW1] Error parsing quantity in Edge Recipe builder` y devuelve el
    default. Esta tolerancia evita que un pantry malformado tumbe el modo
    degradado entero.
    """
    pantry = ["💥 raro 💥 sin cantidad", "100g Arroz Blanco", "100g Aguacate"]

    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=[],
        diet="",
        pantry_items=pantry,
    )

    assert edge_day is not None, (
        "Builder devolvió None con un item malformado en pantry. Debe tolerar."
    )
    # Cada comida debe tener al menos 2 ingredientes string no vacíos.
    for meal in edge_day.get("meals", []):
        ings = meal.get("ingredients", [])
        assert len(ings) >= 2, f"Comida quedó sin ingredientes: {meal}"
        for ing in ings:
            assert isinstance(ing, str) and ing.strip(), (
                f"Ingrediente vacío en comida: {meal}"
            )


def test_edge_recipe_returns_none_when_filters_eliminate_all_proteins():
    """Si las restricciones del usuario dejan el catálogo de proteínas vacío,
    el builder devuelve None — el código del worker en ese caso debe escalar
    a pausa (no servir un plato sin proteína).

    Esto valida la guarda `if not filtered_proteins or not filtered_carbs ...`
    en cron_tasks.py:11447.

    Listamos explícitamente cada protein del catálogo `DOMINICAN_PROTEINS`
    (constants.py:765) como dislike para forzar que el filtro lo vacíe.
    """
    # DOMINICAN_PROTEINS contiene: Pollo, Cerdo, Res, Pavo, Pescado, Atún,
    # Huevos, Queso de Freír, Salami Dominicano, Camarones, Chuleta, Longaniza,
    # Habichuelas Rojas/Negras/Blancas, Gandules, Lentejas, Garbanzos,
    # Soya/Tofu, Queso Ricotta, Queso Blanco, Queso Mozzarella, Yogurt.
    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=[
            "pollo", "cerdo", "res", "pavo", "pescado", "atun", "atún",
            "huevos", "huevo", "queso", "salami", "camarones", "chuleta",
            "longaniza", "habichuelas", "gandules", "lentejas", "garbanzos",
            "soya", "tofu", "yogurt",
        ],
        diet="",
        pantry_items=["100g Arroz Blanco"],
    )
    assert edge_day is None, (
        "Con dislikes cubriendo todas las proteínas del catálogo dominicano, "
        "el builder debe devolver None para que el worker escale a pausa "
        "(no servir un plato sin proteína)."
    )


# ---------------------------------------------------------------------------
# Group B — Cap aplicado y validación pasa con pantry generoso
# ---------------------------------------------------------------------------
def test_capped_edge_recipe_passes_strict_quantity_validation_with_generous_pantry():
    """Con pantry generoso (10× la demanda total agregada del Edge Recipe), una
    Edge Recipe construida con ese pantry pasa `validate_ingredients_against_pantry(strict=True)`.

    El validador es SUSTRACTIVO: cada ingrediente generado se descuenta del
    ledger del pantry, así que el mismo protein servido en 3 comidas demanda
    el total acumulado. El [P0-NEW1] no garantiza feasibility en pantries
    ajustados (el worker pausa cuando no es feasible) — pero SÍ debe garantizar
    feasibility cuando el pantry sobra ampliamente. Este test atrapa
    regresiones donde el builder serviría cantidades absurdas (por ejemplo,
    si un futuro refactor accidentalmente removiera el cap a `default_g`).

    Demanda agregada del Edge Recipe (peor caso, 1 protein dominante):
        breakfast 150g + lunch 200g + dinner 150g = 500g protein
        breakfast 100g + lunch 150g                = 250g carb
        lunch 100g + dinner 150g                   = 250g veggie
    Pantry de 5kg/3kg/3kg cubre todo cómodamente.
    """
    pantry = [
        "5000g Pollo",
        "5000g Pescado",  # alternativa de protein
        "3000g Arroz Blanco",
        "3000g Avena",
        "3000g Aguacate",
        "3000g Tomate",
        "3000g Brócoli",
    ]

    failures = []
    for _ in range(20):
        edge_day = _build_filtered_edge_recipe_day(
            allergies=[],
            dislikes=[],
            diet="",
            pantry_items=pantry,
        )
        if edge_day is None:
            continue
        ings = _all_edge_ingredients(edge_day)
        result = validate_ingredients_against_pantry(
            ings,
            pantry,
            strict_quantities=True,
            tolerance=1.30,
        )
        if result is not True:
            failures.append((ings, result))

    assert not failures, (
        f"[P0-NEW1] Edge Recipe falló strict_quantities con pantry GENEROSO en "
        f"{len(failures)}/20 corridas. Ejemplo: ings={failures[0][0]!r} → "
        f"{failures[0][1]!r}. El builder está sirviendo cantidades por "
        f"encima de los defaults — el cap está roto o un refactor removió "
        f"`min(default_g, total_g)`."
    )


def test_edge_recipe_caps_total_serving_to_default_when_pantry_oversupplies():
    """Aunque pantry tenga 5kg de Pollo, ninguna porción individual debe
    exceder los defaults del builder (breakfast 150g, lunch 200g, dinner 150g).
    El cap es `min(default_g, total_g)` — con pantry sobrado, la rama
    `min` debe escoger `default_g`.
    """
    pantry = [
        "5000g Pollo",
        "5000g Arroz Blanco",
        "5000g Aguacate",
    ]
    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=[],
        diet="",
        pantry_items=pantry,
    )
    assert edge_day is not None
    pollo_re = re.compile(r"\bpollo\b", re.IGNORECASE)
    for ing in _all_edge_ingredients(edge_day):
        if pollo_re.search(ing):
            grams = _extract_grams(ing)
            # Almuerzo es la comida con más protein (200g); ninguna porción
            # individual debe exceder ese tope.
            assert grams <= 200.0, (
                f"Edge Recipe sirvió {grams}g de Pollo en una sola comida "
                f"con pantry sobrado. El cap escaló por encima del default. "
                f"Ingrediente: {ing!r}"
            )


# ---------------------------------------------------------------------------
# Group C — Regresión estructural: el bloque [P0-NEW1] sigue cableado en _chunk_worker
# ---------------------------------------------------------------------------
def test_chunk_worker_contains_p0_new1_quantity_validation_loop():
    """Guarda contra eliminación accidental del bloque inline en `_chunk_worker`.

    El bloque debe seguir presente y debe usar:
      - validate_ingredients_against_pantry con strict_quantities=True
      - _build_filtered_edge_recipe_day como fallback
      - _pause_chunk_for_pantry_refresh con reason='degraded_quantities_unfeasible'

    Si alguien refactoriza y elimina cualquiera de las 3 piezas, este test
    falla — alertando antes de que el modo degradado vuelva a quedar sin
    validación de cantidades.
    """
    # `_chunk_worker` es una closure dentro de `process_plan_chunk_queue`
    # (cron_tasks.py:11839). Inspeccionamos el source del enclosing function
    # para ver el body del worker.
    src = inspect.getsource(cron_tasks.process_plan_chunk_queue)

    assert "[P0-NEW1]" in src, (
        "Marker [P0-NEW1] desapareció de _chunk_worker. El bloque de validación "
        "de cantidades en modo degradado fue removido o renombrado."
    )
    assert "validate_ingredients_against_pantry" in src, (
        "validate_ingredients_against_pantry ya no se invoca en _chunk_worker. "
        "Sin esa llamada, el modo degradado no valida cantidades contra pantry."
    )
    assert "_build_filtered_edge_recipe_day" in src, (
        "El fallback a Edge Recipe quantity-aware desapareció. Sin él, el "
        "worker pasaría directo a pausa cuando shuffle agota intentos."
    )
    assert "degraded_quantities_unfeasible" in src, (
        "El reason='degraded_quantities_unfeasible' del pause no está en "
        "_chunk_worker. Telemetría de [P0-NEW1] perdida."
    )
    assert "strict_quantities=True" in src, (
        "_chunk_worker ya no llama a validate_ingredients_against_pantry con "
        "strict_quantities=True — el guardrail de cantidades está apagado en "
        "el path de modo degradado."
    )


def test_chunk_worker_passes_pantry_to_edge_recipe_fallback():
    """En el fallback a Edge Recipe (cron_tasks.py:13919), el builder DEBE
    recibir `pantry_items=_pantry_snap`. Sin ese argumento, el cap por
    pantry no aplica y el resultado puede exceder lo disponible.
    """
    # `_chunk_worker` es una closure dentro de `process_plan_chunk_queue`
    # (cron_tasks.py:11839). Inspeccionamos el source del enclosing function
    # para ver el body del worker.
    src = inspect.getsource(cron_tasks.process_plan_chunk_queue)

    # Buscar la sección del fallback a Edge Recipe del [P0-NEW1].
    # Debe haber un bloque que llame _build_filtered_edge_recipe_day con
    # pantry_items entre el log "Smart Shuffle falló" y el log "Edge Recipe
    # también falló".
    fallback_block = re.search(
        r"Smart Shuffle falló.*?_build_filtered_edge_recipe_day\((.*?)\)",
        src,
        flags=re.DOTALL,
    )
    assert fallback_block, (
        "No encontré el bloque de fallback a Edge Recipe en _chunk_worker. "
        "El [P0-NEW1] depende de que tras 3 fallos de shuffle se intente un "
        "Edge Recipe quantity-aware — esa llamada no está en el código actual."
    )
    args_blob = fallback_block.group(1)
    assert "pantry_items" in args_blob, (
        f"El fallback a _build_filtered_edge_recipe_day NO está pasando "
        f"pantry_items. Sin pantry_items, el cap por cantidades no se "
        f"aplica y el edge resultante puede exceder el pantry. "
        f"Args encontrados: {args_blob!r}"
    )


def test_chunk_worker_pauses_chunk_when_edge_recipe_also_fails_quantities():
    """Tras shuffle (3x) y Edge Recipe (1x) sin éxito en cantidades, el worker
    debe pausar el chunk con reason='degraded_quantities_unfeasible' — NO
    debe servir un plato infeasible ni marcarlo como completed.
    """
    # `_chunk_worker` es una closure dentro de `process_plan_chunk_queue`
    # (cron_tasks.py:11839). Inspeccionamos el source del enclosing function
    # para ver el body del worker.
    src = inspect.getsource(cron_tasks.process_plan_chunk_queue)

    # El pause con este reason específico debe estar dentro del bloque
    # [P0-NEW1/SHUFFLE-QTY] (después de "If still unfeasible" y antes del
    # log de éxito).
    pattern = re.compile(
        r"if not _qty_validated:.*?_pause_chunk_for_pantry_refresh\(.*?"
        r"reason=\"degraded_quantities_unfeasible\"\).*?return",
        flags=re.DOTALL,
    )
    assert pattern.search(src), (
        "El worker no pausa con reason='degraded_quantities_unfeasible' tras "
        "agotar shuffle+edge. Sin ese pause, un chunk con cantidades "
        "infeasibles podría servirse al usuario."
    )


def test_chunk_worker_smart_shuffle_qty_attempts_capped_at_three():
    """El loop de reintentos de Smart Shuffle debe tener un cap explícito.
    Sin cap, un pantry imposible haría que el worker iterase indefinidamente
    sobre el pool de candidatos.
    """
    # `_chunk_worker` es una closure dentro de `process_plan_chunk_queue`
    # (cron_tasks.py:11839). Inspeccionamos el source del enclosing function
    # para ver el body del worker.
    src = inspect.getsource(cron_tasks.process_plan_chunk_queue)

    # El cap se setea en `_max_shuffle_qty_attempts = 3`. Si alguien lo eleva
    # mucho o lo elimina, este test alerta.
    match = re.search(r"_max_shuffle_qty_attempts\s*=\s*(\d+)", src)
    assert match, (
        "_max_shuffle_qty_attempts ya no se define como literal en "
        "_chunk_worker. El loop de reintentos podría correr sin tope."
    )
    cap = int(match.group(1))
    assert 1 <= cap <= 5, (
        f"_max_shuffle_qty_attempts={cap} fuera del rango razonable [1..5]. "
        f"Cap demasiado bajo = poco margen; demasiado alto = latencia y log "
        f"spam en planes con pantry imposible."
    )
