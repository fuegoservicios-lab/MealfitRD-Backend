"""[P1-D] Tests del warning de pantry-drift entre chunk N y chunk N+1.

Bug original: el LLM del chunk N+1 recibía el inventario actual pero no sabía qué
había cambiado desde el chunk N. Si pollo bajó 80% por consumo no planeado, el LLM
no priorizaba usar lo que quedaba.

Fix:
  1. Helper puro `_compute_pantry_diff_warning(prev_snapshot, current_inventory)`
     detecta drops/increases significativos y retorna dict estructurado.
  2. Helper auxiliar `_extract_pantry_snapshot_from_inventory(inventory, top_n)`
     normaliza el inventario heterogéneo a dict {name_lowercase: qty}.
  3. process_chunk_task inyecta `form_data._pantry_drift_warning` antes del LLM call
     si chunk >= 2 y existe snapshot del chunk previo en plan_data.
  4. Tras merge exitoso, persiste snapshot del chunk actual en
     plan_data._pantry_snapshot_per_chunk[str(week_number)] (cap 6 chunks).

Cubre:
  HELPER `_compute_pantry_diff_warning`:
    1. Snapshot vacío → None.
    2. Drop dentro del threshold → no aparece en critical_drops.
    3. Drop >threshold → aparece con delta_pct correcto.
    4. Increase >threshold → aparece en notable_increases.
    5. Item nuevo (no estaba en prev) → aparece en new_items.
    6. Mezcla drops/increases/new ordenados por magnitud.
    7. current_inventory aceptado como dict O como lista.
    8. Sin cambios significativos → None.

  HELPER `_extract_pantry_snapshot_from_inventory`:
    9. Suma cantidades de duplicados (varios lotes del mismo ingrediente).
    10. Top_N respetado.
    11. Lista vacía → dict vacío.
    12. Items con qty<=0 ignorados.

  WIRING (shape tests):
    13. process_chunk_task lee `_pantry_snapshot_per_chunk` del chunk previo.
    14. process_chunk_task escribe `form_data._pantry_drift_warning` cuando aplica.
    15. Merge final persiste `_pantry_snapshot_per_chunk[week_number]`.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_d_pantry_diff_warning.py -v
"""
import sys
import os
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


# Stubs estándar.
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
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_kw: None,
    execute_sql_write=lambda *_a, **_kw: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    get_inventory_activity_since=lambda *_a, **_kw: [],
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: 0,
    # [test fix] db_inventory.reserve_plan_ingredients returns int (count of items reserved).
    # Stubbing with a dict caused TypeError: '>=' not supported between dict and int in
    # cron_tasks.py:5239/16519. Mirror production: count ingredients with len>=3 in days[2].
    reserve_plan_ingredients=lambda *_a, **_kw: sum(
        1 for d in (_a[2] if len(_a) >= 3 else (_kw.get("days") or []))
        for m in ((d or {}).get("meals") or [])
        for i in (m.get("ingredients") or [])
        if i and len(str(i).strip()) >= 3
    ),
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_recent_plans=lambda *_a, **_kw: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_kw: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_kw: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_kw: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_kw: {})


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_shopping_list_delta=lambda *_a, **_kw: [],
        _parse_quantity=_stub_parse_quantity,
    )
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


import cron_tasks


# ===========================================================================
# Helpers para construir inventory mocks
# ===========================================================================
def _inv_dict(name, qty):
    return {"ingredient_name": name, "quantity": qty}


# ===========================================================================
# HELPER: _compute_pantry_diff_warning
# ===========================================================================
def test_compute_diff_empty_snapshot_returns_none():
    assert cron_tasks._compute_pantry_diff_warning({}, [_inv_dict("pollo", 1.0)]) is None
    assert cron_tasks._compute_pantry_diff_warning(None, [_inv_dict("pollo", 1.0)]) is None


def test_compute_diff_drop_below_threshold_not_reported():
    """Pollo bajó 20% (1.0 → 0.8): por debajo del threshold default 30%, no reporta."""
    prev = {"pollo": 1.0}
    current = [_inv_dict("pollo", 0.8)]
    assert cron_tasks._compute_pantry_diff_warning(prev, current) is None


def test_compute_diff_drop_above_threshold_reported():
    """Pollo bajó 80% (1.0 → 0.2): supera threshold 30%, reporta con delta correcto."""
    prev = {"pollo": 1.0}
    current = [_inv_dict("pollo", 0.2)]
    result = cron_tasks._compute_pantry_diff_warning(prev, current)
    assert result is not None
    drops = result["critical_drops"]
    assert len(drops) == 1
    assert drops[0]["ingredient"] == "pollo"
    assert drops[0]["prev_qty"] == 1.0
    assert drops[0]["current_qty"] == 0.2
    assert drops[0]["delta_pct"] == -80.0


def test_compute_diff_increase_above_threshold_reported():
    """Arroz subió 50% (2.0 → 3.0): aparece en notable_increases."""
    prev = {"arroz": 2.0}
    current = [_inv_dict("arroz", 3.0)]
    result = cron_tasks._compute_pantry_diff_warning(prev, current)
    assert result is not None
    inc = result["notable_increases"]
    assert len(inc) == 1
    assert inc[0]["ingredient"] == "arroz"
    assert inc[0]["delta_pct"] == 50.0


def test_compute_diff_new_item_reported():
    """Plátano no estaba en prev_snapshot: aparece en new_items."""
    prev = {"pollo": 1.0}
    current = [_inv_dict("pollo", 1.0), _inv_dict("plátano", 6.0)]
    result = cron_tasks._compute_pantry_diff_warning(prev, current)
    assert result is not None
    new_items = result["new_items"]
    assert any(item["ingredient"] == "plátano" and item["current_qty"] == 6.0 for item in new_items)


def test_compute_diff_mixed_sorted_by_magnitude():
    """Drops ordenados por delta más negativo primero; increases por más positivo primero."""
    prev = {"pollo": 1.0, "arroz": 2.0, "queso": 5.0}
    current = [
        _inv_dict("pollo", 0.2),  # -80%
        _inv_dict("arroz", 0.8),  # -60%
        _inv_dict("queso", 10.0),  # +100%
    ]
    result = cron_tasks._compute_pantry_diff_warning(prev, current)
    assert result is not None
    drops = result["critical_drops"]
    assert drops[0]["ingredient"] == "pollo"  # -80% es más negativo
    assert drops[1]["ingredient"] == "arroz"


def test_compute_diff_accepts_dict_for_current_inventory():
    """Si pasamos current_inventory como dict (ya extraído), funciona igual."""
    prev = {"pollo": 1.0}
    current_dict = {"pollo": 0.2}
    result = cron_tasks._compute_pantry_diff_warning(prev, current_dict)
    assert result is not None
    assert len(result["critical_drops"]) == 1


def test_compute_diff_no_significant_changes_returns_none():
    """Inventario casi idéntico → None (no inyecta warning vacío)."""
    prev = {"pollo": 1.0, "arroz": 2.0}
    current = [_inv_dict("pollo", 1.0), _inv_dict("arroz", 2.0)]
    assert cron_tasks._compute_pantry_diff_warning(prev, current) is None


# ===========================================================================
# HELPER: _extract_pantry_snapshot_from_inventory
# ===========================================================================
def test_extract_snapshot_sums_duplicate_ingredients():
    """Dos lotes de pollo (1kg + 0.5kg) → snapshot {pollo: 1.5}."""
    inventory = [
        _inv_dict("pollo", 1.0),
        _inv_dict("pollo", 0.5),
        _inv_dict("arroz", 2.0),
    ]
    snapshot = cron_tasks._extract_pantry_snapshot_from_inventory(inventory)
    assert snapshot.get("pollo") == 1.5
    assert snapshot.get("arroz") == 2.0


def test_extract_snapshot_respects_top_n():
    inventory = [_inv_dict(f"item-{i}", float(100 - i)) for i in range(50)]
    snapshot = cron_tasks._extract_pantry_snapshot_from_inventory(inventory, top_n=5)
    assert len(snapshot) == 5
    # Los top 5 son los con qty más alta: items 0..4 (qty 100..96).
    assert "item-0" in snapshot
    assert "item-4" in snapshot
    assert "item-5" not in snapshot


def test_extract_snapshot_empty_input():
    assert cron_tasks._extract_pantry_snapshot_from_inventory([]) == {}
    assert cron_tasks._extract_pantry_snapshot_from_inventory(None) == {}


def test_extract_snapshot_skips_zero_or_negative_qty():
    inventory = [
        _inv_dict("pollo", 0.0),
        _inv_dict("arroz", -1.0),
        _inv_dict("queso", 2.0),
    ]
    snapshot = cron_tasks._extract_pantry_snapshot_from_inventory(inventory)
    assert "pollo" not in snapshot
    assert "arroz" not in snapshot
    assert snapshot.get("queso") == 2.0


def test_extract_snapshot_lowercase_keys():
    inventory = [_inv_dict("Pollo Crudo", 1.0), _inv_dict("ARROZ", 2.0)]
    snapshot = cron_tasks._extract_pantry_snapshot_from_inventory(inventory)
    assert "pollo crudo" in snapshot
    assert "arroz" in snapshot


# ===========================================================================
# WIRING (shape tests sobre el código fuente)
# ===========================================================================
def _read_source():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        return f.read()


def test_process_chunk_task_reads_prev_snapshot_for_chunks_ge_2():
    """process_chunk_task debe leer `_pantry_snapshot_per_chunk` del chunk previo."""
    source = _read_source()
    # La inyección debe estar guardada por `if int(week_number) >= 2`.
    assert "if int(week_number) >= 2 and isinstance(prior_plan_data, dict):" in source
    # Y leer la key correcta de plan_data.
    assert 'prior_plan_data.get("_pantry_snapshot_per_chunk")' in source


def test_process_chunk_task_writes_warning_to_form_data():
    """Cuando hay drift, el call site debe escribir form_data['_pantry_drift_warning']."""
    source = _read_source()
    assert 'form_data["_pantry_drift_warning"]' in source
    # Llama al helper puro.
    assert "_compute_pantry_diff_warning(" in source


def test_merge_persists_pantry_snapshot_per_chunk():
    """Tras merge exitoso, plan_data._pantry_snapshot_per_chunk[str(week_number)] se setea."""
    source = _read_source()
    assert "plan_data['_pantry_snapshot_per_chunk']" in source, (
        "El merge final debe persistir snapshot del chunk actual en "
        "plan_data._pantry_snapshot_per_chunk para que el chunk N+1 pueda detectar drift."
    )
    # Llama al helper de extracción.
    assert "_extract_pantry_snapshot_from_inventory(" in source


def test_snapshot_storage_capped_to_avoid_unbounded_growth():
    """El persist debe limitar la cantidad de snapshots almacenados (planes 30d generan 8+ chunks)."""
    source = _read_source()
    # Buscar el cap implementado: comparación contra > 6.
    idx = source.find("plan_data['_pantry_snapshot_per_chunk']")
    assert idx > -1
    nearby = source[max(0, idx - 800):idx + 800]
    assert "len(_p1d_per_chunk)" in nearby and "> 6" in nearby, (
        "El persist debe capear `_pantry_snapshot_per_chunk` (e.g., a 6 chunks) para "
        "que planes 30d (8 chunks) no inflen plan_data indefinidamente."
    )
