"""[P1-PANTRY-STRICT-CONSENT · 2026-08-02] "Nevera estricta + consentimiento".

Decisión del owner: tras la compra inicial, los cambios de plato ("Actualizar
platos"/Cambiar Plato/regenerate-day) cocinan SOLO de la Nevera FÍSICA real por
default; si no hay alternativa nevera-only, el sistema PREGUNTA (nombre + cantidad
+ precio estimado) en vez de introducir el ingrediente en silencio.

Caso real que lo motivó: un swap (Cambiar Plato) metió catibías de YUCA — 75g de
un día YA ARCHIVADO del plan, jamás registrada en `user_inventory` — sin preguntar.
La lista de compras "renació" con 1 ítem y el botón "Ya compré la lista" reapareció
sin que el usuario hubiera consentido comprar nada.

Causa raíz (reconocimiento verificado leyendo código, no adivinado): el universo
que valida `constants.validate_ingredients_against_pantry` dentro de
`agent.py::swap_meal` (`clean_ingredients`) se construía con
`get_realtime_pantry(plan_data, consumed_ingredients)` — TODOS los ingredientes de
TODOS los días del plan (acumulativo, nunca expira, solo decrementa por consumo
LOGGEADO en el diario), NO la Nevera física (`user_inventory`). La yuca de un día
archivado nunca se "consumió" en el diario → seguía "disponible" para el guard.
`regenerate-day` YA usaba la Nevera real (`_inventory_grams_ledger` +
`pantry_override=True`, P2-REGEN-DAY-PANTRY-OVERRIDE) — el leak era exclusivo de
`/swap-meal` y `/fix-sodium-day` (ambos invocan `agent.swap_meal` SIN setear
`pantry_override`).

Fix: knob `MEALFIT_PANTRY_STRICT_UPDATES` (default True). ON ⇒ el universo pasa a
ser `_swap_real_pantry_ledger_lines(user_id)` (mirror de
`routers/plans.py::_inventory_grams_ledger`, `user_inventory` con quantity>0). Si
el chef no converge nevera-only, `swap_meal_with_consent()` (SSOT usado por
`/swap-meal` y `/fix-sodium-day`) hace 1 probe de descubrimiento interno para
nombrar QUÉ falta (vía `validate_ingredients_against_pantry(...,
return_unauthorized=True)`, reusando el 100% de la lógica de matching ya
endurecida) y responde soft `needs_new_ingredients` con nombre/cantidad/precio
RD$ estimado (`shopping_calculator.estimate_new_ingredient_price_rd`, mismo piso
del Supermercado RD que cotiza la lista) — cero persist, cero cobro. El nuevo
parámetro `allow_new_ingredients` (consentimiento explícito del usuario) suma esos
nombres al universo autorizado y el flujo se re-corre.

Cross-link con `test_p2_hist_audit_14_marker_test_link`: slug
`p1_pantry_strict_consent` ↔ filename `test_p1_pantry_strict_consent.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
_PLANS_PY = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_CONSTANTS_PY = (_BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
_SHOPPING_PY = (_BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — marker + knob wiring (parser-based)
# ---------------------------------------------------------------------------

def test_marker_anchored_in_agent_py():
    assert "P1-PANTRY-STRICT-CONSENT" in _AGENT_PY


def test_marker_bumped_in_app_py():
    # [de-pin · 2026-08-02] `_LAST_KNOWN_PFIX` es single-valued → pinear el literal
    # "P1-PANTRY-STRICT-CONSENT" quedó stale apenas P1-CONTAINER-SERVABLE bumpeó el marker el
    # mismo día (main no puede quedar rojo por un P-fix posterior legítimo). El contrato durable
    # del bump (formato + floor de fecha + cross-link slug↔test) ya vive en
    # test_p3_1_last_known_pfix_freshness.py + test_p2_hist_audit_14_marker_test_link.py — este
    # test solo verifica que `_LAST_KNOWN_PFIX` EXISTE con el formato esperado. Mismo patrón
    # establecido en test_p2_update_intelligence_3.py::test_last_known_pfix_bumped.
    assert re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P\d+-[A-Z0-9-]+ · \d{4}-\d{2}-\d{2}"', _APP_PY), (
        "_LAST_KNOWN_PFIX debe existir con formato `Pn-... · YYYY-MM-DD`."
    )


def test_marker_anchor_filename():
    expected_slug = "p1_pantry_strict_consent"
    assert expected_slug in __file__.replace("\\", "/").lower()


def test_knob_default_true():
    assert 'os.environ.get("MEALFIT_PANTRY_STRICT_UPDATES", "true")' in _AGENT_PY


def test_swap_meal_with_consent_and_ledger_helper_exist():
    assert "def swap_meal_with_consent(form_data: dict)" in _AGENT_PY
    assert "def _swap_real_pantry_ledger_lines(user_id: str)" in _AGENT_PY
    assert "def _pantry_strict_updates_enabled()" in _AGENT_PY


def test_ledger_helper_reads_user_inventory_not_plan():
    """El universo real DEBE nacer de `get_raw_user_inventory` (tabla física), NO de
    `get_realtime_pantry`/`plan_data` — la firma de la función ni siquiera acepta
    `plan_data` como argumento (prueba estructural del fix)."""
    idx = _AGENT_PY.index("def _swap_real_pantry_ledger_lines(user_id: str)")
    body = _AGENT_PY[idx:idx + 2200]
    assert "get_raw_user_inventory" in body
    assert "plan_data" not in body
    assert "get_realtime_pantry" not in body


def test_routers_import_and_use_wrapper_for_swap_and_fix_sodium():
    assert "swap_meal_with_consent" in _PLANS_PY
    assert re.search(r"result\s*=\s*swap_meal_with_consent\(data\)", _PLANS_PY), (
        "/swap-meal debe llamar swap_meal_with_consent(data), no swap_meal(data) directo."
    )
    assert re.search(r"new_meal\s*=\s*swap_meal_with_consent\(meal_form\)", _PLANS_PY), (
        "/fix-sodium-day debe llamar swap_meal_with_consent(meal_form)."
    )


def test_regenerate_day_deliberately_untouched():
    """`regenerate-day` sigue llamando `swap_meal(...)` directo (ya usa la Nevera real
    vía pantry_override — no tenía el leak) — decisión documentada inline."""
    idx = _PLANS_PY.index("def api_regenerate_day(")
    end = _PLANS_PY.index("\ndef ", idx + 100)
    body = _PLANS_PY[idx:end]
    assert "P1-PANTRY-STRICT-CONSENT" in body, (
        "Falta el comentario que documenta POR QUÉ regenerate-day no usa el wrapper."
    )
    # [P1-SWAP-LUNA · 2026-08-05] El regex tolera argumentos extra: el bucle del dia
    # ahora pasa `surface="day"` para pedir su propio reasoning_effort. Lo vigilado
    # sigue siendo QUE se llame a swap_meal ahi, no su lista exacta de argumentos.
    assert re.search(r"nm\s*=\s*swap_meal\(_form_v\b", body)
    assert not re.search(r"=\s*swap_meal_with_consent\(", body), (
        "regenerate-day NO debe LLAMAR al wrapper de consentimiento (decisión deliberada, "
        "ver comentario P1-PANTRY-STRICT-CONSENT en el loop) — mención en prosa OK."
    )


def test_needs_new_ingredients_checked_before_charge_in_swap_meal_endpoint():
    """(e) Ordering guard: el early-return de `needs_new_ingredients` DEBE aparecer
    ANTES de `log_api_usage` — de lo contrario el soft-fail informativo cobraría
    crédito. Ancla contra regresión silenciosa (reordenar código sin querer)."""
    idx_fn = _PLANS_PY.index('@router.post("/swap-meal")')
    idx_next = _PLANS_PY.index("\ndef ", idx_fn + 200)
    body = _PLANS_PY[idx_fn:idx_next]
    idx_needs = body.index("result.get(\"needs_new_ingredients\")")
    idx_charge = body.index('log_api_usage(user_id, "llm_swap_meal")', idx_needs)
    assert idx_needs < idx_charge, (
        "El check de needs_new_ingredients debe preceder al log_api_usage post-éxito."
    )


def test_needs_new_ingredients_checked_before_charge_in_fix_sodium_day_endpoint():
    idx_fn = _PLANS_PY.index("def api_fix_sodium_day(")
    idx_next = _PLANS_PY.index("\ndef ", idx_fn + 200)
    body = _PLANS_PY[idx_fn:idx_next]
    idx_needs = body.index('new_meal.get("needs_new_ingredients")')
    idx_charge = body.index('log_api_usage(verified_user_id, "llm_fix_sodium_day")')
    assert idx_needs < idx_charge


# ---------------------------------------------------------------------------
# Section B — `_swap_real_pantry_ledger_lines` (funcional, aislado)
# ---------------------------------------------------------------------------

class _FakeNutriInfo:
    def __init__(self, name):
        self.name = name


class _FakePantryDB:
    """DB determinista: solo reconoce los alimentos que el test necesita."""
    _CANON = {"pollo": "Pollo", "yuca": "Yuca", "arroz": "Arroz"}

    def lookup(self, raw_name):
        key = str(raw_name or "").strip().lower()
        for token, canon in self._CANON.items():
            if token in key:
                return _FakeNutriInfo(canon)
        return None

    def to_grams(self, qty, unit, info):
        u = str(unit or "").strip().lower()
        qty = float(qty or 0)
        if u in ("kg",):
            return qty * 1000.0
        if u in ("lb", "lbs"):
            return qty * 453.592
        return qty  # tratamos "g"/"" como ya-gramos (inputs de test controlados)


@pytest.fixture
def _pantry_ledger_env(monkeypatch):
    import agent
    import db as db_module
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakePantryDB)
    return agent, db_module


def test_ledger_lines_built_from_real_inventory(_pantry_ledger_env, monkeypatch):
    agent, db_module = _pantry_ledger_env
    monkeypatch.setattr(db_module, "get_raw_user_inventory", lambda uid: [
        {"ingredient_name": "Pollo", "quantity": 500.0, "unit": "g", "available_quantity": 500.0},
        {"ingredient_name": "Arroz", "quantity": 2.0, "unit": "lb", "available_quantity": 2.0},
    ])
    lines = agent._swap_real_pantry_ledger_lines("user-123")
    joined = " | ".join(lines)
    assert "Pollo" in joined
    assert "Arroz" in joined
    assert "Yuca" not in joined  # no estaba en la Nevera real


def test_ledger_lines_empty_for_guest(_pantry_ledger_env, monkeypatch):
    agent, db_module = _pantry_ledger_env
    called = {"n": 0}

    def _boom(uid):
        called["n"] += 1
        return []

    monkeypatch.setattr(db_module, "get_raw_user_inventory", _boom)
    assert agent._swap_real_pantry_ledger_lines("guest") == []
    assert agent._swap_real_pantry_ledger_lines(None) == []
    assert called["n"] == 0, "guest/None NUNCA debe tocar la DB"


def test_ledger_lines_fail_open_on_db_error(_pantry_ledger_env, monkeypatch):
    agent, db_module = _pantry_ledger_env

    def _raise(uid):
        raise RuntimeError("db down")

    monkeypatch.setattr(db_module, "get_raw_user_inventory", _raise)
    assert agent._swap_real_pantry_ledger_lines("user-123") == []


def test_ledger_lines_skip_zero_or_unresolved_items(_pantry_ledger_env, monkeypatch):
    agent, db_module = _pantry_ledger_env
    monkeypatch.setattr(db_module, "get_raw_user_inventory", lambda uid: [
        {"ingredient_name": "Ingrediente Fantasma XYZ", "quantity": 100.0, "unit": "g"},
        {"ingredient_name": "Pollo", "quantity": 0.0, "unit": "g", "available_quantity": 0.0},
    ])
    assert agent._swap_real_pantry_ledger_lines("user-123") == []


# ---------------------------------------------------------------------------
# Section C — `validate_ingredients_against_pantry(..., return_unauthorized=True)`
# ---------------------------------------------------------------------------

def test_return_unauthorized_backcompat_default_false():
    from constants import validate_ingredients_against_pantry
    res = validate_ingredients_against_pantry(["150 g de Yuca"], ["500 g de Pollo"])
    assert isinstance(res, str)  # rechazo → sigue siendo str, NO tuple


def test_return_unauthorized_true_reports_missing_item():
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["150 g de Yuca"], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res != True  # rechazado
    assert any("Yuca" in item for item in unauthorized)


def test_return_unauthorized_true_empty_when_approved():
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["100 g de Pollo"], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res is True
    assert unauthorized == []


def test_return_unauthorized_true_empty_pantry_shortcircuit():
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["150 g de Yuca"], [], return_unauthorized=True,
    )
    assert res is True
    assert unauthorized == []


# ---------------------------------------------------------------------------
# Section C2 — condimentos exentos por WORD-BOUNDARY, no substring [review finding]
#
# Pre-fix: `any(c in item_lower for c in allowed_condiments)` era substring plano —
# "sal" ⊂ "Salami" / "Salmón" aprobaba esos platos como "condimento" sin serlo, dejando
# pasar carnes/pescados enteros sin estar en la Nevera (15ª aparición documentada de esta
# clase de bug — ver culinary_coherence.py::CONDIMENT_EXEMPT, IMPORTANT-5, que cerró la
# 14ª con el mismo patrón word-boundary). Pantry SIN salami/salmón/pollo real para que el
# resultado dependa 100% de si el matching de condimento fue correcto.
# ---------------------------------------------------------------------------

def test_salami_is_not_approved_as_condiment_by_substring():
    """'200 g de Salami' NO debe colarse como condimento vía 'sal' ⊂ 'Salami' — sin
    salami en la Nevera, debe quedar `unauthorized`."""
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["200 g de Salami"], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res != True
    assert any("Salami" in item for item in unauthorized), (
        f"Salami debió quedar unauthorized (no es condimento), pero: {unauthorized}"
    )


def test_salmon_is_not_approved_as_condiment_by_substring():
    """'150 g de Salmón fresco' NO debe colarse vía 'sal' ⊂ 'Salmón'."""
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["150 g de Salmon fresco"], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res != True
    assert any("Salmon" in item for item in unauthorized), (
        f"Salmón debió quedar unauthorized (no es condimento), pero: {unauthorized}"
    )


@pytest.mark.parametrize("item", [
    "1 pizca de sal",
    "aceite de oliva",
    "ajo en polvo",
    "2 sales de mesa",   # plural — sigue exento
])
def test_real_condiments_still_exempt_after_word_boundary_fix(item):
    """Los condimentos REALES (palabra completa, con o sin plural) siguen exentos —
    el fix es de PRECISIÓN (rechaza falsos positivos por substring), no una restricción
    nueva sobre el set de condimentos ya aceptado."""
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        [item], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res is True, f"{item!r} debería seguir exento de condimento: {res}"
    assert unauthorized == []


def test_agua_not_approved_as_condiment_substring_of_aguacate():
    """Bonus del mismo fix (no pedido explícitamente, pero el mismo mecanismo lo cierra):
    'agua' ⊂ 'Aguacate' ya NO aprueba aguacate como condimento."""
    from constants import validate_ingredients_against_pantry
    res, unauthorized = validate_ingredients_against_pantry(
        ["1 Aguacate mediano"], ["500 g de Pollo"], return_unauthorized=True,
    )
    assert res != True
    assert any("guacate" in item.lower() for item in unauthorized)


# ---------------------------------------------------------------------------
# Section D — `shopping_calculator.estimate_new_ingredient_price_rd`
# ---------------------------------------------------------------------------

def test_estimate_price_uses_cheapest_variant(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "cheapest_supermarket_variant", lambda name: {
        "price_rd": 45.0, "presentation": "1 lb",
    })
    monkeypatch.setattr(sc, "_variant_price_per_g", lambda v: 45.0 / 453.592)
    price = sc.estimate_new_ingredient_price_rd("Yuca", 1000.0)
    assert price == pytest.approx(45.0 / 453.592 * 1000.0, rel=1e-3)


def test_estimate_price_none_when_no_catalog_match(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "cheapest_supermarket_variant", lambda name: None)
    assert sc.estimate_new_ingredient_price_rd("Alimento Rarísimo", 500.0) is None


def test_estimate_price_none_for_zero_or_negative_qty(monkeypatch):
    import shopping_calculator as sc
    assert sc.estimate_new_ingredient_price_rd("Yuca", 0) is None
    assert sc.estimate_new_ingredient_price_rd("Yuca", -5) is None
    assert sc.estimate_new_ingredient_price_rd("", 500) is None


# ---------------------------------------------------------------------------
# Section E — `swap_meal_with_consent` (funcional, `agent.swap_meal` mockeado)
# ---------------------------------------------------------------------------

@pytest.fixture
def _consent_env(monkeypatch):
    import agent
    monkeypatch.setenv("MEALFIT_PANTRY_STRICT_UPDATES", "true")
    return agent


def test_c_nevera_only_success_no_consent_prompt(_consent_env, monkeypatch):
    """(c) Éxito nevera-only ⇒ NO pregunta — retorna el plato tal cual, UNA sola
    llamada a swap_meal (sin discovery)."""
    agent = _consent_env
    calls = []

    def _fake_swap(form_data):
        calls.append(dict(form_data))
        return {"name": "Pollo al Horno", "ingredients": ["300 g de pollo"], "cals": 400}

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    result = agent.swap_meal_with_consent({"user_id": "u1", "rejected_meal": "x"})
    assert result == {"name": "Pollo al Horno", "ingredients": ["300 g de pollo"], "cals": 400}
    assert len(calls) == 1


def test_d_knob_off_delegates_1to1_legacy(monkeypatch):
    """(d) Knob OFF ⇒ comportamiento legacy exacto: 1 sola llamada a swap_meal,
    excepciones se propagan SIN wrapping (sin discovery, sin needs_new_ingredients)."""
    import agent
    monkeypatch.setenv("MEALFIT_PANTRY_STRICT_UPDATES", "false")
    calls = []

    def _fake_swap(form_data):
        calls.append(dict(form_data))
        raise ValueError("SWAP_LLM_RETRIES_EXHAUSTED: no candidate")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    with pytest.raises(ValueError, match="SWAP_LLM_RETRIES_EXHAUSTED"):
        agent.swap_meal_with_consent({"user_id": "u1", "allow_new_ingredients": None})
    assert len(calls) == 1, "knob OFF no debe intentar discovery"


def test_needs_new_ingredients_names_and_prices_the_yuca_case(_consent_env, monkeypatch):
    """El caso motivador end-to-end a nivel wrapper: 1er intento (nevera-only) agota
    retries proponiendo Yuca (que NO está en el universo real); el wrapper dispara 1
    discovery probe (relajado), diffea contra el universo real, y responde
    needs_new_ingredients con la Yuca nombrada + precio — SIN levantar, SIN 3er intento."""
    agent = _consent_env
    calls = []

    def _fake_swap(form_data):
        calls.append(dict(form_data))
        if form_data.get("_pantry_discovery_mode"):
            return {"name": "Catibías de Yuca", "ingredients": ["150 g de Yuca", "50 g de aceite"]}
        raise ValueError("SWAP_LLM_RETRIES_EXHAUSTED: el chef IA no pudo generar una alternativa")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    monkeypatch.setattr(agent, "_swap_real_pantry_ledger_lines", lambda uid: ["500g de Pollo"])
    monkeypatch.setattr(
        "shopping_calculator.estimate_new_ingredient_price_rd",
        lambda name, grams: 107.0 if "yuca" in name.lower() else None,
    )

    result = agent.swap_meal_with_consent({"user_id": "u1", "rejected_meal": "x"})

    assert len(calls) == 2, "debe haber exactamente 2 llamadas: intento normal + 1 discovery"
    assert calls[0].get("_pantry_discovery_mode") is not True
    assert calls[1].get("_pantry_discovery_mode") is True
    assert result["needs_new_ingredients"] is True
    assert result["code"] == "needs_new_ingredients"
    names = [m["name"] for m in result["missing_ingredients"]]
    assert names == ["Yuca"], f"solo la Yuca debe faltar (el aceite es condimento permitido): {names}"
    assert result["missing_ingredients"][0]["est_price_rd"] == 107.0
    assert "Yuca" in result["message"]
    assert "RD$107" in result["message"]


def test_b_consent_already_given_does_not_retry_discovery(_consent_env, monkeypatch):
    """(b) Si el caller YA mandó `allow_new_ingredients` (consintió) y el swap AÚN
    falla, el wrapper NO reintenta discovery de nuevo — propaga el soft-fail normal
    (evita loop; el universo ya se amplió y no alcanzó)."""
    agent = _consent_env
    calls = []

    def _fake_swap(form_data):
        calls.append(dict(form_data))
        raise ValueError("SWAP_LLM_RETRIES_EXHAUSTED: sigue sin converger")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    with pytest.raises(ValueError, match="SWAP_LLM_RETRIES_EXHAUSTED"):
        agent.swap_meal_with_consent({
            "user_id": "u1", "rejected_meal": "x", "allow_new_ingredients": ["Yuca"],
        })
    assert len(calls) == 1, "con consentimiento previo, cero discovery adicional"


def test_b_consent_given_and_swap_succeeds_returns_real_meal(_consent_env, monkeypatch):
    """(b) Consentimiento + swap ahora SÍ converge (universo ampliado real, no
    mockeado en este test — se prueba a nivel del wrapper que el resultado exitoso
    pasa intacto) ⇒ retorna el plato real, no needs_new_ingredients."""
    agent = _consent_env

    def _fake_swap(form_data):
        assert form_data.get("allow_new_ingredients") == ["Yuca"]
        return {"name": "Catibías de Yuca", "ingredients": ["150 g de Yuca"], "cals": 380}

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    result = agent.swap_meal_with_consent({
        "user_id": "u1", "rejected_meal": "x", "allow_new_ingredients": ["Yuca"],
    })
    assert result.get("needs_new_ingredients") is None
    assert result["name"] == "Catibías de Yuca"


def test_discovery_probe_itself_fails_propagates_original_error(_consent_env, monkeypatch):
    agent = _consent_env

    def _fake_swap(form_data):
        raise ValueError("SWAP_LLM_RETRIES_EXHAUSTED: primero")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    monkeypatch.setattr(agent, "_swap_real_pantry_ledger_lines", lambda uid: [])
    with pytest.raises(ValueError, match="primero"):
        agent.swap_meal_with_consent({"user_id": "u1", "rejected_meal": "x"})


def test_discovery_candidate_fits_universe_propagates_original_error(_consent_env, monkeypatch):
    """Caso raro: el probe relajado devuelve un candidato que en realidad SÍ cabía
    en el universo real (p.ej. race con un restock a mitad de request) — no hay nada
    honesto que ofrecer como "falta"; se preserva el soft-fail original."""
    agent = _consent_env
    calls = []

    def _fake_swap(form_data):
        calls.append(form_data)
        if form_data.get("_pantry_discovery_mode"):
            return {"name": "Pollo al Horno", "ingredients": ["100 g de Pollo"]}
        raise ValueError("SWAP_LLM_RETRIES_EXHAUSTED: x")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    monkeypatch.setattr(agent, "_swap_real_pantry_ledger_lines", lambda uid: ["500g de Pollo"])
    with pytest.raises(ValueError, match="SWAP_LLM_RETRIES_EXHAUSTED"):
        agent.swap_meal_with_consent({"user_id": "u1", "rejected_meal": "x"})


def test_other_valueerrors_not_wrapped(_consent_env, monkeypatch):
    """Errores que NO son de pantry (p.ej. CLINICAL_VIOLATION) deben propagarse
    directo, sin intentar discovery — el wrapper solo interviene en los 2 códigos
    de pantry."""
    agent = _consent_env
    calls = []

    def _fake_swap(form_data):
        calls.append(form_data)
        raise ValueError("CLINICAL_VIOLATION: alergeno detectado")

    monkeypatch.setattr(agent, "swap_meal", _fake_swap)
    with pytest.raises(ValueError, match="CLINICAL_VIOLATION"):
        agent.swap_meal_with_consent({"user_id": "u1", "rejected_meal": "x"})
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Section F — `_price_missing_ingredients` + `_build_consent_message` (unitario)
# ---------------------------------------------------------------------------

def test_price_missing_ingredients_dedupes_and_prices(monkeypatch):
    import agent
    monkeypatch.setattr(
        "shopping_calculator.estimate_new_ingredient_price_rd",
        lambda name, grams: 107.0 if name == "Yuca" else None,
    )
    out = agent._price_missing_ingredients(["150 g de Yuca", "otra 100 g de Yuca", "50 g de Ñame"])
    names = [m["name"] for m in out]
    assert names.count("Yuca") == 1, "de-duplicado por nombre normalizado"
    yuca_entry = next(m for m in out if m["name"] == "Yuca")
    assert yuca_entry["est_price_rd"] == 107.0
    name_entry = next(m for m in out if "ame" in m["name"].lower() or "ñame" in m["name"].lower())
    assert name_entry["est_price_rd"] is None


def test_build_consent_message_mentions_name_qty_price():
    import agent
    msg = agent._build_consent_message([
        {"name": "Yuca", "qty_needed": 150.0, "unit": "g", "est_price_rd": 107.0},
    ])
    assert "Yuca" in msg
    assert "150" in msg
    assert "RD$107" in msg
    assert "?" in msg or "¿" in msg  # pregunta explícita de consentimiento


def test_build_consent_message_caps_at_three_and_counts_rest():
    import agent
    missing = [
        {"name": n, "qty_needed": 100.0, "unit": "g", "est_price_rd": None}
        for n in ("Yuca", "Ñame", "Auyama", "Batata", "Plátano")
    ]
    msg = agent._build_consent_message(missing)
    assert "y 2 más" in msg


def test_build_consent_message_empty_list_fail_safe():
    import agent
    msg = agent._build_consent_message([])
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# Section H — Nevera vacía = bypass INTENCIONAL del modo estricto [review finding]
#
# Verificado ejecutando (review): con `user_inventory` REALMENTE vacía, el swap normal
# NO levanta `SWAP_STRICT_PANTRY_NO_INVENTORY` ni dispara `needs_new_ingredients` — el
# guard (`validate_ingredients_against_pantry`) se auto-desactiva ante una lista vacía
# (`if not pantry_ingredients: return True`, constants.py), así que `swap_meal()`
# converge en FREE_GENERATION con lo que el chef proponga. Decisión de producto
# documentada en el bypass (constants.py) y aquí: "Nevera estricta" es opt-in por USO
# de la Nevera — un universo vacío no es cocinable, no hay nada que "estrictar".
#
# Este test corre el pipeline REAL de `swap_meal()` (no mockea `agent.swap_meal`
# como el resto de este archivo) para probar el mecanismo end-to-end, mismo patrón
# que `test_p1_sodium_aware_placement.py::_sodium_swap_env` (fake ChatDeepSeek +
# fake circuit breaker + fake IngredientNutritionDB + guards hermanos desactivados
# vía knob — con `clean_ingredients` vacío, la mayoría de los guards hermanos
# skip-en-silencio por su propio `not (strict_pantry and not clean_ingredients)`).
# ---------------------------------------------------------------------------

class _FakeCBAlwaysOpen:
    def can_proceed(self):
        return True

    def record_success(self):
        pass

    def record_failure(self):
        pass


class _FakeSwapLLMOnce:
    def __init__(self, envelope):
        self._envelope = envelope
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self._envelope


class _FakeChatDeepSeekOnce:
    def __init__(self, envelope):
        self._envelope = envelope
        self.swap_llm = None

    def with_structured_output(self, *a, **kw):
        self.swap_llm = _FakeSwapLLMOnce(self._envelope)
        return self.swap_llm


@pytest.fixture
def _empty_pantry_swap_env(monkeypatch):
    import agent
    import nutrition_db
    import db as db_module
    import db_plans

    monkeypatch.setenv("MEALFIT_PANTRY_STRICT_UPDATES", "true")
    monkeypatch.setattr(agent, "UPDATE_CLINICAL_GUARD", False)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *a, **kw: _FakeCBAlwaysOpen())
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakePantryDB)
    monkeypatch.setattr(db_module, "get_raw_user_inventory", lambda uid: [])  # Nevera REALMENTE vacía
    monkeypatch.setattr(
        db_plans, "get_latest_meal_plan_with_id",
        lambda uid: {"plan_data": {"days": []}},  # sin created_at → se salta get_consumed_meals_since
    )
    monkeypatch.setenv("MEALFIT_SODIUM_AWARE_SWAP", "false")
    monkeypatch.setenv("MEALFIT_SWAP_BASE_REPEAT_GATE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_MACRO_TRUTHUP", "false")
    monkeypatch.setenv("MEALFIT_SWAP_DETERMINISTIC_RESCALE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_PROTEIN_CLOSER", "false")
    monkeypatch.setenv("MEALFIT_SWAP_FATS_TRIM", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_SUPERPERS", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_CONDITION_DIRECTIVES", "false")
    # [review finding] `target_calories` no viene en el form_data del test → swap_meal
    # deriva un target vía `get_nutrition_targets` con biométricos DEFAULT (25yo, 'lb'
    # legacy) → 765kcal/46g proteína, muy lejos del candidato fijo del test (400kcal).
    # Ortogonal a lo que este test prueba (el bypass del guard de PANTRY, no de macros)
    # — se aísla apagando el validador de macros, mismo patrón que
    # `MEALFIT_UPDATE_MACRO_TRUTHUP=false` arriba para el resto de guards hermanos.
    monkeypatch.setenv("MEALFIT_SWAP_MACROS_VALIDATE", "false")
    return agent


def test_empty_real_pantry_bypasses_strict_mode_free_generation(_empty_pantry_swap_env, monkeypatch):
    """(b) Nevera REALMENTE vacía (`get_raw_user_inventory` → []) → el swap normal
    CONVERGE con lo que el chef proponga (aquí: camarones, nada que ver con una Nevera
    vacía) en UNA sola llamada LLM — NO `needs_new_ingredients`, NO
    `SWAP_STRICT_PANTRY_NO_INVENTORY`. Corrige la Concern #2 original del reporte, que
    afirmaba lo segundo sin haberlo verificado ejecutando."""
    agent = _empty_pantry_swap_env
    from schemas import MealModel
    envelope = {
        "raw": None,
        "parsed": MealModel(
            meal="Cena", name="Camarones al Ajillo", desc="Camarones salteados con ajo.",
            prep_time="15 min", cals=400, protein=25, carbs=20, fats=15,
            ingredients=["300 g de camarones frescos", "2 dientes de ajo"],
            recipe=["Mise en place: limpia los camarones.",
                    "El Toque de Fuego: saltea con ajo.",
                    "Montaje: sirve caliente."],
        ),
        "parsing_error": None,
    }
    holder = {}

    def _fake_chat_deepseek(*a, **kw):
        inst = _FakeChatDeepSeekOnce(envelope)
        holder["inst"] = inst
        return inst

    # [P1-SWAP-LUNA · 2026-08-05] El punto de intercepcion se movio: `swap_meal` ya no
    # instancia `ChatDeepSeek` directamente, sino que pide el cliente a la fabrica por
    # proveedor (`build_chat_llm`), porque el modelo del swap paso a ser de OpenAI.
    # Parchear `agent.ChatDeepSeek` aqui dejaria de interceptar EN SILENCIO y este test
    # llamaria al proveedor DE VERDAD (medido: la suite tardo 149s haciendo llamadas
    # reales antes de corregir esto).
    monkeypatch.setattr(agent, "build_chat_llm", _fake_chat_deepseek)

    result = agent.swap_meal_with_consent({
        "user_id": "user-empty-nevera", "rejected_meal": "Ensalada vieja",
        "meal_type": "Cena", "swap_reason": "dislike", "diet_type": "balanced",
    })

    assert holder["inst"].swap_llm.calls == 1, (
        "1 sola llamada LLM: con la Nevera vacía el guard se auto-desactiva y el "
        "candidato converge de inmediato — cero retry por rechazo de pantry."
    )
    assert not (isinstance(result, dict) and result.get("needs_new_ingredients")), (
        f"Nevera vacía NO debe disparar needs_new_ingredients (bypass intencional): {result}"
    )
    name = result.get("name") if isinstance(result, dict) else getattr(result, "name", None)
    assert name == "Camarones al Ajillo", (
        f"el swap debió converger con la propuesta libre del chef, resultado: {result}"
    )
