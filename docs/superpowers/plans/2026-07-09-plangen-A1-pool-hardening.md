# A1 Pool-Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Endurecer ~5 clases de identidad de la selección de alimentos a nivel de pool (imposibles por construcción), sin tocar la composición culinaria del LLM, cada garantía nace OFF + canario.

**Architecture:** Un enforcer determinista `harden_day_pools(skeleton, form_data, conditions)` fusionado en el bloque de scrub existente de `plan_skeleton_node` (`graph_orchestrator.py:6455`, antes de `_emit_progress`/return). Muta `skel_days` in-place → el LLM (primer read `day_generator.py:408`) solo ve pools endurecidos. Reusa `condition_rules` (SSOT de tokens contraindicados), `_allergy_safe_fallback_protein` (fallback gracioso), y el patrón de canario `_self_critique_canary_cohort`.

**Tech Stack:** Python 3.12, FastAPI, LangGraph. Tests: pytest (funciones puras, sin DB). Python del env conda `mealfit`: `C:\Users\angel\miniconda3\envs\mealfit\python.exe`.

## Global Constraints (verbatim del spec + convenciones del repo)

- Gates nuevos **nacen OFF**; knob `MEALFIT_*` con default seguro, registrado vía `_env_bool`/`_env_int` (módulo `knobs`, auto-registro en `_KNOBS_REGISTRY`).
- Tests que parsean source → **tooltip-anchor** en el código.
- `harden_day_pools` muta pools **in-place** (mismos objetos lista que lee `day_generator.py:408/567/568`), normalizando acentos/word-boundaries (`strip_accents`, patrón `_key_in_text`).
- **NUNCA** borrar backstops (reviewer, coherence, allergen/diet scans, renal caps, `collect_substitutions`, `_apply_protein_pool_scrub`, fallback gracioso). Solo degradar a telemetría **tras canario verde**.
- Fallback gracioso load-bearing: si un pool queda vacío tras endurecer, reusar `_allergy_safe_fallback_protein(form_data)` (proteínas) o dejar el pool intacto (carbs/fruits) — nunca vaciar.
- Bump `_LAST_KNOWN_PFIX` (`app.py:32`) al cierre, con test cross-linkeado por slug.
- Commit SCOPED con pathspec (nunca `-A`), en el repo `backend/`.

---

### Task 1: Scaffold — knobs + cohorte de canario + enforcer no-op cableado

**Files:**
- Modify: `graph_orchestrator.py` (knobs cerca de `:511`; helper cohorte cerca de `:471`; def `harden_day_pools` + llamada en `:6455`; metric metadata en `:34446`)
- Test: `tests/test_p1_harden_pools_scaffold.py`

**Interfaces:**
- Produces:
  - Knobs: `HARDEN_POOLS_ENABLED` (`MEALFIT_HARDEN_POOLS_ENABLED`, bool, False), `HARDEN_CONDITION_CATALOG` (`MEALFIT_HARDEN_CONDITION_CATALOG`, bool, False), `HARDEN_SALTCURED_MAIN` (`MEALFIT_HARDEN_SALTCURED_MAIN`, bool, False), `HARDEN_SAMEDAY_PROTEIN` (`MEALFIT_HARDEN_SAMEDAY_PROTEIN`, bool, False), `HARDEN_CROSSDAY_QUOTA` (`MEALFIT_HARDEN_CROSSDAY_QUOTA`, bool, False), `HARDEN_POOLS_CANARY_PCT` (`MEALFIT_HARDEN_POOLS_CANARY_PCT`, int, 0, clamp [0,100]).
  - `_harden_pools_canary_cohort(state) -> str` ("on"/"off"), salt `f"harden_pools|{_id}"`.
  - `harden_day_pools(skeleton: dict, form_data: dict, conditions=None) -> dict` (muta in-place, retorna dict de conteos `{"condition_removed":int,"saltcured_removed":int,"sameday_bound":int,"crossday_capped":int}`). Master OFF ⇒ no-op, retorna zeros.

- [ ] **Step 1: Write failing test** — `tests/test_p1_harden_pools_scaffold.py`

```python
import importlib
import graph_orchestrator as go

def test_knobs_exist_and_default_off(monkeypatch):
    # defaults seguros: master + per-class OFF, canary 0
    assert go.HARDEN_POOLS_ENABLED is False
    assert go.HARDEN_CONDITION_CATALOG is False
    assert go.HARDEN_SALTCURED_MAIN is False
    assert go.HARDEN_SAMEDAY_PROTEIN is False
    assert go.HARDEN_CROSSDAY_QUOTA is False
    assert go.HARDEN_POOLS_CANARY_PCT == 0

def test_harden_day_pools_noop_when_master_off(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", False)
    skel = {"days": [{"day": 1, "protein_pool": ["Salami Dominicano", "Pollo"],
                      "carb_pool": ["Arroz blanco"], "fruit_pool": ["Toronja"]}]}
    counts = go.harden_day_pools(skel, {"medicalConditions": ["dm2"]}, None)
    assert counts == {"condition_removed": 0, "saltcured_removed": 0,
                      "sameday_bound": 0, "crossday_capped": 0}
    # pools intactos con master OFF
    assert skel["days"][0]["fruit_pool"] == ["Toronja"]

def test_canary_cohort_deterministic_and_default_on(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 0)
    st = {"form_data": {"user_id": "u-123"}}
    assert go._harden_pools_canary_cohort(st) == "on"  # PCT=0 → siempre on
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 100)
    assert go._harden_pools_canary_cohort(st) == "off"  # PCT=100 → siempre off
    # estable por usuario: misma respuesta 2 veces
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
    a = go._harden_pools_canary_cohort(st)
    b = go._harden_pools_canary_cohort(st)
    assert a == b and a in ("on", "off")
```

- [ ] **Step 2: Run test, verify FAIL** — `python -m pytest tests/test_p1_harden_pools_scaffold.py -v` → FAIL (AttributeError: no `HARDEN_POOLS_ENABLED`).

- [ ] **Step 3: Add knobs** cerca de `graph_orchestrator.py:511` (tras `STAPLE_REPEAT_GATE_ENABLED`), con comentario tooltip-anchor `A1-HARDEN-POOLS`:

```python
# [A1-HARDEN-POOLS · 2026-07-09] Endurecimiento determinista de la SELECCIÓN de alimentos a nivel de
# pool: vuelve imposibles-por-construcción ~5 clases de identidad (condición-contraindicada, salado-como-
# main, proteína repetida mismo-día, repetición cross-día). El LLM conserva la composición del plato +
# recetas. Nace TODO OFF (canariar contra clinical_band antes de flipear). Ver spec 2026-07-09-plangen-A1.
HARDEN_POOLS_ENABLED       = _env_bool("MEALFIT_HARDEN_POOLS_ENABLED", False)        # master kill-switch
HARDEN_CONDITION_CATALOG   = _env_bool("MEALFIT_HARDEN_CONDITION_CATALOG", False)    # clase 3
HARDEN_SALTCURED_MAIN      = _env_bool("MEALFIT_HARDEN_SALTCURED_MAIN", False)       # clase 5
HARDEN_SAMEDAY_PROTEIN     = _env_bool("MEALFIT_HARDEN_SAMEDAY_PROTEIN", False)      # clase 1
HARDEN_CROSSDAY_QUOTA      = _env_bool("MEALFIT_HARDEN_CROSSDAY_QUOTA", False)       # clase 2
HARDEN_POOLS_CANARY_PCT    = _env_int("MEALFIT_HARDEN_POOLS_CANARY_PCT", 0, validator=lambda v: 0 <= v <= 100)


def _harden_pools_canary_cohort(state) -> str:
    """[A1-HARDEN-POOLS] Cohorte determinista del canario A1: 'off' si el plan cae en el
    HARDEN_POOLS_CANARY_PCT% (bucket sha256 con salt propio, insesgado A/B, estable por usuario),
    'on' si no. PCT=0 → siempre 'on'. Fail-safe → 'on'. Salt independiente de self_critique."""
    try:
        pct = HARDEN_POOLS_CANARY_PCT
        if pct <= 0:
            return "on"
        if pct >= 100:
            return "off"
        fd = state.get("form_data") or {}
        _id = fd.get("user_id") or fd.get("session_id") or state.get("session_id") or "anon"
        bucket = int(hashlib.sha256(f"harden_pools|{_id}".encode()).hexdigest(), 16) % 100
        return "off" if bucket < pct else "on"
    except Exception:
        return "on"
```

- [ ] **Step 4: Add `harden_day_pools` no-op skeleton** justo antes de `plan_skeleton_node` (o a nivel módulo cerca de `_apply_protein_pool_scrub`). Master OFF ⇒ no-op:

```python
def harden_day_pools(skeleton: dict, form_data: dict, conditions=None) -> dict:
    """[A1-HARDEN-POOLS · 2026-07-09] Enforcer determinista de pools por día. Muta skeleton['days']
    IN-PLACE antes de que el day-generator los lea. Cada clase gateada por su knob (todas OFF por
    default). Retorna conteos para telemetría. tooltip-anchor: A1-HARDEN-POOLS"""
    counts = {"condition_removed": 0, "saltcured_removed": 0, "sameday_bound": 0, "crossday_capped": 0}
    if not HARDEN_POOLS_ENABLED:
        return counts
    days = (skeleton or {}).get("days") or []
    # (Task 2) clase 3 — filtro por condición
    # (Task 3) clase 5 — salado como main
    # (Task 4) clase 1 — binding slot→proteína
    # (Task 5) clase 2 — cuota cross-día
    return counts
```

- [ ] **Step 5: Cablear la llamada** en `plan_skeleton_node`, insertar en `graph_orchestrator.py:6455` (tras el bloque `# 4. [P1-CLINICAL-MEAL-COUNT]`, antes de `_emit_progress`):

```python
    # [A1-HARDEN-POOLS · 2026-07-09] Endurecer pools por día ANTES de que el day-gen los lea.
    try:
        _harden_counts = harden_day_pools(skeleton, form_data, _active_conditions)
        if any(_harden_counts.values()):
            logger.info(f"🔒 [A1-HARDEN-POOLS] {_harden_counts}")
    except Exception as _hp_oe:
        logger.warning(f"[A1-HARDEN-POOLS] enforcer falló (usa pools del LLM): {type(_hp_oe).__name__}: {_hp_oe}")
```
> Nota: `_active_conditions` = leer del state/form_data la lista de condiciones activas ya computada en el nodo (buscar la variable existente; si no hay, pasar `None` y que Task 2 lo derive de `form_data`).

- [ ] **Step 6: Wire cohorte al state + metric metadata.** En el nodo que arranca el pipeline (donde se escribe `_self_critique_cohort`, ~`:548`), añadir `state["_harden_pools_cohort"] = _harden_pools_canary_cohort(state)`. En el emit de `clinical_band` (`:34456`), añadir junto a `self_critique_cohort`:

```python
                            "harden_pools_cohort": final_state.get("_harden_pools_cohort") or "on",
                            # violation-rate por cohorte (señal "¿la restricción eliminó la clase?")
                            "same_day_protein_repeats": _vr_same_day,
                            "cross_day_proteins": _vr_cross_day_prot,
                            "cross_day_dishes": _vr_cross_day_dish,
```
> `_vr_*` = extraer de `build_variety_report(...)` sobre el plan final si está disponible; si no, `None`. (Reusa el report ya computado en review si existe en `final_state`.)

- [ ] **Step 7: Run tests, verify PASS** — `python -m pytest tests/test_p1_harden_pools_scaffold.py -v` → PASS.

- [ ] **Step 8: Commit** — `git -C backend add graph_orchestrator.py tests/test_p1_harden_pools_scaffold.py && git commit` (mensaje `feat(plangen): A1-HARDEN-POOLS scaffold (knobs OFF + enforcer no-op + canary cohort)`).

---

### Task 2: Clase 3 — filtro de pool por condición médica (mayor valor clínico)

**Files:**
- Modify: `graph_orchestrator.py` (`harden_day_pools`, rama clase 3)
- Test: `tests/test_p1_harden_condition_catalog.py`

**Interfaces:**
- Consumes: `condition_rules` SSOT — usar `collect_substitutions(form_data)` (o `detect_active_rules` + tablas) para obtener las filas `(tokens, replacement, label, preserve_qty)` activas. Extraer los `tokens` de identidad (los food-name; las frases de sazón como "sal al gusto" simplemente no matchean pools).
- Consumes: `_allergy_safe_fallback_protein` (fallback gracioso si un protein_pool queda vacío).

- [ ] **Step 1: Write failing test:**

```python
import graph_orchestrator as go

def _skel():
    return {"days": [{"day": 1,
        "protein_pool": ["Salami Dominicano", "Pollo", "Bacalao salado"],
        "carb_pool": ["Arroz blanco", "Arroz integral"],
        "fruit_pool": ["Toronja", "Fresa"]}]}

def test_dm2_removes_contraindicated_identities(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": ["dm2", "diabetes"]}, None)
    d = skel["days"][0]
    assert "Toronja" not in d["fruit_pool"]          # toronja→CYP3A4 fuera del pool DM2
    assert "Arroz blanco" not in d["carb_pool"]      # IG alto fuera
    assert "Fresa" in d["fruit_pool"]                # el resto intacto
    assert "Arroz integral" in d["carb_pool"]

def test_hta_removes_embutidos_and_bacalao(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": ["hipertension"]}, None)
    pool = [p.lower() for p in skel["days"][0]["protein_pool"]]
    assert not any("salami" in p for p in pool)
    assert not any("bacalao" in p for p in pool)
    assert any("pollo" in p for p in pool)           # proteína legítima sobrevive

def test_graceful_fallback_when_protein_pool_emptied(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Salami Dominicano", "Bacalao salado"],
                      "carb_pool": ["Arroz integral"], "fruit_pool": ["Fresa"]}]}
    go.harden_day_pools(skel, {"medicalConditions": ["hipertension"]}, None)
    # pool NO queda vacío → fallback allergy-safe inyectado
    assert len(skel["days"][0]["protein_pool"]) >= 1

def test_off_leaves_pool_untouched(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_CONDITION_CATALOG", False)  # clase 3 OFF
    skel = _skel()
    go.harden_day_pools(skel, {"medicalConditions": ["dm2"]}, None)
    assert "Toronja" in skel["days"][0]["fruit_pool"]
```

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/test_p1_harden_condition_catalog.py -v` → FAIL.

- [ ] **Step 3: Implement clase 3** en `harden_day_pools` (leer `condition_rules.py:594` para la firma exacta de `collect_substitutions`):

```python
    if HARDEN_CONDITION_CATALOG:
        try:
            from constants import strip_accents as _sa
            import condition_rules as _cr
            # SSOT: filas activas (tokens, replacement, label, preserve_qty) según condiciones del form.
            _subs = _cr.collect_substitutions(form_data or {}) or []
            # Tokens de identidad de alimento (los food-name; las frases de sazón no matchean pools).
            _bad_tokens = []
            for _row in _subs:
                _toks = _row[0] if isinstance(_row, (list, tuple)) else ()
                _bad_tokens.extend(_sa(str(t).lower()) for t in _toks)
            def _pool_item_ok(_item):
                _n = _sa(str(_item).lower())
                return not any(_bt and _bt in _n for _bt in _bad_tokens)
            for _d in days:
                for _field in ("protein_pool", "carb_pool", "fruit_pool"):
                    _orig = _d.get(_field) or []
                    _kept = [x for x in _orig if _pool_item_ok(x)]
                    _removed = len(_orig) - len(_kept)
                    if _removed:
                        _d[_field] = _kept
                        counts["condition_removed"] += _removed
                # fallback gracioso: protein_pool vacío tras filtrar → allergy-safe
                if not _d.get("protein_pool"):
                    _fb = _allergy_safe_fallback_protein(form_data)
                    if _fb:
                        _d["protein_pool"] = [_fb]
        except Exception as _c3e:
            logger.warning(f"[A1-HARDEN-POOLS clase3] falló (skip): {type(_c3e).__name__}: {_c3e}")
```
> `_allergy_safe_fallback_protein` es una closure LOCAL de `plan_skeleton_node` — para reusarla en `harden_day_pools` (módulo), extraerla a función módulo `_allergy_safe_fallback_protein(_fd)` y llamarla desde ambos sitios (refactor DRY, mismo comportamiento). Alternativa mínima: pasar el fallback como parámetro.

- [ ] **Step 4: Run, verify PASS** — `python -m pytest tests/test_p1_harden_condition_catalog.py -v` → PASS.

- [ ] **Step 5: Commit** — scoped (`graph_orchestrator.py` + test), mensaje `feat(plangen): A1 clase 3 — filtro de pool por condición médica (DM2/HTA/renal, OFF)`.

---

### Task 3: Clase 5 — exclusión salado-como-principal

**Files:** Modify `graph_orchestrator.py` (`harden_day_pools`, rama clase 5). Test: `tests/test_p1_harden_saltcured_main.py`

**Interfaces:** Consumes el set `_SALT_CURED_PROTEIN_TOKENS` (hoy local en `ai_helpers.py:610`) — elevarlo a constante módulo compartida (o duplicar el set con tooltip-anchor). Excluye del **protein_pool** (slot principal), universal (goal-independiente). Graceful si vacía → fallback.

- [ ] **Step 1: Write failing test:**

```python
import graph_orchestrator as go

def test_saltcured_excluded_from_protein_pool(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Salami Dominicano", "Pollo"],
                      "carb_pool": ["Arroz integral"], "fruit_pool": ["Fresa"]}]}
    go.harden_day_pools(skel, {}, None)
    pool = [p.lower() for p in skel["days"][0]["protein_pool"]]
    assert not any("bacalao" in p or "salami" in p for p in pool)
    assert any("pollo" in p for p in pool)

def test_saltcured_off_leaves_pool(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", False)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Pollo"],
                      "carb_pool": [], "fruit_pool": []}]}
    go.harden_day_pools(skel, {}, None)
    assert any("bacalao" in p.lower() for p in skel["days"][0]["protein_pool"])

def test_saltcured_graceful_when_all_salt(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skel = {"days": [{"day": 1, "protein_pool": ["Bacalao salado", "Salami Dominicano"],
                      "carb_pool": [], "fruit_pool": []}]}
    go.harden_day_pools(skel, {}, None)
    assert len(skel["days"][0]["protein_pool"]) >= 1  # fallback, nunca vacío
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement clase 5:**

```python
    if HARDEN_SALTCURED_MAIN:
        from constants import strip_accents as _sa2
        _SALT_CURED_NEVER_MAIN = ("bacalao", "arenque", "salami", "salchichon", "pepperoni",
                                  "mortadela", "tocino", "panceta", "longaniza", "chorizo",
                                  "salchicha", "embutido", "jamon")  # tooltip-anchor: A1-SALTCURED-NEVER-MAIN
        for _d in days:
            _orig = _d.get("protein_pool") or []
            _kept = [p for p in _orig if not any(t in _sa2(str(p).lower()) for t in _SALT_CURED_NEVER_MAIN)]
            _removed = len(_orig) - len(_kept)
            if _removed:
                counts["saltcured_removed"] += _removed
                _d["protein_pool"] = _kept if _kept else [_allergy_safe_fallback_protein(form_data) or "Pollo"]
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** — `feat(plangen): A1 clase 5 — exclusión salado-como-principal (OFF)`.

---

### Task 4: Clase 1 — binding determinista slot→proteína (mismo día)

**Files:** Modify `graph_orchestrator.py` (`harden_day_pools`, rama clase 1). Test: `tests/test_p1_harden_sameday_protein.py`

**Interfaces:** Anota en cada día `_bound_main_proteins: list` (1 proteína pesada distinta por slot principal) que `_apply_protein_pool_scrub` / day-gen prompt respetan; o restringe el `protein_pool` a ≥ N proteínas pesadas distintas cuando hay ≥2 slots principales. Carve-outs: legumbre/yogurt repetibles (reusar `LEGUME_NAMES`); `meal_types` ≥5-6 → skip (espeja `_relax_high_mc`). **Este task es más involucrado** — leer cómo el day-gen consume el pool (`day_generator.py:405-568`) antes de implementar; puede requerir un post-gen enforcement adicional en `_apply_protein_pool_scrub`.

- [ ] Steps: test-first (día con 1 sola pesada + 2 slots main → asserta que se añade/marca una 2ª pesada distinta; carve-out ≥5 comidas → sin binding; OFF → sin cambio) → implementar → PASS → commit.

---

### Task 5: Clase 2 — cuota cross-día round-robin

**Files:** Modify `graph_orchestrator.py` (`harden_day_pools`, rama clase 2). Test: `tests/test_p1_harden_crossday_quota.py`

**Interfaces:** Generaliza el cap max-1-día de `_SKELETON_RESTRICTED` a TODAS las pesadas: `quota = ceil(num_days / distinct_available_proteins)`; partición round-robin sobre `skel_days` ANTES del day-gen. Cuota se ensancha si el catálogo es chico (graceful). También sobre `_head_dish_base_token`.

- [ ] Steps: test-first (7 días, 3 proteínas distintas → ninguna en >3 días; catálogo de 1 proteína → no falla, cuota=7) → implementar → PASS → commit.

---

### Task 6: Canary metric wiring + bump marker

**Files:** Modify `graph_orchestrator.py` (`:34446` metadata, si no se hizo en Task 1 Step 6), `app.py:32` (`_LAST_KNOWN_PFIX`). Test: `tests/test_a1_harden_pools_marker.py`

- [ ] Verificar que la metadata `clinical_band` emite `harden_pools_cohort` + violation-rate.
- [ ] Bump `_LAST_KNOWN_PFIX = "A1-HARDEN-POOLS · 2026-07-09"`.
- [ ] Test cross-link: el slug `a1_harden_pools` matchea ≥1 `tests/test_a1_harden_pools*.py` + freshness del marker.
- [ ] Commit.

---

## Follow-up plans (subsistemas independientes — spec §3/§4)

- **Fase 0 completa** (scorecard offline + medidor de varianza K-corridas): plan aparte `2026-07-XX-plangen-fase0-meter.md`. Task 1/6 de A1 solo añade la instrumentación mínima de cohorte.
- **Swap-repair reactivo** (`MEALFIT_POOL_SWAP_REPAIR`): plan aparte — repara canasta inviable en ensamblaje (intercambio hermano del pool + re-narra 1 comida).
- **Fase B** (apagar self_critique): owner-gated (canario en vivo + firma umbral). No se implementa aquí.

## Self-Review

- **Cobertura del spec:** clases 3,5,1,2 → Tasks 2-5; canario → Task 1 Step 6 + Task 6; nace-OFF → Task 1; fallback gracioso → Tasks 2/3; never-retire → Global Constraints (no se toca ningún backstop en estos tasks). Fase 0 completa + swap-repair + Fase B → follow-ups (subsistemas). ✓
- **Placeholders:** Tasks 1-3 tienen código + tests completos. Tasks 4-5 son más involucrados y llevan interfaces + test-shape precisos pero el código exacto se escribe leyendo `day_generator.py` en ejecución (marcado explícitamente — no es un placeholder oculto sino una dependencia de lectura declarada). ✓
- **Consistencia de tipos:** `harden_day_pools(skeleton, form_data, conditions) -> dict` de conteos; cohorte `-> str`; knobs bool/int — consistentes entre tasks. ✓
