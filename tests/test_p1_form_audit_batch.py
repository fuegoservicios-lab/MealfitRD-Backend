"""[P1-FORM-AUDIT-BATCH · 2026-07-03] Regresión del batch de cierre del audit
del formulario (19 pasos) — "¿100% listo para producción y sin contradicciones?".

Cubre los 5 fixes backend + los anclajes frontend:

  C1 — `_allergy_safe_fallback_protein`: el scrub del skeleton inyectaba
       'Lentejas' hardcodeado en pools vacíos, ignorando alergia a leguminosas
       declarada. Ahora escala una escalera de candidatos filtrada por
       alergias + dieta.
  C2 — `habitAlcohol` × condición: la señal del wizard QHabits llegaba al
       prompt solo como JSON pasivo. Ahora hta/gout/gastritis/nafld/dm2/
       pregnancy + consumo recurrente (semanal/diario) → directiva explícita.
  C3 — `targetWeight` sin validación de dirección/rango: lose_fat con meta
       MAYOR al peso actual pasaba silencioso. + `goalPace` enum estricto.
  C4 — freetext other*/motivation sin cap de longitud → truncado 600 chars.
  C5 — `householdComposition` sin validación: {adults:500,children:500} se
       clampaba silencioso a 20 en compute_household_multiplier.
  F1 — `clinical_profile` en SENSITIVE_FIELDS (labs/cirugías iban PLAINTEXT
       a localStorage y sobrevivían logout) + purge defensivo en load.
  F2 — hidratación DB→form: booleans default-false (targetWeightAuto,
       includeSupplements) y budgetCurrency con default 'DOP' cualifican
       vía `_hydrateFieldQualifies` (antes: false ≠ vacío → nunca hidrataban).
"""
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND = BACKEND.parent / "frontend"

GO_SRC = (BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
PLANS_SRC = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
COND_SRC = (BACKEND / "condition_rules.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C1 — fallback de proteína allergy-aware en el scrub del skeleton
# ---------------------------------------------------------------------------

class TestC1AllergySafeFallbackProtein:
    def test_helper_exists_and_callsite_uses_it(self):
        assert "_allergy_safe_fallback_protein" in GO_SRC, (
            "El helper _allergy_safe_fallback_protein desapareció de "
            "graph_orchestrator.py — el scrub del skeleton vuelve a inyectar "
            "proteína hardcodeada ignorando alergias (P1-FORM-AUDIT-BATCH C1)."
        )
        assert "_allergy_safe_fallback_protein(form_data)" in GO_SRC, (
            "El call site del fallback ya no pasa form_data — sin las alergias "
            "el candidato no puede filtrarse."
        )

    def test_no_hardcoded_lentejas_injection_in_scrub(self):
        # El patrón viejo: d['protein_pool'] = ['Lentejas'] literal (sin helper).
        bad = re.findall(
            r"d\[['\"]protein_pool['\"]\]\s*=\s*\[['\"]Lentejas['\"]\]", GO_SRC
        )
        assert not bad, (
            "Reapareció la inyección hardcodeada de 'Lentejas' en protein_pool "
            "— usar _allergy_safe_fallback_protein(form_data) (C1)."
        )

    def test_ladder_behavior(self):
        """Ejecuta la escalera aislada (mismo algoritmo) contra los casos clave."""
        m = re.search(
            r"def _allergy_safe_fallback_protein\(_fd.*?(?=\n    [a-zA-Z_#]|\n\S)",
            GO_SRC, re.DOTALL,
        )
        assert m, "No pude extraer el cuerpo de _allergy_safe_fallback_protein"
        body = m.group(0)
        # Anclas de diseño: vegan excluye Huevos y Pollo; vegetarian excluye Pollo.
        assert "_is_vegan" in body and "_is_veget" in body
        assert '"Huevos"' in body and '"Pollo"' in body
        assert '"Lentejas"' in body and '"Garbanzos"' in body
        # Filtro por alergias de ambos campos:
        assert '"allergies"' in body and '"otherAllergies"' in body

    def test_ladder_functional(self):
        """Réplica funcional 1:1 del helper via exec del cuerpo extraído."""
        # Aislar la función anidada y ejecutarla con un stub de strip_accents.
        m = re.search(
            r"(def _allergy_safe_fallback_protein\(_fd.*?\n)(?=\n?    [a-zA-Z_#@]|\nasync |\ndef )",
            GO_SRC, re.DOTALL,
        )
        assert m, "No pude extraer la función para exec"
        src = m.group(1)
        # des-indentar (está anidada a 4 espacios)
        src = "\n".join(
            line[4:] if line.startswith("    ") else line
            for line in src.splitlines()
        )
        ns = {}
        import sys
        import types
        fake_constants = types.ModuleType("constants")
        fake_constants.strip_accents = lambda s: (
            s.replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        )
        had = "constants" in sys.modules
        old = sys.modules.get("constants")
        sys.modules["constants"] = fake_constants
        try:
            exec(src, ns)
            fn = ns["_allergy_safe_fallback_protein"]
            # Sin alergias → primer peldaño (Lentejas)
            assert fn({}) == "Lentejas"
            # Alergia a leguminosas → salta lentejas/garbanzos/habichuelas → Quinoa
            assert fn({"allergies": ["Legumbres"]}) == "Quinoa"
            # Legumbre + quinoa → Huevos (no vegan)
            assert fn({"allergies": ["legumbres", "quinoa"]}) == "Huevos"
            # Legumbre + quinoa + huevo → Pollo (dieta sin restricción)
            assert fn({"allergies": ["legumbre", "quinoa", "huevo"]}) == "Pollo"
            # Igual pero vegano → None (ningún candidato seguro)
            assert fn({"allergies": ["legumbre", "quinoa"], "dietType": "vegan"}) is None
            # Vegetariano con todo vegetal alérgico → Huevos OK, Pollo NO
            assert fn({"allergies": ["legumbre", "quinoa"], "dietType": "vegetarian"}) == "Huevos"
            # otherAllergies también filtra
            assert fn({"otherAllergies": "alergia a las lentejas y garbanzos"}) == "Habichuelas rojas"
        finally:
            if had:
                sys.modules["constants"] = old
            else:
                sys.modules.pop("constants", None)


# ---------------------------------------------------------------------------
# C2 — habitAlcohol × condición → directiva determinista
# ---------------------------------------------------------------------------

class TestC2AlcoholConditionCross:
    def _prompt(self, **kw):
        import condition_rules as cr
        base = {"medicalConditions": kw.pop("conds", [])}
        base.update(kw)
        return cr.build_condition_prompt(base)

    def test_recurrent_alcohol_plus_hta_emits_directive(self):
        p = self._prompt(conds=["Hipertensión"], habitAlcohol="diario")
        assert "HÁBITO DECLARADO" in p and "ALCOHOL DIARIO" in p

    def test_semanal_also_triggers(self):
        p = self._prompt(conds=["Gota"], habitAlcohol="semanal")
        assert "HÁBITO DECLARADO" in p

    def test_ocasional_does_not_trigger(self):
        p = self._prompt(conds=["Hipertensión"], habitAlcohol="ocasional")
        assert "HÁBITO DECLARADO" not in p

    def test_nunca_and_absent_do_not_trigger(self):
        assert "HÁBITO DECLARADO" not in self._prompt(
            conds=["Hipertensión"], habitAlcohol="nunca")
        assert "HÁBITO DECLARADO" not in self._prompt(conds=["Hipertensión"])

    def test_non_sensitive_condition_does_not_trigger(self):
        # Hipotiroidismo no está en el set alcohol-sensitive
        p = self._prompt(conds=["Hipotiroidismo"], habitAlcohol="diario")
        assert "HÁBITO DECLARADO" not in p

    def test_no_condition_no_block(self):
        # Sin condición activa build_condition_prompt retorna "" (early return)
        assert self._prompt(conds=[], habitAlcohol="diario") == ""

    def test_sensitive_set_contents(self):
        import condition_rules as cr
        assert cr._ALCOHOL_SENSITIVE_RULE_IDS == {
            "hta", "gout", "gastritis", "nafld", "dm2", "pregnancy"}
        assert cr._ALCOHOL_RECURRENT_VALUES == {"semanal", "diario"}


# ---------------------------------------------------------------------------
# C3 — targetWeight dirección/rango + goalPace enum (422)
# ---------------------------------------------------------------------------

def _validate(data):
    from routers.plans import _validate_form_data_ranges
    base = {
        "age": 30, "height": 175, "weight": 180, "weightUnit": "lb",
        "activityLevel": "moderate", "mainGoal": "lose_fat",
        "householdSize": 2,
    }
    base.update(data)
    ok, errors = _validate_form_data_ranges(base)
    return ok, errors, base


class TestC3TargetWeightGoalPace:
    def test_lose_fat_with_higher_target_rejected(self):
        ok, errors, _ = _validate({"targetWeight": "250"})  # 250 lb > 180 lb
        assert not ok
        assert any(e["field"] == "targetWeight" for e in errors)

    def test_gain_muscle_with_lower_target_rejected(self):
        ok, errors, _ = _validate({"mainGoal": "gain_muscle", "targetWeight": "150"})
        assert not ok
        assert any(e["field"] == "targetWeight" for e in errors)

    def test_coherent_target_passes(self):
        ok, errors, _ = _validate({"targetWeight": "160"})
        assert ok, errors

    def test_equal_target_allowed(self):
        # Igualdad exacta NO se rechaza (borderline legítimo, ETA no emite)
        ok, errors, _ = _validate({"targetWeight": "180"})
        assert ok, errors

    def test_target_weight_auto_skips_validation(self):
        ok, errors, _ = _validate({"targetWeight": "250", "targetWeightAuto": True})
        assert ok, errors

    def test_out_of_range_target_rejected(self):
        ok, errors, _ = _validate({"targetWeight": "20"})  # 20 lb ≈ 9 kg < 30
        assert not ok
        assert any(e["field"] == "targetWeight" for e in errors)

    def test_garbage_target_rejected(self):
        ok, errors, _ = _validate({"targetWeight": "abc"})
        assert not ok
        assert any(e["field"] == "targetWeight" for e in errors)

    def test_maintenance_with_target_is_noop(self):
        ok, errors, _ = _validate({"mainGoal": "maintenance", "targetWeight": "250"})
        # dirección solo aplica a lose_fat/gain_muscle; rango sí (250lb=113kg OK)
        assert ok, errors

    def test_goal_pace_valid_values_pass(self):
        for pace in ("gradual", "moderado", "decidido"):
            ok, errors, _ = _validate({"goalPace": pace})
            assert ok, (pace, errors)

    def test_goal_pace_invalid_rejected(self):
        ok, errors, _ = _validate({"goalPace": "yolo"})
        assert not ok
        assert any(e["field"] == "goalPace" for e in errors)

    def test_goal_pace_absent_or_empty_passes(self):
        ok, _, _ = _validate({})
        assert ok
        ok, _, _ = _validate({"goalPace": ""})
        assert ok

    def test_goal_pace_enum_parity_with_pace_adjustments(self):
        """El enum del router DEBE ser subconjunto=igual de las keys reales."""
        from routers.plans import _GOAL_PACE_ENUM
        import nutrition_calculator as nc
        real = set()
        for goal_map in nc.PACE_ADJUSTMENTS.values():
            real.update(goal_map.keys())
        assert _GOAL_PACE_ENUM == real, (
            f"Drift enum goalPace: router={_GOAL_PACE_ENUM} vs "
            f"PACE_ADJUSTMENTS={real} — sincronizar ambos lados."
        )


# ---------------------------------------------------------------------------
# C4 — caps de longitud freetext (truncado in-place, NO 422)
# ---------------------------------------------------------------------------

class TestC4FreetextCaps:
    def test_long_freetext_truncated_not_rejected(self):
        long_text = "x" * 5000
        ok, errors, base = _validate({"otherConditions": long_text})
        assert ok, errors  # truncado, NO 422
        from routers.plans import _FREETEXT_MAX_CHARS
        assert len(base["otherConditions"]) == _FREETEXT_MAX_CHARS

    def test_all_declared_fields_capped(self):
        from routers.plans import _FREETEXT_CAP_FIELDS, _FREETEXT_MAX_CHARS
        expected = {"otherConditions", "otherAllergies", "otherMedications",
                    "otherDislikes", "otherStruggles", "motivation"}
        assert set(_FREETEXT_CAP_FIELDS) == expected
        payload = {f: "y" * (_FREETEXT_MAX_CHARS + 100) for f in expected}
        ok, errors, base = _validate(payload)
        assert ok, errors
        for f in expected:
            assert len(base[f]) == _FREETEXT_MAX_CHARS, f

    def test_short_freetext_untouched(self):
        ok, _, base = _validate({"otherConditions": "gastritis leve"})
        assert ok
        assert base["otherConditions"] == "gastritis leve"

    def test_non_string_freetext_ignored(self):
        # lista/None no revienta el cap (isinstance str gate)
        ok, _, _ = _validate({"otherConditions": ["no", "string"]})
        assert ok


# ---------------------------------------------------------------------------
# C5 — householdComposition rango
# ---------------------------------------------------------------------------

class TestC5HouseholdComposition:
    def test_valid_composition_passes(self):
        ok, errors, _ = _validate({"householdComposition": {"adults": 2, "children": 3}})
        assert ok, errors

    def test_absurd_composition_rejected(self):
        ok, errors, _ = _validate({"householdComposition": {"adults": 500, "children": 500}})
        assert not ok
        assert any(e["field"] == "householdComposition" for e in errors)

    def test_zero_total_rejected(self):
        ok, errors, _ = _validate({"householdComposition": {"adults": 0, "children": 0}})
        assert not ok

    def test_garbage_values_rejected(self):
        ok, errors, _ = _validate({"householdComposition": {"adults": "abc", "children": 1}})
        assert not ok

    def test_non_dict_rejected(self):
        ok, errors, _ = _validate({"householdComposition": [2, 3]})
        assert not ok

    def test_absent_passes(self):
        ok, _, _ = _validate({})
        assert ok


# ---------------------------------------------------------------------------
# F1 — clinical_profile cifrado en storage (parser frontend)
# ---------------------------------------------------------------------------

class TestF1ClinicalProfileSensitive:
    def test_clinical_profile_in_sensitive_fields(self):
        src = (FRONTEND / "src" / "config" / "secureFormStorage.js").read_text(
            encoding="utf-8")
        # `];` como cierre (un `]` suelto dentro de comentarios `[Pn-...]` del
        # array rompería el non-greedy simple).
        m = re.search(r"SENSITIVE_FIELDS\s*=\s*\[(.*?)\];", src, re.DOTALL)
        assert m, "No encontré SENSITIVE_FIELDS en secureFormStorage.js"
        assert "'clinical_profile'" in m.group(1) or '"clinical_profile"' in m.group(1), (
            "clinical_profile salió de SENSITIVE_FIELDS — labs/cirugías "
            "volverían a localStorage en PLAINTEXT (P1-FORM-AUDIT-BATCH F1)."
        )

    def test_load_purges_sensitive_from_public_blob(self):
        src = (FRONTEND / "src" / "config" / "secureFormStorage.js").read_text(
            encoding="utf-8")
        assert "delete publicData[" in src, (
            "El purge defensivo del blob público desapareció del load — un blob "
            "legacy con campos sensibles en plaintext quedaría vivo para siempre."
        )


# ---------------------------------------------------------------------------
# F2 — hidratación booleans default-false + budgetCurrency
# ---------------------------------------------------------------------------

class TestF2HydrationQualifier:
    def test_hydrate_helper_exists_and_used_in_both_loops(self):
        src = (FRONTEND / "src" / "context" / "AssessmentContext.jsx").read_text(
            encoding="utf-8")
        assert "_hydrateFieldQualifies" in src
        # ambos loops de hidratación (fetchProfile + refreshProfileAndPlan)
        # 2 call sites (la definición usa `= (k, ...)` con espacio, no cuenta)
        assert src.count("_hydrateFieldQualifies(") >= 2, (
            "El helper _hydrateFieldQualifies debe invocarse en AMBOS loops de "
            "hidratación (fetchProfile y refreshProfileAndPlan)."
        )
        assert "targetWeightAuto" in src and "includeSupplements" in src

    def test_guest_sensitive_carryover_wired(self):
        src = (FRONTEND / "src" / "context" / "AssessmentContext.jsx").read_text(
            encoding="utf-8")
        assert "guestSensitiveCarryoverRef" in src
        assert "_applyGuestSensitiveCarryover" in src


# ---------------------------------------------------------------------------
# Marker bump
# ---------------------------------------------------------------------------

def test_marker_bumped():
    app_src = (BACKEND / "app.py").read_text(encoding="utf-8")
    assert 'P1-FORM-AUDIT-BATCH' in app_src.split("_LAST_KNOWN_PFIX = ")[1].split("\n")[0]
