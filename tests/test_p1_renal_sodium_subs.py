"""[P1-RENAL-SODIUM-SUBS · 2026-06-19] (audit fresco P1-4) ERC enforza el sodio, no solo en el prompt.

Bug (audit 2026-06-19): la condición renal modelaba la proteína de forma determinista (cap KDIGO +
Guards 3.6/4d) pero la restricción de SODIO quedaba SOLO en el `prompt_block` — la fila renal NO tenía
`substitutions`, a diferencia de HTA (que sí swapea embutidos/cubitos/bacalao→fresco). Un perfil ERC-puro
(sin HTA; los terms son disjuntos) recibía los ofensores de sodio salvo que el LLM los quitara solo.

Fix: la fila renal reusa `_HTA_SODIUM_SUBS`/`_HTA_SODIUM_NEGATIVES` (enforcement determinista vía Guard 3);
+ `_has_renal` en el panel de micros surface un condition_target de sodio (paridad de display con HTA);
+ el degrade-de-panel keya `("sodium_mg","renal","techo")` (paridad de observabilidad con HTA).
"""
from __future__ import annotations

from pathlib import Path

import condition_rules as cr
import micronutrients as mn


# ── A. Enforcement determinista: la fila renal lleva los swaps de sodio ──
def test_renal_rule_carries_hta_sodium_subs():
    renal = next(r for r in cr.CONDITION_RULES if r.id == "renal")
    assert renal.substitutions == cr._HTA_SODIUM_SUBS, "la fila renal debe reusar la tabla de sodio de HTA"
    assert renal.sub_negatives == cr._HTA_SODIUM_NEGATIVES


def test_collect_substitutions_includes_renal_sodium_for_pure_erc():
    # Perfil ERC-puro (sin HTA): antes NO aportaba ningún swap de sodio.
    subs = cr.collect_substitutions({"medicalConditions": ["Enfermedad renal crónica"]})
    renal_subs = [s for s in subs if s["condition"] == "renal"]
    assert renal_subs, "ERC-puro debe aportar sustituciones de sodio (condition='renal')"
    flat_tokens = [t for s in renal_subs for t in s["tokens"]]
    assert any("embutido" in t for t in flat_tokens)
    assert any("cubito" in t for t in flat_tokens)
    assert any("bacalao" in t for t in flat_tokens)


# ── B. Panel: _has_renal + condition_target de sodio (paridad con HTA) ──
def test_has_renal_helper():
    assert mn._has_renal(["enfermedad renal"]) is True
    assert mn._has_renal(["erc", "nefropatia"]) is True
    assert mn._has_renal(["hipertension"]) is False
    assert mn._has_renal([]) is False


class _StubMicroDB:
    def micros_from_ingredient_string(self, s):
        return {}


def test_renal_condition_target_surfaced_in_panel():
    plan = {"days": [{"meals": [{"ingredients": ["100g de arroz", "150g de pollo"]}]}]}
    report = mn.build_micronutrient_report(plan, _StubMicroDB(), sex="male",
                                           conditions=["enfermedad renal"])
    cts = report.get("condition_targets", [])
    assert any(ct.get("condicion") == "Enfermedad renal crónica" for ct in cts), \
        "el panel debe surfacear un condition_target renal de sodio"
    renal_ct = next(ct for ct in cts if ct["condicion"] == "Enfermedad renal crónica")
    assert "Sodio" in renal_ct["regla"]


def test_no_renal_no_renal_target():
    plan = {"days": [{"meals": [{"ingredients": ["100g de arroz"]}]}]}
    report = mn.build_micronutrient_report(plan, _StubMicroDB(), sex="male", conditions=["hipertension"])
    cts = report.get("condition_targets", [])
    assert not any(ct.get("condicion") == "Enfermedad renal crónica" for ct in cts)


# ── C. Parser-anchor: el degrade-de-panel keya sodio→renal ──
def test_degrade_checks_include_renal_sodium():
    src = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '("sodium_mg", "renal", "techo")' in src, \
        "el tuple _checks de _maybe_mark_panel_degraded debe keyar sodio→renal (paridad con HTA)"


# ── D. Comorbilidad renal+HTA: el swap de sodio se aplica UNA sola vez (idempotente, sin doble-conteo) ──
def test_renal_hta_sodium_swap_applied_once():
    # collect_substitutions DUPLICA _HTA_SODIUM_SUBS (una copia por regla activa: renal + hta)…
    subs = cr.collect_substitutions({"medicalConditions": ["Enfermedad renal", "Hipertensión"]})
    embutido_subs = [s for s in subs if any("longaniza" in t for t in s["tokens"])]
    assert len(embutido_subs) >= 2, "renal+HTA debe aportar el sub de embutidos por AMBAS reglas (duplicado)"
    # …pero el motor first-match-wins lo aplica una sola vez (tras el swap el token ya no re-matchea).
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"name": "Almuerzo", "ingredients": ["100g de longaniza", "100g de arroz"]}]}]}
    n = go._apply_condition_substitutions(plan, {"medicalConditions": ["Enfermedad renal", "Hipertensión"]})
    assert n >= 1
    ings = plan["days"][0]["meals"][0]["ingredients"]
    joined = " ".join(ings).lower()
    assert "pechuga de pollo" in joined, "el embutido debe haberse swapeado a Pechuga de pollo"
    assert "longaniza" not in joined, "tras el swap no debe quedar el token de embutido (idempotente)"


# ── E. Renal+vegano: el redirect diet-aware lleva el reemplazo a Lentejas (comportamiento anclado, ver P2) ──
def test_renal_vegan_redirects_to_lentejas():
    subs = cr.collect_substitutions({"medicalConditions": ["Enfermedad renal"]}, diet_type="vegano")
    embutido = next(s for s in subs if s["condition"] == "renal" and any("longaniza" in t for t in s["tokens"]))
    assert embutido["replacement"] == "Lentejas", \
        "renal+vegano redirige el reemplazo animal a Lentejas (trade-off K/P documentado, gate nefrólogo)"
