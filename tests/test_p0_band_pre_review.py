"""[P0-BAND-PRE-REVIEW · 2026-07-10] El chain de calidad/banda del shield pre-INSERT
corre también (1) al FINAL de assemble_plan_node ANTES de construir la lista de compras
y del review, y (2) en el merge T1 del chunk worker (semanas 2+).

Root cause (renovación en vivo corr=2451c8ac, 2026-07-10 11:28-11:33 UTC):
  - El gate de banda del review y el P2-BAND-SCORE-GATE miden el estado PRE-shield:
    el shield del INSERT levantaba la banda 0.583→1.00 DESPUÉS de que el payload SSE
    ya viajó con `_quality_degraded` (reason=low_band_macro:kcal) → el usuario ve
    "La IA no logró un plan óptimo... precisión de las calorías" sobre un plan cuya
    copia persistida quedó 12/12 en banda. Recurrente (reporte del owner).
  - Los rechazos por banda disparan retries que escalan day_generator a PRO
    (5× precio): $1.57 de $2.90 del gasto de 48h era day-gen PRO (forensic
    llm_usage_events 2026-07-10). Cerrar la banda ANTES del review mata la causa
    del retry Y del banner con el MISMO chain ya testeado del shield.
  - Chunk T1 (semanas 2+) mergeaba vía UPDATE con solo fpc → los 6 pases nuevos del
    roadmap de plausibilidad no corrían para ~7/8 de los días de un plan mensual.

Diseño: `db_plans.apply_plan_quality_finalize_chain(plan_data)` = adapter público que
delega en `_finalize_plan_data_for_insert({"plan_data": plan_data})` (SSOT intacto,
cero duplicación del orden de pases). El shield además auto-deriva main_goal +
target_macros para fpc (paridad gainmuscle-refill con el chunk, que antes los pasaba
a mano). Surfaces: assemble tail (pre-shopping/pre-review), chunk T1, INSERT (net).

tooltip-anchor: P0-BAND-PRE-REVIEW
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_DBP = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CT = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Adapter público en db_plans + export vía fachada db
# ---------------------------------------------------------------------------

def test_adapter_defined_and_delegates_to_shield():
    i = _DBP.find("def apply_plan_quality_finalize_chain(")
    assert i > 0, (
        "P0-BAND-PRE-REVIEW: falta `apply_plan_quality_finalize_chain` en db_plans.py. "
        "Es el adapter público que assemble/chunk usan para correr el MISMO chain del "
        "shield pre-INSERT (SSOT). Sin él, el orden de pases se duplicaría en 3 sitios."
    )
    body = _DBP[i: i + 2500]
    assert "_finalize_plan_data_for_insert(" in body, (
        "P0-BAND-PRE-REVIEW: el adapter debe DELEGAR en _finalize_plan_data_for_insert "
        "(SSOT del orden de pases) — no reimplementar la cadena."
    )
    assert '"plan_data"' in body or "'plan_data'" in body, (
        "P0-BAND-PRE-REVIEW: el adapter envuelve el plan en {'plan_data': plan_data} "
        "porque el shield opera sobre el dict de INSERT."
    )


def test_adapter_exported_via_db_facade():
    import sys
    sys.path.insert(0, str(_BACKEND))
    from db import apply_plan_quality_finalize_chain  # noqa: F401
    import db_plans
    assert callable(db_plans.apply_plan_quality_finalize_chain)


# ---------------------------------------------------------------------------
# 2. El shield auto-deriva main_goal + target_macros para fpc (paridad chunk)
# ---------------------------------------------------------------------------

def test_shield_fpc_self_derives_goal_and_target_macros():
    i = _DBP.find("def _finalize_plan_data_for_insert(")
    assert i > 0
    body = _DBP[i: _DBP.find("\ndef ", i + 10)]
    assert "main_goal" in body and "target_macros" in body, (
        "P0-BAND-PRE-REVIEW: el shield debe derivar main_goal + target_macros del "
        "propio plan_data y pasarlos a finalize_plan_data_coherence — sin esto, el "
        "gainmuscle-refill (P1-CHUNK-GAINMUSCLE-PARITY) se pierde cuando el chunk "
        "delega en el adapter, y los paths que saltan assemble nunca lo tuvieron."
    )


# ---------------------------------------------------------------------------
# 3. Assemble tail: chain ANTES de la lista de compras (y por ende del guard/review)
# ---------------------------------------------------------------------------

def test_assemble_runs_chain_before_shopping_build():
    i_asm = _GO.find("async def assemble_plan_node(")
    assert i_asm > 0
    asm = _GO[i_asm: i_asm + 200_000]
    i_chain = asm.find("apply_plan_quality_finalize_chain")
    assert i_chain > 0, (
        "P0-BAND-PRE-REVIEW regresión: assemble_plan_node ya no invoca el chain de "
        "calidad/banda pre-review. Sin él: (a) el gate de banda del review vuelve a "
        "medir estado sin cerrar → retries por banda (que escalan day-gen a PRO, el "
        "driver #1 del gasto), (b) el payload SSE viaja con _quality_degraded stale "
        "→ banner 'precisión de calorías' falso (dolor recurrente del owner)."
    )
    i_shop = asm.find("# Calcular shopping lists")
    assert i_shop > 0, "ancla '# Calcular shopping lists' desapareció de assemble"
    assert i_chain < i_shop, (
        "P0-BAND-PRE-REVIEW: el chain debe correr ANTES de construir la lista de "
        "compras — el closer de banda muta cantidades; si la lista se construye "
        "primero, el coherence guard mide magnitudes divergentes (falso block)."
    )
    i_phantom = asm.find("PHANTOM-PROTEIN-NAMEFIX")
    if i_phantom > 0:
        assert i_chain > i_phantom, (
            "P0-BAND-PRE-REVIEW: el chain debe correr DESPUÉS de la última mutación "
            "de assemble (phantom-namefix) para barrer el estado final."
        )


def test_assemble_sameday_reintro_telemetry_after_chain():
    i_asm = _GO.find("async def assemble_plan_node(")
    asm = _GO[i_asm: i_asm + 200_000]
    i_chain = asm.find("apply_plan_quality_finalize_chain")
    i_tele = asm.find("_days_with_same_day_protein_repeat", i_chain)
    i_shop = asm.find("# Calcular shopping lists")
    assert 0 < i_chain < i_tele < i_shop, (
        "P1-SAMEDAY-REINTRO-TELEMETRY: falta la telemetría warn de proteína repetida "
        "same-day sobre el estado FINAL de assemble (post-chain, pre-shopping). Es la "
        "evidencia para detectar futuros pases que reintroduzcan repeats después del "
        "autofix (hoy: closer no-dup-cheese eligió 'Huevo' 3s después del autofix)."
    )


# ---------------------------------------------------------------------------
# 4. Chunk T1: chain completo (no solo fpc) en el merge de semanas 2+
# ---------------------------------------------------------------------------

def test_chunk_t1_merge_runs_full_chain():
    i_merge = _CT.find("Merge normal: primera vez")
    assert i_merge > 0, "ancla del merge T1 desapareció de cron_tasks"
    region = _CT[i_merge: i_merge + 8_000]
    assert "apply_plan_quality_finalize_chain" in region, (
        "P0-BAND-PRE-REVIEW regresión: el merge T1 del chunk ya no corre el chain "
        "completo. Antes solo corría fpc → los pases del shield (protein-step-parity, "
        "polish-refire, cap condimentos, bigfruit, count-agreement, banda all-4, "
        "detectores pairing/batch) NO cubrían las semanas 2+ (~7/8 de los días de un "
        "plan mensual). El chain es idempotente y corre bajo el mismo lock del merge."
    )


# ---------------------------------------------------------------------------
# 5. Funcional ligero: el adapter muta in-place y es fail-safe sin DB
# ---------------------------------------------------------------------------

def test_adapter_mutates_plan_in_place_and_survives_without_db():
    import sys
    sys.path.insert(0, str(_BACKEND))
    from db import apply_plan_quality_finalize_chain

    plan = {
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "meal": "Almuerzo",
                        "name": "Pollo guisado con arroz",
                        "protein": 40, "carbs": 50, "fats": 15, "cals": 495,
                        "ingredients": ["150g de pechuga de pollo", "1 taza de arroz blanco"],
                        "ingredients_raw": ["150g de pechuga de pollo", "1 taza de arroz blanco"],
                        "recipe": ["MISE EN PLACE: Corta el pollo.",
                                   "EL TOQUE DE FUEGO: Guisa el pollo 20 min y hierve el arroz.",
                                   "MONTAJE: Sirve caliente."],
                    }
                ],
            }
        ],
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": "2000 kcal",
        "main_goal": "gain_muscle",
    }
    days_ref = plan["days"]
    apply_plan_quality_finalize_chain(plan)  # sin DB viva: cada pase interno es fail-safe
    assert plan["days"] is days_ref, "el chain debe mutar el MISMO objeto (in-place)"
    assert plan.get("grocery_start_date"), (
        "el chain incluye _ensure_grocery_start_date (idempotente; respeta valor existente)"
    )
    marker = plan["grocery_start_date"]
    apply_plan_quality_finalize_chain(plan)
    assert plan["grocery_start_date"] == marker, "re-ejecución debe ser idempotente"
