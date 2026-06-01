"""[P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Regression guard del cap-aware
coherence guard. Los caps recortan magnitudes intencionalmente por storage
realism; el guard DEBE excluir esas divergencias de su counter.

Bug observado en test E2E del 2026-05-15 21:51:54 (plan_id=ae29c7a9):
  - `🛒 [COH-GUARD/warn] 61 divergencias (presence=1, magnitude=60,
    multiplier=1.0). Hipótesis: {'cap_swallowed_modifier': 1, 'unknown': 32,
    'unit_mismatch': 27, 'yield_uncovered': 1}.`
  - Caps aplicados durante ese mismo run (logs aggregator):
    [P3-HERB-CAP] Cilantro 933g→100g
    [P5-VEG-CAP] Guineo 11200g→1920g, Auyama 6300g→4400g
    [P6-LEGUMES-DRY-CAP] Gandules 4667g→1814g
    [P6-EGGS-AGGREGATE-CAP] Huevo 10→2 cartones
    [P6-LACTEOS-PERISHABLE-CAP] Yogurt 5717g→2722g
    [P6-SPICE-CAP] Canela 48g→28g
  - Esas 6 magnitudes están entre las "60 divergencias críticas" reportadas.
  - Mode=warn evitaba bloquear el plan, pero documentaba el gap.

Root cause: `run_shopping_coherence_guard` comparaba `expected_sum_from_recipes`
(suma de recetas pre-cap) vs `aggregated_shopping_list` (post-cap). Las
magnitudes divergían por DISEÑO.

Fix:
  1. Tracker module-level `_CAPS_APPLIED_LAST_RUN` + helpers
     `reset_caps_applied_last_run` / `_record_cap_applied` / `get_caps_applied_last_run`.
  2. `aggregate_and_deduct_shopping_list` invoca `reset_caps_applied_last_run`
     al inicio.
  3. 5 callsites de cap (HERB, VEG, LEGUMES-DRY, EGGS-AGGREGATE,
     LACTEOS-PERISHABLE, SPICE) registran metadata via `_record_cap_applied`.
  4. `run_shopping_coherence_guard` filtra `magnitude_divs` cuyo food matchea
     un cap aplicado (canonicalmente).
  5. Knob `MEALFIT_COHERENCE_CAP_AWARE` (default True) kill switch sin redeploy.

[P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Extensión: instrumentados 3 caps
perecederos adicionales que SÍ llegan al guard (no son staples, son
recipe-mentioned) — FRUITS-PERISHABLE (fresas/berries, recort ~68%),
FRUITS-LARGE (melón/sandía) y CANNED-PROTEIN (atún/sardinas) — además de la
rama 'mazo' de P3-HERB-CAP (el disparador COMÚN '1 mazo de cilantro', que antes
solo registraba la rama 'g'/'cda', caso raro).

[P3-CAPS-COHERENCE-RECONCILE-3 · 2026-05-30] CIERRE DE LA CLASE ENTERA. Los 7
caps restantes que el scoping previo dejó diferidos ("cerrar solo lo que produce
FP observados") ahora se instrumentan en TODAS sus ramas: OLIVE (4 ramas:
unit/weight/count/volumétric), CITRUS (unit+g), SAUCE (lata+weight), OIL
(botella+weight), CARBS (paquete+weight), SWEETENER (caja+weight), BROTHS (g+lb).
Además se cerraron las ramas PARCIALES de los caps viejos (instrumentación
incompleta del fix 2026-05-16, que registraba solo UNA rama por cap): VEG rama
unit, SPICE rama sobre, LEGUMES rama paquete, CANNED rama weight, EGGS rama
unidad, LACTEOS ramas lb+pote.

Razón del cierre total (vs el defer previo): añadir `_record_cap_applied` es
puramente ADITIVO y dirección-SEGURA (los caps solo reducen over-buy, nunca
under-buy ni corrupción) → no hay riesgo. El guard ahora NUNCA ve una divergencia
de magnitud falsa de un cap, sin importar qué unidad nativa emitió el LLM. CITRUS
era el FP genuino (perecedero no-staple no-líquido que llega a la capa de
magnitud del guard); los líquidos/condimentos (OIL/SAUCE/SWEETENER) son
belt-and-suspenders sobre la tolerancia de líquidos (50%) + el filtro is_staple
de presence/absence. Estado actual: 16/16 caps instrumentados en todas sus ramas
(cero diferidos).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PATH = _BACKEND_ROOT / "shopping_calculator.py"


def _read_shopping() -> str:
    return _SHOPPING_PATH.read_text(encoding="utf-8")


def test_caps_tracker_module_level_exists():
    """`_CAPS_APPLIED_LAST_RUN` lista module-level + helpers de
    reset/record/get deben existir."""
    text = _read_shopping()
    assert re.search(r"^_CAPS_APPLIED_LAST_RUN\s*:\s*list\s*=", text, re.MULTILINE), (
        "Falta declaración module-level `_CAPS_APPLIED_LAST_RUN: list = []`."
    )
    for fn in (
        "reset_caps_applied_last_run",
        "_record_cap_applied",
        "get_caps_applied_last_run",
    ):
        assert f"def {fn}" in text, f"Falta helper `def {fn}(...)`."


def test_aggregate_calls_reset_at_start():
    """`aggregate_and_deduct_shopping_list` debe llamar `reset_caps_applied_last_run`
    al inicio. Sin esto, runs consecutivos acumulan caps de runs previos."""
    text = _read_shopping()
    fn_match = re.search(
        r"def aggregate_and_deduct_shopping_list\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match, "No se encontró `def aggregate_and_deduct_shopping_list(...)`."
    body = fn_match.group(1)
    # El reset debe estar entre las primeras 50 líneas de la función
    # (antes de cualquier procesamiento real). Tomamos las primeras
    # ~3500 chars de la función para ser tolerantes pero estrictos.
    head = body[:3500]
    assert "reset_caps_applied_last_run()" in head, (
        "`aggregate_and_deduct_shopping_list` no invoca `reset_caps_applied_last_run()` "
        "al inicio. Sin esto, caps de runs previos contaminan al guard."
    )


@pytest.mark.parametrize("cap_marker", [
    "P3-HERB-CAP",
    "P5-VEG-CAP",
    "P6-LEGUMES-DRY-CAP",
    "P6-EGGS-AGGREGATE-CAP",
    "P6-LACTEOS-PERISHABLE-CAP",
    "P6-SPICE-CAP",
    # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Caps perecederos que SÍ
    # llegan al guard (no-staple, recipe-mentioned), ahora instrumentados.
    "P6-FRUITS-PERISHABLE-CAP",
    "P6-FRUITS-LARGE-CAP",
    "P6-CANNED-PROTEIN-CAP",
    # [P3-CAPS-COHERENCE-RECONCILE-3 · 2026-05-30] Cierre de la clase entera:
    # los 7 caps restantes que estaban diferidos ("solo cerrar FPs observados")
    # ahora instrumentados en TODAS sus ramas. Registrar es puramente aditivo y
    # dirección-segura (los caps solo reducen over-buy) → el guard nunca ve una
    # divergencia de magnitud falsa sin importar qué unidad emitió el LLM. CITRUS
    # es el FP genuino (perecedero no-staple no-líquido que llega al guard); los
    # líquidos/staples (OIL/SAUCE/SWEETENER) son belt-and-suspenders sobre la
    # tolerancia de líquidos + el filtro is_staple del guard. Además se cerraron
    # las ramas parciales de los caps viejos (VEG unit, SPICE sobre, LEGUMES
    # paquete, CANNED weight, EGGS unidad, LACTEOS lb+pote) que el fix original
    # de 2026-05-16 había dejado sin registrar.
    "P5-OLIVE-CAP",
    "P6-CITRUS-CAP",
    "P6-SAUCE-CAP",
    "P6-OIL-CAP",
    "P6-CARBS-CAP",
    "P6-SWEETENER-CAP",
    "P6-BROTHS-CAP",
])
def test_cap_callsite_records_metadata(cap_marker: str):
    """Cada uno de los 5 caps instrumentados debe invocar `_record_cap_applied`.
    Si renombras un cap o eliminas su record_cap_applied, este test falla."""
    text = _read_shopping()
    # Buscar el callsite que loguea `[<cap_marker>]` Y dentro de 5 líneas
    # invoca `_record_cap_applied(...)`. Pattern más flexible: buscar
    # primero el log y verificar que existe un _record_cap_applied cerca.
    log_pattern = re.compile(
        rf"\[{re.escape(cap_marker)}\]",
        re.MULTILINE,
    )
    log_matches = list(log_pattern.finditer(text))
    assert log_matches, f"No se encontró ningún log `[{cap_marker}]`."
    # Para cada match del log, chequear si _record_cap_applied aparece
    # dentro de ±15 líneas (~600 chars). Tolerante a orden record-antes
    # o record-después del log.
    found_any = False
    for m in log_matches:
        window_start = max(0, m.start() - 600)
        window_end = min(len(text), m.end() + 600)
        window = text[window_start:window_end]
        if f'_record_cap_applied' in window and cap_marker in window:
            # Verificar que el `_record_cap_applied(...)` referenció este reason
            record_calls = re.findall(
                r'_record_cap_applied\([^)]*\)',
                window,
                re.DOTALL,
            )
            if any(cap_marker in c for c in record_calls):
                found_any = True
                break
    assert found_any, (
        f"Cap `[{cap_marker}]` no tiene callsite invocando "
        f"`_record_cap_applied(..., \"{cap_marker}\")` cerca. Sin registro "
        f"el guard NO lo verá como cap intencional y reportará FP."
    )


def test_guard_consumes_caps_metadata():
    """`run_shopping_coherence_guard` debe consultar `get_caps_applied_last_run()`
    para filtrar divergencias magnitude legítimas."""
    text = _read_shopping()
    fn_match = re.search(
        r"def run_shopping_coherence_guard\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match, "No se encontró `def run_shopping_coherence_guard(...)`."
    body = fn_match.group(1)
    assert "get_caps_applied_last_run" in body, (
        "El guard no invoca `get_caps_applied_last_run()`. Sin esto el filtro "
        "cap-aware no funciona y volverán los falsos positivos."
    )
    assert "MEALFIT_COHERENCE_CAP_AWARE" in body, (
        "El guard no lee el knob `MEALFIT_COHERENCE_CAP_AWARE` (kill switch). "
        "Sin knob, no hay rollback sin redeploy si el filtro produce regresión."
    )


def test_kill_switch_default_enabled():
    """El knob `MEALFIT_COHERENCE_CAP_AWARE` debe parsear default True
    cuando no está set en env. Si flip a False, todas las divergencias
    magnitudes (legítimas o no) vuelven a contaminar al guard."""
    text = _read_shopping()
    # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Regex corregido. El previo
    # exigía `MEALFIT_COHERENCE_CAP_AWARE ... .environ.get(... "true")` (var ANTES
    # de .environ.get), orden imposible para el patrón real
    # `.environ.get("MEALFIT_COHERENCE_CAP_AWARE", "true")` donde la var es el
    # PRIMER argumento de get() → el test fallaba en falso contra código correcto.
    # Ahora ancla el patrón real: el knob se lee con default "true".
    pattern = re.search(
        r'\.environ\.get\(\s*["\']MEALFIT_COHERENCE_CAP_AWARE["\']\s*,\s*["\']true["\']',
        text,
    )
    assert pattern, (
        "El knob `MEALFIT_COHERENCE_CAP_AWARE` no tiene default `\"true\"`. "
        "Default debe ser True para que el fix esté activo sin necesidad de "
        "configurar env vars."
    )
