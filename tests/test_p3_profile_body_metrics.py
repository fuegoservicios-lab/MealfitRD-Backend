"""[P3-PROFILE-BODY-METRICS · 2026-05-20] Tests anti-regresión del editor
de Peso + Altura en el card Perfil de Settings.

Bundle de 3 sub-fixes integrados:
  1. [P3-PROFILE-BODY-METRICS] Inputs editables Peso + Altura persistiendo
     en health_profile (jsonb merge via RPC update_health_profile_merge).
  2. [P3-PROFILE-UNITS-TOGGLE] Toggle kg/lb + cm/ft con conversión
     bidireccional automática. weight persiste en la unit visible;
     height SIEMPRE en cm canonical (backend interpreta `weightUnit`).
  3. [P3-PROFILE-METRICS-COMMIT] Flow de commit explícito: body metrics
     son "draft" hasta que el user click "Actualizar Plan con Nuevos
     Datos" (que persiste + regenera plan). Si sale sin commit, los
     drafts se revierten via cleanup useEffect en activeSection change
     y component unmount. `updateData` sincroniza formData del context
     ANTES de `regeneratePlan` para que Plan.jsx (que se monta post-
     navigate y lee formData fresh) envíe el payload con valores nuevos.

Verifica el flujo end-to-end de UX que el user reportó como crítico:
"si salgo sin actualizar el plan que no se guarden los datos nuevos
de peso y altura que coloque".
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# P3-PROFILE-BODY-METRICS: inputs persisten en health_profile
# ============================================================

def test_weight_height_inputs_exist():
    """[P3-PROFILE-BODY-METRICS] Inputs de peso y altura deben existir
    con placeholders apropiados según unidad (75/165 para kg/lb, 175 para
    cm, 5ft/9in para ft)."""
    src = _read(_SETTINGS_JSX)
    # Estados: weightInput, heightInput, heightFeet, heightInches.
    for state_name in ["weightInput", "heightInput", "heightFeet", "heightInches"]:
        assert re.search(
            rf"const\s*\[\s*{state_name}\s*,\s*set\w+\s*\]\s*=\s*useState",
            src,
        ), f"useState para `{state_name}` ausente. Ver P3-PROFILE-BODY-METRICS."


def test_safe_update_health_profile_called_with_overrides():
    """[P3-PROFILE-BODY-METRICS] `safeUpdateHealthProfile` debe invocarse
    con overrides `{weight, weightUnit, height}` en el handler de commit
    (handleUpdatePlanWithMetrics). Persistencia jsonb merge backend."""
    src = _read(_SETTINGS_JSX)
    # Buscar el bloque de handleUpdatePlanWithMetrics.
    fn_match = re.search(
        r"const\s+handleUpdatePlanWithMetrics\s*=\s*async\s*\(\s*\)\s*=>",
        src,
    )
    assert fn_match, (
        "Handler `handleUpdatePlanWithMetrics` ausente. Sin él, el botón "
        "'Actualizar Plan con Nuevos Datos' no funciona. Ver "
        "P3-PROFILE-METRICS-COMMIT · 2026-05-20."
    )
    # Extraer cuerpo (balanced brace).
    body = _extract_arrow_fn(src, "handleUpdatePlanWithMetrics")
    assert body
    assert "safeUpdateHealthProfile" in body, (
        "Handler no llama `safeUpdateHealthProfile` — body metrics no se "
        "persisten en backend."
    )
    assert "overrides.weight" in body and "overrides.weightUnit" in body, (
        "Handler no construye overrides con `weight` y `weightUnit`."
    )
    assert "overrides.height" in body, (
        "Handler no construye overrides con `height`."
    )


# ============================================================
# P3-PROFILE-UNITS-TOGGLE: kg/lb + cm/ft con conversión correcta
# ============================================================

def test_weight_unit_toggle_state():
    """[P3-PROFILE-UNITS-TOGGLE] State `weightUnit` debe inicializarse
    desde formData.weightUnit o default basado en locale (delegado al
    context). Toggle UI debe permitir alternar entre kg/lb."""
    src = _read(_SETTINGS_JSX)
    assert re.search(
        r"const\s*\[\s*weightUnit\s*,\s*setWeightUnit\s*\]\s*=\s*useState",
        src,
    ), "State `weightUnit` ausente. Ver P3-PROFILE-UNITS-TOGGLE."
    # El componente _UnitToggle debe rendear `[kg, lb]` para peso.
    assert re.search(
        r"options=\{\s*\[\s*['\"]kg['\"]\s*,\s*['\"]lb['\"]\s*\]\s*\}",
        src,
    ), "Toggle de peso no expone opciones [kg, lb]."


def test_height_unit_toggle_with_conversion():
    """[P3-PROFILE-UNITS-TOGGLE] State `heightUnit` con toggle [cm, ft].
    Handler `handleHeightUnitToggle` convierte entre unidades sin perder
    el valor."""
    src = _read(_SETTINGS_JSX)
    assert re.search(
        r"const\s*\[\s*heightUnit\s*,\s*setHeightUnit\s*\]\s*=\s*useState",
        src,
    ), "State `heightUnit` ausente."
    assert re.search(
        r"options=\{\s*\[\s*['\"]cm['\"]\s*,\s*['\"]ft['\"]\s*\]\s*\}",
        src,
    ), "Toggle de altura no expone opciones [cm, ft]."
    assert "handleHeightUnitToggle" in src, (
        "Handler de conversión `handleHeightUnitToggle` ausente. Sin él, "
        "el toggle no convierte el valor visible entre unidades."
    )
    # Conversión cm → ft+in vía helper _cmToFtIn.
    assert "_cmToFtIn" in src, (
        "Helper `_cmToFtIn` ausente. Conversión cm→ft+in rota."
    )


def test_height_persisted_in_cm_canonical():
    """[P3-PROFILE-UNITS-TOGGLE] El height SIEMPRE se persiste en cm
    canonical (independiente de la unit visible). Si user trabaja en ft,
    la conversión ocurre antes del save: cm = ft * 30.48 + in * 2.54."""
    src = _read(_SETTINGS_JSX)
    body = _extract_arrow_fn(src, "handleUpdatePlanWithMetrics")
    assert body
    # Anchor: conversión a cm canonical.
    has_conversion = bool(re.search(
        r"ft\s*\*\s*30\.48\s*\+\s*inches\s*\*\s*2\.54",
        body,
    ))
    assert has_conversion, (
        "Conversión `ft * 30.48 + in * 2.54 → cm` ausente del handler. "
        "Altura se persistiría con valor incorrecto. Ver P3-PROFILE-UNITS-TOGGLE."
    )


# ============================================================
# P3-PROFILE-METRICS-COMMIT: draft + cleanup + commit explícito
# ============================================================

def test_body_metrics_changed_detection():
    """[P3-PROFILE-METRICS-COMMIT] Computed `bodyMetricsChanged` compara
    weight/height/weightUnit contra snapshot `_bodyMetricsOriginalRef`.
    Si cambian, se muestra el banner ámbar + botón 'Actualizar Plan'."""
    src = _read(_SETTINGS_JSX)
    assert "_bodyMetricsOriginalRef" in src, (
        "Ref `_bodyMetricsOriginalRef` ausente. Sin snapshot original, no "
        "se puede detectar drift. Ver P3-PROFILE-METRICS-COMMIT."
    )
    assert "bodyMetricsChanged" in src, (
        "Computed `bodyMetricsChanged` ausente."
    )


def test_cleanup_reverts_on_section_change_and_unmount():
    """[P3-PROFILE-METRICS-COMMIT] 2 useEffects de cleanup deben existir:
      (a) revertir cuando activeSection !== 'profile' (user cambia tab
          dentro de Settings).
      (b) revertir on unmount (user navega fuera de Settings)."""
    src = _read(_SETTINGS_JSX)
    # Helper de revert.
    assert "_revertBodyMetricsToOriginal" in src, (
        "Helper `_revertBodyMetricsToOriginal` ausente. Sin él, cleanup "
        "no puede revertir state. Ver P3-PROFILE-METRICS-COMMIT."
    )
    # useEffect (a): deps [activeSection], llama revert si !== 'profile'.
    has_section_cleanup = bool(re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{[^}]*activeSection\s*!==\s*['\"]profile['\"][^}]*_revertBodyMetricsToOriginal",
        src,
        re.DOTALL,
    ))
    assert has_section_cleanup, (
        "Cleanup `useEffect` con check `activeSection !== 'profile'` y "
        "revert ausente. Sin esto, drafts persisten al cambiar a otra "
        "sección de Settings. Ver P3-PROFILE-METRICS-COMMIT."
    )
    # useEffect (b): unmount cleanup. Retorna función que llama revert.
    has_unmount_cleanup = bool(re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{\s*return\s*\(\s*\)\s*=>\s*\{[^}]*_revertBodyMetricsToOriginal",
        src,
        re.DOTALL,
    ))
    assert has_unmount_cleanup, (
        "Cleanup de unmount (return en useEffect) ausente. Sin esto, "
        "drafts persisten en state residual si user vuelve a Settings "
        "desde otra ruta. Ver P3-PROFILE-METRICS-COMMIT."
    )


def test_save_profile_no_longer_persists_body_metrics():
    """[P3-PROFILE-METRICS-COMMIT] `handleSaveProfile` SOLO persiste
    full_name. Body metrics requieren el flow separado
    `handleUpdatePlanWithMetrics` (commit explícito + regenerate plan).

    Anti-pattern bloqueado: `safeUpdateHealthProfile` dentro de
    `handleSaveProfile` (que sería 'Guardar Cambios' silenciosamente
    persistiendo body metrics sin regenerar plan)."""
    src = _read(_SETTINGS_JSX)
    body = _extract_arrow_fn(src, "handleSaveProfile")
    assert body, "handleSaveProfile no encontrado"
    assert "safeUpdateHealthProfile" not in body, (
        "`handleSaveProfile` invoca `safeUpdateHealthProfile` — body "
        "metrics se persistirían vía 'Guardar Cambios' (botón normal) "
        "sin regenerar plan, violando el contrato de commit explícito. "
        "Ver P3-PROFILE-METRICS-COMMIT · 2026-05-20."
    )
    assert "full_name" in body, (
        "handleSaveProfile no actualiza full_name — funcionalidad rota."
    )


def test_update_data_sync_before_regenerate():
    """[P3-PROFILE-METRICS-COMMIT] `handleUpdatePlanWithMetrics` DEBE llamar
    `updateData('weight'/'height'/'weightUnit', ...)` ANTES de `regeneratePlan`.

    Razón: el hook `useRegeneratePlan` solo hace `navigate('/plan')`. Plan.jsx
    se monta post-navigate y lee `formData` del context para construir el
    payload al backend. Si `updateData` se llamara DESPUÉS del navigate
    (o no se llamara), Plan.jsx leería formData con weight/height viejos."""
    src = _read(_SETTINGS_JSX)
    body = _extract_arrow_fn(src, "handleUpdatePlanWithMetrics")
    assert body
    # Anchor: updateData calls antes del regeneratePlan invoke.
    update_data_pos = body.find("updateData(")
    regenerate_pos = body.find("regeneratePlan(")
    assert update_data_pos != -1, (
        "`updateData` no se invoca en handleUpdatePlanWithMetrics. Sin esto, "
        "formData del context queda con peso/altura viejos cuando Plan.jsx "
        "se monta post-navigate. Ver P3-PROFILE-METRICS-COMMIT."
    )
    assert regenerate_pos != -1, "regeneratePlan no se invoca."
    assert update_data_pos < regenerate_pos, (
        f"`updateData` (pos {update_data_pos}) viene DESPUÉS de "
        f"`regeneratePlan` (pos {regenerate_pos}) — order incorrecto. "
        f"Plan.jsx leerá formData stale. Ver P3-PROFILE-METRICS-COMMIT."
    )


def test_commit_button_distinct_from_save_button():
    """[P3-PROFILE-METRICS-COMMIT] Cuando `bodyMetricsChanged === true`,
    se muestra un botón distinto ('Actualizar Plan con Nuevos Datos',
    color ámbar) en lugar del 'Guardar Cambios' normal. Banner ámbar
    visible avisando que cambios se descartan si sale sin commit."""
    src = _read(_SETTINGS_JSX)
    # Botón condicional: ternary `{bodyMetricsChanged ? ... : ...}`.
    assert re.search(
        r"\{\s*bodyMetricsChanged\s*\?",
        src,
    ), (
        "Render condicional `{bodyMetricsChanged ?` ausente. Sin esto, "
        "no hay distinción visual entre el save normal y el commit con "
        "regenerate. Ver P3-PROFILE-METRICS-COMMIT."
    )
    # Botón "Actualizar Plan con Nuevos Datos" texto literal.
    assert "Actualizar Plan con Nuevos Datos" in src, (
        "Texto 'Actualizar Plan con Nuevos Datos' ausente del botón "
        "commit. Ver P3-PROFILE-METRICS-COMMIT."
    )
    # Banner de aviso "Cambios pendientes".
    assert "Cambios pendientes" in src, (
        "Banner 'Cambios pendientes en peso/altura' ausente. Sin él, el "
        "user no sabe que su edit es draft."
    )


# ============================================================
# Helpers
# ============================================================

def _extract_arrow_fn(src: str, name: str) -> str:
    """Extrae el cuerpo de `const <name> = async (...) => { ... }` con
    balanced braces. Naive `\\}` corta en el primer brace de un object
    literal interno."""
    pat = rf"const\s+{name}\s*="
    match = re.search(pat, src)
    if not match:
        return ""
    arrow = src.find("=>", match.end())
    if arrow < 0:
        return ""
    brace = src.find("{", arrow)
    if brace < 0:
        return ""
    depth = 0
    for j in range(brace, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[match.start():j + 1]
    return ""
