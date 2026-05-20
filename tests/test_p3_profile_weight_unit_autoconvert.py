"""[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT · 2026-05-20] Test anti-regresión
del auto-convert de peso al togglear kg↔lb en Settings.jsx → card Perfil.

Bug pre-fix:
    El callsite `<_UnitToggle ... onChange={setWeightUnit} />` solo cambiaba
    la unidad sin tocar el valor numérico del input. Resultado: 70 kg al
    togglear a lb se quedaba como "70" (interpretado como 70 lb = 31.7 kg
    → BMR significativamente bajo). La validación de rango (55-660 lb /
    25-300 kg) bloqueaba extremos pero valores mid-range pasaban silenciosos.

Fix:
    Wrapper `handleWeightUnitToggle(newUnit)` análogo a
    `handleHeightUnitToggle`. Convierte el valor numérico usando
    `_WEIGHT_LB_PER_KG = 2.20462`, redondea a 1 decimal (match con
    `step="0.1"` del input), preserva NaN/0/empty sin tocar.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_handler_defined():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] El wrapper
    `handleWeightUnitToggle` existe y contiene el marker como
    tooltip-anchor (un renombre futuro falla este test antes de tocar
    producción)."""
    src = _read(_SETTINGS_JSX)
    assert "P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT" in src, (
        "Marker `P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT` ausente — tooltip-anchor "
        "removido. Si quieres remover el fix, primero remueve este test."
    )
    assert re.search(r"const\s+handleWeightUnitToggle\s*=\s*\(", src), (
        "`handleWeightUnitToggle` no definido en Settings.jsx. Pre-fix el "
        "callsite pasaba `setWeightUnit` directo — restaurar el wrapper que "
        "auto-convierte el valor numérico al togglear."
    )


def test_conversion_constant_present():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] La constante de conversión
    debe estar explícita y match con el factor canónico (NIST: 1 kg =
    2.20462262 lb, truncado a 5 decimales para JS float)."""
    src = _read(_SETTINGS_JSX)
    assert "_WEIGHT_LB_PER_KG" in src, (
        "Constante `_WEIGHT_LB_PER_KG` ausente. La conversión hardcoded en "
        "el handler sin nombrar la constante hace el renombre/refactor "
        "menos seguro."
    )
    # Sanity sobre el valor exacto (factor NIST). Si alguien lo cambia a
    # 2.2 o 2.205 silencioso, el drift de ~0.21% se acumula en redondeos.
    assert re.search(r"_WEIGHT_LB_PER_KG\s*=\s*2\.20462\b", src), (
        "`_WEIGHT_LB_PER_KG` debe ser 2.20462 (factor NIST). Cualquier "
        "approximación menor introduce drift de macros."
    )


def test_handler_converts_kg_to_lb():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] Cuando toggle es kg→lb, el
    handler multiplica el valor por el factor antes de `setWeightInput`."""
    src = _read(_SETTINGS_JSX)
    # Busca el branch que detecta kg→lb: `newUnit === 'lb' && weightUnit === 'kg'`.
    pattern = re.compile(
        r"newUnit\s*===\s*['\"]lb['\"]\s*&&\s*weightUnit\s*===\s*['\"]kg['\"]"
    )
    assert pattern.search(src), (
        "Branch kg→lb (`newUnit==='lb' && weightUnit==='kg'`) ausente — el "
        "handler no auto-convierte en esta dirección."
    )


def test_handler_converts_lb_to_kg():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] Cuando toggle es lb→kg, el
    handler divide por el factor."""
    src = _read(_SETTINGS_JSX)
    pattern = re.compile(
        r"newUnit\s*===\s*['\"]kg['\"]\s*&&\s*weightUnit\s*===\s*['\"]lb['\"]"
    )
    assert pattern.search(src), (
        "Branch lb→kg (`newUnit==='kg' && weightUnit==='lb'`) ausente — el "
        "handler no auto-convierte en esta dirección."
    )


def test_handler_idempotent_on_same_unit():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] Toggle a la misma unidad debe
    ser no-op (early return). Sin esto, click repetido sobre el mismo
    chip multiplicaría el valor por 2.20462 cada vez (efecto compuesto)."""
    src = _read(_SETTINGS_JSX)
    # Extraer cuerpo del handler (heurística por braces).
    start = src.find("const handleWeightUnitToggle")
    assert start != -1
    body_start = src.index("=>", start)
    depth = 0
    i = body_start
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body = src[body_start:i + 1]
                break
        i += 1
    else:
        raise AssertionError("No se pudo extraer body de handleWeightUnitToggle")
    # Primer statement DEBE ser early return en same-unit.
    early_return = re.search(
        r"if\s*\(\s*newUnit\s*===\s*weightUnit\s*\)\s*return",
        body,
    )
    assert early_return, (
        "Falta `if (newUnit === weightUnit) return;` early-exit. Sin esto, "
        "toggles repetidos compunden multiplicando el valor cada vez."
    )


def test_unittoggle_callsite_uses_handler_not_raw_setter():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] El `<_UnitToggle>` del peso
    DEBE pasar `handleWeightUnitToggle` (no `setWeightUnit` directo). El
    callsite raw fue exactamente el bug pre-fix."""
    src = _read(_SETTINGS_JSX)
    # Buscar el _UnitToggle del peso (options=['kg', 'lb']).
    weight_toggle = re.search(
        r"<_UnitToggle\s+unit=\{weightUnit\}\s+options=\{\['kg',\s*'lb'\]\}\s+onChange=\{([^}]+)\}",
        src,
    )
    assert weight_toggle, (
        "Callsite `<_UnitToggle ... options={['kg', 'lb']} ... />` no "
        "encontrado. Si lo renombraste, actualiza este test."
    )
    on_change = weight_toggle.group(1).strip()
    assert on_change == "handleWeightUnitToggle", (
        f"Callsite del weight toggle pasa `onChange={{{on_change}}}` — debe "
        "ser `handleWeightUnitToggle`. Pasar `setWeightUnit` directo es el "
        "bug pre-fix (cambia unit sin convertir el valor)."
    )


def test_last_known_pfix_bumped():
    """[P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT] El marker debe reflejar este
    fix en `backend/app.py` para que `/health/version` lo exponga."""
    app_py = _REPO_ROOT / "backend" / "app.py"
    src = app_py.read_text(encoding="utf-8")
    match = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert match, "_LAST_KNOWN_PFIX no encontrado en backend/app.py"
    marker = match.group(1)
    assert (
        "P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT" in marker
        or "2026-05-20" in marker
    ), (
        f"_LAST_KNOWN_PFIX={marker!r} no refleja este P-fix. Bumpear a "
        "'P3-PROFILE-WEIGHT-UNIT-AUTOCONVERT · 2026-05-20' o equivalente."
    )
