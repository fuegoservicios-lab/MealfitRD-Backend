"""[P3-PROFILE-NUMERIC-COERCE · 2026-05-20] Test anti-regresión de la
coerción string→number para los campos numéricos de `health_profile`
(weight, height, age, bodyFat).

Bug pre-fix:
    Wizard `InteractiveQuestions` usa `e.target.value` (string) directo
    en `updateData('weight', ...)`. `health_profile` JSONB termina con
    `{"weight": "70", "height": "168", "age": "20"}` (strings) cuando
    semánticamente son números. Todos los readers coerce — cosmetic
    pero hace queries SQL `WHERE health_profile->'weight' > 80` requieran
    casts manuales.

Fix:
    1. Frontend coerce en `buildHealthProfilePayload` (capa SSOT de
       persistencia) — toda escritura nueva produce JSON numbers.
    2. Migración SSOT `p3_profile_numeric_coerce_2026_05_20.sql`
       normaliza filas legacy con `jsonb_set` + `to_jsonb(::numeric)`.
       Filtro defensivo `~ '^-?[0-9]+(\\.[0-9]+)?$'` ignora valores con
       coma decimal o sufijos (deja al siguiente write normalizarlos).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SECURE_STORAGE = _REPO_ROOT / "frontend" / "src" / "config" / "secureFormStorage.js"
_MIGRATION_SQL = _REPO_ROOT / "migrations" / "p3_profile_numeric_coerce_2026_05_20.sql"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_frontend_helper_defined():
    """[P3-PROFILE-NUMERIC-COERCE] El helper `_coerceNumericHealthFields`
    está definido en secureFormStorage.js y se invoca dentro de
    `buildHealthProfilePayload`."""
    src = _read(_SECURE_STORAGE)
    assert "P3-PROFILE-NUMERIC-COERCE" in src, (
        "Marker `P3-PROFILE-NUMERIC-COERCE` ausente — tooltip-anchor "
        "removido. Si quieres remover el fix, primero remueve este test."
    )
    assert "_coerceNumericHealthFields" in src, (
        "Helper `_coerceNumericHealthFields` no definido. Pre-fix el spread "
        "`{...stripInternalFlags(formData), ...overrides}` no normalizaba "
        "tipos — restaurar el helper que coerce los 4 campos numéricos."
    )


def test_numeric_fields_list_complete():
    """[P3-PROFILE-NUMERIC-COERCE] La constante `NUMERIC_HEALTH_FIELDS`
    debe incluir los 4 campos numéricos canónicos. Si añadimos un nuevo
    field numérico al wizard, debe registrarse aquí también."""
    src = _read(_SECURE_STORAGE)
    match = re.search(
        r"NUMERIC_HEALTH_FIELDS\s*=\s*\[([^\]]+)\]",
        src,
    )
    assert match, "Constante `NUMERIC_HEALTH_FIELDS` no encontrada."
    body = match.group(1)
    for field in ("weight", "height", "age", "bodyFat"):
        assert f"'{field}'" in body or f'"{field}"' in body, (
            f"`{field}` ausente de NUMERIC_HEALTH_FIELDS. Si añadiste un "
            f"campo numérico nuevo, registralo aquí; si removiste uno, "
            f"actualiza el test."
        )


def test_buildhealthprofile_invokes_coerce():
    """[P3-PROFILE-NUMERIC-COERCE] `buildHealthProfilePayload` debe
    invocar el helper antes de retornar. Sin esto, el coerce existe
    pero el dead-code lo deja inerte."""
    src = _read(_SECURE_STORAGE)
    # Buscar el body de buildHealthProfilePayload.
    start = src.find("export const buildHealthProfilePayload")
    assert start != -1, "buildHealthProfilePayload no encontrado."
    # Hasta el primer cierre `};` que match al arrow function.
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
        raise AssertionError("No se pudo extraer body de buildHealthProfilePayload.")
    assert "_coerceNumericHealthFields(" in body, (
        "`buildHealthProfilePayload` no invoca `_coerceNumericHealthFields`. "
        "El helper existe pero está dead-code — restaurar el wrapper sobre "
        "el merge `{...stripInternalFlags(formData), ...overrides}`."
    )


def test_coerce_uses_parseFloat_and_isFinite():
    """[P3-PROFILE-NUMERIC-COERCE] El helper usa `parseFloat` + `Number.isFinite`
    (NO `Number(x)` ni `parseInt`). Razón: `parseFloat('70,5')` retorna 70
    sin la coma decimal (deja al P1-9 _normalizeDecimal normalizar); `Number(x)`
    coerce strings vacíos a 0 silencioso; `parseInt` pierde decimales (e.g.
    bodyFat=15.5)."""
    src = _read(_SECURE_STORAGE)
    # Heurística: helper extraído.
    start = src.find("_coerceNumericHealthFields")
    assert start != -1
    # Asumimos function body inicia con `=` (es const = (...) => {...} o function expr).
    fn_window = src[start:start + 800]
    assert "parseFloat" in fn_window, (
        "`parseFloat` ausente del helper. Usar `Number(x)` permite '' → 0 "
        "silent; usar `parseInt` pierde decimales para bodyFat. Restaurar parseFloat."
    )
    assert "Number.isFinite" in fn_window, (
        "`Number.isFinite` ausente. Sin el check, `parseFloat('NaN')` o "
        "'Infinity' colaría como number — DB queda con valores inválidos."
    )


def test_migration_exists_and_idempotent():
    """[P3-PROFILE-NUMERIC-COERCE] La migración SSOT existe + tiene el
    marker + es idempotente (el filtro `jsonb_typeof = 'string'` + regex
    garantiza no-op en filas ya numéricas)."""
    assert _MIGRATION_SQL.exists(), (
        f"Migración SSOT ausente: {_MIGRATION_SQL}. Sin ella, la normalización "
        "de filas legacy depende de ejecutarse vía MCP runtime — riesgo "
        "operacional (no replicable, no documentado)."
    )
    sql = _read(_MIGRATION_SQL)
    assert "P3-PROFILE-NUMERIC-COERCE" in sql, "Marker ausente en migration SQL."
    # Idempotencia: filter por typeof string + regex numeric.
    assert "jsonb_typeof(health_profile->'weight') = 'string'" in sql, (
        "Filtro idempotente `jsonb_typeof = 'string'` ausente del UPDATE "
        "weight — segundo run intentaría castear `::numeric` sobre números, fallaría."
    )
    # Regex defensivo presente (escape de backslash para Python source — el SQL contiene literal `\.`).
    assert "~ '^-?[0-9]+(\\.[0-9]+)?$'" in sql, (
        "Regex defensivo numeric ausente — sin él, un valor legacy con coma "
        "decimal ('70,5') haría fallar el cast `::numeric` con error duro."
    )


def test_migration_covers_all_four_fields():
    """[P3-PROFILE-NUMERIC-COERCE] La migración debe normalizar los 4
    campos canónicos. Si añades un field a NUMERIC_HEALTH_FIELDS (frontend)
    sin añadir UPDATE acá, las filas legacy de ese field quedan stale."""
    sql = _read(_MIGRATION_SQL)
    for field in ("weight", "height", "age", "bodyFat"):
        assert f"'{{{field}}}'" in sql, (
            f"`{{{field}}}` no aparece como path de `jsonb_set` en la migración. "
            f"Si añadiste el field al frontend, añade UPDATE correspondiente acá."
        )


def test_migration_has_sanity_check():
    """[P3-PROFILE-NUMERIC-COERCE] La migración cierra con un RAISE
    EXCEPTION si quedan filas con strings numéricos tras los UPDATEs
    (patrón estándar de migración defensiva, P3-MIGRATION-IDEMPOTENCE-DOC).
    Sin esto, un bug del UPDATE pasaría silencioso."""
    sql = _read(_MIGRATION_SQL)
    assert "v_remaining_strings" in sql, (
        "Sanity check `v_remaining_strings` ausente. Sin él, un bug en el "
        "UPDATE (e.g. condición WHERE incorrecta) pasaría sin alerta."
    )
    assert "RAISE EXCEPTION" in sql, (
        "RAISE EXCEPTION ausente — el sanity check no aborta si detecta drift."
    )
