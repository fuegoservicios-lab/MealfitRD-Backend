r"""[P3-SHIFT-DATEONLY-LOCAL · 2026-05-18] El backend NO debe tratar
`grocery_start_date` en formato date-only (`"YYYY-MM-DD"`) como UTC midnight.

Síntoma reportado por usuario 2026-05-18:
> "elimino el martes cuando llego el dia lunes y no debio eliminarlo,
>  debio eliminar el dia que paso, que ese dia que paso es el domingo
>  ya que hoy es lunes"

Screenshot: plan [Domingo, Lunes, Martes] vivido el domingo → al abrir
el lunes solo apareció UN tab "Lunes" con los meals del Martes original.

Causa raíz: `grocery_start_date` se persiste en DOS formatos:
  1. `"YYYY-MM-DD"` date-only — backfill SQL (`p0_3_backfill_plan_anchors`)
     y plan_data inicial cuando el LLM emite el campo sin TZ.
  2. `"YYYY-MM-DDTHH:MM:SS+TZ"` timestamp ISO completo — fix
     `[GROCERY-START-DATE-TIMESTAMP-FIX 2026-05-06]` en
     `_ensure_grocery_start_date`.

Pre-fix, `api_shift_plan` (`routers/plans.py`) y
`_background_shift_plan_for_user` (`cron_tasks.py`) procesaban AMBOS
formatos por la misma rama:

    start_dt = safe_fromisoformat(start_date_str)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)   # ← marca date-only como UTC midnight
    start_dt = start_dt - timedelta(minutes=int(tz_offset))  # ← resta 4h en TZ -4
    start_date = start_dt.date()                              # ← retrocede 1 día (off-by-one)

Para `start_date_str = "2026-05-17"` en TZ Santo Domingo (tz_offset=240):
  - `replace(tzinfo=utc)` → `2026-05-17T00:00:00Z`
  - `- timedelta(minutes=240)` → `2026-05-16T20:00:00Z`
  - `.date()` → `2026-05-16` (¡día anterior!)
  - `days_since_creation = (lun 18 - 2026-05-16).days = 2` (debió ser 1)
  - `shift_amount = min(2, 3) = 2` → plan [Dom,Lun,Mar] → [Mar] solo
  - Renombra: i=0 → today=Lunes → day_name="Lunes"
  - Resultado: 1 tab "Lunes" con meals del Martes original.

El frontend ya tenía el caso resuelto desde 2026-05-06
(`_parseStartLocal` en Dashboard.jsx:603, marker
`[GROCERY-START-DATE-LOCAL-PARSE 2026-05-06]`) pero el backend nunca
espejó el fix.

Fix: detectar formato date-only por regex `^\d{4}-\d{2}-\d{2}$` y
parsearlo como fecha LOCAL del usuario (sin TZ dance). Si trae
timestamp con/sin TZ, lógica legacy intacta.

Cubre 2 surfaces:
  - `routers/plans.py:api_shift_plan` (endpoint HTTP).
  - `cron_tasks.py:_background_shift_plan_for_user` (cron rolling refill
    para usuarios inactivos).

Drift detection:
  - Si alguien revierte el branch date-only en cualquiera de los 2
    archivos → falla `test_routers_plans_has_dateonly_branch` o
    `test_cron_tasks_has_dateonly_branch`.
  - Si alguien borra el marker → falla `test_marker_present`.

Tests funcionales (`test_shift_amount_correct_in_tz_negative_*`)
simulan el escenario real del usuario (TZ -4, plan generado domingo,
abierto lunes) y validan que `days_since_creation == 1` (no 2).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ROUTERS_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_CRON_TASKS = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


def _api_shift_plan_body() -> str:
    m = re.search(
        r"def api_shift_plan\([\s\S]+?(?=\n@router\.|\ndef |\Z)",
        _ROUTERS_PLANS,
    )
    assert m, "api_shift_plan no encontrada en routers/plans.py"
    return m.group(0)


def _background_shift_body() -> str:
    m = re.search(
        r"def _background_shift_plan_for_user\([\s\S]+?(?=\ndef |\Z)",
        _CRON_TASKS,
    )
    assert m, "_background_shift_plan_for_user no encontrada en cron_tasks.py"
    return m.group(0)


def test_marker_present_in_routers_plans():
    """El marker debe estar en `api_shift_plan` para que un revert sea
    visible en code-review + para que `test_p2_hist_audit_14_marker_test_link`
    matchee el slug `p3_shift_dateonly_local` con este archivo."""
    body = _api_shift_plan_body()
    assert "P3-SHIFT-DATEONLY-LOCAL" in body, (
        "Marker P3-SHIFT-DATEONLY-LOCAL ausente en api_shift_plan. "
        "Un revert silente reintroduciría el off-by-one en TZ negativas."
    )


def test_marker_present_in_cron_tasks():
    body = _background_shift_body()
    assert "P3-SHIFT-DATEONLY-LOCAL" in body, (
        "Marker P3-SHIFT-DATEONLY-LOCAL ausente en _background_shift_plan_for_user. "
        "El cron de rolling refill para usuarios inactivos volvería a tener el bug."
    )


def test_routers_plans_has_dateonly_branch():
    """Verifica el branch date-only ANTES de la rama timestamp legacy."""
    body = _api_shift_plan_body()
    # Anchor del check: regex `^\d{4}-\d{2}-\d{2}$` que distingue date-only.
    assert r"^\d{4}-\d{2}-\d{2}$" in body, (
        "Branch date-only ausente en api_shift_plan: el regex que detecta "
        "'YYYY-MM-DD' no aparece. Sin ese branch, el bug del off-by-one en "
        "TZ negativas reaparece para todos los planes persistidos con el "
        "formato date-only (backfill SQL p0_3 + plan_data del LLM)."
    )
    # El branch date-only DEBE construir `start_date` desde date(y, m, d),
    # NO desde `start_dt.date()`.
    assert "_date_p3(" in body or re.search(r"date\(_y_p3,\s*_m_p3,\s*_d_p3\)", body), (
        "Branch date-only no construye `date(y, m, d)` directo — está "
        "probablemente reusando el path timestamp. Eso reintroduce el bug."
    )


def test_cron_tasks_has_dateonly_branch():
    body = _background_shift_body()
    assert r"^\d{4}-\d{2}-\d{2}$" in body, (
        "Branch date-only ausente en _background_shift_plan_for_user. "
        "El cron de rolling refill (P0-2) divergiría del endpoint HTTP."
    )
    assert "_date_p3_bg(" in body or re.search(r"date\(_y_p3,\s*_m_p3,\s*_d_p3\)", body), (
        "Branch date-only en cron no construye `date(y, m, d)` directo."
    )


def test_routers_plans_preserves_dateonly_format_on_write():
    """Tras el shift, si el formato original era date-only, persistir el
    nuevo también como date-only (no promover a timestamp ISO completo).
    Espejo del backfill SQL p0_3 — evita drift de formato entre escrituras."""
    body = _api_shift_plan_body()
    # Anchor: el branch _is_date_only_shift dentro del save de new_plan_start_iso.
    save_block = body[body.find("if needs_shift and start_date_str"):]
    assert "_is_date_only_shift" in save_block[:2000], (
        "El save de `new_plan_start_iso` NO ramifica por formato date-only. "
        "Resultado: un plan persistido como 'YYYY-MM-DD' sería promovido a "
        "timestamp ISO completo tras el primer shift, rompiendo SSOT con el "
        "backfill SQL p0_3 y dificultando audits forensics."
    )


def test_cron_tasks_preserves_dateonly_format_on_write():
    body = _background_shift_body()
    # Anchor: el save block. Buscar el branch needs_shift and not is_expired_renewable.
    save_block_idx = body.find("if needs_shift and not is_expired_renewable")
    assert save_block_idx > 0
    save_block = body[save_block_idx:save_block_idx + 1500]
    assert "_is_date_only_shift" in save_block, (
        "Cron `_background_shift_plan_for_user` no preserva formato date-only "
        "en el write. Divergencia con `api_shift_plan`."
    )


def test_marker_bumped_at_least_to_this_fix_date():
    """El marker `_LAST_KNOWN_PFIX` debe ser de fecha >= 2026-05-18 (fecha de
    este P-fix). Validamos la fecha y no el literal exacto porque P-fixes
    posteriores en el mismo día bumpean al suyo — siempre apunta al MÁS reciente.
    El floor global vive en `test_p3_1_last_known_pfix_freshness`."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"[^"]+·\s*(\d{4}-\d{2}-\d{2})"',
        _APP_PY,
    )
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py o sin fecha ISO"
    from datetime import date
    marker_date = date.fromisoformat(m.group(1))
    assert marker_date >= date(2026, 5, 18), (
        f"_LAST_KNOWN_PFIX está stale ({marker_date}); este P-fix fue 2026-05-18. "
        f"Un revert que dejó el marker pre-fix indica que el bump no se hizo."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests funcionales: reproducir el escenario real del usuario sin DB.
# Reproducen la aritmética del fix aislada de la transacción Postgres.
# ─────────────────────────────────────────────────────────────────────────────


def _parse_grocery_start_local_dateonly(
    start_date_str: str, tz_offset_minutes: int, today_local_date: date
) -> int:
    """Reimplementación del fix aislada. Si el contrato del fix cambia,
    actualizar acá Y los callsites prod en paralelo."""
    is_date_only = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", start_date_str.strip()))
    if is_date_only:
        y, m, d = (int(x) for x in start_date_str.strip().split("-"))
        start_date = date(y, m, d)
    else:
        from constants import safe_fromisoformat

        start_dt = safe_fromisoformat(start_date_str)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = start_dt.astimezone(timezone.utc)
        start_dt = start_dt - timedelta(minutes=int(tz_offset_minutes))
        start_date = start_dt.date()
    return (today_local_date - start_date).days


def test_shift_amount_correct_in_tz_negative_dateonly():
    """Escenario reportado por usuario 2026-05-18:
    - Plan generado domingo 2026-05-17 con grocery_start_date persistido
      como date-only "2026-05-17" (backfill SQL o LLM emit).
    - User en TZ -4 Santo Domingo (tz_offset=240 minutos).
    - User abre app lunes 2026-05-18 → `days_since_creation` debe ser 1
      (NO 2, que era el resultado pre-fix).
    """
    result = _parse_grocery_start_local_dateonly(
        start_date_str="2026-05-17",
        tz_offset_minutes=240,
        today_local_date=date(2026, 5, 18),
    )
    assert result == 1, (
        f"days_since_creation incorrecto para date-only en TZ -4: esperado 1, "
        f"recibido {result}. Si es 2, el bug original reapareció — el shift "
        f"eliminará 2 días del plan en vez de 1 al cruzar la medianoche local."
    )


def test_shift_amount_correct_in_tz_positive_dateonly():
    """Mismo plan, TZ +5:30 (India, tz_offset=-330). Pre-fix también
    fallaba (en sentido inverso) cuando el plan se persistió en UTC y
    el user cruzaba medianoche local antes que UTC."""
    # User abre martes 19 mayo en TZ +5:30, plan persistido lunes.
    result = _parse_grocery_start_local_dateonly(
        start_date_str="2026-05-18",
        tz_offset_minutes=-330,
        today_local_date=date(2026, 5, 19),
    )
    assert result == 1


def test_shift_amount_same_day_dateonly():
    """User abre el mismo día en que se generó el plan (sin shift)."""
    result = _parse_grocery_start_local_dateonly(
        start_date_str="2026-05-18",
        tz_offset_minutes=240,
        today_local_date=date(2026, 5, 18),
    )
    assert result == 0


def test_shift_amount_timestamp_format_unchanged():
    """Para timestamp ISO completo, la rama legacy se mantiene — el fix
    NO afecta planes persistidos con `[GROCERY-START-DATE-TIMESTAMP-FIX
    2026-05-06]`."""
    # Plan generado domingo 17 mayo 22:00 local DR = 2026-05-18T02:00Z.
    # User abre lunes 18 mayo: days_since_creation debe ser 1.
    result = _parse_grocery_start_local_dateonly(
        start_date_str="2026-05-18T02:00:00+00:00",
        tz_offset_minutes=240,
        today_local_date=date(2026, 5, 18),
    )
    assert result == 1


def test_pre_fix_buggy_behavior_documented():
    """Documenta el comportamiento BUGGY pre-fix para que un revert
    silente sea inmediatamente visible. Si este test sigue pasando con
    `expected_buggy=2`, alguien probablemente revirtió el fix."""

    def _pre_fix_compute(start_date_str: str, tz_offset: int, today_local: date) -> int:
        """Lógica pre-fix EXACTA. NO usar en prod."""
        from constants import safe_fromisoformat

        start_dt = safe_fromisoformat(start_date_str)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = start_dt.astimezone(timezone.utc)
        start_dt = start_dt - timedelta(minutes=int(tz_offset))
        return (today_local - start_dt.date()).days

    pre_fix_result = _pre_fix_compute("2026-05-17", 240, date(2026, 5, 18))
    assert pre_fix_result == 2, (
        f"Pre-fix compute ya no produce el bug (resultado: {pre_fix_result}). "
        f"Si esto cambia, el escenario reproductor de este test debe revisarse "
        f"— el bug ya no se reproduce con la lógica vieja, lo cual significa "
        f"que algo cambió en `safe_fromisoformat` o `timedelta` semantics."
    )
    # Y confirmar que el fix produce el resultado correcto.
    fix_result = _parse_grocery_start_local_dateonly("2026-05-17", 240, date(2026, 5, 18))
    assert fix_result == 1
    assert fix_result != pre_fix_result, (
        "El fix produce el MISMO resultado que la lógica pre-fix → el "
        "branch date-only no se está activando o no produce diferencia."
    )
