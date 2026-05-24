#!/usr/bin/env python3
"""[P2-PROD-AUDIT-1 · 2026-05-23] Helper para bumpear `_LAST_KNOWN_PFIX`
+ `_PFIX_DATE_FLOOR` atómicamente en un solo paso.

Gap original (audit production-readiness 2026-05-23, B-P2-7):
    `_LAST_KNOWN_PFIX` en `app.py:34` debe bumpearse en CADA cierre de
    P-fix mergeado. Además, `_PFIX_DATE_FLOOR` en
    `tests/test_p3_1_last_known_pfix_freshness.py:44` también debe
    bumpearse en tandem (test enforza floor).

    Modo de fallo:
      - Olvido humano: bump solo uno → test P3-1 falla en CI.
      - Bump del marker sin actualizar el slug-tests cross-link
        (P2-HIST-AUDIT-14): test cross-link falla en CI.

Fix:
    Script con argumentos `--marker "P2-FOO · YYYY-MM-DD"` que:
      (1) Valida formato del marker contra regex canónico.
      (2) Valida que existe al menos un `tests/test_<slug>*.py`.
      (3) Lee y muestra el current `_LAST_KNOWN_PFIX`.
      (4) Updatea ambos files atómicamente (rollback si alguno falla).
      (5) Imprime SOP post-bump (`git diff` review + commit).

Uso:
    ./scripts/bump_last_known_pfix.py --marker "P2-NEW-X · 2026-05-23" \\
        --floor-comment "P2-NEW-X 2026-05-23: short description del fix"

    Para dry-run (mostrar cambios sin aplicar):
    ./scripts/bump_last_known_pfix.py --marker "..." --floor-comment "..." --dry-run

Tooltip-anchor: P2-PROD-AUDIT-1-BUMP-HELPER | audit 2026-05-23.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_TEST_FILE = _BACKEND_ROOT / "tests" / "test_p3_1_last_known_pfix_freshness.py"
_TESTS_DIR = _BACKEND_ROOT / "tests"

_MARKER_PATTERN = re.compile(
    r"^(?P<prefix>P\d+(?:-[A-Z0-9]+)+)\s+·\s+(?P<date>\d{4}-\d{2}-\d{2})$"
)


def _marker_to_slug(marker: str) -> str:
    """`P2-HIST-AUDIT-13 · 2026-05-09` → `p2_hist_audit_13`."""
    prefix = marker.split("·", 1)[0].strip()
    return prefix.replace("-", "_").lower()


def _read_current_marker() -> str:
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        sys.exit("❌ No se pudo leer _LAST_KNOWN_PFIX de app.py")
    return m.group(1)


def _validate_marker(marker: str) -> tuple[str, date]:
    m = _MARKER_PATTERN.match(marker)
    if not m:
        sys.exit(
            f"❌ Marker `{marker}` no sigue formato canónico "
            f"`P<n>-<SLUG>(-<SLUG>)+ · YYYY-MM-DD`.\n"
            f"   Ejemplo válido: `P2-NEW-X · 2026-05-23`"
        )
    prefix = m.group("prefix")
    date_str = m.group("date")
    try:
        marker_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        sys.exit(f"❌ Fecha `{date_str}` no es ISO válida: {e}")
    return prefix, marker_date


def _check_slug_test_exists(slug: str) -> list[Path]:
    matches = list(_TESTS_DIR.glob(f"test_{slug}*.py"))
    return matches


def _update_app_py(new_marker: str, dry_run: bool) -> str:
    text = _APP_PY.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(_LAST_KNOWN_PFIX\s*=\s*)["\'][^"\']+["\']',
        rf'\1"{new_marker}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("❌ No se pudo aplicar el reemplazo en app.py")
    if not dry_run:
        _APP_PY.write_text(new_text, encoding="utf-8")
    return new_text


def _update_test_floor(
    new_marker: str,
    new_date: date,
    floor_comment: str,
    dry_run: bool,
) -> str:
    text = _TEST_FILE.read_text(encoding="utf-8")
    # Pattern: `_PFIX_DATE_FLOOR = date(YYYY, M, D)  # comment`
    new_floor_line = (
        f"_PFIX_DATE_FLOOR = date({new_date.year}, {new_date.month}, {new_date.day})  "
        f"# {floor_comment}"
    )
    new_text, n = re.subn(
        r"_PFIX_DATE_FLOOR\s*=\s*date\(\d{4},\s*\d+,\s*\d+\)\s*#[^\n]*",
        new_floor_line,
        text,
        count=1,
    )
    if n != 1:
        sys.exit("❌ No se pudo aplicar el reemplazo en test_p3_1_*.py")
    if not dry_run:
        _TEST_FILE.write_text(new_text, encoding="utf-8")
    return new_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump _LAST_KNOWN_PFIX + _PFIX_DATE_FLOOR atómicamente",
    )
    parser.add_argument(
        "--marker",
        required=True,
        help="Nuevo marker. Formato: `P<n>-<SLUG>(-<SLUG>)+ · YYYY-MM-DD`. "
             "Ejemplo: `P2-NEW-X · 2026-05-23`",
    )
    parser.add_argument(
        "--floor-comment",
        required=True,
        help="Comment one-line para el floor en el test. Debe describir el "
             "P-fix brevemente.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar cambios sin aplicar.",
    )
    parser.add_argument(
        "--skip-slug-test-check",
        action="store_true",
        help="Skip validation de que existe tests/test_<slug>*.py. Usar solo "
             "cuando el test se crea en el mismo commit.",
    )

    args = parser.parse_args()

    # Validaciones.
    prefix, new_date = _validate_marker(args.marker)
    slug = _marker_to_slug(args.marker)

    current = _read_current_marker()
    print(f"📋 Current  _LAST_KNOWN_PFIX = {current!r}")
    print(f"📋 New      _LAST_KNOWN_PFIX = {args.marker!r}")
    print(f"📋 Slug             = {slug!r}")
    print(f"📋 Floor date       = {new_date}")
    print(f"📋 Floor comment    = {args.floor_comment[:80]}...")

    # Slug cross-link check.
    if not args.skip_slug_test_check:
        slug_matches = _check_slug_test_exists(slug)
        if not slug_matches:
            sys.exit(
                f"\n❌ No existe `tests/test_{slug}*.py` matching el slug "
                f"del marker.\n"
                f"   El test P2-HIST-AUDIT-14 fallaría en CI. Crea el archivo "
                f"de test primero,\n"
                f"   o pasa `--skip-slug-test-check` si lo añades en el mismo "
                f"commit."
            )
        print(f"✓  Slug check OK ({len(slug_matches)} test(s) matching)")

    # Sanity: el nuevo marker NO debe ser igual al current (no bump trivial).
    if current == args.marker:
        sys.exit(
            f"\n⚠️  Nuevo marker es IDÉNTICO al current. "
            f"Bump cosmético no aporta señal. Cambia algún slug o fecha."
        )

    # Sanity: la fecha NO debe retroceder vs el current.
    current_m = _MARKER_PATTERN.match(current)
    if current_m:
        current_date = datetime.strptime(current_m.group("date"), "%Y-%m-%d").date()
        if new_date < current_date:
            sys.exit(
                f"\n❌ Nueva fecha {new_date} es ANTERIOR a la current "
                f"{current_date}.\n"
                f"   Bump retrocedente NO es válido — el marker debe avanzar."
            )

    # Apply.
    if args.dry_run:
        print("\n🔵 DRY RUN — sin cambios aplicados\n")
    else:
        print("\n📝 Aplicando cambios...")

    _update_app_py(args.marker, args.dry_run)
    _update_test_floor(args.marker, new_date, args.floor_comment, args.dry_run)

    print(f"\n✅ {'Dry run completo' if args.dry_run else 'Cambios aplicados'}.")
    print(f"\nSOP post-bump:")
    print(f"  1. git diff app.py tests/test_p3_1_last_known_pfix_freshness.py")
    print(f"  2. git add app.py tests/test_p3_1_last_known_pfix_freshness.py")
    print(f"  3. git commit -m \"feat(backend): bump _LAST_KNOWN_PFIX to {args.marker}\"")
    print(f"  4. git push")
    print(f"  5. Tras deploy: curl /health/version | jq '.last_known_pfix'")
    print(f"     debe matchear {args.marker!r}")
    print(f"  6. Si drift detectado, ver runbook system_alerts_sops.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
