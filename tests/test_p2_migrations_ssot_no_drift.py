"""[P2-MIGRATIONS-SSOT-LINT · 2026-05-23] Enforce parity between the two
canonical migration directories.

Bug pattern this test prevents (audit 2026-05-23, gap 5 of P2 bundle):
    El workspace usa DOS directorios de migrations mantenidos sincronizados
    por convención (P3-MIGRATIONS-SSOT · 2026-05-20 en CLAUDE.md):

      - `supabase/migrations/`         (workspace-root, cross-repo)
      - `backend/supabase/migrations/` (backend repo, propio remote)

    Razón: workspace-root `.gitignore` excluye `backend/` + `frontend/`
    (son repos hermanos con remotes propios). Para que un `git push`
    desde cada repo lleve la migration al deploy correspondiente,
    necesitamos archivos físicos en AMBOS dirs.

    Antes de 2026-05-20 había drift histórico (4 root-only + 1
    backend-only) detectado solo por audit manual. Sin test
    parser-based, el próximo desvío sería igualmente invisible —
    un dev añade `xxxx.sql` solo a uno de los dirs, ese repo
    queda con la migration y el otro NO. En el siguiente deploy
    el binary que está siendo deployado ve schema diferente al
    código.

Cobertura del test:
    1. Listados de archivos `.sql` en ambos dirs son idénticos
       (set-equality).
    2. Lista no vacía (sanity — un sweep accidental no debería
       dejar ambos dirs vacíos y pasar).
    3. Cada par de archivos con el mismo nombre tiene contenido
       idéntico (defensa-en-profundidad: el nombre puede coincidir
       pero el contenido haber drifteado por un copy-paste fallido).

Limitaciones (out of scope):
    - NO valida que las migrations estén APLICADAS en Supabase
      (eso requiere conectividad — fuera del scope de unit test).
    - NO valida orden de aplicación (los nombres de archivo
      empiezan con prefijo de bundle y son auto-orderable).

Tooltip-anchor: P2-MIGRATIONS-SSOT-LINT-START | drift detection 2026-05-23
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_ROOT_MIGRATIONS = _REPO_ROOT / "supabase" / "migrations"
_BACKEND_MIGRATIONS = _REPO_ROOT / "backend" / "supabase" / "migrations"


def _list_sql_files(directory: Path) -> set[str]:
    """Lista archivos `.sql` directos del directorio (no recursivo —
    los dos canonical dirs son flat por convención)."""
    assert directory.exists(), f"Migration dir no encontrado: {directory}"
    return {f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".sql"}


def test_both_migration_dirs_exist():
    """Sanity: ambos directorios canónicos existen. Si refactoreaste la
    estructura del repo, actualizar las constantes al tope de este test."""
    assert _ROOT_MIGRATIONS.exists(), (
        f"Directorio root-level `{_ROOT_MIGRATIONS}` no existe. "
        "Convención P3-MIGRATIONS-SSOT requiere ambos dirs."
    )
    assert _BACKEND_MIGRATIONS.exists(), (
        f"Directorio backend `{_BACKEND_MIGRATIONS}` no existe. "
        "Convención P3-MIGRATIONS-SSOT requiere ambos dirs."
    )


def test_migration_filenames_in_sync():
    """**Test principal**: el set de filenames `.sql` debe ser
    idéntico entre `supabase/migrations/` y
    `backend/supabase/migrations/`.

    Cuando este test falla, el dev añadió una migration a solo uno
    de los dirs. SOP de resolución (CLAUDE.md → 'SSOT de migrations'):

      1. Identifica el archivo huérfano (mensaje del assert).
      2. Verifica que el contenido sea el intendido (`diff` el archivo
         contra cualquier draft en notas).
      3. Copia el archivo al directorio donde falta. Usa `cp` o copy
         literal — NO regeneres desde memoria (drift de comentarios).
      4. Commit en AMBOS repos: workspace-root (referenciando el path
         root) y backend (referenciando el path backend).
    """
    root_files = _list_sql_files(_ROOT_MIGRATIONS)
    backend_files = _list_sql_files(_BACKEND_MIGRATIONS)

    only_in_root = root_files - backend_files
    only_in_backend = backend_files - root_files

    msg_parts = []
    if only_in_root:
        files_list = "\n        - ".join(sorted(only_in_root))
        msg_parts.append(
            f"\n    Solo en `supabase/migrations/` (root) — falta en "
            f"backend:\n        - {files_list}"
        )
    if only_in_backend:
        files_list = "\n        - ".join(sorted(only_in_backend))
        msg_parts.append(
            f"\n    Solo en `backend/supabase/migrations/` — falta en "
            f"root:\n        - {files_list}"
        )

    assert not (only_in_root or only_in_backend), (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Migration SSOT drift detectado (P3-MIGRATIONS-SSOT)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{''.join(msg_parts)}\n"
        f"\n    Resolución: copiar el archivo huérfano al dir que falta.\n"
        f"    Convención completa: CLAUDE.md → 'SSOT de migrations'.\n"
    )


def test_migration_list_non_empty():
    """Sanity: el listado no debe estar vacío. Un sweep accidental
    (e.g., `rm -rf migrations/`) que vacíe ambos dirs pasaría
    `test_migration_filenames_in_sync` falsamente."""
    root_files = _list_sql_files(_ROOT_MIGRATIONS)
    assert len(root_files) >= 10, (
        f"Solo {len(root_files)} migrations en `{_ROOT_MIGRATIONS}`. "
        "Esperaba >=10 (el repo tiene ~40+ históricos). ¿Fue accidentalmente "
        "wipeado? Restaurar desde git history."
    )


def test_migration_file_contents_in_sync():
    """Defensa-en-profundidad: cada par filename-coincidente debe tener
    contenido byte-idéntico. Un copy-paste fallido (e.g., editar el archivo
    en uno de los dirs y olvidar propagar) dejaría los dos repos con
    schema divergente aunque el filename matchee."""
    root_files = _list_sql_files(_ROOT_MIGRATIONS)
    backend_files = _list_sql_files(_BACKEND_MIGRATIONS)
    common = root_files & backend_files

    drifted: list[tuple[str, int, int]] = []
    for name in sorted(common):
        root_text = (_ROOT_MIGRATIONS / name).read_bytes()
        backend_text = (_BACKEND_MIGRATIONS / name).read_bytes()
        if root_text != backend_text:
            drifted.append((name, len(root_text), len(backend_text)))

    if drifted:
        lines = "\n".join(
            f"        - {name}: root={r}B vs backend={b}B (diff={abs(r-b)}B)"
            for name, r, b in drifted
        )
        raise AssertionError(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Migration content drift detectado ({len(drifted)} archivos)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{lines}\n\n"
            f"    Resolución: identificar la versión canónica (la más reciente\n"
            f"    en git log) y copiar a ambos dirs. Bajo ninguna circunstancia\n"
            f"    apliques una sola de las dos versiones a la BD productiva\n"
            f"    sin reconciliar — sería schema-drift entre repos.\n"
        )
