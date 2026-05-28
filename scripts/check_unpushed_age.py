"""[P2-UNPUSHED-AGE · 2026-05-26] Detecta commits locales sin push (y archivos
unstaged/staged sin commit) cuya edad exceda un umbral. Tooling local del
operador para prevenir el modo de fallo "deploy_lag_drift_vs_expected" causado
por commits olvidados sin push.

Contexto del incidente que motivó este script (audit P0-AUDIT 2026-05-25):
    El operador cerró bundle P0-REGEN-BILLING + P1-COH-BENIGN-SKIP en local pero
    NO ejecutó `git push`. 2 commits ahead + 12 files unstaged quedaron sin
    propagar a `origin/main`. El binario en EasyPanel seguía corriendo el commit
    anterior; el cron `_alert_deploy_lag_marker_stale` disparó alert
    `deploy_lag_drift_vs_expected` correctamente, pero el operador asumió
    que el problema era Nixpacks cache hit hasta hacer forensic.

    Lección: el cron del backend NO puede detectar este modo (vive dentro del
    binario deployado, no tiene acceso al árbol git local). El operador debe
    chequear ANTES de hacer push o como parte de su SOP regular.

Uso operacional:

    # Reporte humano:
    py backend/scripts/check_unpushed_age.py

    # Como git hook (en `.git/hooks/pre-push` o `pre-commit`):
    py backend/scripts/check_unpushed_age.py --max-age-hours 24 || exit 1

    # Custom threshold + branch:
    py backend/scripts/check_unpushed_age.py --max-age-hours 48 --branch main

Argumentos:
    --max-age-hours N   Falla si el commit unpushed más antiguo tiene edad > N
                        horas (default 24).
    --branch NAME       Compara HEAD contra origin/<NAME> (default detecta
                        upstream del current branch via @{u}; falla si no hay
                        upstream).
    --include-dirty     También considera unstaged/staged files dirty como
                        "age" del último mtime (útil pre-commit hook).

Exit code:
    0 — todo OK o no aplica (sin commits ahead, sin dirty files).
    1 — commit unpushed más antiguo o dirty file más antiguo excede umbral.
    2 — error de invocación / repo no encontrado / upstream sin configurar.

Diseño:
    NO depende del runtime backend (zero imports de cron_tasks/db/etc.). Cualquier
    Python 3.10+ + git instalado lo corre. Pensado para correr en local del
    operador o en un git hook, NO en el contenedor desplegado.

Tooltip-anchor: P2-UNPUSHED-AGE.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _git(*args: str, cwd: Optional[Path] = None, strip: bool = True) -> str:
    """Ejecuta `git <args...>` y retorna stdout. Si `strip=True` (default)
    aplica `.strip()` global; ojo: eso quita whitespace al inicio del primer
    output (relevante en `git status --porcelain` donde XY puede empezar con
    espacio). Para esos casos usar `strip=False` y splitlines manual."""
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
    )
    return result.stdout.strip() if strip else result.stdout


def _git_root() -> Optional[Path]:
    """Retorna el toplevel del repo donde estamos, o None si no es repo."""
    try:
        root = _git("rev-parse", "--show-toplevel")
        return Path(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _detect_upstream(branch_arg: Optional[str]) -> Optional[str]:
    """Devuelve el ref upstream a comparar (e.g. `origin/main`) o None si no hay.
    Si `--branch` está dado, retorna `origin/<branch>` directo."""
    if branch_arg:
        return f"origin/{branch_arg}"
    try:
        upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        return upstream
    except subprocess.CalledProcessError:
        return None


def _unpushed_commits(upstream: str) -> list[str]:
    """Lista de SHAs ahead de upstream (ordenadas oldest-first)."""
    try:
        out = _git("rev-list", "--reverse", f"{upstream}..HEAD")
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _commit_age_hours(sha: str) -> float:
    """Edad del commit en horas (UTC)."""
    try:
        ts_str = _git("log", "-1", "--format=%ct", sha)
        commit_ts = int(ts_str)
        return (time.time() - commit_ts) / 3600.0
    except (subprocess.CalledProcessError, ValueError):
        return -1.0


def _dirty_files() -> list[tuple[str, str]]:
    """Lista (status, path) de archivos modificados/añadidos pero no commiteados.
    Filtra archivos generados (.pyc, __pycache__, *.log).

    IMPORTANTE: usa `strip=False` en `_git` porque `git status --porcelain`
    emite cada línea con formato `XY <space> <path>` donde X puede ser `' '`
    (espacio = sin modif staged). Un `.strip()` global comería ese espacio
    inicial de la PRIMERA línea, rompiendo el slicing solo de esa.
    """
    try:
        out = _git("status", "--porcelain", "-uno", strip=False)
    except subprocess.CalledProcessError:
        return []
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line or len(line) < 4:
            continue
        status = line[:2].strip()
        path = line[3:].strip()
        # Filtrar artifacts:
        if path.endswith(('.pyc', '.log')) or '__pycache__' in path:
            continue
        rows.append((status, path))
    return rows


def _dirty_max_mtime_hours(paths: list[str], root: Path) -> float:
    """Edad en horas del archivo dirty con MTIME más antiguo. Sirve como proxy
    para "dirty hace X horas sin commit"."""
    oldest = -1.0
    now = time.time()
    for rel in paths:
        full = root / rel
        try:
            mtime = full.stat().st_mtime
        except OSError:
            continue
        age_h = (now - mtime) / 3600.0
        if oldest < 0 or age_h > oldest:
            oldest = age_h
    return oldest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Falla con exit 1 si hay commits locales unpushed o dirty files "
            "con edad > umbral. Pensado como tooling local pre-deploy / git hook."
        )
    )
    parser.add_argument(
        "--max-age-hours", type=float, default=24.0,
        help="Umbral en horas. Default 24.",
    )
    parser.add_argument(
        "--branch", type=str, default=None,
        help="Branch upstream (default detecta @{u}).",
    )
    parser.add_argument(
        "--include-dirty", action="store_true",
        help="También considera dirty files (unstaged/staged).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Solo emite warnings/errors; suprime reportes OK.",
    )
    args = parser.parse_args(argv)

    root = _git_root()
    if root is None:
        print(
            "[P2-UNPUSHED-AGE] ERROR: directorio actual no es un repo git "
            "(o `git` no disponible en PATH).",
            file=sys.stderr,
        )
        return 2

    upstream = _detect_upstream(args.branch)
    if upstream is None:
        print(
            "[P2-UNPUSHED-AGE] ERROR: no se detectó upstream para HEAD. "
            "Usa `--branch <name>` o configura tracking: "
            "`git branch --set-upstream-to=origin/<name>`.",
            file=sys.stderr,
        )
        return 2

    # 1. Commits ahead del upstream.
    unpushed = _unpushed_commits(upstream)
    oldest_commit_age_h = -1.0
    oldest_commit_sha = None
    if unpushed:
        oldest_commit_sha = unpushed[0]
        oldest_commit_age_h = _commit_age_hours(oldest_commit_sha)

    # 2. Dirty files (si --include-dirty).
    dirty: list[tuple[str, str]] = []
    oldest_dirty_age_h = -1.0
    if args.include_dirty:
        dirty = _dirty_files()
        if dirty:
            paths = [p for (_, p) in dirty]
            oldest_dirty_age_h = _dirty_max_mtime_hours(paths, root)

    # 3. Decision + reporte.
    threshold = float(args.max_age_hours)
    lag_commits = oldest_commit_age_h > threshold
    lag_dirty = oldest_dirty_age_h > threshold

    if not unpushed and not dirty:
        if not args.quiet:
            print(
                f"[P2-UNPUSHED-AGE] OK: 0 commits ahead, 0 dirty files. "
                f"Upstream={upstream}."
            )
        return 0

    # Reporte detallado.
    print(f"[P2-UNPUSHED-AGE] Upstream={upstream}, max_age_hours={threshold}.")
    if unpushed:
        oldest_short = oldest_commit_sha[:8] if oldest_commit_sha else "?"
        sigil = "[WARN]" if lag_commits else "[info]"
        print(
            f"  {sigil}  {len(unpushed)} commit(s) unpushed. "
            f"Oldest={oldest_short} age={oldest_commit_age_h:.1f}h."
        )
        for sha in unpushed[:5]:
            short = sha[:8]
            try:
                msg = _git("log", "-1", "--format=%s", sha)[:80]
            except subprocess.CalledProcessError:
                msg = "(no message)"
            print(f"     - {short}  {msg}")
        if len(unpushed) > 5:
            print(f"     ... +{len(unpushed) - 5} más")
    if dirty:
        sigil = "[WARN]" if lag_dirty else "[info]"
        print(
            f"  {sigil}  {len(dirty)} dirty file(s). "
            f"Oldest mtime age={oldest_dirty_age_h:.1f}h."
        )
        for status, path in dirty[:10]:
            print(f"     - [{status}] {path}")
        if len(dirty) > 10:
            print(f"     ... +{len(dirty) - 10} más")

    if lag_commits or lag_dirty:
        print(
            f"[P2-UNPUSHED-AGE] FAIL: edad > {threshold}h. "
            f"Considera `git push` antes de declarar el bundle cerrado.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(
            f"[P2-UNPUSHED-AGE] OK (within threshold {threshold}h)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
