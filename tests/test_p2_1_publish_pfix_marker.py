"""[P2-1 · 2026-05-10] Regression guard: el script
`backend/scripts/publish_pfix_marker.py` automatiza la publicación del
`_LAST_KNOWN_PFIX` actual a `app_kv_store["expected_last_known_pfix"]`,
cerrando el bucle del detector de deploy-lag (P0-1 implementó la mitad
del wiring; esto cierra la otra mitad).

Bug latente que esto cubre:
    P0-1 instaló el cron `_alert_deploy_lag_marker_stale` con dos señales:
      A) marker_stale por edad >168h (no depende del operador).
      B) drift vs `app_kv_store.expected_last_known_pfix` (requiere que
         alguien publique el "expected").
    Sin un script reproducible, el operador olvidaría publicar y la
    señal B queda silenciada permanentemente. P2-1 resuelve eso con un
    script único y testeado para correr post-`git push`.

Cobertura del test:
    1. Script existe en la ruta esperada.
    2. `read_marker_from_app_py()` extrae el marker actual.
    3. Helper usa el MISMO regex que P3-1 freshness y P0-1 detector
       (drift entre parsers significaría detector no-activable).
    4. Script tiene `--dry-run` para CI / pre-flight checks sin escribir
       a DB.
    5. Estructura defensiva: `main()` retorna exit code 0/1 sin levantar.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _BACKEND_ROOT / "scripts" / "publish_pfix_marker.py"


# ---------------------------------------------------------------------------
# 1. Script existe y es importable
# ---------------------------------------------------------------------------
def test_script_file_exists():
    assert _SCRIPT.exists(), (
        f"Script `publish_pfix_marker.py` no encontrado en {_SCRIPT}. "
        f"P2-1 requiere este script para que el operador publique el "
        f"marker tras `git push`."
    )


def test_script_is_importable():
    """Importable como módulo (no rompe en parse)."""
    sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        import scripts.publish_pfix_marker as mod  # noqa: F401
        assert hasattr(mod, "read_marker_from_app_py")
        assert hasattr(mod, "upsert_kv")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_KV_KEY")
        assert mod._KV_KEY == "expected_last_known_pfix"
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# 2. Lectura del marker
# ---------------------------------------------------------------------------
def test_read_marker_returns_current_value():
    sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from scripts.publish_pfix_marker import read_marker_from_app_py
        marker = read_marker_from_app_py()
    finally:
        sys.path.pop(0)
    assert isinstance(marker, str)
    assert marker.strip() == marker
    # Debe seguir el formato Pn-X · YYYY-MM-DD documentado en CLAUDE.md.
    assert re.match(
        r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+\d{4}-\d{2}-\d{2}$",
        marker,
    ), f"Marker `{marker}` no cumple formato de CLAUDE.md."


# ---------------------------------------------------------------------------
# 3. Drift detection: regex del script == regex de los detectores
# ---------------------------------------------------------------------------
def test_script_uses_same_marker_regex_as_freshness_test():
    """El script usa el regex `^_LAST_KNOWN_PFIX\\s*=\\s*["\\']` MULTILINE
    para extraer el marker. Si el regex divergiera del usado en P3-1
    freshness o en el cron P0-1, podríamos publicar el marker correcto
    pero el detector compararía contra otro string → falso positivo
    perpetuo. Anchor de drift detection."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX\\s*=\\s*["\\\']' in src, (
        "El regex del script no usa el patrón canónico. Sincronizar con "
        "test_p3_1_last_known_pfix_freshness.py + cron_tasks._parse_pfix_marker."
    )
    assert "re.MULTILINE" in src, (
        "Debe usar `re.MULTILINE` para que el `^` matchee inicio de línea."
    )


# ---------------------------------------------------------------------------
# 4. --dry-run no requiere DB
# ---------------------------------------------------------------------------
def test_dry_run_does_not_require_db():
    """`--dry-run` debe imprimir el marker y exit 0 sin tocar DB. Útil
    para CI / pre-flight checks."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_BACKEND_ROOT),
    )
    assert result.returncode == 0, (
        f"--dry-run debió exit 0; got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Marker en HEAD" in result.stdout
    assert "dry-run" in result.stdout


# ---------------------------------------------------------------------------
# 5. main() retorna exit code (no levanta sin DB env)
# ---------------------------------------------------------------------------
def test_main_returns_1_when_db_url_missing(monkeypatch):
    """Sin connection string a la DB, `main([])` debe retornar 1
    (no excepción no-capturada).

    [P1-NEON-DB-MIGRATION · 2026-06-12] El script dejó de leer
    `SUPABASE_DB_URL`; ahora resuelve `DATABASE_URL or NEON_DATABASE_URL`
    (ver `upsert_kv`). Hay que borrar las TRES (la vieja Supabase por
    higiene + las dos Neon que el `.env` cargado por `db_core` puebla) para
    que la precondición "sin DB URL" realmente se cumpla; de lo contrario
    `NEON_DATABASE_URL` del `.env` haría que `upsert_kv` conecte y `main`
    retorne 0."""
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from scripts.publish_pfix_marker import main
        rc = main([])
    finally:
        sys.path.pop(0)
    assert rc == 1, "Sin DB URL, main debe exit 1 sin levantar."


# ---------------------------------------------------------------------------
# 6. KV key consistency con el detector P0-1
# ---------------------------------------------------------------------------
def test_kv_key_matches_detector():
    """El script escribe a la key `expected_last_known_pfix` que el cron
    `_alert_deploy_lag_marker_stale` (P0-1) lee. Si difiere, la señal B
    nunca dispara."""
    cron_src = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
    assert '_DEPLOY_LAG_KV_KEY = "expected_last_known_pfix"' in cron_src, (
        "El cron P0-1 cambió la key sin actualizar el script (o vice "
        "versa). Sincronizar `_DEPLOY_LAG_KV_KEY` en cron_tasks.py con "
        "`_KV_KEY` en publish_pfix_marker.py."
    )

    sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from scripts.publish_pfix_marker import _KV_KEY
    finally:
        sys.path.pop(0)
    assert _KV_KEY == "expected_last_known_pfix"
