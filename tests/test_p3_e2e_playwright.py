"""[P3-E2E-PLAYWRIGHT · 2026-05-12] Anchor + regression guard.

El repo debe tener cobertura e2e mínima vía Playwright para detectar
modos de fallo que escapan Vitest unit + parser-based tests (hydration
crash, blank page tras deploy, SPA rewrite roto, FOUT de fuentes).

Defensas que el test enforza:
  1. Anchor `P3-E2E-PLAYWRIGHT` en config + spec.
  2. `frontend/playwright.config.js` existe con `defineConfig({...})` valid.
  3. `frontend/e2e/golden_path.spec.js` existe con al menos 1 `test(...)`.
  4. `frontend/package.json` declara `@playwright/test` en devDependencies
     y los scripts `test:e2e`, `test:e2e:install`.
  5. La config apunta a `testDir: './e2e'` y `baseURL` http(s).

Test parser-based — no instala Playwright ni ejecuta el browser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYWRIGHT_CONFIG = _REPO_ROOT / "frontend" / "playwright.config.js"
_SPEC = _REPO_ROOT / "frontend" / "e2e" / "golden_path.spec.js"
_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"
_README = _REPO_ROOT / "frontend" / "e2e" / "README.md"


def test_playwright_config_exists():
    assert _PLAYWRIGHT_CONFIG.is_file(), (
        f"Falta {_PLAYWRIGHT_CONFIG.relative_to(_REPO_ROOT)}. "
        "Sin config, `npm run test:e2e` fallaría."
    )


def test_anchor_present_in_config():
    src = _PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    assert "P3-E2E-PLAYWRIGHT" in src


def test_config_uses_defineconfig():
    src = _PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    assert re.search(r"defineConfig\s*\(", src), (
        "playwright.config.js debe usar `defineConfig({...})` de @playwright/test."
    )


def test_config_test_dir_points_to_e2e():
    src = _PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    pat = re.compile(r"testDir\s*:\s*['\"]\./e2e['\"]")
    assert pat.search(src), (
        "playwright.config.js debe declarar `testDir: './e2e'` (path canónico)."
    )


def test_config_declares_webserver_or_baseurl_override():
    """Debe haber `webServer` para auto-arrancar `npm run preview` O un
    `baseURL` override por env var. Sin esto, los tests intentan conectar
    a un server que no existe y fallan inmediatamente."""
    src = _PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    has_ws = "webServer" in src
    has_baseurl_env = "PLAYWRIGHT_BASE_URL" in src
    assert has_ws and has_baseurl_env, (
        "playwright.config.js debe tener BOTH `webServer` (auto-arrancar preview) "
        "Y `PLAYWRIGHT_BASE_URL` env var override (para staging/prod)."
    )


def test_golden_path_spec_exists():
    assert _SPEC.is_file(), (
        f"Falta {_SPEC.relative_to(_REPO_ROOT)}. "
        "Smoke spec del golden-path es el contrato mínimo de P3-E2E."
    )


def test_anchor_present_in_spec():
    src = _SPEC.read_text(encoding="utf-8")
    assert "P3-E2E-PLAYWRIGHT" in src


def test_spec_has_at_least_one_test():
    src = _SPEC.read_text(encoding="utf-8")
    # `test('name', async ({ page }) => {...})` o `test.describe(...)` con
    # tests anidados — al menos 1 caso ejecutable.
    pat = re.compile(r"^\s*test\(\s*['\"]", re.MULTILINE)
    matches = pat.findall(src)
    assert len(matches) >= 1, (
        f"Spec golden-path debe tener >=1 `test(...)`. Encontrados: {len(matches)}."
    )


def test_spec_covers_critical_regressions():
    """El spec debe cubrir las regresiones específicas que cierran modos
    de fallo conocidos: pageerror listener (P0-FRONTEND-ANALYTICS) y
    no requests a fonts.gstatic.com (P3-SELF-HOST-FONTS)."""
    src = _SPEC.read_text(encoding="utf-8")
    # pageerror listener (process.env crash regression guard)
    assert "pageerror" in src, (
        "Spec debe registrar listener `pageerror` para capturar JS errors "
        "durante hydration (P0-FRONTEND-ANALYTICS regression guard)."
    )
    # fonts.gstatic.com check (self-host regression guard)
    assert "fonts.gstatic.com" in src or "fonts.googleapis.com" in src, (
        "Spec debe verificar que NO se hace request a fonts remotos "
        "(P3-SELF-HOST-FONTS regression guard)."
    )


def test_package_json_declares_playwright_dev_dep():
    pkg = json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))
    dev_deps = pkg.get("devDependencies", {})
    assert "@playwright/test" in dev_deps, (
        "package.json debe declarar `@playwright/test` en devDependencies."
    )


def test_package_json_declares_e2e_scripts():
    pkg = json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))
    scripts = pkg.get("scripts", {})
    for required in ("test:e2e", "test:e2e:install"):
        assert required in scripts, (
            f"package.json scripts falta `{required}`. "
            "Sin él, contributors no saben cómo ejecutar/instalar Playwright."
        )


def test_e2e_readme_exists():
    """README documentando setup local + scope + follow-up es parte del
    contrato. Sin él, un contributor nuevo no sabe por dónde empezar."""
    assert _README.is_file(), (
        f"Falta {_README.relative_to(_REPO_ROOT)}. "
        "Documenta setup local + scope + follow-up para staging Supabase."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-E2E-PLAYWRIGHT" in src
