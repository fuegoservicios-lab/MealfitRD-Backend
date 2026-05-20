"""[P3-AGENT-BUNDLE-CAP · 2026-05-19] Enforcer de tamaño del chunk
AgentPage tras `vite build`.

Pre-fix: el chunk era ~37 KB gzip pero no había CI gate que detectara
una regresión grande (alguien añade una dep heavy, code-split mal hecho,
polyfill que crece sin notar). El audit production-readiness del Agente
(2026-05-19) marcó esto como P3 — útil pero NO bloqueante para ship.

Cierra el último P3 abierto del audit. Los demás del audit ya cerrados:
  - **P3.1** (`role="log" aria-live="polite"` en messages-container) cerrado
    por `[P1-CHAT-A11Y-LIVE · 2026-05-19]` — aplicado en ambas ramas
    (virtualizada y simple) con `aria-relevant="additions text"` +
    `aria-label="Historial de conversación con el asistente"`.
  - **"virtualización >200 mensajes"** del audit ya cubierto por
    `[P1-CHAT-VIRTUALIZE · 2026-05-19]` (threshold 100, react-virtuoso).

Este test valida que el script + npm registration están en el repo
(no corre el build en CI del backend — el script se ejecuta en CI del
frontend tras `npm run build`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_FP = _REPO_ROOT / "frontend" / "scripts" / "check-agent-bundle-size.mjs"
_PACKAGE_JSON_FP = _REPO_ROOT / "frontend" / "package.json"


def _script_src() -> str:
    return _SCRIPT_FP.read_text(encoding="utf-8")


def _package_json() -> dict:
    return json.loads(_PACKAGE_JSON_FP.read_text(encoding="utf-8"))


def test_script_file_exists() -> None:
    """[P3-AGENT-BUNDLE-CAP] el script debe existir en la ruta canónica
    `frontend/scripts/check-agent-bundle-size.mjs`."""
    assert _SCRIPT_FP.exists(), (
        f"[P3-AGENT-BUNDLE-CAP] falta {_SCRIPT_FP}. Si lo moviste, "
        f"actualiza este test + el npm script en package.json."
    )


def test_script_is_esm_module() -> None:
    """[P3-AGENT-BUNDLE-CAP] el script es ESM (`.mjs`) porque
    package.json tiene `"type": "module"` y usamos `import` syntax."""
    assert _SCRIPT_FP.suffix == ".mjs"
    src = _script_src()
    assert "import { " in src or "import {" in src, (
        "[P3-AGENT-BUNDLE-CAP] el script debe usar ESM `import { ... }` syntax."
    )


def test_script_uses_node_stdlib_only() -> None:
    """[P3-AGENT-BUNDLE-CAP] sin deps externas. Solo `node:fs`, `node:zlib`,
    `node:path`, `node:url`. Razón: el script corre en CI antes/después
    del npm install, no debemos depender de devDeps que pueden no estar
    presentes."""
    src = _script_src()
    # Imports válidos (stdlib).
    for stdlib in ("node:fs", "node:zlib", "node:path", "node:url"):
        # Permitido pero NO obligatorio — verifica solo que NO hay imports
        # externos.
        pass
    # Imports prohibidos (devDeps).
    forbidden_patterns = [
        r'from\s+["\']rollup-plugin-visualizer["\']',
        r'from\s+["\']vite-bundle-analyzer["\']',
        r'from\s+["\']gzip-size["\']',
        r'from\s+["\']bundlesize["\']',
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, src), (
            f"[P3-AGENT-BUNDLE-CAP] el script NO debe importar deps externas "
            f"(match: {pat}). Solo `node:fs` + `node:zlib` + `node:path` + "
            f"`node:url` para que corra en CI sin instalar nada extra."
        )


def test_default_cap_and_clamp_constants() -> None:
    """[P3-AGENT-BUNDLE-CAP] constantes canónicas: default 80 KB, clamp
    [10, 1000] KB. Si alguien las cambia, hacerlo intencional (no
    accidental por refactor de globals)."""
    src = _script_src()
    assert "const DEFAULT_CAP_KB = 80;" in src, (
        "[P3-AGENT-BUNDLE-CAP] default debe ser 80 KB (~2× tamaño actual ~37 KB). "
        "Si lo cambias, actualiza este test + la memoria."
    )
    assert "const CLAMP_MIN_KB = 10;" in src
    assert "const CLAMP_MAX_KB = 1000;" in src


def test_env_knob_name_canonical() -> None:
    """[P3-AGENT-BUNDLE-CAP] env var canónica
    `MEALFIT_AGENT_PAGE_GZIP_CAP_KB`. Si renombras, actualiza la doc en
    la memoria + cualquier CI workflow que la setee."""
    src = _script_src()
    assert "MEALFIT_AGENT_PAGE_GZIP_CAP_KB" in src, (
        "[P3-AGENT-BUNDLE-CAP] env var `MEALFIT_AGENT_PAGE_GZIP_CAP_KB` "
        "debe estar referenciada en el script."
    )


def test_script_matches_agent_page_chunks() -> None:
    """[P3-AGENT-BUNDLE-CAP] el filename pattern matchea `AgentPage-*.js`
    (Vite hashea filenames). Si el pattern cambia accidentalmente,
    el script jamás encontraría el chunk y siempre fallaría con
    'no se encontró' — falso positivo en CI."""
    src = _script_src()
    pattern_present = re.search(
        r"/\^AgentPage-\[\\w-\]\+\\\.js\$/",
        src,
    )
    assert pattern_present, (
        "[P3-AGENT-BUNDLE-CAP] el regex `/^AgentPage-[\\w-]+\\.js$/` debe "
        "estar presente en el script. Sin él, no encuentra los chunks "
        "post-Vite-build."
    )


def test_script_uses_gzip_level_9() -> None:
    """[P3-AGENT-BUNDLE-CAP] gzip level 9 (best compression) — refleja
    el comportamiento de servers prod (nginx/CDN típicamente nivel 6-9).
    Level 1 (default Node) sobrestima ~20%; level 9 es realista."""
    src = _script_src()
    assert "level: 9" in src, (
        "[P3-AGENT-BUNDLE-CAP] `gzipSync(raw, { level: 9 })` para que el "
        "tamaño reportado refleje lo que el cliente descarga. Sin `level: 9` "
        "el cap se vuelve muy permisivo (level 1 default es ~20% peor ratio)."
    )


def test_exit_codes() -> None:
    """[P3-AGENT-BUNDLE-CAP] exit 1 en cualquier path de fallo
    (no build, no chunks, excede cap). Exit 0 solo si el cap pasa."""
    src = _script_src()
    # Hay AL MENOS 3 process.exit(1) llamadas (no-dist, no-chunks, excede-cap).
    exit_1_count = len(re.findall(r"process\.exit\(1\)", src))
    assert exit_1_count >= 3, (
        f"[P3-AGENT-BUNDLE-CAP] esperaba ≥3 `process.exit(1)` en el script "
        f"(no-dist, no-chunks, excede-cap, error inesperado). Encontradas: "
        f"{exit_1_count}."
    )


def test_npm_script_registered() -> None:
    """[P3-AGENT-BUNDLE-CAP] el comando `check:bundle-size` está
    registrado en package.json. CI lo invoca como `npm run check:bundle-size`."""
    pkg = _package_json()
    scripts = pkg.get("scripts", {})
    assert "check:bundle-size" in scripts, (
        "[P3-AGENT-BUNDLE-CAP] falta el script `check:bundle-size` en "
        "package.json. Sin él, el CI no puede invocar el enforcer."
    )
    cmd = scripts["check:bundle-size"]
    assert "scripts/check-agent-bundle-size.mjs" in cmd, (
        f"[P3-AGENT-BUNDLE-CAP] el npm script debe apuntar a "
        f"`scripts/check-agent-bundle-size.mjs`. Comando actual: {cmd!r}."
    )


def test_npm_script_uses_node_not_npx() -> None:
    """[P3-AGENT-BUNDLE-CAP] el invoker es `node ...` (NO `npx`).
    `npx` triggea network lookup de paquetes inexistentes en CI cold-cache,
    sumando segundos sin valor. El script es solo Node stdlib."""
    pkg = _package_json()
    cmd = pkg["scripts"]["check:bundle-size"]
    assert cmd.startswith("node "), (
        f"[P3-AGENT-BUNDLE-CAP] el comando debe empezar con `node ` "
        f"(no `npx`, no `tsx`, etc.). Comando actual: {cmd!r}."
    )


def test_tooltip_anchor_present() -> None:
    """[P3-AGENT-BUNDLE-CAP] marker textual preservado en el código
    para que (a) un grep produzca el contexto, (b) el cross-link de
    `test_p2_hist_audit_14_marker_test_link` matchee el slug del test."""
    src = _script_src()
    count = src.count("P3-AGENT-BUNDLE-CAP")
    assert count >= 3, (
        f"[P3-AGENT-BUNDLE-CAP] esperaba ≥3 menciones del marker en el "
        f"script (header comment + tooltip-anchor + log lines). "
        f"Encontradas: {count}."
    )
