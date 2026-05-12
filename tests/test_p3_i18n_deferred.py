"""[P3-I18N-DEFERRED · 2026-05-13] Anchor de la decisión de producto
"i18n: es-DO permanente" tomada tras el audit production-readiness
2026-05-12.

Contexto:
    El audit P3-1 flageó "100% strings hardcoded en español" como deuda
    técnica i18n. La decisión de producto (2026-05-13) fue:
    **deferred-by-design** — el mercado objetivo es República Dominicana
    únicamente, sin roadmap activo de expansión multilocale. Añadir
    `react-i18next` ahora violaría la convención "Don't design for
    hypothetical future requirements" (CLAUDE.md / instrucciones del
    repo) — introduce bundle overhead (~30KB), deuda de mantenimiento
    (cada string nuevo debe pasar por el sistema), y abstracción no-usada.

    Si en el futuro se decide expandir, el refactor incremental cuesta
    lo mismo que el scaffold preventivo de hoy.

Lo que este test enforza:
  A) La sección "Decisiones de producto" existe en CLAUDE.md.
  B) La sub-sección sobre i18n existe con el anchor `P3-I18N-DEFERRED`.
  C) Cero dependencias de i18n en `frontend/package.json` (sanity
     anti-regresión: si alguien añade `react-i18next`, `i18next`,
     `react-intl` u otra lib similar SIN actualizar la doc, el test
     falla con copy explicativo que dice "decisión deferred-by-design,
     reabrir el P3 antes de añadir la dep").

Cómo revertir la decisión (cuando producto decida expandir):
  1. Eliminar la sub-sección de CLAUDE.md (test A fallará, esperado).
  2. Reabrir P3 con scope: scaffold + migración incremental.
  3. Marcar este test como obsoleto o reemplazarlo por el del scaffold.

Tooltip-anchor: P3-I18N-DEFERRED.
"""
from __future__ import annotations

import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"

# Libs de i18n cuyo agregado a package.json reabre el P3.
# Si añades una nueva lib del ecosistema i18n, súmala aquí.
_I18N_LIB_NAMES = (
    "react-i18next",
    "i18next",
    "i18next-browser-languagedetector",
    "i18next-http-backend",
    "react-intl",
    "@formatjs/intl",
    "lingui",
    "@lingui/react",
    "@lingui/core",
    "vue-i18n",  # defensivo cross-framework
    "ngx-translate",  # defensivo cross-framework
)


def _read_claude_md() -> str:
    assert _CLAUDE_MD.exists(), f"CLAUDE.md no encontrado en {_CLAUDE_MD}"
    return _CLAUDE_MD.read_text(encoding="utf-8")


def _read_package_json() -> dict:
    assert _PACKAGE_JSON.exists(), f"package.json no encontrado en {_PACKAGE_JSON}"
    return json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))


# A) Sección "Decisiones de producto" existe.
def test_a_decisiones_de_producto_section_exists():
    src = _read_claude_md()
    assert "## Decisiones de producto" in src, (
        "P3-I18N-DEFERRED: CLAUDE.md perdió la sección "
        "'## Decisiones de producto'. Esta sección es el SSOT de "
        "decisiones que parecen gaps técnicos pero son producto. "
        "Si la moviste, actualizar este test."
    )


# B) Sub-sección sobre i18n con anchor.
def test_b_i18n_decision_subsection_with_anchor():
    src = _read_claude_md()
    assert "P3-I18N-DEFERRED" in src, (
        "P3-I18N-DEFERRED: CLAUDE.md perdió el anchor "
        "`P3-I18N-DEFERRED`. Sin anchor, un futuro audit no sabe "
        "que la decisión está documentada."
    )
    assert "i18n" in src.lower(), (
        "P3-I18N-DEFERRED: CLAUDE.md no menciona 'i18n' en su body."
    )
    assert "es-DO" in src or "es-do" in src.lower(), (
        "P3-I18N-DEFERRED: CLAUDE.md no menciona el locale objetivo "
        "(es-DO). La decisión sin locale concreto es ambigua."
    )


# C) Sanity: cero deps de i18n en package.json.
def test_c_no_i18n_deps_in_package_json():
    pkg = _read_package_json()
    deps = dict(pkg.get("dependencies", {}))
    deps.update(pkg.get("devDependencies", {}))
    deps.update(pkg.get("peerDependencies", {}))
    deps.update(pkg.get("optionalDependencies", {}))

    found_i18n_libs = [name for name in _I18N_LIB_NAMES if name in deps]
    assert not found_i18n_libs, (
        f"P3-I18N-DEFERRED: package.json declara dependencia(s) de "
        f"i18n: {found_i18n_libs}. Pero CLAUDE.md sección 'Decisiones "
        f"de producto' dice que la decisión es 'es-DO permanente' "
        f"(no infra i18n).\n\n"
        f"Si esto es intencional (producto decidió expandir):\n"
        f"  1. Eliminar la sub-sección `P3-I18N-DEFERRED` de CLAUDE.md.\n"
        f"  2. Reabrir el P3 con scope de scaffold i18n + migración "
        f"incremental.\n"
        f"  3. Marcar este test como obsoleto o reemplazarlo.\n\n"
        f"Si NO es intencional (alguien añadió la dep sin pensar):\n"
        f"  1. Quitar la dep del package.json + lockfile.\n"
        f"  2. Re-correr este test para confirmar."
    )
