"""[P3-I18N-DEFERRED · 2026-05-13 · SUPERSEDED por P1-I18N-DASHBOARD 2026-08-15]

## Qué pasó

Este archivo nació guardando la decisión de producto **«i18n: es-DO permanente»**:
el mercado era República Dominicana, no había roadmap multilocale, y añadir
`react-i18next` habría sido diseñar para un requisito hipotético. El test hacía
tres cosas: comprobar que la decisión seguía escrita en CLAUDE.md y fallar si
alguien metía una librería de i18n en `package.json` sin reabrir el debate.

El 2026-08-15 el dueño revirtió la decisión: el dashboard se lee en 5 idiomas
(es-DO base, en-US, pt-BR, fr-FR, it-IT), con selector en Configuración → Idioma.

## Por qué este archivo NO se borró

Porque su assertion mecánica —**cero librerías de i18n en `package.json`**— sigue
siendo exactamente la correcta, pero por una razón nueva y más fuerte que la
original.

`P1-I18N-DASHBOARD` no se construyó sobre `react-i18next`. Se construyó con un
motor propio de ~250 líneas (`frontend/src/i18n/`) por dos motivos que un futuro
lector desharía sin querer:

1. **Bytes.** La librería son ~30 kB gz de backends HTTP, detección de idioma,
   namespaces y Suspense — nada de lo cual se usa. Este repo pasó agosto entero
   recuperando bytes del entry: `P1-APEX-ENTRY-DIET` (33 kB sacando `@sentry` de
   cinco puertas), `P2-LANDING-OLA1-DIET` (181 iconos fuera del vendor chunk).
2. **La clave es el texto español.** El motor propio permite que `es-DO` no tenga
   catálogo: es el fallback, y el 100% de la base actual descarga cero bytes de
   i18n. Una librería con claves simbólicas obliga a un catálogo es-DO y le
   cobra ese peso al usuario dominicano, que es el único que hoy existe.

O sea: si mañana alguien añade `react-i18next` «para hacerlo bien», estaría
pagando 30 kB por duplicar un motor que ya funciona, y probablemente migrando a
claves simbólicas —lo que reintroduce el catálogo es-DO que este diseño evita—.
Eso es lo que este test frena ahora. El fallo dice qué mirar antes.

## Lo que enforza hoy

  A) La sección «Decisiones de producto» sigue existiendo en CLAUDE.md.
  B) El ancla vieja `P3-I18N-DEFERRED` sigue presente y marcada SUPERSEDED, con
     el ancla nueva al lado. Quien grepee la vieja tiene que aterrizar en el
     supersede, no en el vacío — es la convención del repo para decisiones
     revertidas (ver `P3-LANDING-DARK-ONLY`, `P3-CHAT-SAFETY-OFF-DECISION`).
  C) Cero dependencias de librerías i18n en `frontend/package.json`.

Tooltip-anchor: P3-I18N-DEFERRED.
"""
from __future__ import annotations

import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"
_ENGINE_DIR = _REPO_ROOT / "frontend" / "src" / "i18n"

# Libs de i18n cuyo agregado a package.json reabre el debate.
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


# B) El ancla vieja sobrevive, marcada como superseded, junto a la nueva.
def test_b_ancla_vieja_marcada_superseded_con_la_nueva_al_lado():
    src = _read_claude_md()

    assert "P3-I18N-DEFERRED" in src, (
        "P3-I18N-DEFERRED: CLAUDE.md perdió el ancla `P3-I18N-DEFERRED`. Aunque "
        "la decisión esté revertida, el ancla se CONSERVA marcada SUPERSEDED: "
        "hay comentarios de código, memorias y este propio test que la citan, y "
        "quien la grepee tiene que aterrizar en el supersede en vez de en el "
        "vacío. Es la convención del repo (ver `P3-LANDING-DARK-ONLY`)."
    )
    assert "P1-I18N-DASHBOARD" in src, (
        "P3-I18N-DEFERRED: falta el ancla `P1-I18N-DASHBOARD`, que es la decisión "
        "que reemplazó a esta. Un 'SUPERSEDED' sin decir POR QUÉ deja al lector "
        "sabiendo que la regla ya no vale pero no cuál la sustituye."
    )

    # El ancla vieja y la palabra SUPERSEDED tienen que estar en la MISMA
    # entrada, no sueltas por el documento.
    i_old = src.find("P3-I18N-DEFERRED")
    ventana = src[max(0, i_old - 200): i_old + 400]
    assert "SUPERSEDED" in ventana.upper(), (
        "P3-I18N-DEFERRED: el ancla existe pero no dice SUPERSEDED cerca. Una "
        "decisión revertida cuya entrada sigue leyéndose como vigente es peor "
        "que ninguna entrada: manda a alguien a 'arreglar' el selector de idioma "
        "citando la doc."
    )

    assert "es-DO" in src, (
        "P3-I18N-DEFERRED: CLAUDE.md ya no menciona el locale base (es-DO). "
        "Sigue importando tras el supersede: es-DO es el idioma BASE y el único "
        "SIN catálogo — las claves del código son su texto."
    )


# C) Cero librerías de i18n: el motor es propio.
def test_c_no_i18n_libs_in_package_json():
    pkg = _read_package_json()
    deps = dict(pkg.get("dependencies", {}))
    deps.update(pkg.get("devDependencies", {}))
    deps.update(pkg.get("peerDependencies", {}))
    deps.update(pkg.get("optionalDependencies", {}))

    found_i18n_libs = [name for name in _I18N_LIB_NAMES if name in deps]
    assert not found_i18n_libs, (
        f"P3-I18N-DEFERRED (superseded por P1-I18N-DASHBOARD): package.json "
        f"declara librería(s) de i18n: {found_i18n_libs}.\n\n"
        f"El dashboard YA es multiidioma — pero con motor PROPIO "
        f"(frontend/src/i18n/, ~250 líneas, cero deps). Añadir una librería "
        f"encima significa una de dos cosas, y las dos son un retroceso:\n\n"
        f"  1. Duplicas un motor que ya funciona, pagando ~30 kB gz de backends "
        f"HTTP, detección de idioma, namespaces y Suspense que no se usan — "
        f"justo lo que P1-APEX-ENTRY-DIET y P2-LANDING-OLA1-DIET pasaron agosto "
        f"recuperando.\n"
        f"  2. Migras a claves simbólicas (`t('settings.save')`), lo que OBLIGA "
        f"a crear un catálogo es-DO y le cobra ese peso al usuario dominicano, "
        f"que hoy descarga CERO bytes de i18n porque el español es el fallback.\n\n"
        f"Si de verdad hace falta (p.ej. formato ICU completo con género y "
        f"selectores anidados, que el motor propio no cubre):\n"
        f"  1. Leer backend/docs/i18n_dashboard.md §3 antes de decidir.\n"
        f"  2. Migrar las claves y BORRAR frontend/src/i18n/index.js — dos "
        f"motores conviviendo es peor que cualquiera de los dos.\n"
        f"  3. Actualizar este test y la entrada de CLAUDE.md."
    )


# D) El motor propio sigue ahí (si desaparece, el test C guarda un vacío).
def test_d_el_motor_propio_existe():
    """Sin esto, borrar el motor Y no añadir librería dejaría C en verde.

    Un guard que puede quedarse en verde mientras la funcionalidad que protege
    desaparece no informa de nada.
    """
    assert (_ENGINE_DIR / "index.js").exists(), (
        "P1-I18N-DASHBOARD: falta `frontend/src/i18n/index.js`. El test C exige "
        "cero librerías de i18n PORQUE existe un motor propio; sin el motor, esa "
        "exigencia solo garantiza que la app no tiene idiomas."
    )
    assert (_ENGINE_DIR / "locales.js").exists(), (
        "P1-I18N-DASHBOARD: falta `frontend/src/i18n/locales.js`, el SSOT de la "
        "lista de idiomas del que dependen otros cuatro sitios (boot de "
        "index.html, CHECK de la migración ×2, `_LOCALE_VALUES` del backend). "
        "Ver test_p1_i18n_dashboard.py."
    )
