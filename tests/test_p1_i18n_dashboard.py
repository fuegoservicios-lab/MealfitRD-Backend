"""[P1-I18N-DASHBOARD · 2026-08-15] La lista de idiomas vive en CINCO sitios.

El dashboard puede cambiar de idioma. La lista de idiomas soportados es el dato
que hace funcionar esa función, y por la naturaleza del stack acaba escrita en
cinco lugares distintos, en tres lenguajes:

  1. `frontend/src/i18n/locales.js`            → SSOT (JS)
  2. `frontend/index.html`                     → boot síncrono anti-parpadeo (JS inline)
  3. `migrations/p1_i18n_dashboard_locale_*`   → CHECK de la columna (SQL)
  4. `backend/migrations/…` (misma migración)  → P3-MIGRATIONS-SSOT
  5. `backend/routers/user_data.py`            → `_LOCALE_VALUES` (Python)

No se pueden colapsar: el boot corre antes de que exista ningún módulo, el CHECK
tiene que estar en SQL para proteger a escritores que no pasen por el endpoint, y
el backend valida sin poder importar JS. Lo que sí se puede es EXIGIR QUE
COINCIDAN, que es lo que hace este archivo.

Es la misma clase de drift que P1-DIET-CANON-SSOT: había tres tablas de
canonicalización de dieta escritas a mano, drifearon, y a la del filtro se le
olvidó 'vegetariana' — el sistema servía Pollo a vegetarianas. Aquí el fallo
equivalente es más leve pero igual de silencioso: añades un idioma en `locales.js`
y en el selector, el usuario lo elige, y el PATCH devuelve 400 (o peor: el CHECK
revienta con un 500 de psycopg) porque el backend nunca se enteró.

Tooltip-anchor: P1-I18N-DASHBOARD.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND = _REPO_ROOT / "frontend"
_LOCALES_JS = _FRONTEND / "src" / "i18n" / "locales.js"
_INDEX_HTML = _FRONTEND / "index.html"
_USER_DATA_PY = _REPO_ROOT / "backend" / "routers" / "user_data.py"
_MIGRATION_NAME = "p1_i18n_dashboard_locale_2026_08_15.sql"
_MIGRATION_ROOT = _REPO_ROOT / "migrations" / _MIGRATION_NAME
_MIGRATION_BACKEND = _REPO_ROOT / "backend" / "migrations" / _MIGRATION_NAME

# El idioma base del producto. Si esto cambia, cambia el producto entero:
# `es-DO` es el único locale SIN catálogo (las claves del código son su texto).
_BASE_LOCALE = "es-DO"

# Alias legible para el test que explica por qué el CÓDIGO no se neutraliza
# aunque la ETIQUETA sí (ver test_a5).
DEFAULT_LOCALE_ES_DO = _BASE_LOCALE


def _read(p: Path) -> str:
    assert p.exists(), f"P1-I18N-DASHBOARD: no existe {p}"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Extractores — uno por lenguaje. Cada uno ancla su sitio.
# ---------------------------------------------------------------------------

def _locales_from_ssot() -> list[str]:
    """`{ code: 'xx-YY', native: '…' }` en locales.js, EN ORDEN."""
    src = _read(_LOCALES_JS)
    codes = re.findall(r"\{\s*code:\s*'([a-z]{2}-[A-Z]{2})'", src)
    assert codes, (
        "P1-I18N-DASHBOARD: no encontré ninguna entrada `{ code: 'xx-YY' }` en "
        f"{_LOCALES_JS}. Si cambiaste la forma del array LOCALES, actualiza este "
        "extractor — es el SSOT del que dependen los otros cuatro sitios."
    )
    return codes


def _locales_from_index_html() -> list[str]:
    """El array `SUPPORTED` del boot inline."""
    src = _read(_INDEX_HTML)
    m = re.search(r"var\s+SUPPORTED\s*=\s*\[([^\]]+)\]", src)
    assert m, (
        "P1-I18N-DASHBOARD: el boot de idioma de index.html perdió su `var "
        "SUPPORTED = [...]`. Ese bloque fija <html lang> ANTES del primer paint; "
        "sin él, un usuario en francés es anunciado en español por el lector de "
        "pantalla hasta que React monta."
    )
    return re.findall(r"'([a-z]{2}-[A-Z]{2})'", m.group(1))


def _locales_from_sql(path: Path) -> list[str]:
    """La lista del CHECK `locale IN (...)`."""
    src = _read(path)
    m = re.search(r"CHECK\s*\(\s*locale\s+IN\s*\(([^)]+)\)", src, re.IGNORECASE)
    assert m, (
        f"P1-I18N-DASHBOARD: {path.name} perdió su `CHECK (locale IN (...))`. "
        "Ese CHECK es lo que impide que un escritor que NO pase por el endpoint "
        "(script de soporte, backfill, endpoint futuro que reutilice el whitelist "
        "de escalares) deje un locale inválido en la columna."
    )
    return re.findall(r"'([a-z]{2}-[A-Z]{2})'", m.group(1))


def _locales_from_backend() -> list[str]:
    """El frozenset `_LOCALE_VALUES`."""
    src = _read(_USER_DATA_PY)
    m = re.search(r"_LOCALE_VALUES\s*=\s*frozenset\(\{([^}]+)\}\)", src)
    assert m, (
        "P1-I18N-DASHBOARD: `_LOCALE_VALUES` desapareció de routers/user_data.py. "
        "Sin él, el whitelist de escalares acepta cualquier valor para `locale` y "
        "el 400 legible lo sustituye un 500 crudo del CHECK de la DB."
    )
    return re.findall(r'"([a-z]{2}-[A-Z]{2})"', m.group(1))


# ---------------------------------------------------------------------------
# A) El SSOT existe y está bien formado
# ---------------------------------------------------------------------------

def test_a_ssot_declara_es_do_como_base_y_primero():
    src = _read(_LOCALES_JS)
    m = re.search(r"DEFAULT_LOCALE\s*=\s*'([^']+)'", src)
    assert m, "P1-I18N-DASHBOARD: locales.js perdió `DEFAULT_LOCALE`."
    assert m.group(1) == _BASE_LOCALE, (
        f"P1-I18N-DASHBOARD: DEFAULT_LOCALE es {m.group(1)!r}, esperaba "
        f"{_BASE_LOCALE!r}. El idioma base es el ÚNICO sin catálogo: las claves "
        "del código SON su texto. Cambiarlo sin migrar todas las claves deja la "
        "app entera sin traducción."
    )

    codes = _locales_from_ssot()
    assert codes[0] == _BASE_LOCALE, (
        f"P1-I18N-DASHBOARD: el primer idioma de LOCALES es {codes[0]!r}. El base "
        "va primero: es el orden en que se pinta el selector y el usuario "
        "dominicano —el 100% de la base actual— debe ver el suyo arriba."
    )


def test_a2_sin_codigos_duplicados():
    codes = _locales_from_ssot()
    dupes = {c for c in codes if codes.count(c) > 1}
    assert not dupes, (
        f"P1-I18N-DASHBOARD: códigos duplicados en LOCALES: {sorted(dupes)}. "
        "El selector pintaría dos filas idénticas."
    )


def test_a4_el_parentesis_de_pais_solo_donde_hay_algo_que_desambiguar():
    """[P1-I18N-LABEL-NEUTRAL · 2026-08-15] Un idioma con UNA sola variante no
    lleva país; uno con DOS, las dos lo llevan.

    La etiqueta original era «Español (República Dominicana)», y a un cliente
    español eso le dice «esto no es para ti» — justo lo contrario de lo que hace
    falta si el producto se vende fuera de RD. Quitarlo solo del español dejaba
    a los otros cuatro con país, que se lee como descuido; así que la regla es
    general y esto la enforza en las dos direcciones:

      · Una lengua con una sola variante y país en la etiqueta → sobra el país.
      · Dos variantes de la misma lengua y solo una con país → el usuario tiene
        que ADIVINAR cuál es cuál. Es el caso peor y el que más probable es que
        aparezca al añadir es-ES.
    """
    src = _read(_LOCALES_JS)
    entries = re.findall(
        r"\{\s*code:\s*'([a-z]{2})-[A-Z]{2}',\s*native:\s*'([^']+)'", src
    )
    assert entries, "P1-I18N-DASHBOARD: no pude leer los pares (código, etiqueta)."

    por_lengua: dict[str, list[tuple[str, str]]] = {}
    for lang, native in entries:
        por_lengua.setdefault(lang, []).append((lang, native))

    for lang, filas in por_lengua.items():
        con_pais = [n for _, n in filas if "(" in n]
        if len(filas) == 1:
            assert not con_pais, (
                f"P1-I18N-LABEL-NEUTRAL: «{con_pais[0]}» lleva paréntesis de país "
                f"siendo la ÚNICA variante de '{lang}'. El paréntesis existe para "
                "desambiguar; sin gemela no desambigua nada y sí acota el mercado "
                "(fue el caso de «Español (República Dominicana)»)."
            )
        else:
            assert len(con_pais) == len(filas), (
                f"P1-I18N-LABEL-NEUTRAL: '{lang}' tiene {len(filas)} variantes pero "
                f"solo {len(con_pais)} llevan país. O todas o ninguna: una lista "
                "donde una variante lo lleva y su gemela no obliga al usuario a "
                "deducir cuál es cuál."
            )


def test_a5_el_codigo_base_sigue_siendo_es_do_por_el_formato_de_numeros():
    """El CÓDIGO no sigue a la etiqueta, y hay una razón medida.

    `Intl` formatea `es-DO` como 2,000 / 1,234.5 (convención de EE.UU., que es
    la dominicana) y `es`/`es-ES` como 2000 / 1234,5. «Neutralizar» el código a
    `es` porque la etiqueta se neutralizó movería los separadores de miles y
    decimales de TODA la base actual sin que nadie lo pidiera — y encima en
    silencio, porque ningún test de i18n mira cifras.
    """
    assert DEFAULT_LOCALE_ES_DO in _locales_from_ssot(), (
        f"P1-I18N-DASHBOARD: el código {DEFAULT_LOCALE_ES_DO!r} desapareció de "
        "LOCALES. Si se cambió a 'es' o 'es-ES', las cifras de todos los usuarios "
        "actuales pasan de «2,000» a «2000» y de «1,234.5» a «1234,5». Eso es una "
        "migración de datos visible, no un renombre: exige decidirlo a propósito y "
        "reescribir la columna `locale` de los perfiles existentes."
    )


def test_a3_cada_idioma_tiene_etiqueta_nativa():
    src = _read(_LOCALES_JS)
    entries = re.findall(
        r"\{\s*code:\s*'([a-z]{2}-[A-Z]{2})',\s*native:\s*'([^']+)'", src
    )
    assert len(entries) == len(_locales_from_ssot()), (
        "P1-I18N-DASHBOARD: hay entradas de LOCALES sin `native`. La etiqueta va "
        "en el PROPIO idioma («Français», no «Francés») porque quien busca su "
        "idioma en una lista no sabe leer la que tiene delante."
    )
    for code, native in entries:
        assert native.strip(), f"P1-I18N-DASHBOARD: `native` vacío para {code}."


# ---------------------------------------------------------------------------
# B) Los cuatro espejos coinciden con el SSOT
# ---------------------------------------------------------------------------

def test_b_index_html_boot_coincide_con_ssot():
    assert set(_locales_from_index_html()) == set(_locales_from_ssot()), (
        "P1-I18N-DASHBOARD: el boot de index.html y locales.js NO declaran los "
        f"mismos idiomas.\n  index.html: {sorted(_locales_from_index_html())}\n"
        f"  locales.js: {sorted(_locales_from_ssot())}\n"
        "Consecuencia de la divergencia: el idioma que falte en el boot arranca "
        "con <html lang=\"es-DO\"> hasta que React monta — parpadeo y lector de "
        "pantalla en el idioma equivocado durante el arranque en frío."
    )


def test_b2_check_sql_coincide_con_ssot():
    assert set(_locales_from_sql(_MIGRATION_ROOT)) == set(_locales_from_ssot()), (
        "P1-I18N-DASHBOARD: el CHECK de la migración y locales.js NO declaran los "
        f"mismos idiomas.\n  CHECK: {sorted(_locales_from_sql(_MIGRATION_ROOT))}\n"
        f"  locales.js: {sorted(_locales_from_ssot())}\n"
        "Consecuencia: el usuario elige el idioma nuevo, la UI cambia, y el "
        "guardado revienta contra la constraint."
    )


def test_b3_backend_coincide_con_ssot():
    assert set(_locales_from_backend()) == set(_locales_from_ssot()), (
        "P1-I18N-DASHBOARD: `_LOCALE_VALUES` y locales.js NO declaran los mismos "
        f"idiomas.\n  backend: {sorted(_locales_from_backend())}\n"
        f"  locales.js: {sorted(_locales_from_ssot())}\n"
        "Consecuencia: el PATCH devuelve 400 para un idioma que el selector sí "
        "ofrece."
    )


# ---------------------------------------------------------------------------
# C) P3-MIGRATIONS-SSOT: la migración vive idéntica en los dos directorios
# ---------------------------------------------------------------------------

def test_c_migracion_en_ambos_directorios_y_byte_identica():
    assert _MIGRATION_ROOT.exists(), (
        f"P1-I18N-DASHBOARD: falta {_MIGRATION_ROOT}. Toda migration vive en "
        "`migrations/` (workspace-root) Y en `backend/migrations/` — son repos "
        "hermanos y cada `git push` necesita el archivo físico en su dir "
        "(P3-MIGRATIONS-SSOT)."
    )
    assert _MIGRATION_BACKEND.exists(), (
        f"P1-I18N-DASHBOARD: falta {_MIGRATION_BACKEND} (P3-MIGRATIONS-SSOT)."
    )
    assert _MIGRATION_ROOT.read_bytes() == _MIGRATION_BACKEND.read_bytes(), (
        "P1-I18N-DASHBOARD: las dos copias de la migración DIFIEREN. Ese drift ya "
        "ocurrió antes (audit 2026-05-20: 4 files root-only + 1 backend-only). "
        "Copiar la buena sobre la otra y volver a correr."
    )


def test_c2_migracion_es_idempotente():
    """P3-MIGRATION-IDEMPOTENCE-DOC: re-aplicarla no puede fallar."""
    src = _read(_MIGRATION_ROOT)
    assert "ADD COLUMN IF NOT EXISTS" in src, (
        "P1-I18N-DASHBOARD: el ADD COLUMN no lleva `IF NOT EXISTS`. Una "
        "re-aplicación fallaría (P3-MIGRATION-IDEMPOTENCE-DOC)."
    )
    assert "DROP CONSTRAINT IF EXISTS" in src, (
        "P1-I18N-DASHBOARD: falta el `DROP CONSTRAINT IF EXISTS` previo al ADD "
        "CONSTRAINT. Sin él, re-aplicar la migración choca con la constraint ya "
        "existente."
    )
    assert "RAISE EXCEPTION" in src, (
        "P1-I18N-DASHBOARD: la migración no tiene sanity check. El patrón del repo "
        "(p2_next_4, p3_multiplier_db_check) exige un `DO $$ … RAISE EXCEPTION` "
        "que se detenga ANTES de imponer una constraint que los datos vivos violan."
    )


# ---------------------------------------------------------------------------
# D) El endpoint: whitelist + validación de valor
# ---------------------------------------------------------------------------

def test_d_locale_esta_en_el_whitelist_de_escalares():
    src = _read(_USER_DATA_PY)
    m = re.search(r"_PROFILE_SCALAR_WHITELIST\s*=\s*frozenset\(\{([^}]+)\}\)", src)
    assert m, "P1-I18N-DASHBOARD: no encontré `_PROFILE_SCALAR_WHITELIST`."
    assert '"locale"' in m.group(1), (
        "P1-I18N-DASHBOARD: `locale` salió de `_PROFILE_SCALAR_WHITELIST`. Sin él "
        "el PATCH lo rechaza con 400 y el idioma deja de seguir al usuario entre "
        "dispositivos (se queda en localStorage, que es por origen y por navegador)."
    )


def test_d2_el_whitelist_sigue_sin_columnas_de_entitlement():
    """Guard de regresión: abrir `locale` no puede haber abierto el tier.

    Es la razón de existir del whitelist (I-Billing-1 / P0-BILLING-1): aceptar
    `plan_tier` del cliente reabre el upgrade gratis desde DevTools.
    """
    src = _read(_USER_DATA_PY)
    m = re.search(r"_PROFILE_SCALAR_WHITELIST\s*=\s*frozenset\(\{([^}]+)\}\)", src)
    assert m
    body = m.group(1)
    for forbidden in (
        "plan_tier",
        "subscription_status",
        "subscription_end_date",
        "paypal_subscription_id",
        "credits",
        "is_admin",
    ):
        assert forbidden not in body, (
            f"P1-I18N-DASHBOARD / I-Billing-1: `{forbidden}` apareció en "
            "`_PROFILE_SCALAR_WHITELIST`. El tier es server-derived desde PayPal; "
            "aceptarlo del cliente reabre el upgrade gratis vía DevTools."
        )


def test_d3_el_valor_de_locale_se_valida_en_el_endpoint():
    src = _read(_USER_DATA_PY)
    assert "_LOCALE_VALUES" in src and 'if "locale" in fields' in src, (
        "P1-I18N-DASHBOARD: el PATCH ya no valida el VALOR de `locale`. El "
        "whitelist solo mira CLAVES (nunca necesitó mirar valores porque "
        "`full_name` es texto libre). Sin esta validación, un valor inválido llega "
        "al CHECK de la DB y el usuario recibe un 500 de psycopg en vez de un 400 "
        "que dice qué pasó."
    )


def test_d4_la_validacion_corre_ANTES_del_update():
    """El orden importa: validar después del UPDATE no valida nada."""
    src = _read(_USER_DATA_PY)
    i_validation = src.find('if "locale" in fields')
    i_update = src.find("UPDATE user_profiles SET {set_clause}")
    if i_update == -1:
        i_update = src.find("set_clause = ")
    assert i_validation != -1 and i_update != -1
    assert i_validation < i_update, (
        "P1-I18N-DASHBOARD: la validación de `locale` quedó DESPUÉS de construir "
        "el UPDATE. Una guarda que corre después de la escritura no es una guarda."
    )


# ---------------------------------------------------------------------------
# E) El motor: el idioma base NO puede tener catálogo
# ---------------------------------------------------------------------------

def test_e_es_do_no_tiene_catalogo():
    """Si aparece `es-DO.json`, alguien rompió el diseño sin darse cuenta.

    El ahorro entero de este sistema (0 bytes de i18n para la base dominicana)
    depende de que el idioma base sea el FALLBACK y no un catálogo más.
    """
    base_catalog = _FRONTEND / "src" / "i18n" / "locales" / f"{_BASE_LOCALE}.json"
    assert not base_catalog.exists(), (
        f"P1-I18N-DASHBOARD: existe {base_catalog.name}. El idioma base NO lleva "
        "catálogo: las claves del código SON su texto, y por eso un usuario "
        "dominicano no descarga ni un byte de traducciones. Un catálogo es-DO "
        "reintroduce ese peso para el 100% de la base actual y además crea una "
        "segunda fuente de verdad del copy español."
    )


def test_e2_existe_catalogo_para_cada_idioma_no_base():
    faltan = [
        code
        for code in _locales_from_ssot()
        if code != _BASE_LOCALE
        and not (_FRONTEND / "src" / "i18n" / "locales" / f"{code}.json").exists()
    ]
    assert not faltan, (
        f"P1-I18N-DASHBOARD: idiomas declarados en LOCALES sin catálogo: {faltan}. "
        "El selector los ofrecería y `loadLocale` fallaría en silencio (fail-soft: "
        "se queda en español), así que el usuario ve que su clic no hace nada."
    )


def test_e3_los_catalogos_son_json_valido():
    import json

    for code in _locales_from_ssot():
        if code == _BASE_LOCALE:
            continue
        p = _FRONTEND / "src" / "i18n" / "locales" / f"{code}.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(
                f"P1-I18N-DASHBOARD: {p.name} no es JSON válido ({e}). El import "
                "dinámico lanzaría y `loadLocale` devolvería false — el usuario "
                "elige el idioma y no pasa nada."
            )
        assert isinstance(data, dict), (
            f"P1-I18N-DASHBOARD: {p.name} no es un objeto en la raíz."
        )


# ---------------------------------------------------------------------------
# F) El validador de catálogos existe y está cableado
# ---------------------------------------------------------------------------

def test_f_existe_el_script_i18n_check_y_su_npm_script():
    """Sin este script, «la clave es el español» es una trampa.

    Cambiar el copy español huérfana su traducción EN SILENCIO: nadie ve un
    error, esa línea simplemente vuelve al español en los otros 4 idiomas. Es el
    único fallo real de este diseño y no se cierra con disciplina.
    """
    import json

    script = _FRONTEND / "scripts" / "i18n-check.mjs"
    assert script.exists(), (
        "P1-I18N-DASHBOARD: falta `frontend/scripts/i18n-check.mjs`. Es la red que "
        "detecta claves huérfanas (copy español cambiado que dejó atrás 4 "
        "traducciones muertas) y llamadas a t() en ámbito de módulo."
    )
    pkg = json.loads((_FRONTEND / "package.json").read_text(encoding="utf-8"))
    scripts = pkg.get("scripts", {})
    assert "i18n:check" in scripts, (
        "P1-I18N-DASHBOARD: `i18n:check` desapareció de los scripts de "
        "package.json. Un validador que nadie puede invocar por nombre no se "
        "invoca."
    )
