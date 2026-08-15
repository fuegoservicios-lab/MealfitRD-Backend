"""[P2-SHOPPING-TOTALS · 2026-05-16] Mostrar conteos de items en la lista
de compras (header chip "Total: N ítems" + per-section counts).

Antes (plan aeb25e1c): el usuario no sabía a primera vista cuánto iba a
tomar la compra (1 trip vs 2 trips). Tenía que contar mentalmente o
asumir. Beneficio UX:
  - "Total: 25 ítems" en el header → trip planning
  - "PERECEDEROS · 15 ítems" / "ESTABLES · 10 ítems" → balance entre
    secciones visible a primera vista

Fix solo-frontend (PDF rendering en Dashboard.jsx). Cero riesgo backend.

──────────────────────────────────────────────────────────────────────────
[P1-I18N-DASHBOARD · 2026-08-15] Reanclaje: la migración a i18n cambió la
GRAFÍA de dos de las tres propiedades vigiladas, no la CONDUCTA.

  1. Pluralizador. Antes:
         const _fmtItems = (n) => `${n} ${n === 1 ? 'ítem' : 'ítems'}`;
     Ahora:
         const _fmtItems = (n) => tn(n, '{n} ítem', '{n} ítems', { n });
     `tn` (frontend/src/i18n/index.js) cae al español con exactamente
     `n === 1 ? one : other`, así que en es-DO el resultado es byte a byte
     el de antes; en los otros 4 idiomas decide `Intl.PluralRules`. La
     propiedad «distingue 1 de N» sigue viva — lo que se fue es el
     ternario literal. Por eso el guard ya no busca el ternario: busca que
     la forma SINGULAR y la PLURAL sean distintas, que la singular no
     lleve la 's' de la plural, y que ambas conserven el placeholder del
     número (sin `{n}` el conteo desaparecería del PDF).

  2. Chip «Total» del header. Antes el rótulo era texto literal del HTML
     y solo el conteo pasaba por `escapeHtml`:
         Total: ${escapeHtml(_fmtItems(totalItems))}
     Ahora el rótulo entra por `t()` y `escapeHtml` envuelve el resultado
     YA interpolado:
         ${escapeHtml(t('Total: {items}', { items: _fmtItems(totalItems) }))}
     El escapado no se relajó: se aplica DESPUÉS de la interpolación, así
     que cubre lo mismo que antes y además el rótulo. El guard acepta
     ambas grafías y, en la nueva, exige por retro-referencia que el
     nombre del placeholder de la clave sea el MISMO que alimenta
     `_fmtItems(totalItems)` — un `t('Total: {items}', { item: ... })`
     pintaría el literal `{items}` en el PDF (el motor deja los
     placeholders sin valor tal cual) y aquí revienta.

  3. Escapado (SEGURIDAD, no cosmética). El guard viejo contaba
     ocurrencias de la grafía exacta `${escapeHtml(_fmtItems(x))}`, así
     que la envoltura en `t()` lo bajaba de 3 a 2. El nuevo no cuenta
     grafías: enumera TODOS los call sites `_fmtItems(<var>)` del fuente
     y verifica, uno a uno, que `escapeHtml(` aparezca dentro de la misma
     interpolación `${…}` y ANTES del call site — o sea escapado directo
     o transitivo a través de `t()`. Un call site nuevo sin escapar falla
     aunque los tres viejos sigan intactos.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"

# Los 3 conteos que el PDF pinta: header + las 2 secciones.
_EXPECTED_COUNT_ARGS = {"totalItems", "perishableItemCount", "stableItemCount"}


def _src() -> str:
    return _DASHBOARD_JSX.read_text(encoding="utf-8")


def test_section_counts_computed():
    """`perishableItemCount` y `stableItemCount` derivados de
    `Object.values(perishables/stables).reduce(...)`."""
    src = _src()
    assert "const perishableItemCount = Object.values(perishables).reduce(" in src, (
        "perishableItemCount no calculado vía reduce sobre Object.values."
    )
    assert "const stableItemCount = Object.values(stables).reduce(" in src, (
        "stableItemCount no calculado."
    )


def test_total_items_already_exists():
    """`totalItems` ya está calculado más arriba (P1-PDF-3 pre-existente).
    Lo reusamos, no necesitamos duplicar."""
    src = _src()
    assert "const totalItems = Object.values(consData).length;" in src, (
        "totalItems removido en algún refactor — sin él, el chip 'Total' "
        "del header no tiene la fuente esperada."
    )


# ---------------------------------------------------------------------------
# Pluralizador
# ---------------------------------------------------------------------------

# Grafía original: ternario literal dentro del template.
_PLURAL_TERNARY = re.compile(r"n === 1 \? ['\"]ítem['\"]\s*:\s*['\"]ítems['\"]")

# Grafía i18n: tn(count, singular, plural, vars).
_PLURAL_TN = re.compile(
    r"tn\(\s*n\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*\{\s*n\s*(?::\s*\w+\s*)?\}\s*\)"
)

_FMT_ITEMS_DEF = re.compile(r"const _fmtItems\s*=\s*\(?\s*n\s*\)?\s*=>\s*(.+?);\s*$", re.M)


def test_fmt_items_helper_pluralizes():
    """Helper `_fmtItems(n)` debe pluralizar — 1 ítem vs N ítems. Sin esto,
    veríamos textos awkward como '1 ítems' en planes muy chicos.

    [P1-I18N-DASHBOARD] Acepta el ternario original y la forma `tn()`. En
    ambas la propiedad vigilada es la misma: singular ≠ plural, y el
    singular no es la forma plural.
    """
    src = _src()
    m = _FMT_ITEMS_DEF.search(src)
    assert m, (
        "No se encontró la definición de `_fmtItems` (const _fmtItems = (n) => ...). "
        "Sin el helper, header y section labels pierden el conteo."
    )
    body = m.group(1)

    if _PLURAL_TERNARY.search(body):
        return  # grafía pre-i18n, propiedad evidente en el ternario

    tn_match = _PLURAL_TN.search(body)
    assert tn_match, (
        f"`_fmtItems` no pluraliza por ninguna de las dos vías conocidas "
        f"(ternario `n === 1 ? 'ítem' : 'ítems'` o `tn(n, <singular>, <plural>, {{ n }})`). "
        f"Cuerpo encontrado: {body!r}"
    )
    singular, plural = tn_match.group(1), tn_match.group(2)

    assert singular != plural, (
        f"`_fmtItems` pasa la MISMA cadena como singular y plural ({singular!r}) — "
        f"el pluralizador dejó de distinguir 1 de N."
    )
    assert "ítem" in singular and "ítems" not in singular, (
        f"La forma SINGULAR de `_fmtItems` debería ser el 'ítem' sin 's'; es {singular!r}. "
        f"(¿singular y plural invertidos?)"
    )
    assert "ítems" in plural, (
        f"La forma PLURAL de `_fmtItems` debería contener 'ítems'; es {plural!r}."
    )
    for label, form in (("singular", singular), ("plural", plural)):
        assert "{n}" in form, (
            f"La forma {label} de `_fmtItems` ({form!r}) perdió el placeholder `{{n}}` — "
            f"el PDF pintaría 'ítems' sin el número."
        )


# ---------------------------------------------------------------------------
# Chip "Total" del header
# ---------------------------------------------------------------------------

_TOTAL_CHIP_FORMS = (
    # Grafía original: rótulo literal en el HTML, conteo escapado.
    re.compile(r"Total:\s*\$\{escapeHtml\(_fmtItems\(totalItems\)\)\}"),
    # Grafía i18n: rótulo por t(), escapeHtml envuelve el resultado interpolado.
    # La retro-referencia \1 exige que el placeholder de la clave y la key del
    # objeto de vars coincidan (si no, el motor deja el `{...}` literal).
    re.compile(
        r"\$\{escapeHtml\(\s*t\(\s*'Total: \{(\w+)\}'\s*,\s*"
        r"\{\s*\1\s*:\s*_fmtItems\(totalItems\)\s*\}\s*\)\s*\)\}"
    ),
)


def test_header_includes_total_chip():
    """Header del PDF debe incluir chip 'Total: X ítems' además de
    'Ciclo' + 'Generado'.

    [P1-I18N-DASHBOARD] Se acepta la grafía envuelta en `t()`; lo que NO se
    relaja es que el rótulo 'Total' y `_fmtItems(totalItems)` viajen juntos
    y bajo `escapeHtml`.
    """
    src = _src()
    assert any(rx.search(src) for rx in _TOTAL_CHIP_FORMS), (
        "Chip 'Total' ausente del header del PDF (ni la grafía literal "
        "`Total: ${escapeHtml(_fmtItems(totalItems))}` ni la i18n "
        "`${escapeHtml(t('Total: {items}', { items: _fmtItems(totalItems) }))}`). "
        "Sin él, el usuario no ve el conteo agregado del shopping list."
    )


def test_section_labels_include_counts():
    """Los labels de PERECEDEROS y ESTABLES deben incluir el count
    de items inline (e.g. 'PERECEDEROS · 15 ítems').

    [P1-I18N-DASHBOARD] Estos dos no cambiaron de grafía (el rótulo ya venía
    de una variable), pero el assert es agnóstico a la envoltura: basta con
    que el call site exista; el escapado lo verifica `test_counts_escape_html`.
    """
    src = _src()
    assert "_fmtItems(perishableItemCount)" in src, (
        "Section label perishable no muestra el count — '· N ítems' ausente."
    )
    assert "_fmtItems(stableItemCount)" in src, (
        "Section label stable no muestra el count."
    )


# ---------------------------------------------------------------------------
# Escapado (seguridad)
# ---------------------------------------------------------------------------

_FMT_ITEMS_CALL = re.compile(r"_fmtItems\((\w+)\)")


_QUOTES = "'\"`"
_SCAN_CAP = 20_000  # techo defensivo: ningún `${…}` real del PDF es tan largo


def _interpolation_end(src: str, dollar: int) -> int | None:
    """Índice del `}` que cierra el `${` en `dollar`, o `None`.

    Cuenta llaves saltándose los literales de cadena: el `{items}` de
    `t('Total: {items}', …)` vive DENTRO de comillas y no cierra nada. Un
    conteo ingenuo de `}` daba por cerrada la interpolación justo ahí y
    reportaba el chip como no escapado (falso positivo real, visto al
    reanclar este guard).
    """
    i, depth, limit = dollar + 2, 1, min(len(src), dollar + _SCAN_CAP)
    while i < limit:
        c = src[i]
        if c in _QUOTES:
            i += 1
            while i < limit:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == c:
                    break
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _enclosing_interpolation(src: str, pos: int) -> str | None:
    """Texto entre el `${` de la interpolación MÁS INTERNA que contiene `pos`
    y `pos` mismo. `None` si el call site no vive dentro de un `${…}` — es
    decir, si no se interpola al HTML por template string."""
    cursor = pos
    while True:
        start = src.rfind("${", 0, cursor)
        if start == -1:
            return None
        end = _interpolation_end(src, start)
        if end is not None and end > pos:
            return src[start + 2 : pos]
        cursor = start  # esa interpolación ya cerró antes; sigue hacia atrás


def test_counts_escape_html():
    """SEGURIDAD (no cosmética): todo `_fmtItems(...)` que se interpola al
    HTML del PDF debe pasar por `escapeHtml` — directo o a través de `t()`.
    Consistencia con P1-PDF-XSS-AUDITED: si un refactor futuro hace que
    `_fmtItems` acepte input user-controlled, el escapado ya está puesto.

    [P1-I18N-DASHBOARD] El guard ya no cuenta la grafía
    `${escapeHtml(_fmtItems(x))}` (la migración la volvió
    `${escapeHtml(t('Total: {items}', { items: _fmtItems(totalItems) }))}`
    y el conteo cayó de 3 a 2 sin que la seguridad cambiara). Ahora enumera
    los call sites y verifica el escapado de cada uno: un call site NUEVO
    sin escapar falla aunque los tres actuales sigan bien.
    """
    src = _src()
    call_sites = [(m.group(1), m.start()) for m in _FMT_ITEMS_CALL.finditer(src)]

    args = {arg for arg, _ in call_sites}
    missing = _EXPECTED_COUNT_ARGS - args
    assert not missing, (
        f"Faltan call sites de `_fmtItems` en el PDF: {sorted(missing)}. "
        f"Esperaba header (totalItems) + 2 section labels. Encontrados: {sorted(args)}."
    )
    assert len(call_sites) >= 3, (
        f"Esperaba ≥3 call sites de `_fmtItems(...)` en el HTML "
        f"(header + 2 section labels). Encontrados: {len(call_sites)}."
    )

    unescaped = []
    for arg, pos in call_sites:
        prefix = _enclosing_interpolation(src, pos)
        if prefix is None or "escapeHtml(" not in prefix:
            unescaped.append((arg, prefix))
    assert not unescaped, (
        f"Call sites de `_fmtItems` que llegan al HTML del PDF SIN pasar por "
        f"`escapeHtml` (ni directo ni vía `t()`): {unescaped!r}. "
        f"Ver P1-PDF-XSS-AUDITED — todo interpolado al PDF va escapado."
    )
