"""[P3-PANTRY-FRIDGE-LAYOUT · 2026-05-19] Ancla el rediseño de la Nevera
como nevera física: marco con manija lateral + estantes interiores + gavetas
crisper + puerta diferenciada + alacena externa (paleta ámbar).

Cambio de UX (request usuario 2026-05-19):
    "quiero que la nevera le demos un diseño completamenta identico a una
    nevera, asi le daremos un toque original y simbolico"

Pre-fix:
    `Pantry.jsx` renderizaba `Object.keys(filteredInventory).sort().map(...)`
    iterando categorías en orden alfabético del master_ingredients.category.
    Visual: estantes plana azules sin marco realista, sin diferenciación
    semántica entre lácteos/proteínas/frutas/granos secos.

Fix:
    Mapping `CATEGORY_TO_ZONE` (25 categorías → 7 zonas físicas):
      - shelf_dairy   (Lácteos, Quesos, Huevos)
      - shelf_proteins (Carnes, Pollo, Pescados, ...)
      - shelf_ready    (Panadería, Dulces, Frutos secos)
      - door           (Bebidas, Aceites, Condimentos)
      - drawer_fruits  (Frutas)
      - drawer_veggies (Vegetales, Hierbas)
      - pantry         (Granos, Especias, Despensa) — FUERA del marco

    Las 6 primeras viven dentro de `.nevera-fridge-body` (marco con
    `.nevera-fridge-handle` lateral derecha + `.nevera-fridge-feet`
    inferiores). La 7ma vive en `.nevera-pantry-section` con paleta
    ámbar (alacena de madera). Las gavetas se rinden side-by-side en
    `.nevera-drawers-row` con `border-radius` pronunciado abajo
    simulando crispers reales.

Por qué parser-based:
    El rediseño es 100% frontend visual; correr Playwright para validar
    pixels no aporta más señal que asegurar las anclas estructurales:
      1. CATEGORY_TO_ZONE mapea las 25 categorías a 7 zonas
      2. ZONE_DEFINITIONS define las 7 zonas con kind ∈ {shelf,door,drawer,pantry}
      3. Las clases CSS críticas del marco/gavetas/alacena están presentes
      4. El render usa `inventoryByZone` (proyección por zona) NO el sort
         alfabético antiguo de categorías
      5. Marker presente para que el slug matchee este file (cross-link
         enforced por `test_p2_hist_audit_14`)

Si alguien revierte el rediseño y vuelve al render por categoría, este
test falla en el assert #4 — el `Object.keys(filteredInventory).sort()`
del listado principal ya no debe existir.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Pantry.jsx no encontrado en {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


# Categorías representativas de cada zona — si el mapping pierde alguna,
# el render mete items en `pantry` (fallback) y el usuario los ve fuera
# de la nevera. Test cierra esa regresión.
ZONE_REPRESENTATIVES = {
    "shelf_dairy":    ["LÁCTEOS", "QUESOS", "HUEVOS"],
    "shelf_proteins": ["POLLO", "PESCADO", "CARNES ROJAS"],
    "shelf_ready":    ["PANADERÍA", "DULCES", "FRUTOS SECOS"],
    "door":           ["BEBIDAS", "ACEITES", "CONDIMENTOS"],
    "drawer_fruits":  ["FRUTAS"],
    "drawer_veggies": ["VEGETALES", "VERDURAS", "HIERBAS"],
    "pantry":         ["CEREALES Y GRANOS", "DESPENSA", "LEGUMBRES", "ESPECIAS"],
}


def test_p3_pantry_fridge_layout_marker_present():
    """Marker textual presente en JSX — sin él, el slug del marker en
    `_LAST_KNOWN_PFIX` no matchea ningún archivo de test y el cross-link
    de `test_p2_hist_audit_14` falla.
    """
    src = _read_pantry()
    assert "P3-PANTRY-FRIDGE-LAYOUT" in src, (
        "Marker `P3-PANTRY-FRIDGE-LAYOUT` no encontrado en Pantry.jsx. "
        "Si reviertes el rediseño de nevera, también bumpea "
        "`_LAST_KNOWN_PFIX` en backend/app.py al P-fix de reversión."
    )


def test_category_to_zone_mapping_exists():
    """La constante de mapping debe estar declarada al top-level del
    archivo (fuera del componente para no recrearse en cada render).
    """
    src = _read_pantry()
    assert re.search(r"const\s+CATEGORY_TO_ZONE\s*=\s*\{", src), (
        "Constante `CATEGORY_TO_ZONE` no declarada en Pantry.jsx. "
        "Sin ella, el render no puede mapear master_ingredients.category "
        "a una zona física de la nevera."
    )


@pytest.mark.parametrize("zone,categories", list(ZONE_REPRESENTATIVES.items()))
def test_each_zone_has_representative_categories(zone, categories):
    """Cada zona debe tener al menos una categoría representativa en el
    mapping. Si falta una clave (ej. alguien elimina 'POLLO' sin pensar
    qué zona la cubre), los items de esa categoría caen al fallback
    `pantry` y aparecen fuera de la nevera — UX rota silenciosamente.
    """
    src = _read_pantry()
    # Aislamos el bloque del mapping para no matchear menciones casuales
    match = re.search(
        r"const\s+CATEGORY_TO_ZONE\s*=\s*\{(.+?)\};",
        src,
        re.DOTALL,
    )
    assert match, "Bloque CATEGORY_TO_ZONE no parseable"
    block = match.group(1)

    for cat in categories:
        # La línea tiene formato: 'CATEGORIA': 'zone_key',
        pattern = rf"'{re.escape(cat)}'\s*:\s*'{re.escape(zone)}'"
        assert re.search(pattern, block), (
            f"Categoría representativa '{cat}' no mapeada a zona '{zone}' "
            f"en CATEGORY_TO_ZONE. Si la quitas a propósito, mueve los items "
            f"de '{cat}' al mapping de otra zona o actualiza este test."
        )


def test_zone_definitions_includes_seven_zones():
    """ZONE_DEFINITIONS debe enumerar las 7 zonas en orden de render
    (orden importa: el array se itera tal cual para construir el DOM,
    cambiar orden cambia la UX).
    """
    src = _read_pantry()
    match = re.search(
        r"const\s+ZONE_DEFINITIONS\s*=\s*\[(.+?)\];",
        src,
        re.DOTALL,
    )
    assert match, (
        "Bloque ZONE_DEFINITIONS no parseable. Sin él el render no sabe "
        "qué zonas mostrar ni en qué orden."
    )
    block = match.group(1)

    required_keys = [
        "shelf_dairy", "shelf_proteins", "shelf_ready",
        "door", "drawer_fruits", "drawer_veggies", "pantry",
    ]
    for key in required_keys:
        assert f"key: '{key}'" in block, (
            f"Zona '{key}' falta en ZONE_DEFINITIONS — el render la omitirá."
        )

    # Validar kinds (afecta CSS aplicado a cada zona)
    for kind in ["shelf", "door", "drawer", "pantry"]:
        assert f"kind: '{kind}'" in block, (
            f"Kind '{kind}' falta en ZONE_DEFINITIONS — sin él el render "
            f"no aplica el CSS correspondiente (.nevera-zone-{kind})."
        )


CRITICAL_CSS_CLASSES = [
    # Marco exterior y manija
    ".nevera-fridge-body",
    ".nevera-fridge-interior-wrap",
    ".nevera-fridge-handle",
    ".nevera-fridge-feet",
    # Panel superior tipo control (P3-PANTRY-FRIDGE-POLISH)
    ".nevera-fridge-control-panel",
    ".nevera-fridge-led-display",
    ".nevera-fridge-power-dot",
    # Zonas internas
    ".nevera-zone",
    ".nevera-zone-header",
    ".nevera-zone-grid",
    ".nevera-zone-door",
    # Gavetas crisper
    ".nevera-drawers-row",
    ".nevera-drawer",
    ".nevera-drawer-grid",
    # Alacena externa (paleta ámbar)
    ".nevera-pantry-section",
    ".nevera-pantry-header",
]


@pytest.mark.parametrize("css_class", CRITICAL_CSS_CLASSES)
def test_critical_css_classes_defined(css_class):
    """Cada clase CSS crítica del rediseño debe estar declarada en el
    `<style>` block de Pantry.jsx. Si alguien elimina una, el JSX la
    sigue referenciando pero el browser no aplica estilo → la nevera
    se ve plana sin marco/manija/gavetas.
    """
    src = _read_pantry()
    # Match la clase con `{` (selector) — evita falsos positivos por
    # mención en comentarios u otros contextos.
    pattern = rf"{re.escape(css_class)}\s*[\.,\{{>+~:]"
    assert re.search(pattern, src), (
        f"Clase CSS '{css_class}' no declarada en Pantry.jsx. "
        f"Sin ella el rediseño tipo nevera se rompe visualmente."
    )


def test_render_uses_inventory_by_zone_not_legacy_sort():
    """El render principal debe iterar `inventoryByZone` (proyección
    por zona) NO `Object.keys(filteredInventory).sort()` (legacy por
    categoría). Si vuelve el sort alfabético, el rediseño no surte
    efecto — items se renderizan por categoría sin estructura tipo
    nevera.
    """
    src = _read_pantry()

    # Debe existir el derived state nuevo
    assert re.search(r"const\s+inventoryByZone\s*=\s*useMemo", src), (
        "useMemo `inventoryByZone` no encontrado. Sin él, el render no "
        "puede agrupar por zona."
    )

    # Debe usarse en al menos un branch del render (zonas y/o alacena)
    assert "inventoryByZone[zone.key]" in src or "inventoryByZone[pantryZone.key]" in src, (
        "`inventoryByZone[...]` no se referencia en el render. La proyección "
        "se computa pero no se consume — rediseño no efectivo."
    )

    # El sort legacy en el listado principal NO debe existir más
    # (admitimos que `Object.keys(filteredInventory).length === 0` siga
    # siendo el check de empty-state — eso NO es el sort).
    legacy_pattern = r"Object\.keys\(filteredInventory\)\.sort\(\)\.map"
    assert not re.search(legacy_pattern, src), (
        "Se detectó el render legacy `Object.keys(filteredInventory).sort().map(...)` "
        "en Pantry.jsx. Si vuelves al render por categoría, también remueve "
        "el rediseño tipo nevera (marco, gavetas, alacena) y revierte este test."
    )


def test_render_item_card_helper_extracted():
    """El helper `renderItemCard` debe estar extraído para evitar
    duplicar ~125 líneas de JSX × 7 zonas. Si alguien lo elimina y
    vuelve a inline, el archivo crece desproporcionadamente y futuros
    cambios al card divergen entre zonas.
    """
    src = _read_pantry()
    assert re.search(r"const\s+renderItemCard\s*=\s*\(item\)\s*=>", src), (
        "Helper `renderItemCard` no encontrado. Debe estar declarado "
        "antes del `return (` del componente para que las 7 zonas lo "
        "reusen."
    )


def test_drawer_labels_have_no_gaveta_prefix():
    """[P3-PANTRY-FRIDGE-POLISH · 2026-05-19] Los labels de las gavetas
    NO deben llevar el prefijo 'Gaveta de' — la metáfora visual (crisper
    con asita + radius pronunciado) ya comunica que es una gaveta, el
    texto era ruido.
    """
    src = _read_pantry()
    # Buscar las líneas de ZONE_DEFINITIONS con drawer kind
    drawer_lines = re.findall(
        r"\{\s*key:\s*'drawer_\w+'.+?kind:\s*'drawer'.+?\}",
        src,
    )
    assert len(drawer_lines) >= 2, (
        "No se encontraron las 2 definiciones de drawer en ZONE_DEFINITIONS. "
        "Revisa que el array esté intacto."
    )
    for line in drawer_lines:
        assert "Gaveta de" not in line, (
            f"Label de drawer contiene 'Gaveta de' (redundante con la metáfora "
            f"visual). Línea: {line}"
        )


def test_zone_definitions_have_semantic_colors():
    """[P3-PANTRY-FRIDGE-POLISH · 2026-05-19] Cada zona debe tener un
    campo `color` distinto del cyan default. Pre-fix todos los iconos
    eran `#0EA5E9` y la nevera se sentía monocromática. Test ancla que
    cada zona tiene su propia identidad cromática.
    """
    src = _read_pantry()
    match = re.search(
        r"const\s+ZONE_DEFINITIONS\s*=\s*\[(.+?)\];",
        src,
        re.DOTALL,
    )
    assert match, "ZONE_DEFINITIONS no parseable"
    block = match.group(1)

    # Cada línea de zona debe contener `color: '#...'`
    color_decls = re.findall(r"color:\s*'(#[0-9A-Fa-f]{6})'", block)
    assert len(color_decls) >= 7, (
        f"Esperaba ≥7 declaraciones `color` en ZONE_DEFINITIONS (una por zona), "
        f"encontré {len(color_decls)}: {color_decls}"
    )
    # No todos pueden ser el mismo color
    assert len(set(color_decls)) >= 5, (
        f"Solo {len(set(color_decls))} colores únicos en ZONE_DEFINITIONS — "
        f"la diferenciación cromática se pierde. Esperaba ≥5 colores distintos."
    )


def test_drawers_row_uses_align_items_start():
    """[P3-PANTRY-FRIDGE-POLISH · 2026-05-19] `.nevera-drawers-row` debe
    declarar `align-items: start` para que cada gaveta crezca solo lo
    que su contenido necesita. Sin esto, la gaveta corta se estira a la
    altura de la larga (hueco vacío feo).
    """
    src = _read_pantry()
    match = re.search(
        r"\.nevera-drawers-row\s*\{([^}]+)\}",
        src,
    )
    assert match, ".nevera-drawers-row no encontrada"
    body = match.group(1)
    assert "align-items: start" in body, (
        "`.nevera-drawers-row` debe declarar `align-items: start` para "
        "que las gavetas tengan altura independiente — sin esto reaparece "
        "el bug del hueco vacío bajo la gaveta corta (reportado por user "
        "2026-05-19 con screenshot)."
    )


def test_render_passes_zone_color_to_icon():
    """[P3-PANTRY-FRIDGE-POLISH · 2026-05-19] El render debe pasar
    `zone.color` al style del Icon (no `#0EA5E9` hardcoded). Si vuelve
    el hardcode todos los iconos se ven cyan otra vez.
    """
    src = _read_pantry()
    # Buscar referencias del Icon en renderZoneShelf y en el drawer render.
    # Ambos deben usar `color: zone.color` en el style.
    icon_color_refs = re.findall(
        r"<Icon[^>]+style=\{\{\s*color:\s*([^,}]+)",
        src,
    )
    assert len(icon_color_refs) >= 2, (
        f"Esperaba ≥2 referencias a <Icon ... color=...>, encontré "
        f"{len(icon_color_refs)}: {icon_color_refs}"
    )
    for ref in icon_color_refs:
        ref_clean = ref.strip()
        assert "zone.color" in ref_clean, (
            f"<Icon> debe usar `color: zone.color` (de ZONE_DEFINITIONS), "
            f"no hardcoded. Encontré: `{ref_clean}`. Si reviertes a hardcode "
            f"cyan, todos los iconos se ven iguales y se pierde la "
            f"identidad cromática por zona."
        )
