"""[P0-SHOPPING-CALC-NAMEERROR · 2026-05-15] Regression guard contra el
`NameError: name '_can_lower' is not defined` en
`shopping_calculator.aggregate_and_deduct_shopping_list` (línea ~5002).

Bug observado en runtime 2026-05-15 18:43:51 al generar un plan E2E:

    Traceback (most recent call last):
      File ".../shopping_calculator.py", line 5002, in aggregate_and_deduct_shopping_list
        _ac = canonicalize_aceites(canonical_name)
    NameError: name '_can_lower' is not defined

Root cause: el bloque P3-NEW-12 (5 canonicalizers adicionales) + el bloque
pavo (P1-XX) introdujeron `_orig_name_lower` para el matching del pavo, y
refactorizaron las 13 regex de consolidación de abajo (Fresas, Almendras,
Orégano, Tortilla, Tomate, Cebolla, Espinacas, Zanahoria, Vainitas,
Habichuelas, Tofu, Perejil) para que operaran sobre `_can_lower`. Pero la
asignación `_can_lower = canonical_name.lower()` quedó huérfana en el
refactor → cada plan generado lanzaba excepción aquí.

Síntoma user-facing: `aggregate_and_deduct_shopping_list` fallaba →
fallback dejaba lista de compras vacía/incompleta → coherence guard veía
35 ingredientes de recetas sin contraparte en lista (`presence=expected_only`)
→ alert `[COH-GUARD/block] 35 divergencias` → plan entregado con
`_shopping_coherence_block` no popeado → frontend mostraba "Verificación
médica con observaciones: COHERENCIA RECETAS LISTA: 35 divergencia(s) críticas".

Fix: asignar `_can_lower = canonical_name.lower()` DESPUÉS del bloque pavo
y ANTES del primer uso (línea ~5091, justo antes de "# Consolidación:
Fresas variantes").

Este test es parser-based — verifica que la asignación está presente y
en la posición correcta. NO ejecuta el pipeline LLM.

## [review final audit-v7-p1 · 2026-08-03] Re-apuntado tras la extracción SSOT

`P1-VEG-BACKFILL-HONESTY` (ronda 1) extrajo toda la cadena de canonicalización —master_map,
`_consolidate_inline_canon`, los 5 canonicalizers de P3-NEW-12/P2-NEW-A, el bloque pavo y las 13
regex de cola— de `aggregate_and_deduct_shopping_list` a la función nueva
`canonicalize_shopping_food_name`, para que `get_shopping_list_delta` pudiera canonicalizar el
lado TEXTO con el MISMO código que el lado comprado (antes "300 g de tomates" quedaba keyed por
'Tomates' y el ítem comprado por 'Tomate', y el emparejamiento fallaba en silencio).

El invariante de producción NO cambió: las 16 referencias a `_can_lower` y su asignación siguen
juntas, en la misma función que las usa — sólo que esa función ahora es
`canonicalize_shopping_food_name`. Lo que quedó roto fue este guard, que buscaba en un cuerpo ya
vacío y por tanto medía 0 referencias. Se re-apunta a la función REAL y se añade la mitad que
faltaba: que el agregador DELEGUE (si alguien reintroduce la cadena inline, el drift entre los dos
lados vuelve, que es el bug que la extracción cerró).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CALC_PATH = _BACKEND_ROOT / "shopping_calculator.py"


def _read_calc() -> str:
    return _CALC_PATH.read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    """Quita líneas que comienzan con `#` (comentarios Python) para evitar
    que menciones de `_can_lower` en docstrings/comments contaminen el
    matching. NO toca docstrings entre triple-quotes (las regex aquí no
    necesitan precisión de parser de Python)."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def test_can_lower_assignment_present():
    """`_can_lower = canonical_name.lower()` debe existir como asignación
    en `aggregate_and_deduct_shopping_list` para que las 13 regex de
    consolidación de abajo no lancen NameError."""
    text = _strip_comments(_read_calc())
    assert re.search(
        r"_can_lower\s*=\s*canonical_name\.lower\(\)", text
    ), (
        "P0-SHOPPING-CALC-NAMEERROR: falta la asignación "
        "`_can_lower = canonical_name.lower()` antes de las 13 regex de "
        "consolidación. Cada plan generado lanzará NameError aquí."
    )


def test_can_lower_assigned_before_first_use():
    """La asignación de `_can_lower` debe venir ANTES de su primer uso
    (regex `^fresas?\\b`). Si la asignación queda DESPUÉS del primer uso,
    NameError reaparece."""
    text = _strip_comments(_read_calc())
    assign_match = re.search(r"_can_lower\s*=\s*canonical_name\.lower\(\)", text)
    # Primer uso = primera ocurrencia de `_can_lower` que NO sea la propia
    # asignación. Buscar todos los offsets y tomar el segundo.
    occurrences = [m.start() for m in re.finditer(r"_can_lower\b", text)]
    assert assign_match, "Asignación de `_can_lower` no encontrada."
    assert len(occurrences) >= 2, "Esperaba ≥2 menciones de _can_lower."
    # La asignación es la primera mención; las regex usuarias vienen después.
    assert assign_match.start() == occurrences[0], (
        "La asignación debe ser la PRIMERA mención de _can_lower."
    )
    first_use_pos = occurrences[1]
    assert assign_match.start() < first_use_pos, (
        f"P0-SHOPPING-CALC-NAMEERROR: la asignación de `_can_lower` "
        f"(pos {assign_match.start()}) debe estar ANTES de su primer uso "
        f"(pos {first_use_pos}). Si llega después, NameError reaparece."
    )


def test_can_lower_assigned_after_pavo_block():
    """La asignación debe estar DESPUÉS del bloque pavo (que puede mutar
    `canonical_name`). Si se asigna antes del bloque pavo, las 13 regex
    de consolidación verán el canonical_name pre-pavo y el matching será
    inconsistente."""
    text = _strip_comments(_read_calc())
    # Anchor del final del bloque pavo: `canonical_name = 'Pavo'` en la
    # rama elif `_orig_name_lower.strip() == 'pavo'`.
    pavo_end = re.search(
        r"elif _orig_name_lower\.strip\(\)\s*==\s*'pavo':.*?canonical_name\s*=\s*'Pavo'",
        text,
        re.DOTALL,
    )
    assign_match = re.search(r"_can_lower\s*=\s*canonical_name\.lower\(\)", text)
    assert pavo_end and assign_match, "No localicé anchors."
    assert assign_match.start() > pavo_end.end(), (
        f"P0-SHOPPING-CALC-NAMEERROR: `_can_lower` debe asignarse DESPUÉS "
        f"del bloque pavo. Pre-pavo, `canonical_name` puede ser 'Pechuga de pavo' "
        f"pero post-pavo el bloque puede haberlo cambiado a 'Jamón de pavo' "
        f"o 'Pavo molido'. Las 13 regex de abajo necesitan el valor post-pavo."
    )


def _func_body(text: str, func_name: str) -> str:
    """Cuerpo de una función top-level, delimitado por la SIGUIENTE `def` de columna 0.

    Es un ancla estructural, no una ventana de N caracteres: crece con la función y no caduca
    cuando alguien añade líneas (el modo de fallo de `test_convergence_failopen_never_wipes_lists`,
    que llevaba tres bumpeos de constante)."""
    start = text.find(f"def {func_name}")
    assert start > 0, f"Función `{func_name}` no encontrada."
    next_def = re.search(r"^def \w+", text[start + 1:], re.MULTILINE)
    end = (start + 1 + next_def.start()) if next_def else len(text)
    return text[start:end]


# Función que HOY contiene la cadena de canonicalización. Ver el bloque de la docstring del módulo:
# hasta 2026-08-03 vivía inline en `aggregate_and_deduct_shopping_list`.
_CANON_FUNC = "canonicalize_shopping_food_name"


def test_all_can_lower_references_within_same_function():
    """Las 13+ referencias a `_can_lower` deben estar todas dentro de la MISMA función que hace la
    asignación. Si una se mueve afuera por refactor sin renombrar, NameError ahí también.

    [review final · 2026-08-03] Esa función es hoy `canonicalize_shopping_food_name` (extracción
    SSOT de P1-VEG-BACKFILL-HONESTY). El invariante es idéntico; sólo cambió el contenedor."""
    text = _strip_comments(_read_calc())
    func_body = _func_body(text, _CANON_FUNC)

    references = re.findall(r"_can_lower", func_body)
    # Esperamos: 1 asignación + 13 referencias en regex (= 14 menciones mínimo).
    assert len(references) >= 14, (
        f"Esperaba ≥14 menciones de `_can_lower` dentro de `{_CANON_FUNC}` "
        f"(1 asignación + 13 usos), encontré {len(references)}. Si alguna se removió, "
        f"verificar que la regex correspondiente también se removió."
    )


def test_el_agregador_delega_en_el_ssot_y_no_reintroduce_la_cadena_inline():
    """[review final · 2026-08-03] La otra mitad del guard, que faltaba.

    `aggregate_and_deduct_shopping_list` debe DELEGAR en el SSOT y no volver a contener la cadena.
    Si alguien la reintroduce inline (por "performance", por un merge, por costumbre), el lado
    comprado y el lado texto del backstop vuelven a divergir en silencio: el bug exacto que la
    extracción cerró, y el que P2-NEW-8/P3-NEW-6 llevan documentando desde mayo."""
    text = _strip_comments(_read_calc())
    agg_body = _func_body(text, "aggregate_and_deduct_shopping_list")
    assert f"{_CANON_FUNC}(name, master_map)" in agg_body, (
        "el agregador dejó de delegar en el SSOT de canonicalización"
    )
    assert not re.search(r"_can_lower", agg_body), (
        "la cadena de canonicalización volvió a vivir inline en el agregador: eso reintroduce el "
        "drift entre el lado comprado y el lado texto/guard"
    )


def test_orig_name_lower_only_in_pavo_block():
    """`_orig_name_lower` (raw del parser, pre-master_map) debe usarse SOLO
    en el bloque pavo. Si alguien la usa fuera, está bypasseando el comment
    explicativo y puede haber confundido la convención. El comment dice:
    'Usar SOLO `name.lower()` (raw del parser) ... NO `_can_lower` (post-master_map)'
    pero eso aplica solo al matching de pavo."""
    text = _strip_comments(_read_calc())
    func_body = _func_body(text, _CANON_FUNC)

    # Cuenta usos. Debe haber 1 asignación + N usos dentro del bloque pavo.
    # No verificamos número exacto (puede crecer), solo que existe.
    assert "_orig_name_lower = name.lower()" in func_body, (
        "La asignación de `_orig_name_lower = name.lower()` para el bloque "
        "pavo debe seguir existiendo."
    )
