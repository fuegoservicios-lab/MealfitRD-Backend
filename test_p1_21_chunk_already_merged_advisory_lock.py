"""[P1-21] Tests para que el path `chunk_already_merged` (backfill de lección
tras crash entre T1 y T2) re-adquiera el advisory lock `purpose='general'`
explícitamente antes del UPDATE de plan_data.

Bug original (audit P1-21):
  El UPDATE del backfill en chunk_already_merged (~línea 18305 de
  `cron_tasks.py`) reescribía plan_data confiando ÚNICAMENTE en el lock
  heredado de T1 (adquirido al inicio del FOR UPDATE, ~línea 17790).

  T2 (~línea 19177) sí re-adquiere `acquire_meal_plan_advisory_lock(...,
  purpose='general')` explícitamente justo antes de su UPDATE — es la
  defensa local que sobrevive a refactors que splitten transacciones.

  El path chunk_already_merged carecía de esa defensa: si un dev futuro
  movía el bloque de backfill fuera de T1 (p.ej. para acelerar shopping
  list), el UPDATE quedaba SIN lock y se reabría la race contra
  `/shift-plan` y otros workers que también modifican plan_data — el
  invariante "todo escritor de plan_data adquiere purpose='general'" se
  rompía silenciosamente.

Fix:
  Re-adquisición explícita de `acquire_meal_plan_advisory_lock(cursor,
  meal_plan_id, purpose="general")` justo antes del `UPDATE meal_plans
  SET plan_data` en el path chunk_already_merged. La llamada es
  idempotente vía `pg_advisory_xact_lock` (mismo lock dentro de misma
  transacción es no-op), así que no hay coste en producción mientras
  T1 lo siga sosteniendo. Si la transacción se split-tea en el futuro,
  esta llamada garantiza la serialización local.

Cobertura:
  - test_chunk_already_merged_path_present
  - test_chunk_already_merged_acquires_advisory_lock_explicitly
  - test_lock_acquired_with_purpose_general
  - test_lock_acquired_before_update_in_chunk_already_merged
  - test_lock_uses_canonical_helper_acquire_meal_plan_advisory_lock
  - test_t2_path_also_acquires_general_lock_symmetry
  - test_documentation_p1_21_present
"""
import inspect
import re

import pytest

import cron_tasks


# Source del módulo (lo cargamos una vez para todos los tests).
_SRC = inspect.getsource(cron_tasks)


def _strip_comments(src: str) -> str:
    """Filtra líneas-comentario del source para evitar falsos positivos en
    búsquedas de invocaciones (los comentarios mencionan función/símbolo
    para documentar el rationale)."""
    return "\n".join(
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    )


def _chunk_already_merged_block() -> str:
    """Extrae el bloque de código del path `chunk_already_merged`.

    Buscamos desde la primera mención `if chunk_already_merged:` hasta la
    primera línea con `else:` (que abre el path "Merge normal"). Esto
    aísla la rama backfill para que las búsquedas no contaminen con el
    path normal."""
    code = _strip_comments(_SRC)
    start = code.find("if chunk_already_merged:")
    assert start > -1, "No se encontró el guard `if chunk_already_merged:`"
    # Buscamos el primer `else:` con la indentación inicial del bloque
    # (24 espacios — la rama `else: # Merge normal` está al mismo nivel
    # que el if). Más conservador: cortamos en 50_000 chars o el `else`.
    rest = code[start:]
    # Match: una línea con `else:` o `else: ...` al mismo nivel de indent.
    m = re.search(r"\n( {24}else:.*)", rest)
    if m:
        return rest[: m.start()]
    return rest[:60_000]  # fallback razonable


# ---------------------------------------------------------------------------
# 1. Estructura: el path existe y aún cubre el caso backfill.
# ---------------------------------------------------------------------------
def test_chunk_already_merged_path_present():
    """El bloque `if chunk_already_merged:` aún existe en cron_tasks.py
    (defensa contra deletes accidentales del path)."""
    code = _strip_comments(_SRC)
    assert "if chunk_already_merged:" in code, (
        "P1-21: el guard `if chunk_already_merged:` desapareció. Si fue "
        "intencional, actualizar este test; si no, regression."
    )


def test_chunk_already_merged_writes_plan_data():
    """El path debe seguir conteniendo el UPDATE a plan_data — sin él, no
    hay nada que proteger con el lock."""
    block = _chunk_already_merged_block()
    assert "UPDATE meal_plans SET plan_data" in block, (
        "P1-21: el UPDATE de plan_data del backfill desapareció del path "
        "chunk_already_merged. Verificar que el rationale del test sigue "
        "aplicando."
    )


# ---------------------------------------------------------------------------
# 2. Fix: el lock se adquiere dentro del path.
# ---------------------------------------------------------------------------
def test_chunk_already_merged_acquires_advisory_lock_explicitly():
    """El path debe contener una llamada explícita a
    `acquire_meal_plan_advisory_lock` (re-adquisición idempotente para
    sobrevivir refactors que splitten T1)."""
    block = _chunk_already_merged_block()
    assert "acquire_meal_plan_advisory_lock" in block, (
        "P1-21: el path chunk_already_merged debe re-adquirir el advisory "
        "lock explícitamente antes del UPDATE para mantener simetría con "
        "T2 (~línea 19177) y sobrevivir a refactors de transacción."
    )


def test_lock_acquired_with_purpose_general():
    """La adquisición debe usar `purpose='general'` (el namespace canónico
    para escritores de plan_data — `/shift-plan`, T1, T2 y BG-REFILL ya
    usan ese mismo string). Otro purpose rompería la serialización.

    Aceptamos tanto invocación directa como vía alias (los call sites del
    módulo importan `acquire_meal_plan_advisory_lock as _pXX_acquire_lock`
    para evitar sombras de scope). Verificamos que (a) el módulo
    importa el helper canónico desde `db_plans` dentro del block y (b)
    `purpose="general"` aparece en una invocación dentro del block."""
    block = _chunk_already_merged_block()
    # (a) Import del helper canónico presente.
    assert (
        "from db_plans import acquire_meal_plan_advisory_lock" in block
    ), (
        "P1-21: el path debe importar el helper canónico "
        "`acquire_meal_plan_advisory_lock` desde db_plans dentro del "
        "block (puede ser via alias `as _pXX_acquire_lock`)."
    )
    # (b) Invocación con purpose='general' en el block. Aceptamos tanto
    # la invocación directa como una invocación al alias `_pXX_acquire_lock`
    # — lo que importa es que purpose='general' esté en una llamada a un
    # helper de adquisición (callable seguido de paréntesis).
    pattern = re.compile(
        r"_acquire_lock\([^)]*purpose\s*=\s*['\"]general['\"]"
        r"|acquire_meal_plan_advisory_lock\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.DOTALL,
    )
    assert pattern.search(block), (
        "P1-21: la re-adquisición debe usar purpose='general' para "
        "serializar contra los demás escritores de plan_data."
    )


def test_lock_acquired_before_update_in_chunk_already_merged():
    """La invocación de adquisición del lock debe aparecer ANTES del
    `UPDATE meal_plans SET plan_data` dentro del path. Si el orden se
    invierte, el lock no protege la escritura.

    Acepta tanto invocación directa como vía alias `_pXX_acquire_lock(`."""
    block = _chunk_already_merged_block()
    # Encontrar la primera invocación de adquisición (alias o directa).
    pattern = re.compile(
        r"(?:acquire_meal_plan_advisory_lock|_acquire_lock)\([^)]*purpose",
        re.DOTALL,
    )
    m = pattern.search(block)
    assert m, "P1-21: ninguna invocación de adquisición encontrada en el block"
    lock_idx = m.start()
    update_idx = block.find("UPDATE meal_plans SET plan_data")
    assert update_idx > -1, "UPDATE no encontrado en el block"
    assert lock_idx < update_idx, (
        "P1-21: la adquisición del advisory lock debe ir ANTES del "
        "UPDATE de plan_data — invertirlos deja la escritura sin "
        "protección durante la ventana entre ambas líneas."
    )


def test_lock_uses_canonical_helper_acquire_meal_plan_advisory_lock():
    """La invocación debe pasar por `acquire_meal_plan_advisory_lock`
    (helper canónico de `db_plans`), NO `pg_advisory_xact_lock` raw.
    El helper centraliza el namespacing de la key (`meal_plan:purpose:id`)
    y emite warning ante `purpose` desconocido."""
    block = _chunk_already_merged_block()
    # Negativo: no debe haber un raw `pg_advisory_xact_lock(` inline en
    # este path (eso bypasaría el namespace estable y podría usar otra key).
    assert "pg_advisory_xact_lock" not in block, (
        "P1-21: el path debe usar el helper canónico "
        "`acquire_meal_plan_advisory_lock` en lugar de un "
        "`pg_advisory_xact_lock` raw — el helper garantiza la key "
        "estable `meal_plan:general:<id>` que comparten todos los "
        "escritores."
    )


# ---------------------------------------------------------------------------
# 3. Simetría: T2 mantiene la misma garantía (defensa contra divergencia).
# ---------------------------------------------------------------------------
def test_t2_path_also_acquires_general_lock_symmetry():
    """T2 (transacción final que estampa shopping list + status='completed')
    también debe acquirir el lock con purpose='general'. Si alguien
    elimina el lock de T2, este test rompe — antes de que se note en
    producción. Acepta invocación directa o vía alias `_pXX_acquire_lock`."""
    code = _strip_comments(_SRC)
    # Buscamos un landmark UNICO de T2: `learning_persisted_at = NOW()`
    # solo aparece en T2 (rolling refill no lo escribe). Es la última
    # query del with-cursor del bloque T2 atómico.
    t2_idx = code.find("learning_persisted_at = NOW()")
    assert t2_idx > -1, (
        "No se encontró el bloque T2 (landmark `learning_persisted_at = NOW()`)"
    )
    # Tomamos una ventana de 5000 chars antes del landmark para incluir
    # la adquisición del lock al inicio del with-cursor de T2.
    window_start = max(0, t2_idx - 5000)
    t2_window = code[window_start:t2_idx]
    # Acepta direct + alias (T2 también usa alias `_p04_acquire_lock`).
    pattern = re.compile(
        r"(?:acquire_meal_plan_advisory_lock|_acquire_lock)"
        r"\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.DOTALL,
    )
    assert pattern.search(t2_window), (
        "P1-21/symmetry: T2 también debe adquirir advisory lock "
        "purpose='general'. Si este test rompe, alguien quitó el lock "
        "de T2 — la consistencia entre T1 backfill, T2 final y "
        "/shift-plan se pierde."
    )


# ---------------------------------------------------------------------------
# 4. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_21_present():
    """Comentario `[P1-21]` debe documentar el rationale del lock."""
    assert "[P1-21]" in _SRC, (
        "P1-21: falta marker de documentación que explique por qué se "
        "re-adquiere el lock localmente (sobrevivir refactors)."
    )


def test_documentation_mentions_idempotency_or_refactor_safety():
    """El comentario debe mencionar que la re-adquisición es defensiva
    (idempotente / refactor-safety / simetría con T2). Esto evita que un
    futuro lector piense que es código muerto y lo borre."""
    # Tomamos una ventana de 1500 chars alrededor de la primera mención
    # de [P1-21] para verificar el rationale.
    idx = _SRC.find("[P1-21]")
    assert idx > -1
    window = _SRC[idx : idx + 1500]
    rationale_terms = ["refactor", "idempotent", "no-op", "simetr", "split"]
    assert any(t in window.lower() for t in rationale_terms), (
        f"P1-21: el comentario debe explicar el rationale (refactor "
        f"safety / idempotencia / simetría con T2). Encontrado: "
        f"{window[:300]!r}"
    )
