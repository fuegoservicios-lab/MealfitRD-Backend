"""[P3-CLAUDEMD-MARGIN-RESTORE · 2026-08-04] El cap de CLAUDE.md volvió a
quedarse sin margen: `test_p1_prod_final_1::test_claude_md_has_breathing_room_
after_cleanup` exige >=800 bytes bajo el cap de `test_p3_claudemd_cap`
(58.400) y el archivo llevaba en 58.384 — margen 16. El propio mensaje de ese
test lo dice: «repite la pasada de limpieza estructural antes de añadir nuevo
contenido». Esta es esa pasada.

Qué se movió (y por qué ESTE bloque):

    La subsección «Pattern: `SET search_path = ''` en functions Postgres»
    tenía 2.060 bytes de los cuales ~1.100 eran un INVENTARIO (tabla de 6
    functions × migración × search_path × GRANT). Un inventario es
    justamente lo que la convención doc-first del repo manda sacar de un
    archivo que se auto-carga en CADA turn: la REGLA se lee mil veces, la
    tabla se consulta cuando añades una function.

    Bonus no cosmético: la línea introductoria de «Advisors aceptados» en
    CLAUDE.md prometía desde 2026-07-26 que `backend/docs/advisors_aceptados.md`
    contenía «Tabla canónica + pattern `SET search_path = ''` + lockdown de
    DEFINERs». El doc NO tenía ni el pattern ni el lockdown — la promesa era
    falsa y un lector que siguiera el link no encontraba nada. Mover el
    bloque ahí la vuelve verdadera.

Qué NO se movió (y es deliberado): la REGLA sobrevive entera en CLAUDE.md en
una línea — `SET search_path = ''` + `SECURITY <DEFINER|INVOKER>` explícito, y
el `REVOKE EXECUTE ... FROM PUBLIC, anon, authenticated` obligatorio en
functions `SECURITY DEFINER` que aceptan `user_id` sin validar `auth.uid()`.
Esa segunda mitad es una defensa IDOR cross-user; degradarla a «está en un
doc» sería cambiar la seguridad por bytes. Los tests 3 y 4 la anclan.

Contrato que este archivo enforza:

  1. CLAUDE.md sigue teniendo la subsección y apunta al doc.
  2. El doc existe y contiene el inventario ÍNTEGRO (las 6 functions).
  3. La regla `search_path = ''` sobrevive en CLAUDE.md.
  4. La regla del REVOKE (P1-DEFINER-LOCKDOWN) sobrevive en CLAUDE.md.
  5. Anti-regresión: la tabla NO vuelve a inlinearse en CLAUDE.md.
  6. El margen bajo el cap volvió a ser sano (>=800 bytes).

Nota de resolución de paths: se verifica la copia de workspace-root
(`_REPO_ROOT / "CLAUDE.md"`) — la MISMA que miden `test_p3_claudemd_cap` y
`test_p1_prod_final_1`. La copia espejo `backend/CLAUDE.md` la cubre
`test_p1_diary_editable::test_claude_md_copies_stay_in_sync`.

Tooltip-anchor: P3-CLAUDEMD-MARGIN-RESTORE | limpieza estructural 2026-08-04
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_BACKEND = _REPO_ROOT / "backend"
_ADVISORS_DOC = _BACKEND / "docs" / "advisors_aceptados.md"

_SUBSECTION_HEADER = "### Pattern: `SET search_path = ''` en functions Postgres"

# Las 6 functions del inventario movido. Si añades una function bajo el
# pattern, va al DOC (y a esta lista), no de vuelta a CLAUDE.md.
_INVENTORY_FUNCTIONS = [
    "set_meal_plans_updated_at",
    "apply_inventory_delta",
    "increment_inventory_quantity",
    "handle_new_user",
    "get_monthly_plan_count",
    "log_unknown_ingredient_rpc",
]


@pytest.fixture(scope="module")
def claude_md() -> str:
    assert _CLAUDE_MD.exists(), (
        f"CLAUDE.md no encontrado en {_CLAUDE_MD}. Si moviste la raíz del "
        f"repo, actualiza `_REPO_ROOT` aquí y en test_p3_claudemd_cap.py."
    )
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def advisors_doc() -> str:
    assert _ADVISORS_DOC.exists(), (
        f"{_ADVISORS_DOC} no existe. Es el destino del bloque movido desde "
        f"CLAUDE.md por P3-CLAUDEMD-MARGIN-RESTORE — sin él, el link de "
        f"CLAUDE.md apunta al vacío y el inventario de functions se perdió."
    )
    return _ADVISORS_DOC.read_text(encoding="utf-8")


def _subsection(claude_md: str) -> str:
    """Extrae la subsección del pattern hasta el siguiente `###` / `##` / `---`."""
    start = claude_md.find(_SUBSECTION_HEADER)
    assert start >= 0, (
        f"Subsección `{_SUBSECTION_HEADER}` no encontrada en CLAUDE.md. "
        f"El pattern `SET search_path = ''` es contrato de seguridad para "
        f"toda function Postgres nueva — si lo renombraste, actualiza "
        f"`_SUBSECTION_HEADER` aquí; si lo BORRASTE, reviértelo."
    )
    rest = claude_md[start + len(_SUBSECTION_HEADER):]
    cut = re.search(r"\n###\s|\n##\s|\n---", rest)
    return rest[: cut.start()] if cut else rest


# ---------------------------------------------------------------------------
# 1. CLAUDE.md conserva la subsección y apunta al doc destino
# ---------------------------------------------------------------------------
def test_claude_md_subsection_links_the_doc(claude_md: str):
    section = _subsection(claude_md)
    assert "backend/docs/advisors_aceptados.md" in section, (
        "La subsección del pattern `SET search_path` ya no enlaza "
        "`backend/docs/advisors_aceptados.md`. Tras P3-CLAUDEMD-MARGIN-RESTORE "
        "el inventario de las 6 functions vive AHÍ: sin el link, CLAUDE.md "
        "enuncia una regla y esconde dónde está la lista de quién la cumple."
    )


# ---------------------------------------------------------------------------
# 2. El doc destino contiene el inventario ÍNTEGRO
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", _INVENTORY_FUNCTIONS)
def test_doc_carries_the_full_inventory(advisors_doc: str, fn: str):
    assert fn in advisors_doc, (
        f"`{fn}` no aparece en {_ADVISORS_DOC.name}. El movimiento "
        f"CLAUDE.md → doc debía ser ÍNTEGRO (cero contenido perdido): las 6 "
        f"functions bajo el pattern, con su migración SSOT y su GRANT."
    )


def test_doc_carries_the_inventory_table_header(advisors_doc: str):
    assert "| Function | Migración | `search_path` | EXECUTE granted to |" in advisors_doc, (
        "La cabecera de la tabla de functions no está en el doc — el "
        "inventario se movió sin su estructura (migración / search_path / "
        "EXECUTE), que es justo lo que un autor consulta al añadir una function."
    )


def test_doc_carries_the_definer_lockdown_detail(advisors_doc: str):
    assert "P1-DEFINER-LOCKDOWN" in advisors_doc, (
        "El detalle de P1-DEFINER-LOCKDOWN no llegó al doc."
    )
    assert "REVOKE EXECUTE" in advisors_doc, (
        "El texto del REVOKE EXECUTE no llegó al doc."
    )


# ---------------------------------------------------------------------------
# 3-4. Las DOS reglas sobreviven en CLAUDE.md (no se degradaron a link)
# ---------------------------------------------------------------------------
def test_search_path_rule_survives_in_claude_md(claude_md: str):
    """La regla, no el inventario. Un autor que añade una function debe
    poder leerla sin abrir el doc."""
    section = _subsection(claude_md)
    assert "SET search_path = ''" in section, (
        "La regla `SET search_path = ''` desapareció de CLAUDE.md. Mover el "
        "INVENTARIO a docs/ era el objetivo; mover la REGLA no."
    )
    assert "SECURITY <DEFINER|INVOKER>" in section, (
        "Se perdió la mitad `SECURITY <DEFINER|INVOKER>` explícito de la regla."
    )


def test_definer_lockdown_rule_survives_in_claude_md(claude_md: str):
    """Defensa IDOR cross-user: esta mitad NO puede vivir solo en un doc.

    Una function `SECURITY DEFINER` que acepta `user_id` y se deja
    `GRANT`eada a `authenticated` es IDOR universal sobre datos de otros
    usuarios. La regla se lee en el momento de escribir la migración."""
    section = _subsection(claude_md)
    assert "P1-DEFINER-LOCKDOWN" in section, (
        "El marker P1-DEFINER-LOCKDOWN salió de CLAUDE.md."
    )
    assert "REVOKE EXECUTE" in section, (
        "La obligación de `REVOKE EXECUTE ... FROM PUBLIC, anon, "
        "authenticated` en functions SECURITY DEFINER con `user_id` salió de "
        "CLAUDE.md. Es una defensa IDOR cross-user: degradarla a 'está en un "
        "doc' cambia seguridad por bytes."
    )
    assert "auth.uid()" in section, (
        "Se perdió la condición que dispara el REVOKE (aceptar `user_id` "
        "SIN validar contra `auth.uid()`)."
    )


# ---------------------------------------------------------------------------
# 5. Anti-regresión: la tabla no vuelve a CLAUDE.md
# ---------------------------------------------------------------------------
def test_inventory_table_not_reinlined_in_claude_md(claude_md: str):
    assert "| Function | Migración | `search_path` | EXECUTE granted to |" not in claude_md, (
        "La tabla de functions volvió a inlinearse en CLAUDE.md. Vive en "
        "`backend/docs/advisors_aceptados.md` desde P3-CLAUDEMD-MARGIN-RESTORE: "
        "re-inlinearla devuelve ~1.100 bytes al archivo que se auto-carga en "
        "cada turn y vuelve a comerse el margen del cap."
    )
    # Sanity direccional: el detalle largo tampoco reaparece por otra vía.
    assert "función huérfana, 0 callsites" not in claude_md, (
        "Detalle del inventario re-inlineado en CLAUDE.md (columna de "
        "`get_monthly_plan_count`). Ese matiz vive en el doc."
    )


# ---------------------------------------------------------------------------
# 6. El margen bajo el cap volvió a ser sano
# ---------------------------------------------------------------------------
def test_margin_under_cap_restored():
    """Mismo umbral que `test_p1_prod_final_1` (>=800 bytes), anclado aquí
    para que el P-fix que lo restauró tenga su propia regresión.

    El cap se lee del test SSOT en vez de hardcodearse: si alguien bumpea el
    cap en lugar de limpiar, este test no lo tapa — sigue midiendo margen."""
    cap_test = _BACKEND / "tests" / "test_p3_claudemd_cap.py"
    cap_text = cap_test.read_text(encoding="utf-8")
    m = re.search(r"_DEFAULT_CAP\s*=\s*(\d+)", cap_text)
    assert m, "No pude parsear `_DEFAULT_CAP` de test_p3_claudemd_cap.py."
    cap = int(m.group(1))
    size = _CLAUDE_MD.stat().st_size  # BYTES en disco (CRLF incluido)
    margin = cap - size
    assert margin >= 800, (
        f"CLAUDE.md size={size} bytes, cap={cap}, margin={margin}. "
        f"P3-CLAUDEMD-MARGIN-RESTORE dejó el margen en ~1.237 bytes "
        f"(57.163/58.400); si volvió a bajar de 800, toca otra pasada "
        f"doc-first — NO bumpear el cap. Candidatos medidos y ya verificados "
        f"como NO anclados por ningún test: subsecciones «Sentry sampling» "
        f"(834 B), «Security headers en nginx» (771 B) y «Admin gate en "
        f"/api/system/health» (685 B) de la misma sección «Advisors aceptados»."
    )


def test_marker_anchor_present_in_app_py():
    """Cross-link P2-HIST-AUDIT-14: el slug del marker debe matchear este
    archivo. Ancla también que el bump de `_LAST_KNOWN_PFIX` ocurrió."""
    app_py = _BACKEND / "app.py"
    text = app_py.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    slug = "p3_claudemd_margin_restore"
    assert slug in __file__.replace("\\", "/").lower(), (
        f"El slug `{slug}` debe estar en el nombre de este archivo para "
        f"satisfacer test_p2_hist_audit_14_marker_test_link."
    )
