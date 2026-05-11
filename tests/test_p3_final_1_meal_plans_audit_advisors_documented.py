"""[P3-FINAL-1 . 2026-05-11] Drift detection: los 4 advisors INFO/WARN
intencionales sobre `meal_plans_audit` (introducidos por P2-NEW-5) deben
estar documentados en la seccion "Advisors aceptados" de CLAUDE.md y
respaldados por COMMENTs SSOT en la migracion canonica.

Por que existe este test:
  Tras crear `meal_plans_audit` en P2-NEW-5, `get_advisors` devolvio 4
  advisors nuevos (1 security + 3 performance). Sin documentacion, un
  futuro audit los flagearia como gaps recurrentes; un operador SRE leyendo
  el dashboard del linter no sabria que son intencionales y podria
  "arreglarlos" eliminando los indices o anadiendo policies que rompen el
  modelo service_role-only.

Que enforza:
  1. CLAUDE.md menciona los 4 nombres de advisor exactos en la seccion
     "Advisors aceptados".
  2. La migracion `p3_final_1_meal_plans_audit_advisor_anchors.sql` existe
     y contiene los 4 COMMENTs (1 ON TABLE + 3 ON INDEX) anclando la
     justificacion en pg_catalog.
  3. Cross-link a la memoria de cierre.

Mantenimiento:
  Si en el futuro se decide remover/cambiar alguno de estos advisors
  (e.g., Supabase emite supresion nativa o se decide rotar el modelo),
  actualizar la lista `_EXPECTED_ADVISORS` y la migracion juntos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_MIGRATION = (
    _REPO_ROOT
    / "supabase"
    / "migrations"
    / "p3_final_1_meal_plans_audit_advisor_anchors.sql"
)
_MEMORY_FILENAME = "project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md"

# Los 4 advisors que P3-FINAL-1 cierra. Si en el futuro se anaden mas,
# extender la lista; si se eliminan, removerlos junto con la migracion y
# las filas de CLAUDE.md (no dejar huerfanos).
_EXPECTED_ADVISORS: tuple[tuple[str, str], ...] = (
    # (advisor_name, target_object) — ambos deben aparecer en la fila.
    ("rls_enabled_no_policy", "meal_plans_audit"),
    ("unused_index", "idx_meal_plans_audit_meal_plan_id"),
    ("unused_index", "idx_meal_plans_audit_user_id"),
    ("unused_index", "idx_meal_plans_audit_action_created"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_claude_md_documents_all_four_advisors():
    """Cada advisor (par advisor_name + objeto) debe aparecer en una fila
    de la seccion "Advisors aceptados" de CLAUDE.md."""
    assert _CLAUDE_MD.exists(), (
        f"CLAUDE.md no encontrado en {_CLAUDE_MD}. Si renombraste/moviste, "
        "actualiza el path en este test."
    )
    text = _read(_CLAUDE_MD)
    # Localizar la seccion para que el match no agarre menciones aleatorias
    # en otras secciones (e.g., el SOP P3-AUDIT-6 menciona meal_plans_audit
    # en otro contexto).
    section_start = text.find("## Advisors aceptados")
    assert section_start != -1, (
        "Seccion '## Advisors aceptados' no encontrada en CLAUDE.md. "
        "Si la renombraste, actualizar el header en este test."
    )
    # La seccion termina en el siguiente '## ' top-level o EOF.
    section_end = text.find("\n## ", section_start + 1)
    section_text = text[section_start : section_end if section_end != -1 else None]

    missing = []
    for advisor_name, target in _EXPECTED_ADVISORS:
        # La fila tiene el formato:
        #   `advisor_name` (`target_object`) ... | razon | memoria |
        # Asi que ambos tokens deben aparecer cerca (en la misma fila).
        # Comprobamos por simple co-presencia en la seccion + presencia
        # del par exacto en alguna linea del markdown.
        rows_with_both = [
            line
            for line in section_text.splitlines()
            if advisor_name in line and target in line and line.startswith("|")
        ]
        if not rows_with_both:
            missing.append((advisor_name, target))

    assert not missing, (
        "Advisors P3-FINAL-1 sin fila en CLAUDE.md 'Advisors aceptados':\n"
        + "\n".join(f"  - {name}({target})" for name, target in missing)
        + "\n\nAccion: anade fila con razon + link a memoria "
        f"`{_MEMORY_FILENAME}`. Patron paralelo a P2-PERF-1."
    )


def test_migration_file_exists_with_all_four_comments():
    """La migracion SSOT debe existir y contener los 4 COMMENTs canonicos."""
    assert _MIGRATION.exists(), (
        f"Migracion P3-FINAL-1 no encontrada en {_MIGRATION}. "
        "Sin esto, los COMMENTs viven solo en runtime — un `db reset` desde "
        "migrations los pierde y futuros operadores no veran la justificacion."
    )
    sql = _read(_MIGRATION)

    expected_statements = (
        "COMMENT ON TABLE public.meal_plans_audit",
        "COMMENT ON INDEX public.idx_meal_plans_audit_meal_plan_id",
        "COMMENT ON INDEX public.idx_meal_plans_audit_user_id",
        "COMMENT ON INDEX public.idx_meal_plans_audit_action_created",
    )

    missing = [stmt for stmt in expected_statements if stmt not in sql]
    assert not missing, (
        "Migracion P3-FINAL-1 falta los siguientes COMMENTs:\n"
        + "\n".join(f"  - {stmt}" for stmt in missing)
        + "\n\nSin estos, advisors quedan sin anchor SSOT en pg_catalog."
    )


def test_migration_references_p3_final_1_marker():
    """La migracion debe contener el marker P3-FINAL-1 (no solo P2-NEW-5)
    para que el grep `_LAST_KNOWN_PFIX` localice la consolidacion correcta."""
    sql = _read(_MIGRATION)
    assert "P3-FINAL-1" in sql, (
        "Migracion P3-FINAL-1 no contiene el marker en su header. "
        "Operador buscando el origen del fix por `git grep P3-FINAL-1` "
        "no encontraria la migracion."
    )


def test_claude_md_links_to_memory_file():
    """Las filas de los 4 advisors deben referenciar el archivo de memoria
    canonico (cierre del cross-link CLAUDE.md ↔ memoria)."""
    text = _read(_CLAUDE_MD)
    section_start = text.find("## Advisors aceptados")
    section_end = text.find("\n## ", section_start + 1)
    section_text = text[section_start : section_end if section_end != -1 else None]

    # Las 4 filas deben mencionar el filename de memoria. No exigimos URL exacta
    # porque el path con `~/` se renderiza distinto segun viewer.
    occurrences = section_text.count(_MEMORY_FILENAME)
    assert occurrences >= 4, (
        f"Esperaba >=4 referencias a `{_MEMORY_FILENAME}` en la seccion "
        f"'Advisors aceptados' (1 por advisor); encontradas: {occurrences}. "
        "Cada fila debe linkear a la memoria de cierre para forensics."
    )


def test_no_extra_p3_final_1_advisors_undocumented():
    """Floor anchor: si alguien anade un 5to COMMENT en la migracion sin
    extender CLAUDE.md, el operador deberia tener visibilidad. Este test
    cuenta los `COMMENT ON` en la migracion y exige que los nombres de
    objetos coincidan con `_EXPECTED_ADVISORS`."""
    sql = _read(_MIGRATION)
    # Cuenta de COMMENT ON statements (lineas que empiezan con "COMMENT ON")
    comment_lines = [
        line.strip()
        for line in sql.splitlines()
        if line.strip().startswith("COMMENT ON ")
    ]
    # Esperamos exactamente 4 (1 tabla + 3 indices). Si suben a 5+, anadir
    # entrada a `_EXPECTED_ADVISORS` y al test 1.
    assert len(comment_lines) == 4, (
        f"Esperaba exactamente 4 statements COMMENT ON en la migracion "
        f"(1 tabla + 3 indices); encontrados: {len(comment_lines)}.\n\n"
        "Si anadiste un nuevo COMMENT, extiende `_EXPECTED_ADVISORS` "
        "y la documentacion en CLAUDE.md. Si removiste uno, valida que "
        "el advisor correspondiente realmente este resuelto por otro medio."
    )
