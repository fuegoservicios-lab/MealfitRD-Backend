"""Guards de G30: el checkout backend debe poder colectar sin el repo frontend.

El CI del repo backend no contiene ``../frontend``.  Leer ese repo hermano al
importar un módulo de tests aborta toda la colección antes de ejecutar una sola
prueba.  Este guard mira la propiedad (I/O de frontend en scope de módulo), no
los 68 nombres de archivo que exhibían el defecto el 2026-08-23.
"""

from __future__ import annotations

import ast
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = BACKEND_ROOT / "tests"


class _TopLevelCalls(ast.NodeVisitor):
    """Visita expresiones ejecutadas al importar, pero no cuerpos diferidos."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - API de ast
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        # Decoradores/defaults sí corren al importar; el cuerpo no.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        # El cuerpo de una clase sí se ejecuta al definirla.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        # Crear el lambda no ejecuta su cuerpo.
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)


def _assigned_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        targets.extend(item.optional_vars for item in node.items if item.optional_vars)
    return {
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name)
    }


def _source(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def _function_loaders(tree: ast.Module) -> set[str]:
    """Helpers de I/O; el call site decide si la ruta pertenece a frontend."""

    loaders: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
            name = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else ""
            )
            if name in {"open", "read", "read_text", "read_bytes"}:
                loaders.add(node.name)
                break
    return loaders


def _module_frontend_io(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    # Propaga aliases de ruta: _FRONT -> _QPB -> _QPB_SRC.
    frontend_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            value_source = _source(text, value).lower()
            used = {child.id for child in ast.walk(value) if isinstance(child, ast.Name)}
            if "frontend" not in value_source and not used.intersection(frontend_names):
                continue
            for name in _assigned_names(node):
                if name not in frontend_names:
                    frontend_names.add(name)
                    changed = True

    loaders = _function_loaders(tree)
    offenders: list[int] = []
    for statement in tree.body:
        visitor = _TopLevelCalls()
        visitor.visit(statement)
        statement_source = _source(text, statement).lower()
        statement_names = {
            child.id for child in ast.walk(statement) if isinstance(child, ast.Name)
        }
        touches_frontend = (
            "frontend" in statement_source
            or bool(statement_names.intersection(frontend_names))
        )
        for call in visitor.calls:
            call_name = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else ""
            )
            if (call_name in loaders and touches_frontend) or (
                call_name in {"open", "read", "read_text", "read_bytes"}
                and touches_frontend
            ):
                offenders.append(statement.lineno)
                break
    return offenders


def test_no_test_module_reads_frontend_during_import() -> None:
    offenders = {
        path.relative_to(BACKEND_ROOT).as_posix(): lines
        for path in sorted(TESTS_ROOT.glob("test_*.py"))
        if path != Path(__file__) and (lines := _module_frontend_io(path))
    }
    assert not offenders, (
        "El checkout backend no trae ../frontend. Mueve estas lecturas a una "
        "fixture que dependa de `frontend_repo_path` (con skip si falta):\n"
        + "\n".join(f"  {name}: líneas {lines}" for name, lines in offenders.items())
    )


def test_backend_ci_has_no_collection_failure_ceiling() -> None:
    workflow = (BACKEND_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    pytest_command = next(
        line.strip() for line in workflow.splitlines() if line.strip().startswith("pytest tests/")
    )
    assert "--maxfail" not in pytest_command
    assert " -x" not in pytest_command


def test_shared_fixture_skips_when_frontend_repo_is_absent() -> None:
    conftest = (TESTS_ROOT / "conftest.py").read_text(encoding="utf-8")
    tree = ast.parse(conftest)
    fixture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "frontend_repo_path"
    )
    body = _source(conftest, fixture)
    assert "pytest.skip" in body
    assert "repo hermano ausente" in body
    assert 'parents[2] / "frontend"' in body


def test_last_known_pfix_exposes_g30_closure() -> None:
    app_source = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert (
        '_LAST_KNOWN_PFIX = "P2-CI-BACKEND-CERO-TESTS · 2026-08-23"'
        in app_source
    )
