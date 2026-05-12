"""[P2-LAZY-PDF · 2026-05-13] Bloquear imports estáticos top-level de
`html2pdf.js` en `frontend/src/pages/`.

Contexto del audit production-readiness 2026-05-12:
    `html2pdf.js` produce un chunk de 976 KB (el más grande del bundle).
    Pre-fix, Dashboard.jsx y Recipes.jsx tenían `import html2pdf from
    'html2pdf.js'` top-level → Vite/Rollup auto-split el chunk pero el
    import estático lo declara como dependencia HARD de la ruta → el
    browser lo descarga eagerly al entrar al Dashboard/Recipes, ANTES
    de que el usuario haga click en "Descargar PDF". 100% de usuarios
    autenticados pagan el costo aunque nunca exporten un PDF.

Patrón correcto (post-fix):
    const html2pdf = (await import('html2pdf.js')).default;
    await html2pdf().set(opt).from(element).save();

Esto deja el chunk fuera del initial waterfall del Dashboard y solo lo
fetch cuando el handler ejecuta — análogo a la decisión P3-LAZY-MARKDOWN
para react-markdown (closure 2026-05-12).

Si necesitas excepción legítima (e.g. precarga deliberada en una landing
de marketing donde sabes que el conversion es 100% PDF), añade marker
`// [P2-LAZY-PDF WHITELIST: <razón ≥1 char>]` en las 30 líneas previas
al import. El test lo respeta y NO falla — análogo al patrón de
test_p1_new_a_frontend_no_direct_meal_plans_write.py para meal_plans
direct writes.

Tooltip-anchor: P2-LAZY-PDF.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = REPO_ROOT / "frontend" / "src" / "pages"

# `import html2pdf from 'html2pdf.js'` o `import h from "html2pdf.js"` (cualquier alias).
STATIC_IMPORT_RE = re.compile(
    r"""^\s*import\s+\w+\s+from\s+['"]html2pdf\.js['"]""",
    re.MULTILINE,
)
WHITELIST_RE = re.compile(r"//\s*\[P2-LAZY-PDF\s+WHITELIST:\s*\S")

# `(await import('html2pdf.js')).default` o variantes con destructuring.
DYNAMIC_IMPORT_RE = re.compile(
    r"""await\s+import\(\s*['"]html2pdf\.js['"]\s*\)""",
)


def _window_before(text: str, offset: int, lines_back: int = 30) -> str:
    """Retorna las N líneas previas al offset (para buscar el whitelist marker)."""
    head = text[:offset]
    lines = head.split("\n")
    return "\n".join(lines[-lines_back:])


def test_p2_lazy_pdf_no_static_imports_in_frontend_pages():
    """Bloquea `import html2pdf from 'html2pdf.js'` top-level no-whitelisted."""
    assert PAGES_DIR.exists(), f"Pages dir no existe: {PAGES_DIR}"

    violations: list[str] = []
    for jsx in sorted(PAGES_DIR.glob("*.jsx")):
        text = jsx.read_text(encoding="utf-8")
        for match in STATIC_IMPORT_RE.finditer(text):
            window = _window_before(text, match.start(), lines_back=30)
            if WHITELIST_RE.search(window):
                continue  # excepción legítima, marker presente
            line_no = text[: match.start()].count("\n") + 1
            violations.append(f"  {jsx.name}:{line_no}  {match.group(0).strip()}")

    assert not violations, (
        "[P2-LAZY-PDF] Static top-level import of 'html2pdf.js' detectado en "
        "frontend/src/pages/. Cada uno hace que el chunk de 976 KB se fetch "
        "eagerly al entrar a la ruta — usuarios que no descarguen PDF pagan "
        "el costo. Migrar al patrón dynamic-inside-handler:\n\n"
        "    const html2pdf = (await import('html2pdf.js')).default;\n"
        "    await html2pdf().set(opt).from(element).save();\n\n"
        "Violations:\n" + "\n".join(violations) + "\n\n"
        "Si la excepción es deliberada, añadir comment "
        "`// [P2-LAZY-PDF WHITELIST: <razón>]` en las 30 líneas previas al import."
    )


def test_p2_lazy_pdf_dynamic_imports_present_in_expected_pages():
    """Confirma que Dashboard + Recipes mantienen el dynamic import.

    Cierre del gap inverso: si alguien elimina la funcionalidad de PDF
    completa, este test debe fallar y obligar al refactor a también
    eliminar el test (en lugar de quedar con un anchor stale apuntando
    a código inexistente).
    """
    expected = ["Dashboard.jsx", "Recipes.jsx"]
    missing: list[str] = []
    for name in expected:
        path = PAGES_DIR / name
        assert path.exists(), f"Expected page {name} no existe en {PAGES_DIR}"
        text = path.read_text(encoding="utf-8")
        if not DYNAMIC_IMPORT_RE.search(text):
            missing.append(name)

    assert not missing, (
        "[P2-LAZY-PDF] Las siguientes páginas ya NO tienen `await import("
        "'html2pdf.js')` esperado:\n  " + "\n  ".join(missing) + "\n\n"
        "Si removiste la feature de descarga PDF de la página, "
        "elimina también el nombre de esta lista en `expected`. "
        "Si la moviste a otra ubicación, actualiza el path. "
        "Si solo refactorizaste el import a un wrapper, actualiza el regex."
    )
