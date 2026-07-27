"""[P1-2-MEMORY-COMPACT · 2026-05-10] Regression guard: invariantes del
índice de memoria persistente del asistente (`MEMORY.md`).

Bug observado en el audit 2026-05-10:
    `MEMORY.md` creció a 65.2KB (target ~24.4KB del sistema). Cada entrada
    tenía cuerpo multi-cláusula con 600-1000+ chars (UTF-8). El sistema
    truncaba al cargar → tareas futuras perdían contexto P-fix histórico.

Fix:
    1. Hooks truncados a <=200 bytes por línea (convención CLAUDE.md
       "Convenciones del repo" — `Keep index entries to one line under
       ~200 chars`).
    2. Entradas con fecha <2026-05-09 dejadas como header-only (link sin
       hook). El detalle vive en su archivo topic referenciado.
    3. Reducción total 65.2KB → 28.9KB (56%).

Cobertura de este test:
    1. Toda línea bullet (`- [...]`) ocupa <=200 bytes UTF-8. Bloquea
       futuros copy-paste de bodies largos sin truncar.
    2. Cada link `(file.md)` resuelve a un archivo existente en el mismo
       directorio. Bloquea typos y deletions accidentales.
    3. Archivo total <=35KB (margen sobre el target 24.4KB sin ser
       agresivo). Si crece más, el test falla y obliga a recompactar
       (ejecutar el script de pase 1+2+3 documentado en la memoria de
       cierre P1-2).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


# [2026-07-26] El path estaba hardcodeado al directorio VIEJO, de cuando el repo no vivía dentro
# de `Nodalia/`. Al mover el proyecto, Claude Code abrió un directorio de memoria NUEVO (la clave
# es la ruta del workspace) y el viejo quedó congelado:
#
#     c--Users-angel-OneDrive-Escritorio-MealfitRD-IA               576 archivos, MEMORY.md 456 KB
#     c--...-Nodalia-MealfitRD-Software-MealfitRD-IA (ACTIVO)       251 archivos, MEMORY.md  19 KB
#
# El test medía el abandonado. De ahí los 456 KB contra un techo de 35: nadie lo compacta porque
# nadie le escribe. Un rojo que no pedía compactar nada, pedía mirar otro sitio.
#
# Ahora se deriva de la ruta REAL del repo con el mismo esquema de Claude Code (ruta absoluta con
# los separadores convertidos en `-`), con fallback al viejo si el nuevo no existe. El viejo se
# conserva como archivo histórico — 576 entradas de contexto que no se tiran, pero que tampoco
# tiene sentido someter al techo de un índice vivo.
def _memory_dir_activo() -> Path:
    _repo = Path(__file__).resolve().parent.parent.parent
    _slug = str(_repo).replace(":", "").replace("\\", "-").replace("/", "-").lower()
    _base = Path.home() / ".claude" / "projects"
    _cand = _base / f"c--{_slug.lstrip('c-')}" / "memory"
    if _cand.exists():
        return _cand
    # Fallback: el directorio cuyo MEMORY.md se tocó más recientemente.
    _dirs = [d / "memory" for d in _base.glob("c--*") if (d / "memory" / "MEMORY.md").exists()]
    if _dirs:
        return max(_dirs, key=lambda d: (d / "MEMORY.md").stat().st_mtime)
    return _base / "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA" / "memory"


_MEMORY_DIR = _memory_dir_activo()
_MEMORY_MD = _MEMORY_DIR / "MEMORY.md"
_MAX_LINE_BYTES = 200
_MAX_FILE_BYTES = 35 * 1024  # ceiling con margen sobre 24.4KB del sistema


def _skip_if_missing():
    """En entornos sin la carpeta de memoria (CI hermético, otra máquina),
    skipear en vez de fallar — el test es local-by-design."""
    if not _MEMORY_MD.exists():
        pytest.skip(f"MEMORY.md no encontrado en {_MEMORY_MD}; test es local-only.")


def test_memory_md_total_size_within_ceiling():
    """Archivo total <=35KB. Si crece, recompactar (ver memoria P1-2)."""
    _skip_if_missing()
    size = _MEMORY_MD.stat().st_size
    assert size <= _MAX_FILE_BYTES, (
        f"MEMORY.md = {size} bytes ({size/1024:.1f}KB), excede el ceiling "
        f"{_MAX_FILE_BYTES} bytes ({_MAX_FILE_BYTES/1024:.1f}KB). "
        f"Recompactar siguiendo el procedimiento documentado en "
        f"`project_p1_2_memory_compact_2026_05_10.md`."
    )


def test_every_bullet_line_within_byte_budget():
    """Cada línea `- [...]` ocupa <=200 bytes UTF-8."""
    _skip_if_missing()
    overshoots = []
    for i, line in enumerate(_MEMORY_MD.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("- ["):
            continue
        b = len(line.encode("utf-8"))
        if b > _MAX_LINE_BYTES:
            overshoots.append((i, b, line[:80]))
    assert not overshoots, (
        f"{len(overshoots)} líneas exceden {_MAX_LINE_BYTES} bytes. "
        f"Sample: {overshoots[:3]}. "
        f"Convención CLAUDE.md → `Keep index entries to one line under ~200 chars`. "
        f"Mover el detalle al archivo topic referenciado y truncar la entrada."
    )


def test_every_link_resolves_to_existing_topic_file():
    """Cada `(file.md)` referenciado en MEMORY.md existe en el mismo directorio.

    Cierra la rotura silenciosa cuando alguien borra un archivo topic sin
    actualizar el índice o renombra sin actualizar links.
    """
    _skip_if_missing()
    text = _MEMORY_MD.read_text(encoding="utf-8")
    # Capturar los `.md` referenciados (excluye anchors `#section`).
    refs = re.findall(r"\(([a-zA-Z0-9_./-]+\.md)(?:#[^\)]+)?\)", text)
    referenced = {Path(r).name for r in refs if not r.startswith(("http://", "https://"))}
    referenced.discard("MEMORY.md")  # auto-referencia no aplica
    existing = {p.name for p in _MEMORY_DIR.glob("*.md")}
    missing = sorted(referenced - existing)
    assert not missing, (
        f"{len(missing)} link(s) en MEMORY.md apunta(n) a archivos inexistentes: "
        f"{missing[:5]}. ¿Renombre/borrado sin actualizar el índice?"
    )


def test_no_orphan_topic_files():
    """Cada archivo topic en el directorio está referenciado desde MEMORY.md.

    Evita acumular archivos huérfanos (ya sea por bug del workflow de cierre
    de P-fix, o por cleanup parcial olvidado).

    Excepciones permitidas:
      - MEMORY.md (es el índice, no se referencia a sí mismo).
      - Archivos auxiliares como `python_env.md`, `feedback_*.md`,
        `reference_*.md` — son entries válidas del índice.
    """
    _skip_if_missing()
    # [reapuntado 2026-07-27] El índice se compactó archivando los meses viejos en
    # `MEMORY_archivo_*.md` (el índice se carga entero cada sesión; el archivo solo bajo demanda).
    # Una referencia sigue viva si está en MEMORY.md O en un archivo de histórico que MEMORY.md
    # enlaza — un nivel de indirección, no un grafo: si el histórico se desengancha del índice,
    # sus 100+ referencias vuelven a contar como huérfanas, que es el comportamiento correcto.
    text = _MEMORY_MD.read_text(encoding="utf-8")
    refs = re.findall(r"\(([a-zA-Z0-9_./-]+\.md)(?:#[^\)]+)?\)", text)
    referenced = {Path(r).name for r in refs if not r.startswith(("http://", "https://"))}
    for archivo in sorted(referenced):
        if not archivo.startswith("MEMORY_archivo"):
            continue
        p = _MEMORY_DIR / archivo
        if p.is_file():
            sub = re.findall(r"\(([a-zA-Z0-9_./-]+\.md)(?:#[^\)]+)?\)",
                             p.read_text(encoding="utf-8"))
            referenced |= {Path(r).name for r in sub
                           if not r.startswith(("http://", "https://"))}
    existing = {p.name for p in _MEMORY_DIR.glob("*.md")} - {"MEMORY.md"}
    orphans = sorted(existing - referenced)
    assert not orphans, (
        f"{len(orphans)} archivo(s) huérfano(s) en {_MEMORY_DIR.name}/ "
        f"(no referenciados desde MEMORY.md ni sus archivos de histórico): {orphans[:5]}. "
        f"Añadir entrada al índice o eliminar el archivo."
    )
