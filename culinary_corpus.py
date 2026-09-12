# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-18 · 2026-09-12 · C0] El corpus FIJO de la medición culinaria.

`plan_data` es VIVO: el shift del cron encoge la ventana de días de un plan ya existente, y la purga de cuentas se
lleva planes enteros (la flota pasó de 96 planes el 09-06 a 6 el 09-12). Una línea base medida sobre la ventana viva
no se puede volver a medir mañana —la del 6-sep dejó de ser reproducible en 14 h—, así que ningún umbral calculado
contra ella es interpretable: el delta mezcla el efecto del código con el del cron.

Este módulo congela en un fichero, con huella, exactamente lo que las capas de medición LEEN:

  · por plan: `days` tal cual (nombre, ingredientes, pasos… lo que `culinary_contract_scan` y `judged_fingerprint`
    leen), `_culinary_judge_history` (lo que el juez ya dijo) y el estado (`revision`, `generation_status`,
    `updated_at`);
  · el catálogo con el que `build_culinary_index` construye el vocabulario del detector (`master_ingredients`:
    name, aliases, category, ready_to_eat, prep_methods) — el vocabulario cambia con el catálogo, así que entra en
    la huella;
  · la huella del corpus = sha256 de las huellas de los planes (ordenadas) + la del catálogo + la versión del formato.

Medir dos veces el mismo fichero da las mismas cifras por construcción; una diferencia entre dos mediciones sobre
el MISMO corpus sólo puede venir del código (`huella_reglas()`), nunca del cron. Puro: sin DB. Quien lee la base es
`scripts/congela_corpus_culinario.py`; quien mide, `scripts/culinary_baseline.py --corpus`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

CORPUS_VERSION = 1

#: Claves de `plan_data` que las capas de medición leen. Sólo estas se congelan: el resto del plan (macros, lista de
#: compras, políticas) no entra ni en el scan ni en el sello del juez.
CLAVES_PLAN_DATA = ("days", "_culinary_judge_history", "generation_status", "name")

#: Columnas de `master_ingredients` con las que `build_culinary_index` construye el vocabulario.
COLUMNAS_CATALOGO = ("name", "aliases", "category", "ready_to_eat", "prep_methods")


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _h16(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def huella_plan(plan_data: dict) -> str:
    """sha256 (16 hex) del contenido congelado de un plan: cambia si cambia un paso, un ingrediente o un juicio."""
    return _h16(_canon(plan_data))


def rebanada_plan(fila: dict) -> dict:
    """La rebanada congelable de una fila de `meal_plans` (`id`, `created_at`, `updated_at`, `revision`, `plan_data`)."""
    pd = fila.get("plan_data") or {}
    datos = {k: pd.get(k) for k in CLAVES_PLAN_DATA if k in pd}
    dias = [d for d in (datos.get("days") or []) if isinstance(d, dict)]
    return {
        "plan_id": str(fila.get("id")),
        "created_at": str(fila["created_at"]) if fila.get("created_at") is not None else None,
        "updated_at": str(fila["updated_at"]) if fila.get("updated_at") is not None else None,
        "revision": fila.get("revision"),
        "generation_status": pd.get("generation_status"),
        "dias": len(dias),
        "comidas": _comidas(datos),
        "huella": huella_plan(datos),
        "plan_data": datos,
    }


def _comidas(plan_data: dict) -> int:
    return sum(len((d or {}).get("meals") or []) for d in (plan_data or {}).get("days") or [] if isinstance(d, dict))


def filas_catalogo(cat: Iterable[dict]) -> list[dict]:
    """El catálogo reducido a las columnas del índice y ordenado por nombre: es lo que se congela y lo que se hashea."""
    out = [{c: r.get(c) for c in COLUMNAS_CATALOGO} for r in (cat or []) if isinstance(r, dict)]
    return sorted(out, key=lambda r: str(r.get("name") or ""))


def huella_catalogo(cat: Iterable[dict]) -> str:
    return _h16(_canon(filas_catalogo(cat)))


def huella_corpus(huellas_planes: Iterable[str], huella_cat: str) -> str:
    return _h16(_canon({"v": CORPUS_VERSION, "planes": sorted(huellas_planes), "catalogo": huella_cat}))


def huella_reglas() -> Optional[str]:
    """La versión de las reglas que producen las cifras: sha del fuente de `culinary_coherence.py`."""
    try:
        p = Path(__file__).resolve().parent / "culinary_coherence.py"
        return _h16(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def congelar(filas: Iterable[dict], cat: Iterable[dict], *, motivo: str, git_sha: Optional[str] = None,
             ahora: Optional[datetime] = None) -> dict:
    """El documento del corpus. Planes ordenados por id (el orden de lectura no entra en la huella)."""
    planes = sorted((rebanada_plan(f) for f in filas), key=lambda p: p["plan_id"])
    cat_f = filas_catalogo(cat)
    h_cat = huella_catalogo(cat_f)
    return {
        "version": CORPUS_VERSION,
        "congelado_at": (ahora or datetime.now(timezone.utc)).isoformat(),
        "motivo": motivo,
        "codigo": {"git_sha": git_sha, "reglas_huella": huella_reglas()},
        "planes_n": len(planes),
        "comidas": sum(p["comidas"] for p in planes),
        "catalogo": {"filas": len(cat_f), "huella": h_cat},
        "huella": huella_corpus((p["huella"] for p in planes), h_cat),
        "nota": ("Corpus FIJO: congela `days`, `_culinary_judge_history` y el estado de cada plan, más el catálogo del "
                 "índice culinario. Medirlo dos veces da las mismas cifras; si dos mediciones sobre este fichero "
                 "difieren, cambió el código (reglas_huella), no el corpus."),
        "planes": planes,
        "catalogo_filas": cat_f,
    }


def verificar_integridad(corpus: dict) -> list[str]:
    """Recalcula cada huella del fichero. Vacío = íntegro; si no, qué se editó o se corrompió."""
    fallos: list[str] = []
    planes = corpus.get("planes") or []
    recalculadas = []
    for p in planes:
        pd = p.get("plan_data") or {}
        h = huella_plan(pd)
        recalculadas.append(h)
        if h != p.get("huella"):
            fallos.append(f"plan {str(p.get('plan_id'))[:8]}: huella {p.get('huella')} ≠ recalculada {h}")
        c = _comidas(pd)
        if c != p.get("comidas"):
            fallos.append(f"plan {str(p.get('plan_id'))[:8]}: comidas {p.get('comidas')} ≠ contadas {c}")
    h_cat = huella_catalogo(corpus.get("catalogo_filas") or [])
    if h_cat != (corpus.get("catalogo") or {}).get("huella"):
        fallos.append(f"catálogo: huella {(corpus.get('catalogo') or {}).get('huella')} ≠ recalculada {h_cat}")
    # desde las huellas RECALCULADAS, no las guardadas: si no, editar un plan dejaría la huella del corpus «íntegra»
    h = huella_corpus(recalculadas, h_cat)
    if h != corpus.get("huella"):
        fallos.append(f"corpus: huella {corpus.get('huella')} ≠ recalculada {h}")
    if corpus.get("planes_n") != len(planes):
        fallos.append("planes_n no coincide con la lista de planes")
    if corpus.get("comidas") != sum(p.get("comidas") or 0 for p in planes):
        fallos.append("comidas no coincide con la suma por plan")
    return fallos


def cargar(path) -> dict:
    """Lee y verifica. Un corpus editado a mano o corrupto no se mide: se rechaza con la lista de fallos."""
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    fallos = verificar_integridad(corpus)
    if fallos:
        raise ValueError("corpus no íntegro: " + "; ".join(fallos))
    return corpus


def filas_para_medir(corpus: dict) -> list[dict]:
    """Las filas con la forma que `culinary_baseline._medir_filas` espera: `{"id", "plan_data"}`."""
    return [{"id": p["plan_id"], "plan_data": p.get("plan_data") or {}} for p in corpus.get("planes") or []]
