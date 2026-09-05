"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Benchmark cultural (roadmap 2.5 §13.4) — Fase 7, subfase C.

Mide, SIN LLM y de forma reproducible, lo que el gate de la Fase 7 exige de cada perfil de cocina, leyendo
los snapshots compilados del Dish Registry (`data/registry/dish_registry_<lib>_v1.json`) y los seis perfiles
de `cultural_profiles.PROFILES`:

  1. resolvabilidad     — % de constituyentes que resuelven al catálogo (el snapshot ya lo trae);
  2. cobertura          — plantillas por franja frente a la barra (≥80; desayuno 18 / almuerzo 28 / cena 22 /
                          merienda 16; ≥10 familias de proteína; ≥12 técnicas);
  3. contaminación      — plantillas cuyo nombre lleva un plato/básico EXCLUSIVO de otra cocina (p. ej. «mangú»
                          en la biblioteca de US); un léxico compartido por dos cocinas no cuenta;
  4. adecuación         — técnica/franja: arroz o pasta como base fuera de almuerzo, fritura por encima del
                          tope, guisos/sopas en merienda;
  5. disponibilidad     — constituyentes marcados `is_dominican_cultivar` usados fuera de DO (riesgo de no
                          encontrarlos en ese mercado) y % con precio en el catálogo (solo DO tiene precios);
  6. diversidad         — familias de proteína, técnicas, bases y cabezas de plato distintas dentro del perfil;
  7. mezcla coherente   — con principal 0,7 / secundaria 0,3, cada día del reparto determinista encuentra
                          candidatos en SU biblioteca para las cuatro franjas;
  8. cero bypass clínico— `template_candidates(..., exclude_allergens=[clase])` jamás devuelve una plantilla
                          que lleve esa clase; y cuántas llevan carne procesada o sodio alto (para el revisor);
  9. revisión humana    — estado editorial y la lista de plantillas marcadas por 3/4/5 para que una persona
                          las mire: el benchmark señala, no cura.

Salida: `run_benchmark()` → dict; `gate_verdict()` → (ok, fallos); `render_markdown()` → informe para
`backend/docs/cultural_benchmark_report.md`. CLI: `python cultural_benchmark.py [--write]`.
Fail-open: sin snapshots devuelve un informe vacío con `gate_ok=False` y el motivo.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import unicodedata
from typing import Any, Iterable, Optional

BAR = {"total": 80, "desayuno": 18, "almuerzo": 28, "cena": 22, "merienda": 16, "proteins": 10, "techniques": 12}
MAX_FRIED_SHARE = 0.20
MAX_AVAILABILITY_RISK_PCT = 15.0
ALLERGEN_CLASSES = ("mariscos", "huevo", "lacteos", "gluten", "mani", "frutos secos", "pescado", "soya")
_HEAVY_FOR_SNACK = ("guisado", "sopa", "sancocho", "asopao", "estofado")
_STARCH_BASES = ("arroz", "pasta")


def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").split())


def _words(s: str) -> set[str]:
    return set(_norm(s).replace(",", " ").replace("(", " ").replace(")", " ").split())


def _load_snapshots() -> dict[str, dict]:
    import dish_registry as dr
    out = {}
    for lib in dr.LIBRARIES:
        p = dr.snapshot_path(lib)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[lib] = json.load(f)
    return out


# Palabras de plato GENÉRICAS (formato, técnica o ingrediente panhispánico): que un perfil las declare no las
# vuelve suyas. Sin esta lista, «huevo» (básico de US) marcaba de contaminación a media biblioteca dominicana.
_GENERIC_DISH_WORDS = {
    "guisado", "guisados", "guiso", "tortilla", "ensalada", "revuelto", "sopa", "sopas", "pasta", "asado",
    "lentejas", "caldo", "caldos", "bowl", "sandwich", "sándwich", "arroz", "huevo", "pollo", "vegetales",
    "avena", "aguacate", "tomate", "frijol", "frijoles", "habichuelas", "papa", "patata", "yuca", "platano",
    "plátano", "pan", "pan integral", "pescado al horno", "legumbres", "aceite de oliva", "chile", "nopal",
}
MAX_CONTAMINATION_SHARE = 0.03


def _profile_lexicons() -> dict[str, set[str]]:
    """Léxico EXCLUSIVO por perfil: PLATOS (`dish_families`, no básicos) que ningún otro perfil declara y que
    no son una palabra genérica. Un plato compartido por dos cocinas (sancocho DO/CO, asopao DO/PR) no cuenta."""
    import cultural_profiles as cp
    raw: dict[str, set[str]] = {}
    for pid, p in cp.PROFILES.items():
        toks: set[str] = set()
        for item in (p.get("dish_families") or []):
            t = _norm(item)
            if len(t) >= 4 and t not in {_norm(g) for g in _GENERIC_DISH_WORDS}:
                toks.add(t)
        raw[pid] = toks
    exclusive = {}
    for pid, toks in raw.items():
        others = set().union(*(v for k, v in raw.items() if k != pid))
        exclusive[pid] = {t for t in toks if t not in others}
    return exclusive


def _lib_profile(lib: str) -> str:
    import cultural_profiles as cp
    for pid, p in cp.PROFILES.items():
        if p.get("library") == lib:
            return pid
    return "dominican_criolla"


def _contamination(lib: str, templates: list, lexicons: dict[str, set[str]]) -> list[dict]:
    own = _lib_profile(lib)
    hits = []
    for t in templates:
        name = _norm(t.get("name"))
        words = _words(name)
        for pid, toks in lexicons.items():
            if pid == own:
                continue
            for tok in toks:
                if (" " in tok and tok in name) or (" " not in tok and tok in words):
                    hits.append({"template": t.get("name"), "foreign_profile": pid, "token": tok})
                    break
    return hits


def _appropriateness(templates: list) -> dict:
    starch_off_slot, fried, heavy_snack = [], 0, []
    for t in templates:
        slots = [_norm(s) for s in (t.get("slots") or [])]
        base = _norm(t.get("base"))
        tech = _norm(t.get("technique"))
        if base in _STARCH_BASES and any(s in ("desayuno", "cena") for s in slots):
            starch_off_slot.append(t.get("name"))
        if "frit" in tech:
            fried += 1
        if "merienda" in slots and any(h in tech for h in _HEAVY_FOR_SNACK):
            heavy_snack.append(t.get("name"))
    n = max(1, len(templates))
    return {"starch_base_off_slot": starch_off_slot, "fried_share": round(fried / n, 3),
            "heavy_technique_in_snack": heavy_snack,
            "ok": not starch_off_slot and fried / n <= MAX_FRIED_SHARE}


def _availability(lib: str, templates: list, catalog_rows: Optional[list]) -> dict:
    if not catalog_rows:
        return {"available": None, "note": "sin catálogo vivo: disponibilidad no evaluada"}
    by_name = {_norm(r.get("name")): r for r in catalog_rows if isinstance(r, dict)}
    total, priced, cultivar_hits = 0, 0, []
    for t in templates:
        for c in t.get("constituents") or []:
            row = by_name.get(_norm(c.get("canonical") or c.get("name")))
            if not row:
                continue
            total += 1
            if row.get("price_per_lb") or row.get("price_per_unit"):
                priced += 1
            if lib != "do" and row.get("is_dominican_cultivar"):
                cultivar_hits.append({"template": t.get("name"), "ingredient": row.get("name")})
    risk_pct = round(100.0 * len(cultivar_hits) / total, 1) if total else 0.0
    # `is_dominican_cultivar` marca producto TÍPICO de DO, no ausente fuera: en PR o CO la yuca, el plátano o
    # la auyama son igual de corrientes. Por eso el porcentaje es una señal para el revisor humano (lista
    # nominal abajo) y solo rompe el gate cuando la biblioteca depende de ese producto de forma extrema.
    return {"constituents": total, "priced_pct": round(100.0 * priced / total, 1) if total else 0.0,
            "dominican_cultivar_outside_do": cultivar_hits, "availability_risk_pct": risk_pct,
            "review_threshold_pct": MAX_AVAILABILITY_RISK_PCT, "needs_review": risk_pct > MAX_AVAILABILITY_RISK_PCT,
            "ok": risk_pct <= 35.0}


def _diversity(templates: list) -> dict:
    prots = collections.Counter(_norm(t.get("protein")) for t in templates)
    techs = collections.Counter(_norm(t.get("technique")) for t in templates)
    bases = collections.Counter(_norm(t.get("base")) for t in templates)
    heads = {" ".join(_norm(t.get("name")).split()[:2]) for t in templates}
    n = max(1, len(templates))
    top_prot = prots.most_common(1)[0][1] / n if prots else 0.0
    return {"proteins": len(prots), "techniques": len(techs), "bases": len(bases),
            "distinct_heads_ratio": round(len(heads) / n, 3), "top_protein_share": round(top_prot, 3),
            "transform_share": round(sum(1 for t in templates if t.get("transform")) / n, 3),
            "ok": len(prots) >= BAR["proteins"] and len(techs) >= BAR["techniques"] and top_prot <= 0.5}


def _coverage(templates: list) -> dict:
    slots = collections.Counter(_norm(s) for t in templates for s in (t.get("slots") or []))
    gaps = []
    if len(templates) < BAR["total"]:
        gaps.append(f"total {len(templates)} < {BAR['total']}")
    for s in ("desayuno", "almuerzo", "cena", "merienda"):
        if slots.get(s, 0) < BAR[s]:
            gaps.append(f"{s} {slots.get(s, 0)} < {BAR[s]}")
    return {"templates": len(templates), "slots": {s: slots.get(s, 0) for s in ("desayuno", "almuerzo", "cena", "merienda")},
            "gaps": gaps, "ok": not gaps}


def _clinical(lib: str, cc: str) -> dict:
    import dish_registry as dr
    leaks = []
    snap = dr.load_registry(cc) or {}
    ts = snap.get("templates") or []
    by_id = {t.get("template_id"): t for t in ts}
    for cls in ALLERGEN_CLASSES:
        for slot in ("desayuno", "almuerzo", "cena", "merienda"):
            # ningún candidato devuelto con la clase excluida puede llevarla en sus atributos de riesgo
            for c in dr.template_candidates(cc, slot, None, k=500, exclude_allergens=[cls]):
                t = by_id.get(c.get("template_id")) or {}
                if cls in {_norm(a) for a in ((t.get("intrinsic_risk_attributes") or {}).get("allergens") or [])}:
                    leaks.append({"template": t.get("name"), "class": cls, "slot": slot})
    processed = [t.get("name") for t in ts if (t.get("intrinsic_risk_attributes") or {}).get("processed_meat")]
    sodium = [t.get("name") for t in ts if (t.get("intrinsic_risk_attributes") or {}).get("sodium_high")]
    return {"allergen_leaks": leaks, "processed_meat": processed, "sodium_high": sodium, "ok": not leaks}


def _mixing(profiles: Iterable[str], days: int = 10) -> dict:
    import cultural_profiles as cp
    import dish_registry as dr
    results = {}
    pids = list(profiles)
    for main in pids:
        for sec in pids:
            if sec == main:
                continue
            ws = [{"profile_id": main, "weight": 0.7}, {"profile_id": sec, "weight": 0.3}]
            seq = [cp.profile_for_day(ws, d) for d in range(days)]
            missing = []
            for d, pid in enumerate(seq):
                cc = cp.country_for_profile(pid)
                for slot in ("desayuno", "almuerzo", "cena", "merienda"):
                    if not dr.template_candidates(cc, slot, None, k=1):
                        missing.append({"day": d, "profile": pid, "slot": slot})
            results[f"{main}+{sec}"] = {"split": dict(collections.Counter(seq)), "missing": missing, "ok": not missing}
    return {"pairs": results, "ok": all(r["ok"] for r in results.values())}


def run_benchmark(*, catalog_rows: Optional[list] = None) -> dict:
    import cultural_profiles as cp
    import dish_registry as dr
    snaps = _load_snapshots()
    report: dict[str, Any] = {"schema_version": 1, "bar": BAR, "profiles": {}, "gate_ok": False, "failures": []}
    if not snaps:
        report["failures"].append("sin snapshots compilados (scripts/compile_dish_registry.py)")
        return report
    lexicons = _profile_lexicons()
    for pid, p in cp.PROFILES.items():
        lib = p["library"]
        snap = snaps.get(lib)
        if not snap:
            report["profiles"][pid] = {"library": lib, "missing_snapshot": True}
            continue
        ts = snap.get("templates") or []
        cc = snap.get("country") or cp.country_for_profile(pid)
        st = snap.get("stats") or {}
        entry = {
            "library": lib, "country": cc, "snapshot_hash": snap.get("snapshot_hash"),
            "resolvability": {"pct": st.get("resolution_pct"), "ok": (st.get("resolution_pct") or 0) >= (99.0 if lib == "do" else 100.0),
                              "excluded": st.get("excluded", 0)},
            "coverage": _coverage(ts),
            "contamination": {"hits": _contamination(lib, ts, lexicons)},
            "appropriateness": _appropriateness(ts),
            "availability": _availability(lib, ts, catalog_rows),
            "diversity": _diversity(ts),
            "clinical": _clinical(lib, cc),
            "review": {"editorial": dict(collections.Counter((t.get("editorial") or {}).get("status") for t in ts))},
        }
        # Gate: hasta un 3 % de plantillas con léxico ajeno (un «taco en hoja de lechuga» dominicano es
        # legítimo); por encima, la biblioteca está copiando otra cocina. Todas van a revisión humana.
        entry["contamination"]["ok"] = len(entry["contamination"]["hits"]) <= MAX_CONTAMINATION_SHARE * max(1, len(ts))
        flagged = [h["template"] for h in entry["contamination"]["hits"]]
        flagged += entry["appropriateness"]["starch_base_off_slot"] + entry["appropriateness"]["heavy_technique_in_snack"]
        flagged += [h["template"] for h in (entry["availability"].get("dominican_cultivar_outside_do") or [])]
        entry["review"]["flagged_for_human_review"] = sorted(set(flagged))
        entry["review"]["human_signoff"] = False
        report["profiles"][pid] = entry
    report["mixing"] = _mixing(cp.PROFILES.keys())
    ok, failures = gate_verdict(report)
    report["gate_ok"], report["failures"] = ok, failures
    return report


def gate_verdict(report: dict) -> tuple[bool, list[str]]:
    failures = []
    for pid, e in (report.get("profiles") or {}).items():
        if e.get("missing_snapshot"):
            failures.append(f"{pid}: sin snapshot")
            continue
        for key in ("resolvability", "coverage", "contamination", "appropriateness", "diversity", "clinical"):
            if not (e.get(key) or {}).get("ok", False):
                failures.append(f"{pid}: {key}")
        av = e.get("availability") or {}
        if av.get("ok") is False:
            failures.append(f"{pid}: availability")
    if report.get("mixing") and not report["mixing"].get("ok", False):
        failures.append("mixing")
    return (not failures), failures


def render_markdown(report: dict) -> str:
    lines = ["# Benchmark cultural (Fase 7, §13.4)", "",
             f"Gate: **{'PASA' if report.get('gate_ok') else 'NO PASA'}**"
             + (f" — fallos: {', '.join(report.get('failures') or [])}" if report.get("failures") else ""), "",
             "| Perfil | Biblioteca | Plantillas | Des/Alm/Cen/Mer | Resueltos | Contaminación | Adecuación | Disponib. riesgo | Proteínas/Técnicas | Clínico | Revisión humana |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pid, e in (report.get("profiles") or {}).items():
        if e.get("missing_snapshot"):
            lines.append(f"| {pid} | {e.get('library')} | — | — | — | — | — | — | — | — | sin snapshot |")
            continue
        cov = e["coverage"]; s = cov["slots"]
        av = e["availability"]
        lines.append(
            f"| {pid} | {e['library']} | {cov['templates']} | {s['desayuno']}/{s['almuerzo']}/{s['cena']}/{s['merienda']} "
            f"| {e['resolvability']['pct']} % | {len(e['contamination']['hits'])} | "
            f"{'ok' if e['appropriateness']['ok'] else 'revisar'} (fritura {int(e['appropriateness']['fried_share'] * 100)} %) | "
            f"{av.get('availability_risk_pct', 'n/a')} % | {e['diversity']['proteins']}/{e['diversity']['techniques']} | "
            f"{'ok' if e['clinical']['ok'] else 'FUGA'} (procesados {len(e['clinical']['processed_meat'])}, sodio {len(e['clinical']['sodium_high'])}) | "
            f"{len(e['review']['flagged_for_human_review'])} marcadas · firma: {'sí' if e['review']['human_signoff'] else 'pendiente'} |")
    mixing = report.get("mixing") or {}
    lines += ["", f"Mezcla 0,7/0,3 en 10 días: {'todas las parejas con candidatos en las 4 franjas' if mixing.get('ok') else 'faltan candidatos'} "
              f"({len(mixing.get('pairs') or {})} parejas).", ""]
    for pid, e in (report.get("profiles") or {}).items():
        fl = (e.get("review") or {}).get("flagged_for_human_review") or []
        if fl:
            lines.append(f"- **{pid}** — para revisión humana: " + "; ".join(fl[:12]) + (" …" if len(fl) > 12 else ""))
        for h in (e.get("contamination") or {}).get("hits") or []:
            lines.append(f"  - contaminación: «{h['template']}» lleva «{h['token']}» ({h['foreign_profile']})")
        for h in (e.get("availability") or {}).get("dominican_cultivar_outside_do") or []:
            lines.append(f"  - disponibilidad: «{h['template']}» usa {h['ingredient']} (cultivar dominicano)")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rows = None
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        import db_core
        db_core.connection_pool.open()
        from shopping_calculator import get_master_ingredients
        rows = list(get_master_ingredients() or [])
    except Exception as e:  # noqa: BLE001
        print(f"[cultural_benchmark] sin catálogo vivo ({type(e).__name__}): disponibilidad no evaluada")
    report = run_benchmark(catalog_rows=rows)
    md = render_markdown(report)
    print(md)
    if "--write" in argv:
        here = os.path.dirname(os.path.abspath(__file__))
        out_json = os.path.join(here, "data", "registry", "cultural_benchmark_v1.json")
        with open(out_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        out_md = os.path.join(here, "docs", "cultural_benchmark_report.md")
        with open(out_md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print("escrito", out_json, "y", out_md)
    return 0 if report.get("gate_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
