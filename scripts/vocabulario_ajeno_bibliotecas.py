# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-855 · 2026-09-29] ¿Cuántas plantillas de las bibliotecas beta hablan dominicano?

G24 (29-sep, 6 planes reales) leyó «guineo», «lechosa», «auyama», «habichuelas» y «ají morrón» en México,
España y EE. UU., y la revisión encontró que el analizador de la batería los EXIMÍA porque ya estaban en la
biblioteca de platos de ese país: `dish_templates_us.json` trae Guineo, Auyama y Habichuelas rojas. Esta
herramienta mide esa contaminación con un criterio explícito por país (la tabla `NATIVO_FUERA_DE_RD`), no con
«lo que la biblioteca ya dice».

SÓLO MIDE. No toca bibliotecas, catálogo ni plan: los `constituents[].name` son nombres EXACTOS de
`master_ingredients` (identificadores del motor — `pantry_names_match`, el guard de coherencia y el backstop de
alergias resuelven por ellos), y cambiar una plantilla cambia lo que se genera. Lo que el usuario ve puede
localizarse en una capa de vista; eso es una propuesta, no esta herramienta.

Uso:  python scripts/vocabulario_ajeno_bibliotecas.py [--json salida.json]
tooltip-anchor: P1-PLAN-LOTE-855-VOCABULARIO-AJENO
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_BIBLIOTECAS = {"ES": "dish_templates_es.json", "US": "dish_templates_us.json", "MX": "dish_templates_mx.json",
                "PR": "dish_templates_pr.json", "CO": "dish_templates_co.json"}
_PAISES_BETA = tuple(_BIBLIOTECAS)

# Palabra de la mesa dominicana (minúscula, sin acentos, singular) → países BETA donde también es la palabra de
# todos los días. En RD todas son nativas (RD nunca es «ajeno»). Criterio conservador: ante la duda, se cuenta
# como nativa (se prefiere no acusar de más). Revisable por el dueño; cambiarla sólo cambia la MEDICIÓN.
NATIVO_FUERA_DE_RD: dict[str, frozenset] = {
    # frutas y víveres
    "guineo": frozenset({"PR"}),                 # ES/MX plátano · CO banano · US banana
    "guineo verde": frozenset({"PR", "CO"}),     # el cayeye de la costa colombiana es de guineo verde
    "guineito": frozenset({"PR"}),
    "lechosa": frozenset({"PR"}),                # papaya en ES/MX/CO/US
    "chinola": frozenset(),                      # PR parcha · maracuyá / fruta de la pasión
    "auyama": frozenset({"CO"}),                 # CO «ahuyama/auyama» · calabaza en ES/MX/PR/US
    "batata": frozenset({"ES", "PR", "CO", "US"}),   # MX camote
    "yautia": frozenset({"PR"}),                 # malanga en MX/CO/ES/US
    "tayota": frozenset(),                       # chayote
    "molondron": frozenset(),                    # okra / quimbombó / guingambó
    "mapuey": frozenset(),
    "lerenes": frozenset({"PR"}),
    "viveres": frozenset(),                      # PR «viandas»
    # legumbres y verduras
    "habichuela": frozenset({"PR", "CO"}),       # frijol (MX/US) · judía/alubia (ES) · en CO es la vaina VERDE
    "habichuelas rojas": frozenset({"PR"}),      # el GRANO: en CO se lee «vaina verde roja»
    "habichuelas negras": frozenset({"PR"}),
    "habichuelas blancas": frozenset({"PR"}),
    "gandul": frozenset({"PR", "CO"}),
    "guandul": frozenset({"PR", "CO"}),
    "vainita": frozenset(),                      # judía verde · ejote · habichuela (CO)
    "aji morron": frozenset(),                   # pimiento (ES/MX/PR/US) · pimentón (CO)
    "aji cubanela": frozenset({"PR"}),
    "recao": frozenset({"PR"}),
    "bija": frozenset(),                         # achiote
    # lácteos y embutidos
    "queso blanco": frozenset({"PR", "CO", "MX", "US"}),   # en España no: queso fresco / de Burgos
    "queso de freir": frozenset(),
    "queso de hoja": frozenset(),
    "tocineta": frozenset({"PR", "CO", "US"}),   # ES beicon/panceta · MX tocino
    "dominicano": frozenset(),                   # «Salami dominicano», «Orégano dominicano», «Longaniza dominicana»
    "dominicana": frozenset(),
    # panes y platos
    "casabe": frozenset({"CO", "PR"}),
    "pan de agua": frozenset({"PR"}),
    "mangu": frozenset(),
    "moro": frozenset(),
    "locrio": frozenset(),
    "chenchen": frozenset(),
    "chaca": frozenset(),
    "yaniqueque": frozenset(),
    "catibia": frozenset(),
    "quipe": frozenset(),
    "concon": frozenset(),
    "pica pollo": frozenset(),
    "revoltillo": frozenset({"PR"}),             # huevos revueltos / pericos / revuelto
    "pastelon": frozenset({"PR"}),
    "mofongo": frozenset({"PR"}),
    "toston": frozenset({"PR", "US"}),           # CO patacón
    "asopao": frozenset({"PR"}),
    "sancocho": frozenset({"PR", "CO"}),
    "arepita": frozenset({"CO"}),
    "fusion criolla": frozenset({"PR"}),         # técnica asignada dominicana
    "ropa vieja": frozenset({"ES", "PR", "US"}),     # Canarias; cubana en PR/US; MX salpicón, CO desmechada
    # marcas de RD (salen en la lista de compras). «funda» (el envase) NO entra: choca con el verbo — «para que el
    # queso funda» (G24 CO, D1 cena) —; los envases de RD ya los mide la lista (C5/C6 de la batería).
    "sosua": frozenset(),
}


def _norm(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii").lower()


_RX = {w: re.compile(r"\b" + re.escape(w) + r"(?:s|es)?\b") for w in NATIVO_FUERA_DE_RD}


def palabras_ajenas(texto, pais) -> list:
    """Las palabras de la tabla presentes en `texto` (frontera de palabra, plural incluido) que en `pais` no son
    la palabra de todos los días. Manda la entrada MÁS LARGA («guineo verde» antes que «guineo»: en Colombia la
    primera es nativa y la segunda no). RD ⇒ siempre []. Orden de la tabla; sin repetidos."""
    cc = str(pais or "").strip().upper()
    if cc not in _PAISES_BETA:
        return []
    t = _norm(texto)
    hits = [(w, m.span()) for w, rx in _RX.items() for m in rx.finditer(t)]
    dentro = {i for i, (w, (a, b)) in enumerate(hits)
              for j, (w2, (a2, b2)) in enumerate(hits) if i != j and a2 <= a and b <= b2 and (b2 - a2) > (b - a)}
    vistas = {w for i, (w, _) in enumerate(hits) if i not in dentro}
    return [w for w in _RX if w in vistas and cc not in NATIVO_FUERA_DE_RD[w]]


def _cargar(pais) -> list:
    d = json.loads((_BACKEND / "data" / _BIBLIOTECAS[pais]).read_text(encoding="utf-8"))
    return d.get("templates") or []


def medir_bibliotecas(max_ejemplos: int = 8) -> dict:
    """Por país beta: plantillas, cuántas llevan alguna palabra ajena (en el nombre o en un constituyente), en qué
    campo, y las palabras más frecuentes."""
    out = {}
    for cc in _PAISES_BETA:
        tpls = _cargar(cc)
        con, en_nombre, en_const, freq, ejemplos = 0, 0, 0, {}, []
        for t in tpls:
            pn = palabras_ajenas(t.get("name"), cc)
            pc = sorted({w for c in (t.get("constituents") or []) for w in palabras_ajenas(c.get("name"), cc)})
            todas = sorted(set(pn) | set(pc))
            if not todas:
                continue
            con += 1
            en_nombre += bool(pn)
            en_const += bool(pc)
            for w in todas:
                freq[w] = freq.get(w, 0) + 1
            if len(ejemplos) < max_ejemplos:
                ejemplos.append({"plantilla": t.get("name"), "palabras": todas,
                                 "constituyentes": [c.get("name") for c in (t.get("constituents") or [])
                                                    if palabras_ajenas(c.get("name"), cc)]})
        out[cc] = {"plantillas": len(tpls), "plantillas_con_ajeno": con, "en_nombre": en_nombre,
                   "en_constituyentes": en_const,
                   "palabras": dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))), "ejemplos": ejemplos}
    return out


def main(argv) -> int:
    m = medir_bibliotecas()
    lineas = []
    for cc, r in m.items():
        pct = 100.0 * r["plantillas_con_ajeno"] / r["plantillas"] if r["plantillas"] else 0.0
        lineas.append(f"{cc}: {r['plantillas_con_ajeno']}/{r['plantillas']} plantillas ({pct:.0f} %) · en el nombre "
                      f"{r['en_nombre']} · en constituyentes {r['en_constituyentes']} · {r['palabras']}")
        lineas.extend(f"     {e['plantilla']}  <- {e['palabras']}" for e in r["ejemplos"][:4])
    # [P2-LOGGER-EXEMPT: salida CLI de la medición, a stdout a propósito]
    print("\n".join(lineas))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
