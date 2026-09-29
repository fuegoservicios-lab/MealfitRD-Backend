# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Replay del autofix de proteína repetida sobre el corpus congelado: el lote sólo QUITA
reescrituras a la base; nunca añade una ni cambia la que la base hacía.

Revisión r6: con las especies nuevas contando, la comida de sardinas se quedaba de guardiana y el autofix reescribía a
pollo la tilapia de la base (d8b10b05 D2, «Espaguetis con sardinas»): una repetición que la base no veía y una comida que
no tocaba. Corpus: los 13 planes de producción con días y los 6 de G24 (`fixtures/replay_857_autofix_2026_09_29.json`),
en tres modos (orden real, orden invertido, dieta pescetariana), y la prueba forzada de la ronda 6: cada comida del mar
del corpus junto a un plato COCINADO de su misma etiqueta, en los dos órdenes. La base se reproduce en el mismo proceso
apagando lo del lote: el mapa de pescado de antes (`graph_orchestrator._PESCADO_ANTES`), sin especies nuevas, sin la
guarda del pez listo y sin la escalera por dieta; medido una vez contra el árbol de la base (fd62378b), da lo mismo byte a
byte en los 141 casos. Sin IA ni DB.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path

import pytest

import graph_orchestrator as go
import pescado_especies as pe
from constants import strip_accents as _sa
from culinary_context import _name_has_token

_BACKEND = Path(__file__).resolve().parent.parent
_CORPUS = json.loads((_BACKEND / "tests" / "fixtures" / "replay_857_autofix_2026_09_29.json")
                     .read_text(encoding="utf-8"))["planes"]
_CAMPOS = ("name", "ingredients", "ingredients_raw", "recipe")
_MAR = ("pescado", "atun", "camarones")
_COCIDO = {
    "pescado": {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
                "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"],
                "recipe": ["Hornea la tilapia 20 minutos a 200 °C."]},
    "atun": {"meal": "Almuerzo", "name": "Atún guisado con arroz", "ingredients": ["150 g de Atún fresco", "1 taza de Arroz"],
             "recipe": ["Guisa el atún 15 minutos en salsa de tomate."]},
    "camarones": {"meal": "Almuerzo", "name": "Camarones al ajillo", "ingredients": ["150 g de Camarones", "1 taza de Arroz"],
                  "recipe": ["Saltea los camarones con ajo 5 minutos."]},
}


class _Motivos(logging.Handler):
    def __init__(self):
        super().__init__()
        self.motivos = []

    def emit(self, rec):
        m = re.search(r"\[P1-AUTOFIX-IMPOTENCE\] Día (\S+): '(\w+)'.*?reason=(\w+)", rec.getMessage())
        if m:
            self.motivos.append(m.groups())


def _cambios(antes, despues) -> dict:
    out = {}
    for i, (da, dd) in enumerate(zip(antes, despues)):
        for j, (ma, mb) in enumerate(zip(da.get("meals") or [], dd.get("meals") or [])):
            if isinstance(ma, dict) and any(ma.get(c) != mb.get(c) for c in _CAMPOS):
                out[(i + 1, j)] = (mb.get("_protein_autofix_applied"), json.dumps([mb.get(c) for c in _CAMPOS],
                                                                                   ensure_ascii=False))
    return out


def _casos():
    """(clave, días, formulario): cada plan en sus tres modos, y la prueba forzada."""
    casos = []
    for k, p in _CORPUS.items():
        for modo in ("real", "inv", "pesc"):
            dias = copy.deepcopy(p["days"])
            form = dict(p["form"])
            if modo == "inv":
                dias = [dict(d, meals=list(reversed(d.get("meals") or []))) for d in dias]
            if modo == "pesc":
                form["dietType"] = "pescetariano"
            casos.append(((k, modo), dias, form))
    # la prueba forzada: la unión de los dos mapas (el de la rama y la base con «dorado»), para no perder ninguna comida
    union = {l: {_sa(a.lower()) for a in go._MAIN_PROTEIN_ALIASES.get(l, ())} for l in _MAR}
    union["pescado"] |= {_sa(a.lower()) for a in go._PESCADO_ANTES}
    vistos = set()
    for k, p in _CORPUS.items():
        for i, d in enumerate(p["days"]):
            for m in d.get("meals") or []:
                if not isinstance(m, dict) or m.get("_recipe_source") == "library":
                    continue
                blob = _sa((str(m.get("name")) + " " + " ".join(str(x) for x in m.get("ingredients") or [])).lower())
                for lbl in _MAR:
                    if (lbl, m.get("name")) in vistos or not any(_name_has_token(a, blob) for a in union[lbl]):
                        continue
                    vistos.add((lbl, m.get("name")))
                    for orden in ("cocido_primero", "corpus_primero"):
                        ms = [copy.deepcopy(_COCIDO[lbl]), copy.deepcopy(m)]
                        casos.append(((k, f"D{i + 1}", lbl, orden, str(m.get("name"))),
                                      [{"day": 1, "meals": ms if orden == "cocido_primero" else ms[::-1]}],
                                      {"dietType": "balanced", "country": p["form"].get("country") or "DO"}))
    return casos


def _correr(casos) -> tuple:
    h = _Motivos()
    lg = logging.getLogger("graph_orchestrator")
    nivel = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    cambios, motivos = {}, {}
    try:
        for clave, dias, form in casos:
            del h.motivos[:]
            dd = copy.deepcopy(dias)
            go._protein_repeat_autofix(dd, form, None)
            cambios[clave] = _cambios(dias, dd)
            motivos[clave] = list(h.motivos)
    finally:
        lg.removeHandler(h)
        lg.setLevel(nivel)
    return cambios, motivos


@pytest.fixture(scope="module")
def replay():
    casos = _casos()
    rama = _correr(casos)
    with pytest.MonkeyPatch.context() as m:                     # la base: lo del lote, apagado
        m.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
        m.setattr(pe, "PEZ_LISTO_GUARD", False)
        m.setattr(pe, "destino_apto_para_la_dieta", lambda *a, **k: True)
        m.setattr(go, "_PEZ_NUEVO", ())
        m.setitem(go._MAIN_PROTEIN_ALIASES, "pescado", list(go._PESCADO_ANTES))
        base = _correr(casos)
    return {"rama": rama[0], "motivos": rama[1], "base": base[0], "base_motivos": base[1]}


def test_el_corpus_es_el_de_la_revision(replay):
    assert sum(k.startswith("prod-") for k in _CORPUS) == 13 and sum(k.startswith("G24-") for k in _CORPUS) == 6
    assert "dorado" in go._PESCADO_ANTES and "trucha" not in go._PESCADO_ANTES
    assert len([c for c in replay["rama"] if len(c) == 5]) >= 80, "la prueba forzada perdió casos"


def test_las_reescrituras_de_la_rama_son_subconjunto_de_las_de_la_base(replay):
    """Cada comida que la rama reescribe, la base también la reescribía, y con el MISMO resultado."""
    fuera = []
    for clave, cambios in replay["rama"].items():
        base = replay["base"][clave]
        for sitio, resultado in cambios.items():
            if base.get(sitio) != resultado:
                fuera.append((clave, sitio, resultado[0], "la base no la tocaba" if sitio not in base else "otra"))
    assert not fuera, fuera[:10]
    assert sum(map(len, replay["rama"].values())) >= 21 and sum(map(len, replay["base"].values())) >= 21, "no es vacío"


def test_d8b10b05_d2_con_sardinas_va_al_gate_entero(replay):
    """El caso real de la revisión r6: la ronda 6 reescribía a pollo la tilapia del día (la base no veía la repetición
    y no tocaba nada); ahora el día entero va al gate (la repetición tilapia + sardinas sí existe)."""
    clave = ("prod-d8b10b05", "real")
    assert any(dia == 2 for dia, _ in replay["base"][clave]) is False, "la base no veía la repetición"
    assert not any(dia == 2 for dia, _ in replay["rama"][clave])
    assert ("2", "pescado", "especie_nueva") in replay["motivos"][clave]


# Las reescrituras de la prueba forzada que cuecen el pez en una cláusula que lo nombra (medidas por el revisor r6 con
# la regla de V7f): siguen reescribiéndose. (plan, día, orden, principio del nombre de la comida del corpus)
_CON_COCCION = {
    ("G24-DO", "D1", "Pescado al vapor con papas"), ("G24-DO", "D2", "Filete de pescado blanco con pur"),
    ("G24-ES", "D1", "Merluza a la plancha con piment"), ("G24-MX", "D2", "Salmón a la parrilla con gorditas"),
    ("G24-US", "D1", "Pescado Blanco Desmenuzado con Espinaca"), ("prod-125e45b1", "D2", "Mapuey crujiente en airfryer"),
    ("prod-1461aeca", "D1", "Filete de pescado en fusi"), ("prod-1461aeca", "D2", "Filete de pescado blanco a la plancha"),
    ("prod-1461aeca", "D3", "Queso blanco fresco al horno con auyama"), ("prod-358a2cdf", "D3", "Tilapia al horno con papa"),
    ("prod-6594aae1", "D2", "Filete de pescado blanco con majado"), ("prod-6594aae1", "D3", "Filete de pescado a la plancha"),
    ("prod-92328ff7", "D4", "Pasta integral criolla con Filete"), ("prod-92328ff7", "D8", "Wrap dominicano de pescado"),
    ("prod-92328ff7", "D8", "Tortitas de plátano maduro con huevo"), ("prod-92328ff7", "D9", "Guineítos verdes con huevo"),
    ("prod-92328ff7", "D10", "Pescado blanco al limón con ensalada"), ("prod-cd1b2fd0", "D3", "Pescado blanco guisado"),
    ("prod-d8b10b05", "D2", "Tilapia al horno con plátano"), ("prod-d8b10b05", "D3", "Pescado blanco a la parrilla"),
    ("prod-d8b10b05", "D3", "Queso blanco fresco a la parrilla"),
}
# Sin cláusula que cueza el pez: van al gate. cd1b2fd0 D2 («al guiso en los últimos minutos») es un acierto; 125e45b1 D1
# y 92328ff7 D2 son falsos positivos (el pez se cuece, pero no en una cláusula que lo nombre): un reintento cada uno.
_SIN_COCCION = {("prod-125e45b1", "D1"), ("prod-92328ff7", "D2"), ("prod-cd1b2fd0", "D2")}


def _reescritas_del_corpus(replay):
    out = {}
    for clave, cambios in replay["rama"].items():
        if len(clave) != 5:
            continue
        plan, dia, _lbl, orden, nombre = clave
        idx = 1 if orden == "cocido_primero" else 0                  # la comida del corpus
        if (1, idx) in cambios:
            out[(plan, dia, orden, nombre)] = cambios[(1, idx)][0]
    return out


def test_las_21_con_coccion_siguen_reescribiendose(replay):
    hechas = _reescritas_del_corpus(replay)
    for plan, dia, prefijo in _CON_COCCION:
        assert any(p == plan and d == dia and n.startswith(prefijo) for p, d, _o, n in hechas), (plan, dia, prefijo)
    assert len(hechas) == 21, sorted(hechas)


def test_las_sin_coccion_van_al_gate_con_su_motivo(replay):
    hechas = _reescritas_del_corpus(replay)
    for plan, dia in _SIN_COCCION:
        assert not any(p == plan and d == dia for p, d, _o, _n in hechas), (plan, dia)
        assert any(c[0] == plan and c[1] == dia and ("1", "pescado", "pez_sin_coccion") in replay["motivos"][c]
                   for c in replay["motivos"] if len(c) == 5), (plan, dia)
