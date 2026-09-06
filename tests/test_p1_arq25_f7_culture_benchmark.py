"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Fase 7 (subfase C): benchmark cultural (roadmap §13.4) sobre los snapshots
del Dish Registry — resolvabilidad, cobertura, contaminación cultural, adecuación técnica/franja, disponibilidad
en el mercado, diversidad, mezcla coherente y cero bypass clínico — y su gate. El benchmark SEÑALA para la
revisión humana; no cura. La firma humana queda `pendiente` en el informe hasta que una persona la dé.
"""
import json
from pathlib import Path

import pytest

import cultural_benchmark as cb
import cultural_profiles as cp

_BACKEND = Path(__file__).resolve().parents[1]


def test_a_el_lexico_exclusivo_es_de_platos_no_de_ingredientes():
    lex = cb._profile_lexicons()
    todo = set().union(*lex.values())
    for generic in ("huevo", "pollo", "arroz", "avena", "aguacate", "tomate", "ensalada", "sopa", "guisados", "revuelto", "tortilla"):
        assert generic not in todo, f"«{generic}» es genérico: no puede marcar contaminación"
    # platos compartidos por dos cocinas no son exclusivos de ninguna
    assert "sancocho" not in lex["colombia_casera"] and "sancocho" not in lex["dominican_criolla"]
    assert "asopao" not in lex["puertorico_criolla"]
    # y los propios sí
    assert "mangu" in lex["dominican_criolla"] and "pozole" in lex["mexico_casera"] and "ajiaco" in lex["colombia_casera"]


def test_b_contaminacion_detecta_un_plato_ajeno_y_respeta_el_propio():
    lex = cb._profile_lexicons()
    hits = cb._contamination("us", [{"name": "Mangú con huevo"}, {"name": "Ensalada de pollo"}, {"name": "Pozole de pollo"}], lex)
    assert {(h["template"], h["foreign_profile"]) for h in hits} == {("Mangú con huevo", "dominican_criolla"), ("Pozole de pollo", "mexico_casera")}
    assert cb._contamination("do", [{"name": "Mangú con huevo"}, {"name": "Locrio de pollo"}], lex) == []


def test_c_adecuacion_y_diversidad_sinteticas():
    ts = [
        {"name": "Arroz con pollo", "slots": ["cena"], "base": "arroz", "technique": "guisado", "protein": "pollo"},
        {"name": "Chicharrón", "slots": ["almuerzo"], "base": "none", "technique": "frito", "protein": "cerdo"},
        {"name": "Sancocho", "slots": ["merienda"], "base": "viveres", "technique": "sopa espesa", "protein": "pollo"},
        {"name": "Ensalada", "slots": ["cena"], "base": "none", "technique": "frío", "protein": "none"},
    ]
    a = cb._appropriateness(ts)
    assert a["starch_base_off_slot"] == ["Arroz con pollo"] and a["heavy_technique_in_snack"] == ["Sancocho"]
    assert a["fried_share"] == 0.25 and a["ok"] is False
    d = cb._diversity(ts)
    assert d["proteins"] == 3 and d["top_protein_share"] == 0.5 and d["ok"] is False


def test_d_gate_verdict_nombra_cada_fallo():
    rep = {"profiles": {"x": {"resolvability": {"ok": True}, "coverage": {"ok": False}, "contamination": {"ok": True},
                              "appropriateness": {"ok": True}, "diversity": {"ok": True}, "clinical": {"ok": False},
                              "availability": {"ok": True}},
                        "y": {"missing_snapshot": True}},
           "mixing": {"ok": False}}
    ok, failures = cb.gate_verdict(rep)
    assert not ok and failures == ["x: coverage", "x: clinical", "y: sin snapshot", "mixing"]


@pytest.mark.skipif(not Path(cb.__file__).with_name("data").joinpath("registry", "dish_registry_es_v1.json").exists(), reason="snapshots no compilados")
def test_e_los_snapshots_reales_pasan_el_gate_y_el_informe_es_reproducible():
    rep = cb.run_benchmark()
    assert rep["gate_ok"], rep["failures"]
    assert set(rep["profiles"]) == set(cp.PROFILES)
    for pid, e in rep["profiles"].items():
        assert e["coverage"]["ok"] and e["clinical"]["ok"] and e["contamination"]["ok"], pid
        # la firma vive en data/registry/cultural_curation_review_v1.json y caduca con el hash del snapshot:
        # tocar una biblioteca sin volver a revisarla deja el perfil «pendiente» y este test en rojo
        so = e["review"]["signoff"]
        assert so and so["snapshot_hash"] == e["snapshot_hash"] and so["by"] and so["decisions"], f"{pid}: revisión curatorial pendiente o caducada"
        # [P1-REVIEW-KIND-HONEST · 2026-09-05] Había UN booleano, `human_signoff`, y valía True con una revisión
        # firmada por Claude. Ahora son tres campos y ninguno se deduce de otro: esta revisión es AUTOMÁTICA, la
        # cultural humana no existe y la clínica tampoco. Si algún día se firma a mano, este test cae y hay que
        # venir a decidirlo — que es exactamente lo que debe pasar.
        assert e["review"]["automated_review"] is True
        assert e["review"]["human_cultural_review"] is False
        assert e["review"]["clinical_review"] is False
        assert "human_signoff" not in e["review"], "el campo que mentía no vuelve por la puerta de atrás"
        assert e["clinical"]["allergen_leaks"] == []
    assert rep["mixing"]["ok"] and len(rep["mixing"]["pairs"]) == 30
    md = cb.render_markdown(rep)
    assert "| Perfil | Biblioteca |" in md and "Gate: **PASA**" in md
    # el informe committed refleja los snapshots committed (sin catálogo vivo la disponibilidad no entra)
    committed = _BACKEND / "data" / "registry" / "cultural_benchmark_v1.json"
    if committed.exists():
        saved = json.loads(committed.read_text(encoding="utf-8"))
        for pid in cp.PROFILES:
            assert saved["profiles"][pid]["snapshot_hash"] == rep["profiles"][pid]["snapshot_hash"], f"{pid}: informe desfasado — corre `python cultural_benchmark.py --write`"


def test_f_la_doc_del_informe_existe_y_nombra_la_firma_pendiente():
    p = _BACKEND / "docs" / "cultural_benchmark_report.md"
    assert p.exists(), "python cultural_benchmark.py --write"
    txt = p.read_text(encoding="utf-8")
    assert "firma: sí (" in txt and "firma: pendiente" not in txt, "informe desfasado: python cultural_benchmark.py --write"
