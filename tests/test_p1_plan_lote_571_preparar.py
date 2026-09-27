# backend/tests/test_p1_plan_lote_571_preparar.py
"""[P1-PLAN-LOTE-571 · 2026-09-27] Preparar el banco: muestrea, baja las PNG y congela su sha256."""
import json

import banco_analizador as ba
from scripts import banco_analizador_preparar as prep


def _csv():
    filas = []
    for i in range(12):
        kcal = 150 + 60 * i
        filas.append(f"dish_{i:02d},{kcal},{300},{kcal * 0.25 / 9},{kcal * 0.5 / 4},{kcal * 0.25 / 4},"
                     f"ingr_1,white rice,150,1,1,1,1")
    return "\n".join(filas).encode()


def _falso_descargar(url, cliente):
    if url.endswith("dish_metadata_cafe1.csv"):
        return _csv()
    if url.endswith("dish_metadata_cafe2.csv"):
        return b""
    if url.endswith("rgb_test_ids.txt"):
        return "\n".join(f"dish_{i:02d}" for i in range(0, 12, 2)).encode()
    return ("PNG-" + url.split("/")[-2]).encode()


def test_congela_la_muestra_y_la_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "descargar", _falso_descargar)
    monkeypatch.setattr(prep, "existe", lambda url, cliente: True)
    monkeypatch.setattr(ba, "MANIFIESTO", tmp_path / "manifest.json")
    assert prep.main(["--cache", str(tmp_path / "cache"), "--n", "6"]) == 0
    man = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert man["n"] == 6 and {p["dish_id"] for p in man["platos"]} == {f"dish_{i:02d}" for i in range(0, 12, 2)}
    assert all(p["sha256_png"] == ba.sha256_de(f"PNG-{p['dish_id']}".encode()) for p in man["platos"])


def test_solo_cache_detecta_una_foto_cambiada(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "descargar", _falso_descargar)
    monkeypatch.setattr(prep, "existe", lambda url, cliente: True)
    monkeypatch.setattr(ba, "MANIFIESTO", tmp_path / "manifest.json")
    cache = tmp_path / "cache"
    assert prep.main(["--cache", str(cache), "--n", "6"]) == 0
    (cache / "dish_00.png").write_bytes(b"otra")
    assert prep.main(["--cache", str(cache), "--solo-cache"]) == 1


def test_descarta_los_platos_sin_foto_cenital(tmp_path, monkeypatch):
    # Medido al congelar el banco real: dish_1550705477 está en rgb_test_ids y su rgb.png da 404.
    monkeypatch.setattr(prep, "descargar", _falso_descargar)
    monkeypatch.setattr(prep, "existe", lambda url, cliente: "dish_04" not in url)
    monkeypatch.setattr(ba, "MANIFIESTO", tmp_path / "manifest.json")
    assert prep.main(["--cache", str(tmp_path / "cache"), "--n", "6"]) == 0
    ids = {p["dish_id"] for p in json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["platos"]}
    assert "dish_04" not in ids and len(ids) == 5
