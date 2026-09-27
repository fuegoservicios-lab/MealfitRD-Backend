# backend/tests/test_p1_plan_lote_572_corrida.py
"""[P1-PLAN-LOTE-572 · 2026-09-27] La corrida del banco: analizador real sustituido por un doble, sin DB."""
import asyncio
import io
import json
from pathlib import Path

from PIL import Image

import banco_analizador as ba
from scripts import banco_analizador_correr as corrida

RAIZ = Path(__file__).resolve().parents[2]


def _png(color):
    b = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(b, "PNG")
    return b.getvalue()


class _VisionFalsa:
    def __init__(self, fallar_primero=(), fallar_siempre=()):
        self.llamadas, self.vistos = 0, {}
        self.fallar_primero, self.fallar_siempre = set(fallar_primero), set(fallar_siempre)
        self._uso = None

    def _reset_last_vision_usage(self):
        self._uso = None

    def get_last_vision_usage(self):
        return self._uso

    async def process_image_with_vision(self, datos):
        self.llamadas += 1
        assert datos[:2] == b"\xff\xd8"                     # llega JPEG, como desde el cliente
        clave = ba.sha256_de(datos)                          # no la longitud: dos JPEG lisos pueden medir igual
        self.vistos[clave] = self.vistos.get(clave, 0) + 1
        self._uso = {"input_tokens": 1000, "output_tokens": 200}
        if clave in self.fallar_siempre or (clave in self.fallar_primero and self.vistos[clave] == 1):
            return {"analysis_failed": True, "is_food": False}
        return {"is_food": True, "photo_kind": "plato", "calories": 330, "protein": 22, "carbs": 30,
                "healthy_fats": 11, "items": [{"name": "Arroz"}]}


def _man_y_fotos(n=3):
    platos, fotos = [], {}
    for i in range(n):
        p = {"dish_id": f"d{i}", "kcal": 300.0, "masa_g": 300.0, "grasa_g": 10.0, "carbs_g": 30.0,
             "proteina_g": 20.0, "ingredientes": [{"nombre": "white rice", "gramos": 200.0}]}
        platos.append(p)
        fotos[p["dish_id"]] = _png((40 * i, 80, 120))
    man = ba.manifiesto(platos, {d: ba.sha256_de(b) for d, b in fotos.items()}, 1, "test")
    return man, fotos


def test_corre_todo_el_banco_con_el_doble():
    man, fotos = _man_y_fotos()
    va = _VisionFalsa()
    filas = asyncio.run(corrida.correr(man, fotos.get, concurrencia=2, va=va))
    assert len(filas) == 3 and va.llamadas == 3
    assert all(f["fallo"] is None and f["errores"]["kcal"] == 0.1 for f in filas)
    assert all(f["tokens"] == {"input": 1000, "output": 200} for f in filas)


def test_reintenta_una_vez_y_anota_el_fallo(monkeypatch):
    monkeypatch.setattr(corrida, "ESPERA_REINTENTO_S", 0)
    man, fotos = _man_y_fotos()
    clave = {d: ba.sha256_de(corrida.a_jpeg(b)) for d, b in fotos.items()}
    va = _VisionFalsa(fallar_primero=[clave["d0"]], fallar_siempre=[clave["d2"]])
    filas = asyncio.run(corrida.correr(man, fotos.get, concurrencia=1, va=va))
    assert sorted(f["fallo"] or "ok" for f in filas) == ["error", "ok", "ok"]
    assert va.llamadas == 5                                   # 3 + un reintento por cada plato que falló


def test_aborta_si_la_cache_no_cuadra(tmp_path, monkeypatch):
    man, fotos = _man_y_fotos()
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(ba, "MANIFIESTO", tmp_path / "manifest.json")
    cache = tmp_path / "cache"
    cache.mkdir()
    for d, b in fotos.items():
        (cache / f"{d}.png").write_bytes(b if d != "d1" else b"otra")
    va = _VisionFalsa()
    monkeypatch.setattr(corrida, "_vision", lambda: va)
    assert corrida.main(["--cache", str(cache), "--salida", str(tmp_path / "r.json")]) == 2
    assert va.llamadas == 0


def test_no_escribe_en_tablas_de_usuario():
    src = (RAIZ / "backend" / "scripts" / "banco_analizador_correr.py").read_text(encoding="utf-8")
    for prohibido in ("log_llm_usage_event", "consumed_meals", "llm_usage_events"):
        assert prohibido not in src, prohibido


def test_migracion_idempotente_en_los_dos_directorios():
    nombre = "p1_plan_lote_572_analyzer_benchmark_runs_2026_09_27.sql"
    a = (RAIZ / "migrations" / nombre).read_text(encoding="utf-8")
    b = (RAIZ / "backend" / "migrations" / nombre).read_text(encoding="utf-8")
    assert a == b
    assert "CREATE TABLE IF NOT EXISTS public.analyzer_benchmark_runs" in a
    assert "RAISE EXCEPTION" in a and "auth." not in a
