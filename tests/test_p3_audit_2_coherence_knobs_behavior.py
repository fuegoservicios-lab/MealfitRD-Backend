"""[P3-AUDIT-2 · 2026-05-10] Behavior tests para 2 knobs deferred del
coherence guard:
  - `MEALFIT_COHERENCE_LIQUID_KEYWORDS` (existía, sin behavior test).
  - `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD` (implementado en P3-AUDIT-2,
    antes hardcoded 0.5).

Bug original (audit 2026-05-10):
    Knobs documentados en CLAUDE.md / docstrings pero NO testeados
    end-to-end. Si alguien borra el `_env_str`/`_env_float` y deja
    la doc, el anchor pasa pero comportamiento se pierde silenciosamente.

Fix:
    Tests behavior que:
      1. Setean el env var.
      2. Importan el módulo (sin caching desde sesión previa).
      3. Verifican que el CÓDIGO respeta el override.

Cobertura:
    - LIQUID_KEYWORDS: default set + knob extiende set + `_is_liquid_food`
      consume el set correctamente.
    - PANTRY_OVERDEDUCT_RATIO_THRESHOLD: default 0.5 + knob bumpea a 0.75 +
      validator rechaza valores fuera de [0,1] + knob registrado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shopping_calculator import (
    _get_coherence_liquid_keywords,
    _is_liquid_food,
    _classify_divergence_hypothesis,
)


# ---------------------------------------------------------------------------
# Section A: MEALFIT_COHERENCE_LIQUID_KEYWORDS
# ---------------------------------------------------------------------------
class TestLiquidKeywordsBehavior:
    def test_default_keywords_returned(self, monkeypatch):
        """Default knob → set incluye keywords clásicas."""
        monkeypatch.delenv("MEALFIT_COHERENCE_LIQUID_KEYWORDS", raising=False)
        kws = _get_coherence_liquid_keywords()
        assert "aceite" in kws
        assert "vinagre" in kws

    def test_knob_override_replaces_keywords(self, monkeypatch):
        """Knob CSV REEMPLAZA el set default (no merge)."""
        monkeypatch.setenv(
            "MEALFIT_COHERENCE_LIQUID_KEYWORDS",
            "leche de coco,agrio de naranja",
        )
        kws = _get_coherence_liquid_keywords()
        assert "leche de coco" in kws
        assert "agrio de naranja" in kws
        # NOTE: el default include "aceite/vinagre" pero el knob REEMPLAZA
        # (no merge). Verificar que default ya no aparece.
        assert "aceite" not in kws, (
            "P3-AUDIT-2: knob debe reemplazar el set default (CSV completo). "
            "Si quieres merge, cambiar el contrato + actualizar este test."
        )

    def test_knob_extends_is_liquid_food_behavior(self, monkeypatch):
        """Behavior end-to-end: extender knob → `_is_liquid_food` retorna
        True para el nuevo keyword. Sin esto, el knob es solo metadata."""
        monkeypatch.setenv(
            "MEALFIT_COHERENCE_LIQUID_KEYWORDS",
            "aceite,vinagre,leche de coco",
        )
        kws = _get_coherence_liquid_keywords()
        assert _is_liquid_food("Leche de coco", kws) is True, (
            "P3-AUDIT-2 regresión: knob añadió 'leche de coco' pero "
            "`_is_liquid_food` no lo detecta. El consumer no respeta "
            "el knob."
        )
        # Sanity: comida no-líquida sigue dando False.
        assert _is_liquid_food("Pollo", kws) is False

    def test_empty_csv_falls_back_to_default(self, monkeypatch):
        """Knob vacío → fallback al default (no set vacío)."""
        monkeypatch.setenv("MEALFIT_COHERENCE_LIQUID_KEYWORDS", "")
        kws = _get_coherence_liquid_keywords()
        assert "aceite" in kws or "vinagre" in kws, (
            f"P3-AUDIT-2: knob vacío debe caer al default, got {kws}"
        )

    def test_malformed_items_skipped(self, monkeypatch):
        """Items vacíos (dobles comas, espacios) se ignoran."""
        monkeypatch.setenv(
            "MEALFIT_COHERENCE_LIQUID_KEYWORDS",
            ",aceite,,  ,vinagre,",
        )
        kws = _get_coherence_liquid_keywords()
        assert "aceite" in kws
        assert "vinagre" in kws
        # No items vacíos en el set final.
        assert "" not in kws
        assert " " not in kws


# ---------------------------------------------------------------------------
# Section B: MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD
# ---------------------------------------------------------------------------
class TestPantryOverdeductThresholdBehavior:
    def _classify(self, exp_qty, act_qty):
        """Helper: ejercita _classify con units consistentes."""
        return _classify_divergence_hypothesis(
            exp_qty=exp_qty,
            act_qty=act_qty,
            exp_units={"g": exp_qty},
            act_units={"g": act_qty},
        )

    def test_default_threshold_0_5(self, monkeypatch):
        """Default 0.5: ratio=0.2 → pantry_overdeduct; ratio=0.6 → unknown.

        Nota: ratio=0.4 caería en la banda yield_uncovered legume
        (0.30-0.40) que tiene precedencia sobre pantry_overdeduct.
        Por eso usamos ratio=0.2 (claramente fuera de yield bands).
        """
        monkeypatch.delenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", raising=False)
        # ratio=0.2 (< 0.5, fuera de yield bands) → pantry_overdeduct.
        assert self._classify(100.0, 20.0) == "pantry_overdeduct"
        # ratio=0.6 (> 0.5, fuera de yield bands) → unknown.
        assert self._classify(100.0, 60.0) == "unknown"

    def test_knob_0_75_expands_overdeduct_bucket(self, monkeypatch):
        """Knob=0.75 captura ratios entre 0.5 y 0.75 que antes caían a
        `unknown`. Cierra el caso documentado: receta 3kg + nevera 2kg
        (ratio=0.67) ahora detecta sobrededucción."""
        monkeypatch.setenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", "0.75")
        # ratio=0.67 — antes unknown, ahora pantry_overdeduct.
        assert self._classify(3000.0, 2000.0) == "pantry_overdeduct"
        # ratio=0.8 — todavía sobre threshold → unknown.
        assert self._classify(100.0, 80.0) == "unknown"

    def test_knob_invalid_falls_back_to_default(self, monkeypatch):
        """Validator `0.0 < v < 1.0` rechaza valores fuera de banda;
        fallback al default 0.5. Uso ratio=0.2 (fuera de yield bands)
        para que el assert reflecte SOLO el behavior del knob."""
        # 0 — inválido (no estrictamente > 0).
        monkeypatch.setenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", "0")
        # Fallback a 0.5 → ratio=0.2 (< 0.5) → pantry_overdeduct.
        assert self._classify(100.0, 20.0) == "pantry_overdeduct"
        # 1.0 — inválido (no estrictamente < 1.0).
        monkeypatch.setenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", "1.0")
        assert self._classify(100.0, 20.0) == "pantry_overdeduct"
        # Garbage — inválido → default.
        monkeypatch.setenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", "garbage")
        assert self._classify(100.0, 20.0) == "pantry_overdeduct"

    def test_knob_0_3_narrows_overdeduct_bucket(self, monkeypatch):
        """Knob=0.25 reduce el bucket: ratio=0.5 cae a `unknown`.
        Permite tuning conservador si SRE quiere bucket más estricto.

        Nota: usamos threshold=0.25 (no 0.3) para que el test no
        compita con la banda yield legume 0.30-0.40. Y testeamos
        con ratio=0.5 (fuera de banda yield) para ver el efecto puro
        del knob.
        """
        monkeypatch.setenv("MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD", "0.25")
        # ratio=0.5 — > 0.25 → unknown (antes pantry_overdeduct con default 0.5).
        assert self._classify(100.0, 50.0) == "unknown"
        # ratio=0.2 — < 0.25 → pantry_overdeduct.
        assert self._classify(100.0, 20.0) == "pantry_overdeduct"

    def test_knob_registered_in_registry(self):
        """Knob debe registrarse en `_KNOBS_REGISTRY` vía `_knob_env_float`
        para aparecer en `/admin/knobs`."""
        _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
        src = _SC.read_text(encoding="utf-8")
        pattern = re.compile(
            r'_knob_env_float\(\s*["\']MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD["\']\s*,',
        )
        assert pattern.search(src), (
            "P3-AUDIT-2 regresión: `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD` "
            "no se lee via `_knob_env_float`. Sin esto, el knob no aparece "
            "en `/admin/knobs` y rollback en caliente no funciona."
        )

    def test_validator_excludes_boundary_values(self):
        """El validator debe rechazar 0.0 y 1.0 (no strictamente entre).
        Verificable por inspección del source."""
        _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
        src = _SC.read_text(encoding="utf-8")
        # Pattern: validator lambda con 0.0 < v < 1.0.
        pattern = re.compile(
            r'validator\s*=\s*lambda\s+v\s*:\s*0\.0\s*<\s*v\s*<\s*1\.0',
        )
        assert pattern.search(src), (
            "P3-AUDIT-2 regresión: validator del knob no usa rango "
            "estricto `0.0 < v < 1.0`. Permitir 0.0 (siempre False) o "
            "1.0 (siempre True) rompe la semántica del threshold."
        )
