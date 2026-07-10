"""[P2-GUARD-PERF-REGEXCACHE · 2026-07-10] El coherence guard tardaba 9.3-9.7s (umbral 5s) en prod.
cProfile en VPS (plan real): 92% del tiempo en normalize_name — ~140 regexes DINÁMICOS compilados
POR LLAMADA (loop de stops) → ~20k compilaciones re.* por corrida (thrash del LRU de 512 de `re`)
+ strip_accents 151k llamadas/corrida sin cache.

Fix: (1) alternación precompilada `_NORMALIZE_STOPS_RE` a nivel módulo (orden length-DESC para que
frases multi-palabra ganen a sus sub-tokens); (2) `strip_accents` con LRU 8192 (pura str→str).
Equivalencia semántica validada por las suites de coherencia/shopping (1052 verdes).

tooltip-anchor: P2-GUARD-PERF-REGEXCACHE
"""
import re
from pathlib import Path

import shopping_calculator as sc
import constants


def test_stops_regex_precompiled_at_module_level():
    assert isinstance(sc._NORMALIZE_STOPS_RE, re.Pattern)
    src = Path(sc.__file__).read_text(encoding="utf-8")
    # el loop viejo (compile-por-llamada) no debe volver
    assert "for s in stops:" not in src


def test_multiword_stops_win_over_subtokens():
    # 'bajo en grasa' (frase) debe removerse completa, no dejar residuo
    assert "queso" in sc.normalize_name("queso bajo en grasa").lower()
    assert "grasa" not in sc.normalize_name("queso bajo en grasa").lower()


def test_normalize_behavior_samples():
    # muestras representativas del comportamiento pre-fix (equivalencia)
    assert sc.normalize_name("cebolla picada finamente") == sc.normalize_name("cebolla")
    assert sc.normalize_name("Taza de avena en hojuelas") == sc.normalize_name("avena")
    assert "pollo" in sc.normalize_name("pechuga de pollo asada").lower()


def test_strip_accents_cached_and_correct():
    assert constants.strip_accents("Torónja") == "Toronja"
    assert constants.strip_accents("piña") == "pina"
    # cache activo (función interna lru_cache)
    assert hasattr(constants._strip_accents_cached, "cache_info")
    constants.strip_accents("plátano")
    assert constants._strip_accents_cached.cache_info().currsize >= 1


def test_strip_accents_non_str_failsafe():
    assert constants.strip_accents(123) == "123"
