"""[P2-SHOPPING-COPY-QUIET · 2026-09-04] Dos copias que confundían al dueño en la lista de compras:
(1) el PDF ponía «Esta compra RD$375» y debajo «Costo real del ciclo de 30 Días RD$1,607» con 1 solo ítem
(«¿y cuál es el objetivo de eso?»); (2) el toast «Tu lista tuvo N revisiones automáticas» contaba
detecciones. Ahora el ciclo es una línea discreta que solo aparece con >2 ítems y >15 % de diferencia
(y sin ella tampoco hay línea de presupuesto), y el toast dice QUÉ pasa (cuántas cantidades) y qué hacer.
"""
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()


def _frontend(*parts):
    for base in (_HERE.parents[2], _HERE.parents[1].parent):
        p = base.joinpath("frontend", "src", *parts)
        if p.exists():
            return p
    pytest.skip("frontend hermano no disponible")


def test_a_el_ciclo_solo_con_sustancia_y_diferencia_real():
    src = _frontend("pages", "Dashboard.jsx").read_text(encoding="utf-8")
    assert "const _showCycleCost = duration !== 'weekly' && totalItems > 2 && _fullCycleCostFinal > _shopTotalCostFinal * 1.15;" in src
    assert src.index("const totalItems = Object.values(consData).length;") < src.index("const _showCycleCost = duration !== 'weekly' && totalItems > 2")
    assert "t('Estimado del ciclo de {duracion}', { duracion: durationText })" in src
    assert "Costo real del ciclo de {duracion}" not in src
    # sin línea de ciclo, sin línea de presupuesto (compara el ciclo)
    assert "${!_showCycleCost ? '' : (() => {" in src


def test_b_el_toast_dice_que_pasa():
    src = _frontend("utils", "renderCoherenceWarnings.js").read_text(encoding="utf-8")
    assert "t('Revisa las cantidades de tu lista de compras')" in src
    assert "revisiones automáticas recientes" not in src
    assert "divergences: n" in src


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_c_catalogos_sin_huerfanas_y_con_las_nuevas(locale):
    cat = json.loads(_frontend("i18n", "locales", f"{locale}.json").read_text(encoding="utf-8"))
    for k in ("Estimado del ciclo de {duracion}", "Incluye ≈{monto} de recompras de frescos", "Despensa 1× + frescos cada 7 días",
              "Revisa las cantidades de tu lista de compras"):
        assert cat.get(k), (locale, k)
    for k in ("Costo real del ciclo de {duracion}", "Tu lista de compras tuvo {n} revisiones automáticas recientes"):
        assert k not in cat, (locale, k, "clave huérfana")
    assert "{n}" in cat["El control automático encontró {n} cantidad(es) que pueden no cuadrar con las recetas. Ajústalas antes de comprar."]
