"""[P1-GUEST-CATALOG · 2026-08-11] Un invitado no podía completar el paso 15 del wizard.

EL DEFECTO. `/assessment` es público (sin login) y su paso «Tus básicos de siempre» busca
contra `GET /api/catalog`. Ese endpoint exigía sesión —el docstring decía «Auth requerida
(paridad con el acceso RLS previo)», o sea que la restricción venía del TRANSPORTE
anterior, no de una necesidad de privacidad: `master_ingredients` es una tabla global de
referencia, no datos de nadie—. Sin sesión el catálogo llegaba vacío.

Y NO FALLABA DE FORMA VISIBLE. El componente se tragaba el 403 en un `catch` mudo, así
que el usuario escribía «arroz», no aparecía nada, y lo que entendía era que ese alimento
no está en el catálogo. *Un fallo que se traga su propio error es indistinguible de un
resultado vacío legítimo* — y aquí el resultado vacío legítimo tiene un significado
completamente distinto.

LO QUE NO SE ABRE: el invitado recibe una proyección reducida. Precios por libra y por
unidad, densidades, envase de mercado y tamaños disponibles se quedan detrás de la
sesión: un buscador necesita nombres, y esas columnas son el trabajo curado del producto.

Este test afirma las dos mitades —que responde sin sesión Y que responde MENOS— porque
abrir el endpoint sin acotar qué devuelve sería cambiar un defecto por otro.
"""
from __future__ import annotations

import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "routers" / "user_data.py").read_text(encoding="utf-8")


def _bloque_del_catalogo() -> str:
    """Del decorador de `/catalog` hasta el siguiente decorador de ruta."""
    i = _SRC.index('@router.get("/catalog")')
    j = _SRC.find("@router.", i + 10)
    return _SRC[i: j if j > 0 else len(_SRC)]


def test_el_catalogo_responde_sin_sesion():
    bloque = _bloque_del_catalogo()
    # `_require_user` es lo que convertía la falta de sesión en un 403.
    assert "_require_user" not in bloque, (
        "el catálogo vuelve a exigir sesión: el paso «Tus básicos de siempre» del wizard "
        "público deja de poder completarse, y sin error visible — el buscador simplemente "
        "no encuentra nada"
    )
    assert re.search(r"verified_user_id:\s*Optional\[str\]\s*=\s*Depends\(get_verified_user_id\)", bloque), (
        "la dependencia de auth dejó de ser opcional"
    )


def test_al_invitado_se_le_sirve_MENOS():
    """Abrir el endpoint sin acotar la proyección sería cambiar un defecto por otro."""
    bloque = _bloque_del_catalogo()
    assert "if not verified_user_id:" in bloque, (
        "desapareció la poda para invitados: se estaría sirviendo el catálogo completo "
        "—precios, densidades, envases— a cualquiera sin sesión"
    )

    m = re.search(r"_CATALOG_CAMPOS_INVITADO\s*=\s*\(([^)]*)\)", _SRC)
    assert m, "desapareció la lista de campos del invitado"
    campos = {c.strip().strip("\"'") for c in m.group(1).split(",") if c.strip()}

    # Lo que el buscador necesita.
    assert {"name", "staple_gate_label"} <= campos, (
        "faltan campos que el buscador de básicos usa: sin `name` no hay búsqueda y sin "
        "`staple_gate_label` se pierde el aviso de que dos alimentos gastan un solo cupo"
    )
    # Lo que no debe salir sin sesión.
    for prohibido in ("price_per_lb", "price_per_unit", "density_g_per_cup",
                      "density_g_per_unit", "market_container", "available_sizes_g"):
        assert prohibido not in campos, (
            f"`{prohibido}` se está sirviendo a invitados: es dato curado del producto y "
            "ningún buscador lo necesita"
        )


def test_la_poda_va_despues_de_anotar_el_rotulo_del_gate():
    """`staple_gate_label` lo calcula el backend DESPUÉS del SELECT (P1-STAPLE-SEARCH-RANK).

    Podar antes lo dejaría fuera —el campo aún no existiría— y el invitado perdería el
    aviso de que dos alimentos colapsan al mismo cupo. El orden es la corrección.
    """
    bloque = _bloque_del_catalogo()
    i_rotulo = bloque.find("staple_gate_label")
    i_poda = bloque.find("if not verified_user_id:")
    assert i_rotulo > 0 and i_poda > 0
    assert i_poda > i_rotulo, (
        "la poda del invitado se movió ANTES de anotar el rótulo del gate: el campo aún "
        "no existe ahí, así que se serviría siempre vacío"
    )


def test_el_camino_sin_sesion_esta_limitado():
    """Un endpoint sin auth que consulta la DB necesita contrapeso.

    El cliente cachea el catálogo 24h, así que un uso legítimo lo pide una vez por
    wizard: el límite no molesta a nadie real.
    """
    bloque = _bloque_del_catalogo()
    assert "_CATALOG_LIMITER" in bloque, (
        "el catálogo quedó sin límite de tasa y ya responde sin sesión: es un grifo de "
        "scraping barato"
    )
    assert re.search(r"_CATALOG_LIMITER\s*=\s*RateLimiter\(", _SRC), "no se declara el limitador"
