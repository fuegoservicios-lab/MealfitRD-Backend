"""[P1-LEGAL-JSX-COPIA-PAIS · 2026-08-23] G29: la copia React de los legales afirmaba que
España/México/Colombia eran expansión FUTURA, cinco días después de que el apex publicara que
el servicio YA se vende allí.

POR QUÉ NO BASTABA EL 301. Las 19 rutas legales devuelven 301 al apex por nginx, y eso hacía
parecer que la copia JSX era inalcanzable. No lo era: `ResearchPage` navegaba con
`<Link to="/data-protection">`, y React Router resuelve eso en el CLIENTE — el 301 del
servidor nunca llega a dispararse. El texto obsoleto estaba delante del usuario a dos clics.

*Una redirección de servidor no protege una ruta a la que se llega sin pedirla al servidor.*

DOS COSAS, y la segunda no es cosmética: una política es una afirmación sobre dónde operas y
bajo qué ley. Decir «cuando ampliemos a Latinoamérica» mientras cobras en España no es una
copia desactualizada: es una declaración falsa sobre el marco legal aplicable, y nombraba dos
países (Brasil, Argentina) que nunca se han ofrecido.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"
_LEGAL = _FRONT / "src" / "pages" / "legal" / "LegalPages.jsx"
_RESEARCH = _FRONT / "src" / "pages" / "ResearchPage.jsx"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _sin_comentarios(src: str) -> str:
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith("//"))


# ── el camino que esquivaba el 301 ────────────────────────────────────────────

def test_los_legales_de_research_salen_al_apex_y_no_por_react_router():
    """`<Link to>` se resuelve en el cliente: el 301 de nginx no se dispara nunca."""
    codigo = _sin_comentarios(_leer(_RESEARCH))
    for ruta in ("/data-protection", "/privacy"):
        assert f'<Link to="{ruta}"' not in codigo, (
            f"{ruta} vuelve a navegarse con <Link>: React Router lo resuelve en el cliente y "
            "el usuario aterriza en la copia JSX en vez de en el apex"
        )
    assert codigo.count("apexUrl('/data-protection')") >= 2
    assert "apexUrl('/privacy')" in codigo


def test_apexurl_se_importa_de_verdad():
    """Sin el import, `apexUrl` sería un ReferenceError en tiempo de render."""
    assert re.search(r"import \{[^}]*apexUrl[^}]*\} from '\.\./config/site';", _leer(_RESEARCH))


def test_no_se_usa_replace_en_el_cuerpo_legal():
    """⚠ En dev/preview `apexUrl(path)` devuelve `path` tal cual: un
    `window.location.replace(apexUrl(path))` desde la propia ruta legal sería un BUCLE
    infinito de redirección. Un `<a href>` no tiene ese problema."""
    codigo = _sin_comentarios(_leer(_LEGAL))
    assert "location.replace(apexUrl" not in codigo, (
        "replace(apexUrl(...)) dentro de la ruta legal: en dev apexUrl devuelve la misma ruta "
        "y el navegador entra en bucle"
    )


# ── y la afirmación que era falsa ─────────────────────────────────────────────

def test_la_politica_ya_no_presenta_como_futuro_lo_que_se_vende_hoy():
    src = _leer(_LEGAL)
    assert "ampliemos el servicio a otros países" not in src, (
        "la política sigue diciendo que la expansión es futura mientras se cobra en España "
        "desde el 18-ago"
    )


def test_nombra_las_normas_de_los_paises_donde_se_vende():
    """Espejo del apex, que es la copia canónica. Si un país entra al selector y su norma no
    aparece aquí, esta copia vuelve a mentir por omisión."""
    src = _leer(_LEGAL)
    for norma in ("RGPD", "2016/679", "CCPA", "LFPDPPP", "Ley 1581"):
        assert norma in src, f"la política no menciona {norma}"


def test_no_nombra_paises_que_nunca_se_han_ofrecido():
    """Decía «la LGPD en Brasil o la Ley 25.326 en Argentina». Ninguno de los dos está en el
    selector: prometer cobertura legal donde no operas es el defecto simétrico."""
    src = _leer(_LEGAL)
    import constants
    paises = {p["name_es"] for p in constants.COUNTRY_PROFILES.values()}
    for fantasma, norma in (("Brasil", "LGPD"), ("Argentina", "25.326")):
        if fantasma not in paises:
            assert norma not in src, (
                f"la política cita {norma} ({fantasma}) y ese país no está en COUNTRY_PROFILES"
            )


def test_la_copia_se_declara_copia():
    """Mientras exista, tiene que decir que la vigente es la del apex — o vuelve a ser una
    segunda fuente con autoridad aparente."""
    src = _leer(_LEGAL)
    assert "bioboros.com" in src and "copia de cortesía" in src
