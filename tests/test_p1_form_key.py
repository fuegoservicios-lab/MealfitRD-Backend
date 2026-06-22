"""[P1-FORM-KEY · 2026-06-21] La llave de cifrado del form sensible se deriva en el
backend de MEALFIT_SESSION_SECRET (estable) en vez del access_token de Neon (que
rotaba → el form "se borraba"). Ancla: determinística, per-usuario, gated en el secreto."""
import auth


def test_derive_form_key_deterministic_and_per_user(monkeypatch):
    monkeypatch.setattr(auth, "_SESSION_SECRET", "x" * 40)
    k_a1 = auth.derive_form_key("user-A")
    k_a2 = auth.derive_form_key("user-A")
    k_b = auth.derive_form_key("user-B")
    assert k_a1, "debe producir una llave con secreto fuerte + uid"
    assert k_a1 == k_a2, "misma cuenta -> misma llave (estable across re-logins)"
    assert k_a1 != k_b, "usuarios distintos -> llaves distintas (aislamiento)"
    # Suficientemente larga para el setFormCryptoSecret del frontend (>=16) y AES-GCM.
    assert len(k_a1) >= 32


def test_derive_form_key_requires_uid(monkeypatch):
    monkeypatch.setattr(auth, "_SESSION_SECRET", "x" * 40)
    assert auth.derive_form_key("") is None
    assert auth.derive_form_key(None) is None


def test_derive_form_key_gated_on_strong_secret(monkeypatch):
    # Sin secreto fuerte (<32) -> None: el frontend degrada al access_token (cero regresión).
    monkeypatch.setattr(auth, "_SESSION_SECRET", "short-secret")
    assert auth.derive_form_key("user-A") is None
