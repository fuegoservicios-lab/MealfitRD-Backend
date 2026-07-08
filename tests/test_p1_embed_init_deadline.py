"""[P1-EMBED-INIT-DEADLINE · 2026-07-08] Bound wall-clock GLOBAL de la init del semantic cache.

Root cause (diagnosticado con faulthandler en la sesión de aceitunas): `get_semantic_cache` →
`_batched_embed_documents` embebe los ~203 nombres del catálogo contra Cohere en batches seriales
(batch=10, delay=0.5s) + retries de 429. Con Cohere lento/rate-limiting la init se arrastra minutos
SIN bound wall-clock global — el thread que hace la init en frío (startup warmer, o un usuario
desafortunado que llega antes de que el warmer termine) queda bloqueado.

Mitigación existente (parcial): lock non-blocking 0.05s (el usuario síncrono cae a fast-path si otro
thread inicializa), timeout per-request del embeddings_provider, cooldown 600s tras fallo, Redis-cache
one-time. Pero NINGUNA acota el wall-clock del thread que SÍ hace la init. Este deadline lo cierra: si
la init global excede `MEALFIT_EMBED_INIT_DEADLINE_S` (default 30s), aborta → Regex Fast-Path + cooldown.
"""
import time

import pytest

import shopping_calculator as sc


class _SlowClient:
    """Cliente de embeddings falso: cada `embed_documents` duerme `sleep_per_call`s."""
    def __init__(self, sleep_per_call):
        self.sleep_per_call = sleep_per_call
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.sleep_per_call:
            time.sleep(self.sleep_per_call)
        return [[0.0] * 4 for _ in texts]


def test_knob_default_and_clamp():
    assert sc.EMBED_INIT_DEADLINE_S >= 5.0, "clamp mínimo 5s para no auto-sabotear la init normal"


def test_deadline_wired_into_get_semantic_cache():
    import inspect
    src = inspect.getsource(sc.get_semantic_cache)
    assert "deadline=" in src, "get_semantic_cache debe pasar un deadline a _batched_embed_documents"
    assert "EMBED_INIT_DEADLINE_S" in src


def test_deadline_aborts_multibatch_init():
    """50 textos, batch 5 → 10 batches. Cada batch duerme 0.2s; deadline +0.3s → aborta ~batch 2."""
    client = _SlowClient(sleep_per_call=0.2)
    texts = [f"item {i}" for i in range(50)]
    deadline = time.monotonic() + 0.3
    with pytest.raises(Exception) as ei:
        sc._batched_embed_documents(client, texts, 5, 0.0, "test", deadline=deadline)
    assert 1 <= client.calls < 10, f"debe abortar antes de los 10 batches: calls={client.calls}"
    assert "deadline" in str(ei.value).lower() or isinstance(ei.value, TimeoutError)


def test_no_deadline_processes_all_backward_compat():
    """Sin deadline (None) → comportamiento pre-fix: procesa todos los batches."""
    client = _SlowClient(sleep_per_call=0.0)
    texts = [f"item {i}" for i in range(50)]
    out = sc._batched_embed_documents(client, texts, 5, 0.0, "test", deadline=None)
    assert len(out) == 50 and client.calls == 10


def test_deadline_not_exceeded_processes_all():
    """Deadline generoso (60s) → init rápida completa normalmente."""
    client = _SlowClient(sleep_per_call=0.0)
    texts = [f"item {i}" for i in range(50)]
    out = sc._batched_embed_documents(client, texts, 5, 0.0, "test", deadline=time.monotonic() + 60)
    assert len(out) == 50 and client.calls == 10


def test_single_batch_ignores_deadline():
    """Un solo batch (len <= batch_size) no entra al loop → per-request timeout ya lo acota."""
    client = _SlowClient(sleep_per_call=0.0)
    out = sc._batched_embed_documents(client, ["a", "b"], 10, 0.0, "test",
                                      deadline=time.monotonic() - 1)  # ya vencido
    assert len(out) == 2 and client.calls == 1, "single-batch corre pese al deadline vencido"
