"""Contratos de adjuntos móviles privados, múltiples e idempotentes."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import os
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture
def db_chat_module():
    import db_chat
    return db_chat


def test_signed_attachment_urls_expire_and_are_tamper_evident(monkeypatch, db_chat_module):
    monkeypatch.setenv("MEALFIT_CHAT_ATTACHMENT_SIGNING_SECRET", "s" * 48)
    monkeypatch.setattr(db_chat_module.time, "time", lambda: 1_000_000)
    attachment_id = "11111111-1111-4111-8111-111111111111"
    url = db_chat_module.build_chat_attachment_url(attachment_id, ttl_seconds=600)
    query = parse_qs(urlparse(url).query)
    expires = int(query["expires"][0])
    signature = query["sig"][0]
    assert db_chat_module.verify_chat_attachment_signature(attachment_id, expires, signature)
    assert not db_chat_module.verify_chat_attachment_signature(
        "22222222-2222-4222-8222-222222222222", expires, signature
    )
    monkeypatch.setattr(db_chat_module.time, "time", lambda: expires + 1)
    assert not db_chat_module.verify_chat_attachment_signature(attachment_id, expires, signature)


def test_message_save_is_atomic_ordered_and_idempotent(monkeypatch, db_chat_module):
    monkeypatch.setattr(db_chat_module, "connection_pool", object())
    metadata = [
        {"attachment_id": "a", "name": "uno.jpg"},
        {"attachment_id": "b", "name": "dos.jpg"},
    ]
    monkeypatch.setattr(db_chat_module, "_owned_attachment_metadata", lambda *args: metadata)
    captured = {}
    monkeypatch.setattr(
        db_chat_module,
        "execute_sql_transaction",
        lambda statements: captured.setdefault("statements", statements),
    )
    monkeypatch.setattr(
        db_chat_module,
        "execute_sql_query",
        lambda *args, **kwargs: {"id": "message-existing"},
    )
    monkeypatch.setitem(
        sys.modules,
        "proactive_agent",
        types.SimpleNamespace(handle_nudge_response=lambda *_args, **_kwargs: None),
    )

    result = db_chat_module.save_message_with_attachments(
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "mira esto",
        "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        ["a", "b", "a"],
        client_message_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        vision_items=[
            {"kind": "plato", "description": "arroz con pollo"},
            {"kind": "unavailable", "reason": "busy"},
        ],
    )
    assert result == "message-existing"
    insert, claim = captured["statements"]
    assert "ON CONFLICT (session_id, client_message_id)" in insert[0]
    assert json.loads(insert[1][4]) == [
        {
            "attachment_id": "a",
            "name": "uno.jpg",
            "kind": "plato",
            "description": "arroz con pollo",
        },
        {
            "attachment_id": "b",
            "name": "dos.jpg",
            "kind": "unavailable",
            "reason": "busy",
        },
    ]
    assert claim[1][3] == ["a", "b"]


def test_attachment_reuse_from_another_turn_is_rejected(monkeypatch, db_chat_module):
    monkeypatch.setattr(db_chat_module, "connection_pool", object())
    monkeypatch.setattr(db_chat_module, "_owned_attachment_metadata", lambda *args: [])
    with pytest.raises(ValueError, match="no pertenecen"):
        db_chat_module.save_message_with_attachments(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "x",
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            ["already-claimed"],
            client_message_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        )


def test_migration_enforces_private_lifecycle_and_max_four():
    sql = (ROOT / "migrations" / "p1_chat_mobile_attachments_2026_08_24.sql").read_text(encoding="utf-8")
    assert "jsonb_array_length(attachments) <= 4" in sql
    assert "ON DELETE CASCADE" in sql
    assert "WHERE client_message_id IS NOT NULL" in sql
    # Neon no tiene `auth.users`/`auth.uid()` de Supabase. La FK real y el
    # aislamiento server-side deben quedar explícitos en la migración.
    assert "REFERENCES public.user_profiles(id) ON DELETE CASCADE" in sql
    assert "SIN RLS" in sql
    assert "AND user_id = %s" in sql
    assert "WHERE message_id IS NULL" in sql


def test_chat_upload_never_uses_public_diary_storage_and_returns_attachment_id():
    src = (BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert 'purpose: str = Form("diary")' in src
    assert 'if _storage_client and purpose == "diary"' in src
    assert 'if purpose == "chat" and actual_user_id and session_id' in src
    assert '"attachment_id": attachment_id' in src
    assert 'if actual_user_id and purpose == "diary"' in src


def test_chat_route_renews_urls_and_checks_owner_or_signature():
    src = (BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    assert "_hydrate_chat_attachment_urls" in src
    assert '"url": build_chat_attachment_url(str(attachment_id))' in src
    assert "signed = verify_chat_attachment_signature" in src
    assert "verified_user_id == attachment.get(\"user_id\")" in src
    assert "if not signed and not owned" in src
    assert '"Cache-Control": "private, max-age=3600"' in src


def test_frontend_does_not_persist_temporary_signature_inside_prompt():
    src = (ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")
    assert ".filter((item) => !item.attachment_id)" in src
    assert "client_message_id: clientMessageId" in src
    assert "mapWithConcurrency(localAttachments, 2, uploadOne)" in src
    assert "uploadForm.append('purpose', 'chat')" in src
