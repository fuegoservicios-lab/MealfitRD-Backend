from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stream_regeneration_replaces_model_and_does_not_duplicate_user():
    source = (ROOT / "routers" / "chat.py").read_text(encoding="utf-8")

    assert "get_model_response_id_for_regeneration(" in source
    assert "replace_model_response_for_regeneration(" in source
    assert "is_regeneration = bool(regenerate_message_id or regenerate_response_content)" in source
    assert "if is_regeneration:" in source
    assert "elif _db_user_id and client_message_id:" in source


def test_replace_helpers_are_scoped_to_session_and_model_role():
    source = (ROOT / "db_chat.py").read_text(encoding="utf-8")

    assert "def get_model_response_id_for_regeneration(" in source
    assert "WHERE id = %s AND session_id = %s AND role = 'model'" in source
    assert "WHERE session_id = %s AND role = 'model' AND content = %s" in source
    assert "ORDER BY created_at DESC, id DESC LIMIT 1" in source
    assert "def replace_model_response_for_regeneration(" in source
    assert "UPDATE public.agent_messages SET content = %s, feedback = NULL" in source
