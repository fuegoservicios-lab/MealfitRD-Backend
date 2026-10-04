"""Execute the actual stream billing guard for text and server-delegated voice."""
import ast
import logging
from pathlib import Path

import pytest


@pytest.mark.parametrize('voice,billed,observed,expected', [
    (True, False, True, 0),
    (False, False, True, 1),
    (False, True, True, 0),
    (False, False, False, 0),
])
def test_voice_exemption_preserves_text_billing(voice, billed, observed, expected):
    path = Path(__file__).resolve().parent.parent / 'routers/chat.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and '_voz_sin_cuota' in ast.unparse(n.test))
    calls = []
    ns = {'_voz_sin_cuota': voice, '_billed': billed, '_chunk_observed': observed,
          'verified_user_id': 'authenticated-owner', 'logger': logging.getLogger('test'),
          'log_api_usage': lambda *args: calls.append(args)}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(path), 'exec'), ns)
    assert calls == [('authenticated-owner', 'llm_chat')] * expected
    assert ns['_billed'] == (billed or bool(expected))
