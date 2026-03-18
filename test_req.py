import urllib.request
import urllib.error
import json

try:
    req = urllib.request.Request(
        'http://127.0.0.1:3001/api/chat',
        data=b'{"session_id":"3f95bd58-1144-47a4-a090-e6adb74f03fb","prompt":"hola","user_id":"guest"}',
        headers={'Content-Type':'application/json'}
    )
    resp = urllib.request.urlopen(req)
    print("OK:", resp.read()[:50])
except urllib.error.HTTPError as e:
    print(e.read().decode('utf-8'))
