"""The hub token goes to the hub only."""

import urllib.request

from data import hub


def test_the_token_is_sent_to_the_hub_and_nowhere_else(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.delenv(hub.TOKEN_FILE_ENV, raising=False)
    sent = {}

    def fake_urlopen(request, timeout):
        sent[request.full_url] = request.get_header("Authorization")
        return object()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    hub.open_url("https://huggingface.co/api/datasets/org/set")
    hub.open_url("https://raw.githubusercontent.com/org/repo/abc/dev.json")
    assert sent["https://huggingface.co/api/datasets/org/set"] == "Bearer hf_secret"
    assert sent["https://raw.githubusercontent.com/org/repo/abc/dev.json"] is None
