"""The API client, against a fake transport. Nothing here reaches the network."""

import json

import pytest

from data.label.client import (
    MAX_ATTEMPTS,
    ApiClient,
    ApiError,
    BudgetExceeded,
    ReasoningBilled,
)

BASE = "https://api.test/v1"
MESSAGES = [{"role": "user", "content": "Merhaba"}]


def completion(cost=0.001, reasoning_tokens=0, provider="prov-x", gen_id="gen-1"):
    body = {
        "id": gen_id,
        "choices": [{"message": {"role": "assistant", "content": "A"}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 1,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }
    return body


class FakeApi:
    """Answers /chat/completions from a queue, or a default."""

    def __init__(
        self,
        chat=None,
    ):
        self.chat_queue = list(chat or [])
        self.default_chat = (200, completion())
        self.requests = []

    def __call__(self, method, url, headers, json_body):
        self.requests.append((method, url, headers, json_body))
        assert method == "POST" and url.endswith("/chat/completions")
        return self.chat_queue.pop(0) if self.chat_queue else self.default_chat

    def paths(self):
        return [url.removeprefix(BASE) for _, url, _, _ in self.requests]

    def chat_bodies(self):
        return [body for _, url, _, body in self.requests if url.endswith("/chat/completions")]


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def make(tmp_path, api, **overrides):
    sleep = FakeSleep()
    options = {
        "cap_usd": 1.0,
        "ledger_path": tmp_path / "ledger.jsonl",
        "base_url": BASE,
        "transport": api,
        "sleep": sleep,
    }
    options.update(overrides)
    return ApiClient("test-key", **options), sleep


def ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]


def test_the_request_turns_reasoning_off_and_asks_for_logprobs(tmp_path):
    api = FakeApi()
    client, _ = make(tmp_path, api)
    client.chat("vendor/model-a", "prov-x", MESSAGES, max_tokens=1, logprobs=True, top_logprobs=10)
    (body,) = api.chat_bodies()
    assert body["model"] == "vendor/model-a"
    assert body["messages"] == MESSAGES
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0.0
    effort = body.get("reasoning_effort")
    assert effort == "none"
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 10
    _, _, headers, _ = api.requests[-1]
    assert headers["Authorization"] == "Bearer test-key"


def test_logprob_fields_are_left_out_unless_asked(tmp_path):
    api = FakeApi()
    client, _ = make(tmp_path, api)
    client.chat("m", "p", MESSAGES, max_tokens=5)
    (body,) = api.chat_bodies()
    assert "logprobs" not in body
    assert "top_logprobs" not in body


def test_spend_is_summed_and_each_call_leaves_one_ledger_line(tmp_path):
    api = FakeApi(
        chat=[
            (200, completion(cost=0.0015, gen_id="gen-1", provider="Prov X")),
            (200, completion(cost=0.0025, gen_id="gen-2", provider="Prov X")),
        ]
    )
    client, _ = make(tmp_path, api)
    client.chat("m", "prov-x", MESSAGES, max_tokens=1)
    client.chat("m", "prov-x", MESSAGES, max_tokens=1)
    assert client.spent == pytest.approx(0.004)
    lines = ledger(tmp_path)
    assert len(lines) == 2
    first = lines[0]
    assert first["model"] == "m"
    assert first["provider"] == "prov-x"
    assert first["generation_id"] == "gen-1"
    assert first["cost"] == 0.0015
    assert first["prompt_tokens"] == 120
    assert first["completion_tokens"] == 1
    assert first["reasoning_tokens"] == 0
    assert first["status"] == "ok"
    assert first["ts"]
    assert "test-key" not in (tmp_path / "ledger.jsonl").read_text()


def test_the_cap_stops_the_call_after_the_total_reaches_it(tmp_path):
    api = FakeApi(chat=[(200, completion(cost=0.006)), (200, completion(cost=0.006))])
    client, _ = make(tmp_path, api, cap_usd=0.012)
    client.chat("m", "p", MESSAGES, max_tokens=1)
    client.chat("m", "p", MESSAGES, max_tokens=1)
    sent = len(api.chat_bodies())
    with pytest.raises(BudgetExceeded):
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert len(api.chat_bodies()) == sent == 2
    assert client.spent == pytest.approx(0.012)


def test_a_client_on_an_existing_ledger_counts_its_spend_toward_the_cap(tmp_path):
    api = FakeApi(chat=[(200, completion(cost=0.006)), (200, completion(cost=0.006))])
    first, _ = make(tmp_path, api, cap_usd=0.012)
    first.chat("m", "p", MESSAGES, max_tokens=1)
    first.chat("m", "p", MESSAGES, max_tokens=1)
    restarted, _ = make(tmp_path, FakeApi(), cap_usd=0.012)
    assert restarted.spent == pytest.approx(0.012)
    with pytest.raises(BudgetExceeded):
        restarted.chat("m", "p", MESSAGES, max_tokens=1)


def test_billed_reasoning_is_rejected_after_its_cost_is_recorded(tmp_path):
    api = FakeApi(chat=[(200, completion(cost=0.003, reasoning_tokens=12))])
    client, _ = make(tmp_path, api)
    with pytest.raises(ReasoningBilled):
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert client.spent == pytest.approx(0.003)
    (line,) = ledger(tmp_path)
    assert line["status"] == "reasoning_billed"
    assert line["reasoning_tokens"] == 12
    assert line["cost"] == 0.003


def test_reasoning_tokens_pass_when_not_forbidden(tmp_path):
    api = FakeApi(chat=[(200, completion(cost=0.003, reasoning_tokens=12))])
    client, _ = make(tmp_path, api)
    client.chat("m", "p", MESSAGES, max_tokens=1, forbid_reasoning_tokens=False)
    assert client.spent == pytest.approx(0.003)


@pytest.mark.parametrize("status", [429, 503])
def test_429_and_5xx_are_retried_with_backoff(tmp_path, status):
    api = FakeApi(chat=[(status, {"error": "busy"})] * 2 + [(200, completion())])
    client, sleep = make(tmp_path, api)
    client.chat("m", "p", MESSAGES, max_tokens=1)
    assert len(api.chat_bodies()) == 3
    assert sleep.calls == [1.0, 2.0]
    assert ledger(tmp_path)[0]["attempts"] == 3


@pytest.mark.parametrize("status", [429, 503])
def test_retries_stop_after_four_attempts(tmp_path, status):
    api = FakeApi(chat=[(status, {"error": "busy"})] * 10)
    client, sleep = make(tmp_path, api)
    with pytest.raises(ApiError) as caught:
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert caught.value.status == status
    assert len(api.chat_bodies()) == MAX_ATTEMPTS == 4
    assert sleep.calls == [1.0, 2.0, 4.0]
    assert ledger(tmp_path)[0]["status"] == f"http_{status}"


def test_a_400_is_not_retried_and_carries_the_body(tmp_path):
    body = {"error": {"message": "top_logprobs not supported"}}
    api = FakeApi(chat=[(400, body)])
    client, sleep = make(tmp_path, api)
    with pytest.raises(ApiError) as caught:
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert caught.value.status == 400
    assert caught.value.body == body
    assert "top_logprobs not supported" in str(caught.value)
    assert len(api.chat_bodies()) == 1
    assert sleep.calls == []


def test_a_response_without_a_cost_stops_the_run(tmp_path):
    response = completion()
    del response["usage"]["cost"]
    api = FakeApi(chat=[(200, response)])
    client, _ = make(tmp_path, api)
    with pytest.raises(ApiError, match="usage.cost"):
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert ledger(tmp_path)[0]["status"] == "no_cost"


def test_a_transport_failure_is_not_retried_and_is_logged(tmp_path):
    def broken(method, url, headers, json_body):
        raise TimeoutError("read timed out")

    client, sleep = make(tmp_path, broken)
    with pytest.raises(TimeoutError):
        client.chat("m", "p", MESSAGES, max_tokens=1)
    assert sleep.calls == []
    assert ledger(tmp_path)[0]["status"] == "transport_error"


def test_the_api_address_comes_from_configuration_not_code(tmp_path, monkeypatch):
    import pytest

    from data.label.client import BASE_URL_ENV

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    with pytest.raises(ValueError, match=BASE_URL_ENV):
        ApiClient("k", cap_usd=1.0, ledger_path=tmp_path / "l.jsonl")
    monkeypatch.setenv(BASE_URL_ENV, "https://api.test/v1/")
    client = ApiClient("k", cap_usd=1.0, ledger_path=tmp_path / "l.jsonl")
    assert client.base_url == "https://api.test/v1"


def test_a_429_waits_as_long_as_the_api_asks(tmp_path):
    calls = []

    def api(method, url, headers, body):
        calls.append(url)
        if len([c for c in calls if c.endswith("/chat/completions")]) == 1:
            return 429, {"error": {"message": "rate limited"}, "_retry_after": "7"}
        return 200, {"choices": [{"message": {"content": "A"}}], "usage": {"cost": 0.001}}

    client, sleep = make(tmp_path, api)
    client.chat("m", "p", [{"role": "user", "content": "x"}], max_tokens=1)
    assert sleep.calls == [7.0], "the Retry-After the API sent was not respected"
