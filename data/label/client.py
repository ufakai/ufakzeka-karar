"""A client for a hosted chat-completions API, built to stop rather than overspend.

Every judge call goes through here. The client turns
reasoning off and rejects a call that billed reasoning tokens anyway, and it keeps
a running total of the cost the API reports for each call (usage.cost, in US
dollars), refusing any call once that total reaches the cap.

Each chat call leaves one JSON line in the ledger, whatever its outcome, so the
spend can be reconciled against the API account's own billing page afterwards.

The transport is a plain callable so the tests can replace the network. The
default uses urllib from the standard library and adds no dependency.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The API's address is configuration, not code: it names a vendor, which public
# code does not (rule 4). It is set on the box that runs the pipeline.
BASE_URL_ENV = "KARAR_API_URL"

# (method, url, headers, json body or None) -> (HTTP status, parsed JSON body)
Transport = Callable[[str, str, dict[str, str], dict[str, Any] | None], tuple[int, Any]]

# 429 and 5xx are retried; any other 4xx is a mistake in the request and is not.
MAX_ATTEMPTS = 4
MAX_RETRY_AFTER_S = 60.0
BACKOFF_BASE_S = 1.0
TIMEOUT_S = 120.0


class ApiError(RuntimeError):
    """The API refused the request or answered with something unusable."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message if body is None else f"{message}: {body!r}")
        self.status = status
        self.body = body


class BudgetExceeded(RuntimeError):
    """The running spend has reached the cap; no further call is made."""


class LowBalance(RuntimeError):
    """The account balance is under the floor; no further call is made."""


class ReasoningBilled(RuntimeError):
    """A call billed reasoning tokens although reasoning was asked off. Its cost is recorded."""


def urllib_transport(
    method: str, url: str, headers: dict[str, str], json_body: dict[str, Any] | None
) -> tuple[int, Any]:
    """The default transport. An HTTP error status is returned, not raised."""
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, _parse_body(response.read())
    except urllib.error.HTTPError as error:
        body = _parse_body(error.read())
        # The API's error guide asks callers to respect Retry-After on a
        # 429 before retrying; the header travels with the body to the retry
        # loop, which is the only place that uses it.
        wait = error.headers.get("Retry-After") if error.headers else None
        if wait is not None and isinstance(body, dict):
            body["_retry_after"] = wait
        return error.code, body


def _parse_body(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _retryable(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _number(value: Any) -> float | None:
    """A finite, non-negative number from a JSON field, or None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value) or value < 0:
        return None
    return value


class ApiClient:
    def __init__(
        self,
        api_key: str,
        *,
        cap_usd: float,
        ledger_path: str | Path,
        base_url: str | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        if cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        self._api_key = api_key
        self.cap_usd = float(cap_usd)
        self.ledger_path = Path(ledger_path)
        base_url = base_url or os.environ.get(BASE_URL_ENV)
        if not base_url:
            raise ValueError(f"no API address: pass base_url or set {BASE_URL_ENV}")
        self.base_url = base_url.rstrip("/")
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self._lock = threading.Lock()
        self._ledger_lock = threading.Lock()
        # The cap covers every call in the ledger, so a build restarted on the
        # same output does not get a second budget.
        self._spent = self._spent_in_ledger()

    def _spent_in_ledger(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += json.loads(line).get("cost") or 0.0
        return total

    @property
    def spent(self) -> float:
        """What the API reported as spent on this client's calls, in US dollars."""
        with self._lock:
            return self._spent

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any, int]:
        """Send with retries on 429 and 5xx. Returns (status, body, attempts made).

        A transport exception is not retried: a request that timed out may still
        have been served and billed, and sending it again could pay twice.
        """
        url = f"{self.base_url}{path}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status, parsed = self._transport(method, url, self._headers(), body)
            if not _retryable(status) or attempt == MAX_ATTEMPTS:
                return status, parsed, attempt
            backoff = BACKOFF_BASE_S * 2 ** (attempt - 1)
            asked = (
                _number(_to_float(parsed.get("_retry_after"))) if isinstance(parsed, dict) else None
            )
            # Wait at least as long as the API asked, capped so one call
            # cannot stall a run for minutes.
            self._sleep(min(max(backoff, asked or 0.0), MAX_RETRY_AFTER_S))
        raise AssertionError("unreachable")

    def _before_call(self) -> None:
        """Refuse the call once the cap is reached."""
        with self._lock:
            if self._spent >= self.cap_usd:
                raise BudgetExceeded(
                    f"spent {self._spent:.4f} of a {self.cap_usd:.4f} USD cap; no further calls"
                )

    def chat(
        self,
        model: str,
        provider: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = "none",
        temperature: float = 0.0,
        forbid_reasoning_tokens: bool = True,
    ) -> dict[str, Any]:
        """One chat completion. Returns the parsed response body.

        `provider` is the host the panel names for the model; the ledger records it.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # A model without reasoning may reject the field, so it is sent only when
        # asked for; reasoning_tokens in the usage is still checked either way.
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if logprobs:
            body["logprobs"] = True
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs

        self._before_call()
        try:
            status, response, attempts = self._request("POST", "/chat/completions", body)
        except Exception:
            # The call may have been billed, and its cost is now unknown; the
            # ledger line lets the spend be reconciled against the account's page.
            self._write_ledger(self._entry(model, provider, {}, {}, None, 0, "transport_error", 0))
            raise

        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        cost = _number(usage.get("cost"))
        details = usage.get("completion_tokens_details")
        reported = details.get("reasoning_tokens") if isinstance(details, dict) else None
        reasoning_tokens = math.ceil(_number(reported) or 0)

        # Whatever the outcome, a reported cost has been spent and is counted.
        if cost is not None:
            with self._lock:
                self._spent += cost

        if status != 200:
            outcome = f"http_{status}"
        elif isinstance(response, dict) and response.get("error"):
            outcome = "error_in_body"
        elif cost is None:
            outcome = "no_cost"
        elif forbid_reasoning_tokens and reasoning_tokens > 0:
            outcome = "reasoning_billed"
        else:
            outcome = "ok"

        self._write_ledger(
            self._entry(model, provider, response, usage, cost, reasoning_tokens, outcome, attempts)
        )

        if outcome.startswith("http_"):
            raise ApiError(f"chat call failed with HTTP {status}", status=status, body=response)
        if outcome == "error_in_body":
            raise ApiError("chat call returned an error", status=status, body=response)
        if outcome == "no_cost":
            # Without a cost the cap cannot be enforced, so the run stops here.
            raise ApiError("response carries no usage.cost", status=status, body=response)
        if outcome == "reasoning_billed":
            raise ReasoningBilled(
                f"{model} via {provider} billed {reasoning_tokens} reasoning tokens"
            )
        return response

    @staticmethod
    def _entry(
        model: str,
        provider: str,
        response: Any,
        usage: dict[str, Any],
        cost: float | None,
        reasoning_tokens: int,
        status: str,
        attempts: int,
    ) -> dict[str, Any]:
        response = response if isinstance(response, dict) else {}
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "model": model,
            "provider": provider,
            "generation_id": response.get("id"),
            "cost": cost,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "status": status,
            "attempts": attempts,
        }
        return entry

    def _write_ledger(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._ledger_lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
