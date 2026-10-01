"""Reading from the Hugging Face hub within its rate limits.

The hub's rate-limit page (huggingface.co/docs/hub/rate-limits, read
2026-09-23) counts requests in 5-minute fixed windows per bucket: 500 API calls
per IP anonymous and 1,000 with a free account's token, and far more for file
downloads through the resolver (/resolve/ URLs). It asks for a token on every
call, for resolver downloads in place of API calls where possible, and on a 429
for the wait the RateLimit header's t= gives. The dataset viewer's rows API is
an API call per hundred rows, and its edge throttled this lab's server after a
few hundred, so datasets are read as parquet files through the resolver.

Every dataset on the hub has a parquet export under the refs/convert/parquet
revision, one folder per config and split.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

TOKEN_FILE_ENV = "KARAR_HF_TOKEN_FILE"
TREE = "https://huggingface.co/api/datasets/{repo}/tree/refs%2Fconvert%2Fparquet/{folder}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{path}"
HUB_HOSTS = ("huggingface.co", "datasets-server.huggingface.co")
RETRIED = (429, 500, 502, 503, 504)
MAX_WAIT_S = 300


def token() -> str | None:
    path = os.environ.get(TOKEN_FILE_ENV)
    if path and Path(path).is_file():
        return Path(path).read_text().strip()
    return os.environ.get("HF_TOKEN")


def reset_after(header: str | None) -> float | None:
    """Seconds until the window resets, from a RateLimit header such as '"api";r=0;t=42'."""
    match = re.search(r"\bt=(\d+)", header or "")
    return float(match.group(1)) if match else None


def open_url(url: str, attempts: int = 6):
    """A response for `url`, with the token when one is set and the waits a 429 asks for."""
    headers = {"User-Agent": "ufakzeka-karar"}
    # The token goes to the hub only, never to another host a caller passes.
    if urllib.parse.urlsplit(url).hostname in HUB_HOSTS and (secret := token()):
        headers["Authorization"] = f"Bearer {secret}"
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
        except urllib.error.HTTPError as error:
            if error.code not in RETRIED or attempt == attempts - 1:
                raise
            wait = reset_after(error.headers.get("RateLimit")) if error.code == 429 else None
            time.sleep(min(MAX_WAIT_S, (wait or 2**attempt) + 1))
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def get_json(url: str) -> Any:
    with open_url(url) as response:
        return json.load(response)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with open_url(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)


def parquet_files(repo: str, config: str, split: str, cache: Path, *, fetch_json=get_json,
                  fetch_file=download) -> list[Path]:  # fmt: skip
    """One config and split's parquet export, downloaded once and checked by size."""
    listing = fetch_json(TREE.format(repo=repo, folder=f"{config}/{split}"))
    entries = sorted(
        (e for e in listing if e["path"].endswith(".parquet")), key=lambda e: e["path"]
    )
    if not entries:
        raise RuntimeError(f"{repo}: no parquet export for {config}/{split}")
    paths = []
    for entry in entries:
        local = cache / repo.replace("/", "__") / entry["path"]
        if not (local.is_file() and local.stat().st_size == entry["size"]):
            fetch_file(RESOLVE.format(repo=repo, path=entry["path"]), local)
        paths.append(local)
    return paths
