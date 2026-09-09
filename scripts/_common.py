"""Shared helpers for the Runpod control-plane scripts.

Uses the Runpod REST API (https://rest.runpod.io/v1). Only the stdlib is
required so these run without installing anything.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE lines from .env without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env(name: str, default=None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        sys.exit(f"error: {name} is not set (put it in .env or export it)")
    return value


def api_key() -> str:
    return env("RUNPOD_API_KEY", required=True)


def request(method: str, path: str, body: dict | None = None) -> dict:
    url = path if path.startswith("http") else f"{REST}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"error: {method} {url} -> HTTP {exc.code}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"error: {method} {url} -> {exc.reason}")
    return json.loads(raw) if raw.strip() else {}


def upsert_env_file(updates: dict, path: Path = ENV_FILE) -> None:
    """Write KEY=VALUE back into .env, replacing any existing key in place."""
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(updates)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in remaining.items())
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {', '.join(updates)} to {path}")
