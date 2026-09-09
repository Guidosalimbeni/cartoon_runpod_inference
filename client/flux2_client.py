"""Standalone client for the FLUX.2 + cartoon-LoRA Runpod endpoint.

Single file, stdlib only — no pip installs. Copy it into any project.

Config comes from the environment (or a .env file beside the script):

    RUNPOD_API_KEY=rpa_...
    RUNPOD_ENDPOINT_ID=wqa9wrxf1chw26

As a library:

    from flux2_client import generate

    png = generate("lora-cartoon, a caricature bust of a smiling man", seed=42)
    open("out.png", "wb").write(png)

    # with a reference image
    png = generate("lora-cartoon, caricature of this person",
                   init_image="face.jpg", guidance_scale=2.5)

As a CLI:

    python flux2_client.py "lora-cartoon, ..." --init-image face.jpg -o out.png
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

__all__ = ["generate", "generate_many", "Flux2Error"]

BASE_URL = "https://api.runpod.ai/v2"

# A cold worker pulls the image and loads ~34GB of weights, so the default
# ceiling is generous. Poll rather than blocking on /runsync, which gives up
# after ~90s and would fail exactly when the endpoint is cold.
DEFAULT_TIMEOUT = 900
POLL_SECONDS = 3

ImageSpec = Union[str, Path, bytes]


class Flux2Error(RuntimeError):
    """Raised when the endpoint or the worker reports a failure."""


def _load_dotenv() -> None:
    """Read a .env sitting next to this file, without clobbering real env vars."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _config(api_key: Optional[str], endpoint_id: Optional[str]) -> tuple[str, str]:
    _load_dotenv()
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
    if not api_key:
        raise Flux2Error("RUNPOD_API_KEY is not set")
    if not endpoint_id:
        raise Flux2Error("RUNPOD_ENDPOINT_ID is not set")
    return api_key, endpoint_id


def _post(url: str, api_key: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    # Runpod sits behind Cloudflare, which 403s urllib's default UA (error 1010).
    req.add_header("User-Agent", "flux2-client/1.0")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise Flux2Error(f"HTTP {exc.code} from {url}: {exc.read().decode(errors='replace')[:500]}") from exc
    except urllib.error.URLError as exc:
        raise Flux2Error(f"cannot reach {url}: {exc.reason}") from exc


def encode_image(spec: ImageSpec) -> str:
    """Turn a path, raw bytes, or URL into what the endpoint expects.

    http(s) URLs pass through untouched — the worker fetches them itself, which
    keeps them out of the request body. Everything else is base64-encoded.
    """
    if isinstance(spec, bytes):
        return base64.b64encode(spec).decode("ascii")
    text = str(spec)
    if text.startswith("http://") or text.startswith("https://"):
        return text
    path = Path(text)
    if not path.is_file():
        raise Flux2Error(f"reference image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate_many(
    prompt: str,
    *,
    init_image: Optional[ImageSpec] = None,
    init_images: Optional[Iterable[ImageSpec]] = None,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    num_images: int = 1,
    lora_scale: Optional[float] = None,
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    on_status=None,
) -> list[bytes]:
    """Generate images and return them as a list of PNG byte strings.

    `on_status(status, seconds)` is called on each poll if given, so a caller can
    show progress during a cold start.
    """
    api_key, endpoint_id = _config(api_key, endpoint_id)

    payload: dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if lora_scale is not None:
        payload["lora_scale"] = float(lora_scale)
    if init_image is not None:
        payload["init_image"] = encode_image(init_image)
    if init_images:
        payload["init_images"] = [encode_image(s) for s in init_images]

    base = f"{BASE_URL}/{endpoint_id}"
    started = time.time()

    job = _post(f"{base}/run", api_key, {"input": payload})
    job_id = job.get("id")
    if not job_id:
        raise Flux2Error(f"endpoint did not queue a job: {job}")

    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
    while job.get("status") not in terminal:
        if time.time() - started > timeout:
            raise Flux2Error(f"job {job_id} still {job.get('status')} after {timeout}s")
        if on_status:
            on_status(job.get("status"), time.time() - started)
        time.sleep(POLL_SECONDS)
        job = _post(f"{base}/status/{job_id}", api_key)

    if job.get("status") != "COMPLETED":
        raise Flux2Error(f"job {job_id} ended as {job.get('status')}: {job.get('error')}")

    output = job.get("output") or {}
    if output.get("error"):
        raise Flux2Error(f"worker error: {output['error']}")

    encoded = output.get("images_base64") or ([output["image_base64"]] if output.get("image_base64") else [])
    if not encoded:
        raise Flux2Error(f"no image in response: {json.dumps(output)[:500]}")
    return [base64.b64decode(b) for b in encoded]


def generate(prompt: str, **kwargs) -> bytes:
    """Generate a single image and return its PNG bytes."""
    return generate_many(prompt, **kwargs)[0]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Call the FLUX.2 cartoon-LoRA endpoint.")
    parser.add_argument("prompt")
    parser.add_argument("-i", "--init-image", help="Reference image: local path or http(s) URL")
    parser.add_argument("-o", "--out", default="output.png")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--endpoint-id")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def progress(status, seconds):
        if not args.quiet:
            print(f"  {status} ({seconds:.0f}s)", flush=True)

    started = time.time()
    images = generate_many(
        args.prompt,
        init_image=args.init_image,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_scale=args.lora_scale,
        num_images=args.num_images,
        endpoint_id=args.endpoint_id,
        on_status=progress,
    )

    out = Path(args.out)
    for index, data in enumerate(images):
        target = out if index == 0 else out.with_stem(f"{out.stem}-{index + 1}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"saved {target} ({len(data) / 1000:.0f} KB) in {time.time() - started:.1f}s")


if __name__ == "__main__":
    _main()
