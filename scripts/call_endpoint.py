#!/usr/bin/env python3
"""Call the deployed endpoint and save the returned PNG.

    python scripts/call_endpoint.py "a cartoon fox riding a skateboard"
    python scripts/call_endpoint.py "same fox, watercolor" --init-image ./ref.png
    python scripts/call_endpoint.py "..." --init-image https://example.com/ref.png --async

Local files given to --init-image are base64-encoded into the request body;
http(s) URLs are passed through and fetched by the worker.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import env, load_env, request  # noqa: E402


def encode_init_image(spec: str) -> str:
    if spec.startswith("http://") or spec.startswith("https://"):
        return spec
    path = Path(spec)
    if not path.exists():
        sys.exit(f"error: init image not found: {spec}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-image", default=None, help="Local path or http(s) URL")
    parser.add_argument("--endpoint-id", default=env("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--out", default="out/output.png")
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Use /run + /status polling instead of /runsync")
    args = parser.parse_args()

    if not args.endpoint_id:
        sys.exit("error: RUNPOD_ENDPOINT_ID is not set — run scripts/create_endpoint.py first")

    payload = {
        "input": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "init_image": encode_init_image(args.init_image) if args.init_image else None,
        }
    }

    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    started = time.time()

    if args.use_async:
        job = request("POST", f"{base}/run", payload)
        job_id = job.get("id")
        print(f"job {job_id} queued; polling…")
        while True:
            job = request("GET", f"{base}/status/{job_id}")
            status = job.get("status")
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            print(f"  {status} ({time.time() - started:.0f}s)")
            time.sleep(3)
    else:
        # A cold start can exceed the 90s runsync budget; the first call may come
        # back IN_QUEUE, in which case rerun or use --async.
        job = request("POST", f"{base}/runsync", payload)

    if job.get("status") != "COMPLETED":
        print(json.dumps(job, indent=2)[:4000])
        sys.exit(f"job did not complete (status={job.get('status')})")

    output = job.get("output") or {}
    if "error" in output:
        sys.exit(f"worker error: {output['error']}")

    b64 = output.get("image_base64")
    if not b64:
        sys.exit(f"no image in output: {json.dumps(output)[:1000]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64))

    print(f"saved {out_path} in {time.time() - started:.1f}s "
          f"(worker {output.get('time_seconds')}s, delay {job.get('delayTime')}ms)")


if __name__ == "__main__":
    main()
