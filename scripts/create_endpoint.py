#!/usr/bin/env python3
"""Create (or update) the scale-to-zero Runpod serverless endpoint.

Picks every GPU type in your account with at least MIN_VRAM_GB of VRAM and
offers them to the scheduler in ascending-memory order, so a 48GB card is used
when available and a larger one is the fallback rather than a hard failure.

    python scripts/create_endpoint.py [--name flux2-cartoon-lora] [--list-gpus]

Writes RUNPOD_ENDPOINT_ID back into .env.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import env, load_env, request, upsert_env_file  # noqa: E402


# The REST API has no GPU-metadata endpoint, so VRAM is tabulated here against
# the exact gpuTypeIds the /endpoints schema accepts. AMD is deliberately absent:
# the image is CUDA-only (bitsandbytes NF4 needs CUDA).
GPU_VRAM_GB = {
    "NVIDIA A40": 48,
    "NVIDIA L40": 48,
    "NVIDIA L40S": 48,
    "NVIDIA RTX A6000": 48,
    "NVIDIA RTX 6000 Ada Generation": 48,
    "NVIDIA RTX PRO 5000 Blackwell": 48,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 PCIe": 80,
    "NVIDIA H100 NVL": 94,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 96,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 96,
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": 96,
    "NVIDIA H200": 141,
    "NVIDIA H200 NVL": 141,
    "NVIDIA B200": 180,
    # Below the 48GB bar; listed so --min-vram-gb can reach them deliberately.
    "NVIDIA A100-SXM4-40GB": 40,
    "NVIDIA GeForce RTX 5090": 32,
    "NVIDIA RTX 5000 Ada Generation": 32,
    "NVIDIA RTX PRO 4500 Blackwell": 32,
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA GeForce RTX 3090": 24,
    "NVIDIA L4": 24,
    "NVIDIA RTX A5000": 24,
}


def gpu_types(min_vram_gb: int) -> list[tuple[str, int]]:
    matching = [(gid, vram) for gid, vram in GPU_VRAM_GB.items() if vram >= min_vram_gb]
    matching.sort(key=lambda item: (item[1], item[0]))
    return matching


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=env("ENDPOINT_NAME", "flux2-cartoon-lora"))
    parser.add_argument("--template-id", default=env("RUNPOD_TEMPLATE_ID"))
    parser.add_argument("--min-vram-gb", type=int, default=int(env("MIN_VRAM_GB", "48")))
    parser.add_argument("--workers-min", type=int, default=int(env("WORKERS_MIN", "0")))
    parser.add_argument("--workers-max", type=int, default=int(env("WORKERS_MAX", "1")))
    parser.add_argument("--idle-timeout", type=int, default=int(env("IDLE_TIMEOUT", "5")))
    parser.add_argument("--gpu-ids", default=env("GPU_TYPE_IDS", ""),
                        help="Comma-separated GPU type ids, overriding VRAM autodetection")
    parser.add_argument("--data-centers", default=env("DATA_CENTER_ID", ""),
                        help="Comma-separated data centers. Must include the network "
                             "volume's data center, since a volume is region-bound.")
    parser.add_argument("--network-volume-id", default=env("NETWORK_VOLUME_ID", ""),
                        help="Attach a network volume at /runpod-volume so the ~34GB "
                             "of weights survive between cold starts. Bills hourly "
                             "even while the endpoint is idle — see scripts/teardown.py")
    parser.add_argument("--list-gpus", action="store_true", help="Print matching GPUs and exit")
    args = parser.parse_args()

    if args.list_gpus:
        for gid, vram in gpu_types(args.min_vram_gb):
            print(f"{vram:>4}GB  {gid}")
        return

    if not args.template_id:
        sys.exit("error: RUNPOD_TEMPLATE_ID is not set — run scripts/create_template.py first")

    if args.gpu_ids.strip():
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    else:
        matching = gpu_types(args.min_vram_gb)
        if not matching:
            sys.exit(f"error: no known GPU type has >= {args.min_vram_gb}GB")
        gpu_ids = [gid for gid, _ in matching]
        print(f"GPUs >= {args.min_vram_gb}GB (smallest first, so 48GB is preferred):")
        for gid, vram in matching:
            print(f"  {vram:>4}GB  {gid}")

    body = {
        "name": args.name,
        "templateId": args.template_id,
        "computeType": "GPU",
        "gpuTypeIds": gpu_ids,
        "gpuCount": 1,
        "workersMin": args.workers_min,
        "workersMax": args.workers_max,
        "idleTimeout": args.idle_timeout,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "executionTimeoutMs": int(env("EXECUTION_TIMEOUT_MS", "900000")),
        "flashboot": True,
    }
    if args.network_volume_id:
        body["networkVolumeId"] = args.network_volume_id
    if args.data_centers.strip():
        body["dataCenterIds"] = [d.strip() for d in args.data_centers.split(",") if d.strip()]

    existing = {e.get("name"): e for e in request("GET", "/endpoints") or []}
    if args.name in existing:
        endpoint_id = existing[args.name]["id"]
        print(f"updating existing endpoint {args.name} ({endpoint_id})")
        # id/computeType/gpuCount are immutable after creation.
        patch = {k: v for k, v in body.items() if k not in ("computeType", "gpuCount")}
        result = request("PATCH", f"/endpoints/{endpoint_id}", patch)
        result.setdefault("id", endpoint_id)
    else:
        print(f"creating endpoint {args.name}")
        result = request("POST", "/endpoints", body)

    endpoint_id = result.get("id")
    if not endpoint_id:
        sys.exit(f"error: no endpoint id in response: {result}")

    print(f"\nendpoint id:  {endpoint_id}")
    print(f"runsync URL:  https://api.runpod.ai/v2/{endpoint_id}/runsync")
    print(f"workersMin={args.workers_min} workersMax={args.workers_max} idleTimeout={args.idle_timeout}s")
    if args.network_volume_id:
        print(f"network volume: {args.network_volume_id} (billed hourly while it exists)")
    else:
        print("no network volume — every cold start re-downloads ~34GB of weights")
    upsert_env_file({"RUNPOD_ENDPOINT_ID": endpoint_id})


if __name__ == "__main__":
    main()
