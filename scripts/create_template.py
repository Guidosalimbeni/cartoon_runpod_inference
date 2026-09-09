#!/usr/bin/env python3
"""Create (or update) the Runpod serverless template for the GHCR worker image.

    python scripts/create_template.py [--name flux2-cartoon-lora] [--image ghcr.io/...]

Writes RUNPOD_TEMPLATE_ID back into .env.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import env, load_env, request, upsert_env_file  # noqa: E402


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=env("TEMPLATE_NAME", "flux2-cartoon-lora"))
    parser.add_argument("--image", default=env("IMAGE"))
    parser.add_argument("--container-disk-gb", type=int, default=int(env("CONTAINER_DISK_GB", "60")))
    args = parser.parse_args()

    if not args.image:
        sys.exit("error: IMAGE is not set (put it in .env or pass --image)")

    worker_env = {
        "BASE_MODEL": env("BASE_MODEL", "black-forest-labs/FLUX.2-dev"),
        "LORA_REPO": env("LORA_REPO", "Guido/cartoon_lora"),
        "LORA_WEIGHT_NAME": env("LORA_WEIGHT_NAME", "my_first_lora_v2.safetensors"),
        "DTYPE": env("DTYPE", "bfloat16"),
        "QUANTIZE": env("QUANTIZE", "4bit"),
        "OFFLOAD": env("OFFLOAD", "1"),
    }
    # Only pass HF_TOKEN through when the repos actually need it.
    hf_token = env("HF_TOKEN")
    if hf_token:
        worker_env["HF_TOKEN"] = hf_token

    body = {
        "name": args.name,
        "imageName": args.image,
        "isServerless": True,
        "category": "NVIDIA",
        "containerDiskInGb": args.container_disk_gb,
        "env": worker_env,
    }

    existing = {t.get("name"): t for t in request("GET", "/templates") or []}
    if args.name in existing:
        template_id = existing[args.name]["id"]
        print(f"updating existing template {args.name} ({template_id})")
        result = request("PATCH", f"/templates/{template_id}", body)
    else:
        print(f"creating template {args.name}")
        result = request("POST", "/templates", body)

    template_id = result.get("id") or result.get("templateId")
    if not template_id:
        sys.exit(f"error: no template id in response: {result}")

    print(f"template id: {template_id}")
    print(f"image:       {args.image}")
    upsert_env_file({"RUNPOD_TEMPLATE_ID": template_id})


if __name__ == "__main__":
    main()
