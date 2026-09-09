#!/usr/bin/env python3
"""Manage the network volume that caches the model weights.

A volume is the only thing here that bills while nothing is running: Runpod
charges for provisioned storage per hour regardless of whether a worker is
awake. Everything else on this endpoint is scale-to-zero and costs nothing idle.

    python scripts/volume.py list
    python scripts/volume.py create --size-gb 60 --data-center EU-RO-1
    python scripts/volume.py delete <volume_id>

Sizing: the NF4 pipeline is ~34GB on disk; 60GB leaves room for the LoRA, a
second model, and HF's temp files during download.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import env, load_env, request, upsert_env_file  # noqa: E402

# Runpod's published network-storage rate; used only for the local estimate below.
USD_PER_GB_MONTH = 0.07


def cmd_list(_args) -> None:
    volumes = request("GET", "/networkvolumes") or []
    if not volumes:
        print("no network volumes — nothing is accruing storage charges")
        return
    total = 0
    for vol in volumes:
        size = vol.get("size", 0)
        total += size
        print(f"{vol.get('id')}  {size:>4}GB  {vol.get('dataCenterId')}  {vol.get('name')}")
    print(f"\ntotal {total}GB ~= ${total * USD_PER_GB_MONTH:.2f}/month while these exist")


def cmd_create(args) -> None:
    body = {"name": args.name, "size": args.size_gb, "dataCenterId": args.data_center}
    vol = request("POST", "/networkvolumes", body)
    volume_id = vol.get("id")
    print(f"created volume {volume_id} ({args.size_gb}GB in {args.data_center})")
    print(f"estimated storage cost ~= ${args.size_gb * USD_PER_GB_MONTH:.2f}/month")
    print("\nNext: python scripts/create_endpoint.py  (picks up NETWORK_VOLUME_ID)")
    print("The endpoint's GPUs must be in this data center, so create_endpoint.py")
    print("will only be able to schedule on GPU types available there.")
    upsert_env_file({"NETWORK_VOLUME_ID": volume_id})


def cmd_delete(args) -> None:
    request("DELETE", f"/networkvolumes/{args.volume_id}")
    print(f"deleted volume {args.volume_id} — storage billing for it stops now")
    print("Cold starts will re-download the weights until you create a new one.")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    create = sub.add_parser("create")
    create.add_argument("--name", default=env("VOLUME_NAME", "flux2-weights"))
    create.add_argument("--size-gb", type=int, default=int(env("VOLUME_SIZE_GB", "60")))
    create.add_argument("--data-center", default=env("DATA_CENTER_ID", "EU-RO-1"))
    create.set_defaults(func=cmd_create)

    delete = sub.add_parser("delete")
    delete.add_argument("volume_id")
    delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
