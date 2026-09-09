#!/usr/bin/env python3
"""Show everything billable on this account, and tear it down.

    python scripts/teardown.py --status         # what exists and what it costs
    python scripts/teardown.py --all            # delete endpoint + template + volume
    python scripts/teardown.py --endpoint --template

Billing model, so you know what actually needs deleting:

  * Endpoint  — scale-to-zero. Costs nothing while no worker runs. Setting
                workersMin=0 is enough; deleting it is optional.
  * Template  — free. Just configuration.
  * Volume    — bills per hour for provisioned size whether or not anything
                runs. This is the one that keeps charging you. Delete it to stop.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import env, load_env, request, upsert_env_file  # noqa: E402

USD_PER_GB_MONTH = 0.07


def status() -> None:
    endpoints = request("GET", "/endpoints") or []
    volumes = request("GET", "/networkvolumes") or []

    print("ENDPOINTS")
    if not endpoints:
        print("  (none)")
    for ep in endpoints:
        idle_cost = "idle: $0" if ep.get("workersMin", 0) == 0 else "IDLE WORKERS BILLING"
        print(f"  {ep.get('id')}  {ep.get('name')}  "
              f"min={ep.get('workersMin')} max={ep.get('workersMax')}  {idle_cost}")

    print("\nNETWORK VOLUMES  (these bill whether or not anything runs)")
    if not volumes:
        print("  (none) — nothing is accruing storage charges")
    total = 0
    for vol in volumes:
        size = vol.get("size", 0)
        total += size
        print(f"  {vol.get('id')}  {size}GB  {vol.get('dataCenterId')}  {vol.get('name')}")
    if total:
        print(f"  ~= ${total * USD_PER_GB_MONTH:.2f}/month while they exist")

    print("\nTo stop all recurring charges: python scripts/teardown.py --volume")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show billable resources and exit")
    parser.add_argument("--endpoint", action="store_true", help="Delete the endpoint")
    parser.add_argument("--template", action="store_true", help="Delete the template")
    parser.add_argument("--volume", action="store_true", help="Delete the network volume")
    parser.add_argument("--all", action="store_true", help="Delete all three")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if args.status or not (args.endpoint or args.template or args.volume or args.all):
        status()
        return

    endpoint_id = env("RUNPOD_ENDPOINT_ID")
    template_id = env("RUNPOD_TEMPLATE_ID")
    volume_id = env("NETWORK_VOLUME_ID")

    targets = []
    if args.all or args.endpoint:
        targets.append(("endpoint", endpoint_id, "/endpoints"))
    if args.all or args.template:
        targets.append(("template", template_id, "/templates"))
    if args.all or args.volume:
        targets.append(("volume", volume_id, "/networkvolumes"))

    targets = [t for t in targets if t[1]]
    if not targets:
        sys.exit("nothing to delete — no matching ids in .env")

    print("About to permanently delete:")
    for kind, rid, _ in targets:
        print(f"  {kind}: {rid}")
    if not args.yes:
        if input("Type 'delete' to confirm: ").strip() != "delete":
            sys.exit("aborted")

    cleared = {}
    for kind, rid, path in targets:
        # An endpoint must be scaled to zero workers before it will delete.
        if kind == "endpoint":
            request("PATCH", f"{path}/{rid}", {"workersMin": 0, "workersMax": 0})
        request("DELETE", f"{path}/{rid}")
        print(f"deleted {kind} {rid}")
        cleared[{"endpoint": "RUNPOD_ENDPOINT_ID",
                 "template": "RUNPOD_TEMPLATE_ID",
                 "volume": "NETWORK_VOLUME_ID"}[kind]] = ""

    upsert_env_file(cleared)
    print("\nRemaining:")
    status()


if __name__ == "__main__":
    main()
