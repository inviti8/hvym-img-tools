"""Create (or update) the RunPod Serverless template + endpoint.

    uv run python scripts/deploy_runpod.py --image ghcr.io/inviti8/hvym-img-tools:0.1.0

Reads RUNPOD_API_KEY from the environment or .env. Idempotent: re-running with
the same --name updates the existing template/endpoint rather than piling up
duplicates.

Private images need registry credentials (--registry-user/--registry-token).
Those are forwarded to RunPod, so prefer a token scoped to *read:packages only* —
never a broad-scoped one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"


def load_env() -> None:
    """Read .env without a dependency, so this runs anywhere."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def call(method: str, path: str, key: str, body: dict | None = None) -> object:
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:1000]
        raise SystemExit(f"RunPod API {method} {path} failed ({exc.code}):\n{detail}") from exc


def find_by_name(items, name):
    for item in items or []:
        if item.get("name") == name:
            return item
    return None


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="container image, e.g. ghcr.io/u/r:tag")
    ap.add_argument("--name", default="hvym-img-tools", help="template + endpoint name")
    ap.add_argument("--gpu-types", default="NVIDIA GeForce RTX 4090,NVIDIA L4,NVIDIA RTX A5000",
                    help="comma-separated; peak VRAM measured at 4.44GB so 24GB is plenty")
    ap.add_argument("--workers-min", type=int, default=0, help="0 = scale to zero")
    ap.add_argument("--workers-max", type=int, default=2)
    ap.add_argument("--idle-timeout", type=int, default=10, help="seconds before a worker sleeps")
    ap.add_argument("--execution-timeout-ms", type=int, default=600_000)
    ap.add_argument("--container-disk-gb", type=int, default=20)
    ap.add_argument("--no-flashboot", action="store_true")
    ap.add_argument("--registry-user", default=os.environ.get("GHCR_USER", ""))
    ap.add_argument("--registry-token", default=os.environ.get("GHCR_TOKEN", ""),
                    help="use a read:packages-only token; omit if the image is public")
    ap.add_argument("--network-volume-id", default="",
                    help="strongly recommended: without shared storage the result "
                         "cache dies with each worker")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set (put it in .env)")

    # 1. registry credentials (only for a private image)
    registry_auth_id = None
    if args.registry_token:
        auths = call("GET", "/containerregistryauth", key) or []
        existing = find_by_name(auths, args.name)
        if existing:
            registry_auth_id = existing["id"]
            print(f"registry auth: reusing {registry_auth_id}")
        elif not args.dry_run:
            created = call("POST", "/containerregistryauth", key, {
                "name": args.name,
                "username": args.registry_user,
                "password": args.registry_token,
            })
            registry_auth_id = created["id"]
            print(f"registry auth: created {registry_auth_id}")

    # 2. template
    template_body = {
        "name": args.name,
        "imageName": args.image,
        "isServerless": True,
        "containerDiskInGb": args.container_disk_gb,
        "env": {
            # Weights are baked into the image; these point at them.
            "HVYM_WARM_ON_STARTUP": "1",
            "HVYM_LOG_LEVEL": "INFO",
        },
    }
    if registry_auth_id:
        template_body["containerRegistryAuthId"] = registry_auth_id

    if args.dry_run:
        print("DRY RUN template:", json.dumps(template_body, indent=2))
    templates = call("GET", "/templates", key) or []
    existing_t = find_by_name(templates, args.name)
    if args.dry_run:
        template_id = existing_t["id"] if existing_t else "<new>"
    elif existing_t:
        template_id = existing_t["id"]
        call("PATCH", f"/templates/{template_id}", key, template_body)
        print(f"template: updated {template_id}")
    else:
        template_id = call("POST", "/templates", key, template_body)["id"]
        print(f"template: created {template_id}")

    # 3. endpoint
    endpoint_body = {
        "name": args.name,
        "templateId": template_id,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": [g.strip() for g in args.gpu_types.split(",") if g.strip()],
        "workersMin": args.workers_min,
        "workersMax": args.workers_max,
        "idleTimeout": args.idle_timeout,
        "executionTimeoutMs": args.execution_timeout_ms,
        "flashboot": not args.no_flashboot,
    }
    if args.network_volume_id:
        endpoint_body["networkVolumeId"] = args.network_volume_id
    else:
        print("WARNING: no --network-volume-id. The result cache is per-worker, so a "
              "cold worker starts cold-cached and re-requests are not free.")

    if args.dry_run:
        print("DRY RUN endpoint:", json.dumps(endpoint_body, indent=2))
        return 0

    endpoints = call("GET", "/endpoints", key) or []
    existing_e = find_by_name(endpoints, args.name)
    if existing_e:
        endpoint_id = existing_e["id"]
        call("PATCH", f"/endpoints/{endpoint_id}", key,
             {k: v for k, v in endpoint_body.items() if k not in ("computeType",)})
        print(f"endpoint: updated {endpoint_id}")
    else:
        endpoint_id = call("POST", "/endpoints", key, endpoint_body)["id"]
        print(f"endpoint: created {endpoint_id}")

    print()
    print(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
    print(f"  runsync: https://api.runpod.ai/v2/{endpoint_id}/runsync")
    print("  point the proxy at it:")
    print(f"    RUNPOD_ENDPOINT_ID={endpoint_id} RUNPOD_API_KEY=... "
          "HVYM_API_KEY=... uv run hvym-img-proxy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
