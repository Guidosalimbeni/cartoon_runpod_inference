# cartoon_runpod_inference

A **Runpod Serverless** worker that runs **FLUX.2-dev** with a **Diffusers LoRA**
(`Guido/cartoon_lora`), built as a container image on **GHCR** and deployed
**scale-to-zero** (`workersMin=0`) so you only pay while a request is running.

```
GitHub Actions ──build──▶ ghcr.io/<you>/flux2-runpod-worker:0.1
                                   │
                          Runpod Template (env: BASE_MODEL, LORA_REPO, HF_TOKEN, …)
                                   │
                          Runpod Endpoint (48GB GPU, min 0 / max 1, idle 5s)
                                   │
              POST https://api.runpod.ai/v2/{endpoint_id}/runsync
```

## Fitting FLUX.2-dev on a 48GB GPU

FLUX.2-dev in bf16 does not fit on a 48GB card — not with offloading, not at
all. The actual weights on the Hub:

| Repo | transformer | text encoder | VAE | total |
|---|---|---|---|---|
| `black-forest-labs/FLUX.2-dev` (bf16) | 64.4 GB | 48.0 GB | 0.3 GB | **113 GB** |
| `diffusers/FLUX.2-dev-bnb-4bit` (NF4) | 18.1 GB | 15.4 GB | 0.3 GB | **34 GB** |

So this repo defaults to **`diffusers/FLUX.2-dev-bnb-4bit`** — the official
diffusers NF4 mirror, a complete `Flux2Pipeline` whose DiT and Mistral-3 text
encoder ship already quantized (the VAE stays fp16). That is a 34GB download
instead of 113GB, and it leaves comfortable headroom on a 48GB GPU. The cost is
some quality loss from 4-bit, which you said is an acceptable trade.

| Setting | Default | Effect |
|---|---|---|
| `BASE_MODEL` | `diffusers/FLUX.2-dev-bnb-4bit` | Weights arrive NF4; no load-time quantization |
| `QUANTIZE` | `none` | Only set `4bit`/`8bit` if you point `BASE_MODEL` at an unquantized repo |
| `OFFLOAD` | `1` | `enable_model_cpu_offload()`; try `0` for more speed, it likely still fits |
| `DTYPE` | `bfloat16` | Compute dtype |

`scripts/create_endpoint.py` selects **every GPU type with ≥ `MIN_VRAM_GB`
(default 48)**, ordered smallest first, so a 48GB card is preferred and a larger
one is a fallback rather than a scheduling failure.

> If you have seen a notebook claiming a 48GB pod runs FLUX.2-dev in plain bf16
> with `pipe.to("cuda")` and no quantization — it does not. 64.4GB of transformer
> weights alone exceed the card.

## Hugging Face token

Set `HF_TOKEN` in `.env`. The default `BASE_MODEL`
(`diffusers/FLUX.2-dev-bnb-4bit`) is **not gated** and the LoRA repo
`Guido/cartoon_lora` is public, so a plain read token is enough — you do not
need to click through any licence for the default configuration.

If you switch `BASE_MODEL` to `black-forest-labs/FLUX.2-dev`, that repo *is*
gated (`gated: auto`): open it while logged in and press **"Agree and access
repository"** once, then any token from that account can download it. A worker
whose token has not cleared the gate fails at load with `GatedRepoError` / 403.

Verify a token before deploying:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -I -L \
  "https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit/resolve/main/vae/diffusion_pytorch_model.safetensors" \
  -H "Authorization: Bearer $HF_TOKEN"
```

`200` means the token can pull the weights; `401`/`403` means it cannot.

## Repo layout

| Path | Purpose |
|---|---|
| [handler.py](handler.py) | The serverless worker: loads the pipeline once, serves requests |
| [Dockerfile](Dockerfile) | CUDA PyTorch base + diffusers from git (FLUX.2 support) |
| [requirements.txt](requirements.txt) | Python deps (torch comes from the base image) |
| [.github/workflows/build-and-push.yml](.github/workflows/build-and-push.yml) | Builds and pushes the image to GHCR |
| [scripts/create_template.py](scripts/create_template.py) | Creates/updates the Runpod template |
| [scripts/create_endpoint.py](scripts/create_endpoint.py) | Creates/updates the scale-to-zero endpoint |
| [scripts/call_endpoint.py](scripts/call_endpoint.py) | Calls the endpoint and saves the PNG |
| [scripts/volume.py](scripts/volume.py) | Creates/lists/deletes the weight-cache network volume |
| [scripts/teardown.py](scripts/teardown.py) | Shows what is billing, and deletes it |
| [.env.example](.env.example) | Template for your local secrets and config |

## Setup

```bash
cp .env.example .env
# then fill in RUNPOD_API_KEY, HF_TOKEN and IMAGE
```

`.env` is gitignored. Nothing in this repo reads a hardcoded secret.

### 1. Build and push to GHCR

Push to `main`, or trigger the workflow manually:

```bash
gh workflow run build-and-push.yml -f tag=0.1
```

The workflow authenticates with the built-in `GITHUB_TOKEN` (it declares
`packages: write`), so no PAT is needed for CI.

A **push to main** publishes two tags: `latest` and `sha-<short-commit>`. The
`0.1`-style tag only appears on a manual `workflow_dispatch` run, since
`inputs.tag` is empty on a push event. **Point `IMAGE` at the `sha-` tag**, not
`latest` — Runpod caches images on its workers, so a mutable tag can leave you
with workers running different code.

**Make the package accessible to Runpod.** A package published from a public
repo is public by default, which is what Runpod needs. Verify with an anonymous
pull — no credentials at all:

```bash
IMG=<owner>/flux2-runpod-worker
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:$IMG:pull&service=ghcr.io" | jq -r .token)
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" \
  "https://ghcr.io/v2/$IMG/manifests/latest"
```

`200` means Runpod can pull it. If you get `401`/`403`, open
`https://github.com/users/<owner>/packages/container/flux2-runpod-worker/settings`
and set visibility to **public**, or add a GHCR pull credential to the template.

To build locally instead:

```bash
docker build -t ghcr.io/<owner>/flux2-runpod-worker:0.1 .
echo $GHCR_TOKEN | docker login ghcr.io -u <owner> --password-stdin
docker push ghcr.io/<owner>/flux2-runpod-worker:0.1
```

### 2. Create the template

```bash
python scripts/create_template.py
```

Reads `IMAGE`, `BASE_MODEL`, `LORA_REPO`, `LORA_WEIGHT_NAME`, `DTYPE`,
`QUANTIZE`, `OFFLOAD` (and `HF_TOKEN` if set) from `.env`, creates a serverless
`NVIDIA` template, and writes `RUNPOD_TEMPLATE_ID` back into `.env`. Rerunning
updates the existing template rather than creating a duplicate.

### 3. (Optional but recommended) Create the weight cache volume

```bash
python scripts/volume.py create --size-gb 60 --data-center EU-RO-1
```

Without this, **every cold start re-downloads 34GB** — slow, and you pay GPU
time while it downloads. With it, `HF_HOME` points at `/runpod-volume/huggingface`
(already set in the Dockerfile) and the weights persist. Writes
`NETWORK_VOLUME_ID` into `.env`, which the endpoint script picks up.

**This is the one thing that bills while nothing is running** — see
[Costs and how to stop billing](#costs-and-how-to-stop-billing) below.

The volume pins the endpoint to its data center, so pick one that actually has
48GB GPUs. Check with `python scripts/create_endpoint.py --list-gpus` first.

### 4. Create the endpoint

```bash
python scripts/create_endpoint.py --list-gpus   # see which ≥48GB GPUs you can get
python scripts/create_endpoint.py
```

Defaults: `workersMin=0`, `workersMax=1`, `idleTimeout=5`, `gpuCount=1`,
`scalerType=QUEUE_DELAY`, FlashBoot on. Writes `RUNPOD_ENDPOINT_ID` into `.env`.

Lower `idleTimeout` is cheaper but means more cold starts; raise it to 15–30 if
you are sending bursts of requests.

## Calling the endpoint

`POST https://api.runpod.ai/v2/{endpoint_id}/runsync` with
`Authorization: Bearer $RUNPOD_API_KEY`.

Use `/runsync` for a blocking call (fine once the worker is warm) and `/run` +
`/status/{job_id}` for anything longer — **a cold start downloads tens of GB of
weights and will exceed the `/runsync` window**, so make your first call async.

### Request body

```json
{
  "input": {
    "prompt": "a cartoon fox riding a skateboard, bold outlines",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 3.5,
    "seed": 123,
    "init_image": null
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | **Required.** |
| `negative_prompt` | string | `""` | Omitted from the pipeline call when empty |
| `width` / `height` | int | 1024 | Rounded down to a multiple of 16, clamped to `MAX_SIDE` (1536) |
| `steps` | int | 30 | Also accepted as `num_inference_steps` |
| `guidance_scale` | float | 3.5 | |
| `seed` | int | `null` | `null` means a random result each call |
| `num_images` | int | 1 | 1–4 |
| `lora_scale` | float | `LORA_SCALE` env (1.0) | LoRA strength for this request |
| `init_image` | string | `null` | Reference image — see below |
| `init_images` | string[] | `null` | Multiple reference images (FLUX.2 accepts several) |

### Providing an image in the request

`init_image` accepts **three** forms, and the worker figures out which it got:

**1. A public http(s) URL** — the worker downloads it (capped by `MAX_IMAGE_BYTES`,
32MB, and `DOWNLOAD_TIMEOUT`, 30s):

```json
{ "input": { "prompt": "make it watercolor", "init_image": "https://example.com/ref.png" } }
```

**2. A data URI** — the `data:image/png;base64,` prefix is stripped for you:

```json
{ "input": { "prompt": "make it watercolor", "init_image": "data:image/png;base64,iVBORw0KG..." } }
```

**3. A bare base64 string** — the most common choice for local files:

```bash
python - <<'PY'
import base64, json, os, requests
b64 = base64.b64encode(open("ref.png","rb").read()).decode()
r = requests.post(
    f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}/runsync",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
    json={"input": {"prompt": "cartoon version of this photo", "init_image": b64,
                    "width": 1024, "height": 1024, "steps": 30}},
    timeout=600,
)
out = r.json()["output"]
open("output.png","wb").write(base64.b64decode(out["image_base64"]))
PY
```

For several reference images, pass `init_images` as a list — each entry may
independently be a URL, data URI, or base64 string.

The Runpod request body has a **10MB limit**, so anything bigger than a few
megapixels should be uploaded somewhere and passed as a URL instead of base64.

### curl

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a cartoon fox on a skateboard","width":1024,"height":1024,"steps":30,"guidance_scale":3.5,"seed":123}}'
```

Async (recommended for the first, cold call):

```bash
JOB=$(curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a cartoon fox on a skateboard"}}' | jq -r .id)

curl -s "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq .status
```

### Helper script

```bash
python scripts/call_endpoint.py "a cartoon fox riding a skateboard" --seed 123
python scripts/call_endpoint.py "cartoon version of this photo" --init-image ./ref.png --async
python scripts/call_endpoint.py "restyle this" --init-image https://example.com/ref.png
```

Local paths are base64-encoded for you; URLs are passed through. Output lands in
`out/output.png` unless you pass `--out`.

### Response

```json
{
  "id": "…",
  "status": "COMPLETED",
  "delayTime": 412,
  "executionTime": 18734,
  "output": {
    "image_base64": "iVBORw0KGgo…",
    "images_base64": ["iVBORw0KGgo…"],
    "seed": 123,
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 3.5,
    "conditioned_on_images": 0,
    "time_seconds": 18.7
  }
}
```

On failure the job still returns `COMPLETED` with `output.error` set to
`"<ExceptionType>: <message>"`; the full traceback is in the Runpod worker logs.

## Worker environment variables

Set on the Runpod template by `scripts/create_template.py`.

| Variable | Default | Purpose |
|---|---|---|
| `BASE_MODEL` | `diffusers/FLUX.2-dev-bnb-4bit` | Base checkpoint (NF4 mirror by default) |
| `LORA_REPO` | `Guido/cartoon_lora` | LoRA repo; empty disables LoRA loading |
| `LORA_WEIGHT_NAME` | `my_first_lora_v2.safetensors` | File within the LoRA repo |
| `LORA_SCALE` | `1.0` | Default LoRA strength; override per request with `lora_scale` |
| `LORA_ADAPTER_NAME` | `cartoon` | PEFT adapter name used by `set_adapters` |
| `REF_MAX_SIDE` | `1024` | Reference images are thumbnailed to fit this |
| `DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `QUANTIZE` | `none` | On-load quantization: `4bit`, `8bit`, `none`. Leave `none` with the prequantized default |
| `OFFLOAD` | `1` | `enable_model_cpu_offload()`; `0` puts everything on the GPU |
| `HF_TOKEN` | unset | Read token; not gate-restricted for the default repos |
| `MAX_SIDE` | `1536` | Upper bound on requested width/height |
| `MAX_IMAGE_BYTES` | `33554432` | Cap on a downloaded/decoded `init_image` |
| `PRELOAD_ON_BOOT` | `1` | Load the model during startup, not on the first job |

## Costs and how to stop billing

Three resources, and only one of them charges you when idle:

| Resource | Bills when idle? | How to stop it |
|---|---|---|
| **Endpoint** | **No** — `workersMin=0` means zero workers, zero cost | Nothing to do; delete only if you want it gone |
| **Template** | No — it is just configuration | Nothing to do |
| **Network volume** | **Yes** — provisioned size, billed hourly, running or not | Delete the volume |

At Runpod's published ~$0.07/GB/month, a 60GB cache volume is roughly
**$4/month** whether or not you generate a single image.

Check what you currently have:

```bash
python scripts/teardown.py --status
```

Stop the recurring storage charge:

```bash
python scripts/teardown.py --volume       # deletes the volume, keeps the endpoint
```

Tear everything down:

```bash
python scripts/teardown.py --all
```

Both prompt for confirmation unless you pass `--yes`. Deleting the volume does
not break the endpoint — cold starts simply go back to re-downloading the
weights each time.

**The trade-off:** keeping the volume costs ~$4/month and makes cold starts
minutes faster. Deleting it costs nothing at rest, but you pay GPU seconds for a
34GB download on every cold start. If you generate more than a handful of images
a month, the volume is cheaper overall. If you are experimenting in bursts and
then leaving it for weeks, delete it between sessions.

## Cold starts

Measured on this endpoint (A40-class 48GB, EU-RO-1):

| | queue delay | worker time |
|---|---|---|
| First ever call (pull 4.5GB image + download 34GB of weights) | **147 s** | 71 s |
| Later cold start, weights already on the volume | **3.9 s** | 96 s |

That 147s → 3.9s difference is the entire case for the network volume. The
first call must be async (`/run` + `/status`); once the volume is warm,
`/runsync` is fine.

To reduce cold starts further:

- Attach the **network volume** above so later cold starts only pay load time.
- Keep **FlashBoot** on (it is, by default).
- Raise `idleTimeout` so a warm worker survives between requests. `5` is the
  cheapest and the most cold-start-prone; `30`–`60` is kinder during a session.
- `PRELOAD_ON_BOOT=1` (default) loads the model during worker startup rather
  than inside your first job.

## Notes

- `Flux2Pipeline` is newer than the latest diffusers release, so the Dockerfile
  installs diffusers from git (`DIFFUSERS_REF`, default `main`). Once a build
  works, pin `DIFFUSERS_REF` to that commit SHA for reproducibility.
- LoRA is attached as a named PEFT adapter (`LORA_ADAPTER_NAME`), so per-request
  `lora_scale` works via `set_adapters` and applies correctly on top of the NF4
  weights. This is why `peft` is in `requirements.txt`.
- `torch` is intentionally absent from `requirements.txt` — it ships with the
  base image built against CUDA 12.8, and installing it from PyPI would replace
  it with a different build.
- Rerunning either create script updates the existing template/endpoint by name
  instead of creating duplicates.

## Deployed instance

Live resources created by this repo (ids also in `.env`, which is gitignored):

| Resource | Id |
|---|---|
| Template | `ctl7j0wlh7` |
| Endpoint | `wqa9wrxf1chw26` |
| Network volume | `jocfg4z26l` (60GB, EU-RO-1) |
| Image | `ghcr.io/guidosalimbeni/flux2-runpod-worker:sha-b3dc1d9` |

```bash
python scripts/call_endpoint.py "lora-cartoon, ..." --seed 42
python scripts/teardown.py --status
```
