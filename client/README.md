# flux2_client.py

Standalone client for the FLUX.2 + cartoon-LoRA Runpod endpoint. **Copy
`flux2_client.py` into another project — that one file is all you need.** It
imports only the Python standard library, so there is nothing to `pip install`.

## Setup in the other repo

```bash
cp flux2_client.py /path/to/other-repo/
```

Then provide two values, either as real environment variables or in a `.env`
file next to the script:

```
RUNPOD_API_KEY=rpa_...
RUNPOD_ENDPOINT_ID=wqa9wrxf1chw26
```

Add that `.env` to the other repo's `.gitignore`.

## As a library

```python
from flux2_client import generate

png = generate("lora-cartoon, a caricature bust of a smiling bald man", seed=42)
open("out.png", "wb").write(png)
```

With a reference image — a local path, raw `bytes`, or an `http(s)` URL:

```python
png = generate(
    "lora-cartoon, a caricature bust of this person, smooth grey clay ZBrush render",
    init_image="face.jpg",
    guidance_scale=2.5,   # lower guidance when conditioning on a photo
    seed=123,
)
```

`generate` returns **PNG bytes**, so it does not need Pillow. Use
`generate_many(...)` for a list when `num_images > 1`.

Show progress during a cold start:

```python
png = generate(prompt, on_status=lambda status, secs: print(status, f"{secs:.0f}s"))
```

Failures raise `Flux2Error` rather than returning something odd:

```python
from flux2_client import generate, Flux2Error

try:
    png = generate(prompt, init_image="face.jpg")
except Flux2Error as exc:
    print("generation failed:", exc)
```

## As a CLI

```bash
python flux2_client.py "lora-cartoon, ..." -o out.png --seed 42
python flux2_client.py "lora-cartoon, ..." -i face.jpg -o out.png --guidance-scale 2.5
```

## Arguments

| Argument | Default | Notes |
|---|---|---|
| `prompt` | — | Required. Lead with `lora-cartoon` — the LoRA's trigger token |
| `init_image` | `None` | Path, `bytes`, or `http(s)` URL |
| `init_images` | `None` | Several references; FLUX.2 accepts a list |
| `negative_prompt` | `""` | |
| `width` / `height` | 1024 | Rounded to a multiple of 16 by the worker, max 1536 |
| `steps` | 28 | |
| `guidance_scale` | 3.5 | Use 2.5–3 with a reference image, ~3.5–4 prompt-only |
| `seed` | `None` | `None` gives a different result each call |
| `num_images` | 1 | 1–4; use `generate_many` to get them all |
| `lora_scale` | endpoint default (1.0) | LoRA strength |
| `timeout` | 900 | Seconds before giving up |

## How images are sent

Local files and `bytes` are base64-encoded into the request body. `http(s)`
URLs are passed through as strings and fetched by the worker itself, which
keeps them out of the body — worth preferring for large references, since
Runpod caps a request at **10MB**.

## Timing

The client uses `/run` + `/status` polling rather than `/runsync`, because
`/runsync` gives up after about 90 seconds and a cold worker takes longer than
that. Expect roughly 90–280s on a cold start and a few seconds once warm.
Raising the endpoint's `idleTimeout` keeps a worker alive between calls.
