"""Runpod serverless worker: FLUX.2-dev + a Diffusers LoRA.

Cold start loads the pipeline once into module state; every subsequent
invocation reuses it. Configuration is entirely environment driven so the same
image can serve a different base model / LoRA by editing the Runpod template.
"""

import base64
import io
import os
import time
import traceback
from typing import Any, Optional

import requests
import runpod
import torch
from PIL import Image

# Defaults to the NF4-prequantized mirror: a complete Flux2Pipeline repo whose
# DiT and text encoder are already 4-bit (~34GB to download, vs ~113GB for the
# bf16 original, which cannot fit on a 48GB GPU at any offload setting).
BASE_MODEL = os.getenv("BASE_MODEL", "diffusers/FLUX.2-dev-bnb-4bit")
LORA_REPO = os.getenv("LORA_REPO", "Guido/cartoon_lora")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "my_first_lora_v2.safetensors")
LORA_ADAPTER_NAME = os.getenv("LORA_ADAPTER_NAME", "cartoon")
LORA_SCALE = float(os.getenv("LORA_SCALE", "1.0"))
DTYPE = os.getenv("DTYPE", "bfloat16")

# Only needed when BASE_MODEL points at an unquantized repo: quantize on load.
# With the prequantized default this stays "none" — the weights arrive NF4.
QUANTIZE = os.getenv("QUANTIZE", "none").lower()

# enable_model_cpu_offload() keeps peak VRAM well under 48GB by holding only the
# active submodule on the GPU. Set OFFLOAD=0 for more speed if it fits.
OFFLOAD = os.getenv("OFFLOAD", "1") not in ("0", "false", "False", "")

MAX_SIDE = int(os.getenv("MAX_SIDE", "1536"))
REF_MAX_SIDE = int(os.getenv("REF_MAX_SIDE", "1024"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(32 * 1024 * 1024)))

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

PIPE = None


def _torch_dtype() -> torch.dtype:
    if DTYPE not in _DTYPES:
        raise ValueError(f"Unsupported DTYPE={DTYPE!r}; choose one of {sorted(_DTYPES)}")
    return _DTYPES[DTYPE]


def _quant_config(dtype):
    """On-load BitsAndBytes config, or None when the weights are already quantized."""
    if QUANTIZE in ("none", "no", "off", ""):
        return None
    from diffusers import PipelineQuantizationConfig

    if QUANTIZE in ("4bit", "8bit"):
        kwargs = (
            {"load_in_8bit": True}
            if QUANTIZE == "8bit"
            else {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": dtype,
            }
        )
        return PipelineQuantizationConfig(
            quant_backend=f"bitsandbytes_{QUANTIZE}",
            quant_kwargs=kwargs,
            components_to_quantize=["transformer", "text_encoder"],
        )
    raise ValueError(f"Unsupported QUANTIZE={QUANTIZE!r}; use none, 4bit or 8bit")


def load_pipeline():
    """Build the FLUX.2 pipeline and attach the LoRA. Called once per worker."""
    global PIPE
    if PIPE is not None:
        return PIPE

    started = time.time()
    dtype = _torch_dtype()

    try:
        from diffusers import Flux2Pipeline
    except ImportError as exc:  # pragma: no cover - depends on installed version
        import diffusers

        raise RuntimeError(
            f"Flux2Pipeline is not available in diffusers {diffusers.__version__}. "
            "Rebuild the image with a diffusers revision that ships FLUX.2 support "
            "(see DIFFUSERS_REF in the Dockerfile)."
        ) from exc

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    quant = _quant_config(dtype)
    if quant is not None:
        kwargs["quantization_config"] = quant

    # The default NF4 repo is public and ungated; a token is only needed if you
    # point BASE_MODEL at the gated black-forest-labs/FLUX.2-dev or a private repo.
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token

    print(f"[boot] loading {BASE_MODEL} dtype={DTYPE} quantize={QUANTIZE} "
          f"offload={OFFLOAD}", flush=True)
    pipe = Flux2Pipeline.from_pretrained(BASE_MODEL, **kwargs)

    if LORA_REPO:
        print(f"[boot] loading LoRA {LORA_REPO}/{LORA_WEIGHT_NAME}", flush=True)
        lora_kwargs: dict[str, Any] = {"adapter_name": LORA_ADAPTER_NAME}
        if LORA_WEIGHT_NAME:
            lora_kwargs["weight_name"] = LORA_WEIGHT_NAME
        if token:
            lora_kwargs["token"] = token
        pipe.load_lora_weights(LORA_REPO, **lora_kwargs)
        pipe.set_adapters(LORA_ADAPTER_NAME, LORA_SCALE)

    if OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    PIPE = pipe
    print(f"[boot] ready in {time.time() - started:.1f}s", flush=True)
    return PIPE


def _decode_image(spec: Any) -> Image.Image:
    """Accept an http(s) URL, a data: URI, or a raw base64 PNG/JPEG string."""
    if isinstance(spec, Image.Image):
        image = spec.convert("RGB")
        image.thumbnail((REF_MAX_SIDE, REF_MAX_SIDE))
        return image
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("init_image must be a URL, data URI, or base64 string")

    spec = spec.strip()
    if spec.startswith("http://") or spec.startswith("https://"):
        resp = requests.get(spec, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        raw = b""
        for chunk in resp.iter_content(64 * 1024):
            raw += chunk
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError(f"init_image exceeds MAX_IMAGE_BYTES={MAX_IMAGE_BYTES}")
    else:
        if spec.startswith("data:"):
            _, _, spec = spec.partition(",")
        try:
            raw = base64.b64decode(spec, validate=True)
        except Exception as exc:
            raise ValueError(f"init_image is not valid base64: {exc}") from exc
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"init_image exceeds MAX_IMAGE_BYTES={MAX_IMAGE_BYTES}")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((REF_MAX_SIDE, REF_MAX_SIDE))
    return image


def _collect_images(payload: dict) -> list[Image.Image]:
    """FLUX.2 conditions on a list of reference images; accept one or many."""
    spec = payload.get("init_image")
    if spec is None:
        spec = payload.get("image")
    extra = payload.get("init_images") or payload.get("images")

    specs = []
    if spec is not None:
        specs.append(spec)
    if extra:
        if not isinstance(extra, list):
            raise ValueError("init_images must be a list")
        specs.extend(extra)
    return [_decode_image(s) for s in specs]


def _round_to_16(value: int) -> int:
    return max(256, min(MAX_SIDE, (int(value) // 16) * 16))


def _as_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handler(job: dict) -> dict:
    payload = job.get("input") or {}

    prompt = payload.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "`prompt` is required"}

    try:
        pipe = load_pipeline()

        width = _round_to_16(payload.get("width", 1024))
        height = _round_to_16(payload.get("height", 1024))
        steps = int(payload.get("steps", payload.get("num_inference_steps", 30)))
        guidance = float(payload.get("guidance_scale", 3.5))
        num_images = max(1, min(4, int(payload.get("num_images", 1))))
        seed = payload.get("seed")

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        call_kwargs: dict[str, Any] = {
            "prompt": str(prompt),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": num_images,
            "generator": generator,
        }

        negative = payload.get("negative_prompt")
        if negative:
            call_kwargs["negative_prompt"] = str(negative)

        images = _collect_images(payload)
        if images:
            # Image-conditioned path: FLUX.2 takes reference images directly.
            call_kwargs["image"] = images

        lora_scale = float(payload.get("lora_scale", LORA_SCALE))
        if LORA_REPO:
            pipe.set_adapters(LORA_ADAPTER_NAME, lora_scale)

        started = time.time()
        result = pipe(**call_kwargs)
        elapsed = time.time() - started

        encoded = [_as_png_b64(img) for img in result.images]
        return {
            "image_base64": encoded[0],
            "images_base64": encoded,
            "seed": int(seed) if seed is not None else None,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "conditioned_on_images": len(images),
            "lora_scale": lora_scale if LORA_REPO else None,
            "time_seconds": round(elapsed, 2),
        }
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if os.getenv("PRELOAD_ON_BOOT", "1") not in ("0", "false", "False", ""):
    # Pay the model load during worker startup rather than inside the first job.
    try:
        load_pipeline()
    except Exception:
        traceback.print_exc()
        print("[boot] preload failed; will retry on first request", flush=True)

runpod.serverless.start({"handler": handler})
