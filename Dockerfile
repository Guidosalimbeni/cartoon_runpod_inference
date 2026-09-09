# CUDA-enabled PyTorch base: torch/torchvision are preinstalled against CUDA 12.8,
# so requirements.txt deliberately does not list torch (installing it from PyPI
# would swap in a different CUDA build).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt hf_transfer

# FLUX.2 (Flux2Pipeline) is newer than the latest diffusers release, so install
# from git. Pin to a commit for reproducible builds once a build is known good.
ARG DIFFUSERS_REF=main
RUN pip install "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REF}"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
