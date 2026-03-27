## syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY setup.py requirements.txt /workspace/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install pytest black ruff mypy

COPY . /workspace

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-deps

CMD ["biofuse", "smoke", "--preset", "custom", "--device", "cuda"]
