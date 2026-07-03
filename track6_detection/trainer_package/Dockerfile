# Trainer-package Dockerfile for AI City Challenge 2026 Track 6, targeting
# the Hafnia Training-as-a-Service platform (Milestone Systems).
#
# Structure confirmed against the public GitHub repos
# github.com/milestone-hafnia/{hafnia,trainer-object-detection,trainer-classification}:
# Hafnia's own reference trainers use a `debian:bookworm-slim` base image
# with the `uv` package manager (not a CUDA base image -- GPU support
# comes from the PyTorch wheels themselves, and Hafnia's runner supplies
# the GPU driver/runtime at the infrastructure level).
#
# STILL UNCONFIRMED: whether `hafnia runc build` on the platform expects
# this exact base image or tolerates substitutes; whether there is a
# required ENTRYPOINT/CMD convention beyond "invoke scripts/train.py or
# scripts/benchmark.py with the args the dashboard collects via the
# paired *.schema.json". Verify with `hafnia runc build-local` /
# `hafnia runc launch-local` once logged in (see README.md).
#
# Keep this image lean: trainer package upload is capped at 2GB.

FROM debian:bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock* .python-version ./
RUN uv sync --no-cache --locked || uv sync --no-cache

COPY . .

# Hafnia's dashboard is expected to invoke scripts/train.py (training
# experiments) or scripts/benchmark.py (benchmarking jobs) with CLI flags
# derived from the paired *.schema.json files. This ENTRYPOINT defaults to
# training; benchmarking jobs override the CMD/args at launch time
# (confirm exact invocation convention via `hafnia runc launch-local`).
ENTRYPOINT ["uv", "run", "python3", "scripts/train.py"]
CMD []
