# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Make the local label_studio_ml package importable without running setup.py
ENV PYTHONPATH=/workspace


# Update base OS and install git (needed by label-studio-sdk git dependency)
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y git; \
    apt-get autoremove -y

# ── Install root package dependencies (Flask, SDK, etc.) ──────────────────────
# Copy only the files needed to install the root package first (better caching)
COPY requirements.txt /tmp/ls_requirements.txt
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r /tmp/ls_requirements.txt

# ── Copy the local label_studio_ml source to /workspace ───────────────────────
# We set PYTHONPATH=/workspace so `import label_studio_ml` resolves here.
# This avoids running setup.py which tries to `import label_studio_ml` itself.
COPY label_studio_ml/ /workspace/label_studio_ml/

# ── Install example-specific requirements ─────────────────────────────────────
COPY label_studio_ml/examples/huggingface_annotation/requirements.txt ./example_requirements.txt
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r ./example_requirements.txt

# ── Copy example source files into /app ──────────────────────────────────────
COPY label_studio_ml/examples/huggingface_annotation/ /app/

EXPOSE ${PORT:-9090}

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
