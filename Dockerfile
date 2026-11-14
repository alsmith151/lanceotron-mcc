# syntax=docker/dockerfile:1.7
# Runtime image for mcc-peaks, with the plot extra by default. The compiled
# Rust kernels are built in the builder stage; the runtime needs no toolchain.
# Without plotting:
#   docker build --build-arg EXTRAS="" -t mcc-peaks:slim .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

ARG EXTRAS="plot"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# C linker and Rust toolchain for the native extension; builder only.
ARG RUST_VERSION=1.95.0
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"

WORKDIR /src

COPY . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-default-groups \
        $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)


FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/alsmith151/lanceotron-mcc" \
      org.opencontainers.image.description="Distance-aware MCC peak calling and ligation-junction footprinting" \
      org.opencontainers.image.licenses="GPL-3.0"

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /data

# No ENTRYPOINT: workflow managers (Snakemake, Nextflow, Apptainer exec) pass full commands.
CMD ["mcc-peaks", "--help"]
