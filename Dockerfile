FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

# Setup git with GITHUB_TOKEN.
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*
ARG GITHUB_TOKEN
RUN git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /olmoearth_projects

COPY pyproject.toml /olmoearth_projects/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --all-extras --no-install-project

ENV PATH="/olmoearth_projects/.venv/bin:$PATH"
COPY ./ /olmoearth_projects
RUN uv sync --all-extras --locked
