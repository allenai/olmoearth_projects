FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy SSH config to the Docker image.
# See https://github.com/webfactory/ssh-agent?tab=readme-ov-file#using-multiple-deploy-keys-inside-docker-builds
COPY root-config /root/
RUN sed 's|/home/runner|/root|g' -i.bak /root/.ssh/config

WORKDIR /olmoearth_projects

COPY pyproject.toml /olmoearth_projects/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --extra extra --extra dev --no-install-project

ENV PATH="/olmoearth_projects/.venv/bin:$PATH"
COPY ./ /olmoearth_projects
RUN uv sync --extra extra --extra dev --locked
