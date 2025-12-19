FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

RUN apt-get update && apt-get install -y git --no-install-recommends

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install tippecanoe, it is used by the forest loss driver inference pipeline for
# making the slippy tiles that appear in the web app.
RUN apt install -y build-essential libsqlite3-dev zlib1g-dev
RUN git clone https://github.com/mapbox/tippecanoe /opt/tippecanoe
WORKDIR /opt/tippecanoe
RUN make -j
RUN make install

WORKDIR /olmoearth_projects

COPY pyproject.toml /olmoearth_projects/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --all-extras --no-install-project

ENV PATH="/olmoearth_projects/.venv/bin:$PATH"
COPY ./ /olmoearth_projects
RUN uv sync --all-extras --locked
