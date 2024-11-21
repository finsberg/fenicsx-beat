# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/lab:v0.9.0

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[demos]"
