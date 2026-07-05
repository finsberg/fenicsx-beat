# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/lab:stable

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[demos]"
