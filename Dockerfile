# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.8.0

ENV PYVISTA_JUPYTER_BACKEND="html"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs

COPY . /repo
WORKDIR /repo

ARG TARGETPLATFORM
RUN echo "Building for $TARGETPLATFORM"
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl"; fi

RUN python3 -m pip install ".[demos]"
