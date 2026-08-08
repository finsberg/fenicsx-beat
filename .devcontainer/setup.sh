#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status

chsh -s $(which zsh) root || true
if ! command -v starship &> /dev/null; then
  echo "Installing Starship prompt..."
  curl -sS https://starship.rs/install.sh | sh -s -- -y
fi
python3 -m pip install pkgconfig
HDF5_MPI=ON HDF5_PKGCONFIG_NAME="hdf5" python3 -m pip install h5py --no-build-isolation --no-binary=h5py
python3 -m pip install scifem --no-build-isolation --no-binary=scifem
python3 -m pip install -e .[dev]
pre-commit install
# 3. Print success message
echo "Container setup complete!"
