#!/bin/bash
set -e

export MPI_DIR=/opt/openmpi-4.0.4
export PETSC_DIR=/opt/petsc
export PETSC_ARCH=arch-linux2-c-opt
export PATH="${MPI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="/opt:${PETSC_DIR}/${PETSC_ARCH}/lib:${MPI_DIR}/lib:${LD_LIBRARY_PATH:-}"

# The SIF injected these settings at launch time. Docker imports only its
# root filesystem, so recreate the runtime environment explicitly. OpenFOAM's
# MPI setup calls mpicc while it is being sourced, hence MPI must be on PATH
# first.
source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc

export PATH="${MPI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="/opt:${PETSC_DIR}/${PETSC_ARCH}/lib:${MPI_DIR}/lib:${LD_LIBRARY_PATH:-}"

exec "$@"
