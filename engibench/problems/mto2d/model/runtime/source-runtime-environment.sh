#!/usr/bin/env bash

# Static equivalent of the OpenFOAM 5.x bashrc for the pinned source image.
# Avoiding the interactive shell setup removes roughly two seconds from every
# amd64-emulated container start. Keep this file image-specific: the legacy
# SIF-derived oracle falls back to its original bashrc in docker-entrypoint.sh.

export WM_PROJECT_INST_DIR=/opt/OpenFOAM
export WM_PROJECT_DIR=/opt/OpenFOAM/OpenFOAM-5.x
export WM_PROJECT=OpenFOAM
export WM_PROJECT_VERSION=5.x
export WM_PROJECT_USER_DIR=/opt/mto2d/user
export WM_THIRD_PARTY_DIR=/opt/OpenFOAM/ThirdParty-5.x
export WM_DIR="$WM_PROJECT_DIR/wmake"

export WM_ARCH=linux64
export WM_ARCH_OPTION=64
export WM_OSTYPE=POSIX
export WM_COMPILER=Gcc
export WM_COMPILER_TYPE=system
export WM_COMPILE_OPTION=Opt
export WM_PRECISION_OPTION=DP
export WM_LABEL_OPTION=Int32
export WM_LABEL_SIZE=32
export WM_MPLIB=SYSTEMOPENMPI
export WM_OPTIONS=linux64GccDPInt32Opt
export WM_CC=gcc
export WM_CXX=g++
export WM_CFLAGS="-m64 -fPIC"
export WM_CXXFLAGS="-m64 -fPIC -std=c++0x"
export WM_LDFLAGS=-m64
export WM_LINK_LANGUAGE=c++

export FOAM_INST_DIR="$WM_PROJECT_INST_DIR"
export FOAM_APP="$WM_PROJECT_DIR/applications"
export FOAM_SRC="$WM_PROJECT_DIR/src"
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export FOAM_SOLVERS="$FOAM_APP/solvers"
export FOAM_UTILITIES="$FOAM_APP/utilities"
export FOAM_TUTORIALS="$WM_PROJECT_DIR/tutorials"
export FOAM_JOB_DIR="$FOAM_INST_DIR/jobControl"
export FOAM_RUN="$WM_PROJECT_USER_DIR/run"
export FOAM_APPBIN="$WM_PROJECT_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_LIBBIN="$WM_PROJECT_DIR/platforms/$WM_OPTIONS/lib"
export FOAM_SITE_APPBIN="$FOAM_INST_DIR/site/$WM_PROJECT_VERSION/platforms/$WM_OPTIONS/bin"
export FOAM_SITE_LIBBIN="$FOAM_INST_DIR/site/$WM_PROJECT_VERSION/platforms/$WM_OPTIONS/lib"
export FOAM_EXT_LIBBIN="$WM_THIRD_PARTY_DIR/platforms/linux64GccDPInt32/lib"
export FOAM_MPI=openmpi-system
export MPI_ARCH_PATH="$MPI_DIR"
export MPI_BUFFER_SIZE=20000000

export PATH="$FOAM_SITE_APPBIN:$FOAM_APPBIN:$WM_PROJECT_DIR/bin:$WM_DIR:$MPI_DIR/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$FOAM_LIBBIN/$FOAM_MPI:$FOAM_EXT_LIBBIN/$FOAM_MPI:$MPI_DIR/lib:$FOAM_SITE_LIBBIN:$FOAM_LIBBIN:$FOAM_EXT_LIBBIN:$FOAM_LIBBIN/dummy:${LD_LIBRARY_PATH:-}"
