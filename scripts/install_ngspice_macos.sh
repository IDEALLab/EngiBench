#!/usr/bin/env bash

set -euo pipefail

readonly VERSION="44.2"
readonly SHA256="e7dadfb7bd5474fd22409c1e5a67acdec19f77e597df68e17c5549bc1390d7fd"
readonly URL="https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/${VERSION}/ngspice-${VERSION}.tar.gz/download"
readonly PREFIX="${1:-${HOME}/.local/ngspice-${VERSION}}"

if [[ -x "${PREFIX}/bin/ngspice" ]] && "${PREFIX}/bin/ngspice" --version 2>&1 | grep -q "ngspice-${VERSION}"; then
    echo "ngspice ${VERSION} is already installed at ${PREFIX}"
    exit 0
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

archive="${work_dir}/ngspice-${VERSION}.tar.gz"
curl --fail --location --retry 3 --output "${archive}" "${URL}"
echo "${SHA256}  ${archive}" | shasum -a 256 --check
tar -xzf "${archive}" -C "${work_dir}"

cd "${work_dir}/ngspice-${VERSION}"
CXXFLAGS="-Wno-invalid-specialization" ./configure \
    --prefix="${PREFIX}" \
    --enable-relpath \
    --without-x \
    --with-readline=no \
    --with-fftw3=no \
    --disable-openmp
make -j2 CXXFLAGS="-Wno-invalid-specialization"
make install CXXFLAGS="-Wno-invalid-specialization"

"${PREFIX}/bin/ngspice" --version
