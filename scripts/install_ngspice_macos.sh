#!/usr/bin/env bash

set -euo pipefail

readonly VERSION="44.2"
readonly SHA256="e7dadfb7bd5474fd22409c1e5a67acdec19f77e597df68e17c5549bc1390d7fd"
readonly URL="https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/${VERSION}/ngspice-${VERSION}.tar.gz/download"
readonly PREFIX="${1:-${HOME}/.local/ngspice-${VERSION}}"
readonly EXECUTABLE="${PREFIX}/bin/ngspice"

if [[ -x "${EXECUTABLE}" ]] \
    && lipo -archs "${EXECUTABLE}" | grep -qw "x86_64" \
    && "${EXECUTABLE}" --version 2>&1 | grep -q "ngspice-${VERSION}"; then
    echo "ngspice ${VERSION} is already installed at ${PREFIX}"
    exit 0
fi

build_command=(/usr/bin/env)
case "$(uname -m)" in
    arm64)
        if ! arch -x86_64 /usr/bin/true 2>/dev/null; then
            echo "Rosetta 2 is required to build the validated x86_64 ngspice binary." >&2
            echo "Install it with: softwareupdate --install-rosetta --agree-to-license" >&2
            exit 1
        fi
        build_command=(arch -x86_64)
        ;;
    x86_64) ;;
    *)
        echo "Unsupported macOS architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

archive="${work_dir}/ngspice-${VERSION}.tar.gz"
curl --fail --location --retry 3 --output "${archive}" "${URL}"
echo "${SHA256}  ${archive}" | shasum -a 256 --check
tar -xzf "${archive}" -C "${work_dir}"

cd "${work_dir}/ngspice-${VERSION}"
CXXFLAGS="-Wno-invalid-specialization" "${build_command[@]}" ./configure \
    --prefix="${PREFIX}" \
    --enable-relpath \
    --without-x \
    --with-readline=no \
    --with-fftw3=no \
    --disable-openmp
"${build_command[@]}" make -j2 CXXFLAGS="-Wno-invalid-specialization"
"${build_command[@]}" make install CXXFLAGS="-Wno-invalid-specialization"

if ! lipo -archs "${EXECUTABLE}" | grep -qw "x86_64"; then
    echo "Expected an x86_64 ngspice executable at ${EXECUTABLE}." >&2
    exit 1
fi

"${EXECUTABLE}" --version
