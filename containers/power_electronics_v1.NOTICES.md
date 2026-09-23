# PowerElectronics v1 runtime notices

The PowerElectronics v1 SIF is a multi-license software aggregate. The
EngiBench repository license and the dataset license do not replace the
licenses of software installed in the image.

The image contains Debian 12 (bookworm) packages, Python packages installed
from PyPI, and ngspice 44.2. Copyright and license notices supplied by Debian
packages are available inside the image under
`/usr/share/doc/<package>/copyright`. Python package metadata and license files
are retained in their installed `.dist-info` directories. The accompanying
SPDX SBOM is an inventory aid; the installed notices remain authoritative.
The publication bundle preserves those notices and records the exact Debian
binary and source package versions in `debian-source-packages.tsv`. Debian
source packages remain available from
[`sources.debian.org`](https://sources.debian.org/) and
[`snapshot.debian.org`](https://snapshot.debian.org/).

The ngspice 44.2 `COPYING` file describes the main license as Modified BSD and
lists exceptions under LGPL, MPL, GPL, MIT-compatible, and public-domain
terms. The exact unmodified ngspice 44.2 source archive, including that
complete `COPYING` file, is distributed with the runtime publication bundle as
`ngspice-44.2.tar.gz`:

```text
e7dadfb7bd5474fd22409c1e5a67acdec19f77e597df68e17c5549bc1390d7fd  ngspice-44.2.tar.gz
```

The SIF was used as a simulation runtime. EngiBench source code and the
generated PowerElectronics datasets are not copied into it; EngiBench was
mounted into the container when the dataset was generated.

The SIF is provided without warranty. Recipients remain responsible for
reviewing the license terms of the components they use or redistribute.
