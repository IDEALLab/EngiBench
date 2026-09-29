# PowerElectronics v1 runtime publication

The canonical Linux x86_64 runtime is published as a SIF artifact in GitHub
Container Registry:

```text
ghcr.io/ideallab/engibench-power-electronics-v1:v1.0.0
```

For an immutable pull, use its OCI manifest digest:

```bash
apptainer pull power-electronics-v1.sif \
  oras://ghcr.io/ideallab/engibench-power-electronics-v1@sha256:3377ea0e2315e9e83a1f042df1a9f9852bf7098a404ed0adc85c8b01c9a3ca85
```

Verify the downloaded SIF itself before use:

```bash
echo "40816f203b7e1c68ae37f4d9353bd302486d77988b98733c021f1ff71f48ae02  power-electronics-v1.sif" \
  | sha256sum --check
```

The OCI manifest digest identifies the registry manifest; the SIF SHA-256
identifies the exact file used to generate the dataset. The `v1.0.0` tag is a
convenience reference, while the digest-pinned form is the reproducible input.

The registry artifact has two OCI referrers: an SPDX SBOM and a provenance
bundle containing the definition file, checksums, package manifests, installed
license metadata, and the exact ngspice 44.2 source archive. See
[`power_electronics_v1.NOTICES.md`](power_electronics_v1.NOTICES.md) for the
mixed-license disclosure.

The corresponding dataset is
[`IDEALLab/power_electronics_v1`](https://huggingface.co/datasets/IDEALLab/power_electronics_v1)
at tag `v1.0.0` and commit
`eefad7d727ea1e5bef5e1b7088dea20a9b0cd67f`.
