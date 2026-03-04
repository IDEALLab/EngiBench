"""
HuggingFace dataset loader for the Wings3D problem.
"""

from __future__ import annotations

from datasets import DatasetDict, load_dataset


def load_wings3d_dataset(dataset_id: str) -> DatasetDict:
    if not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")

    try:
        return load_dataset(dataset_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Hugging Face dataset '{dataset_id}'.\n"
            f"- If it hasn't been uploaded yet, this is expected.\n"
            f"- If it's private, run: huggingface-cli login\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e