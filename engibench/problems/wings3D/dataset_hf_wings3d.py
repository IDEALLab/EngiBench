"""
Dataset loader for the Wings3D problem.

Tries Hugging Face first. Optionally falls back to a local file for development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def _pandas_to_datasetdict(df: pd.DataFrame, split: str = "train") -> DatasetDict:
    # Convert numpy arrays to lists so HF Dataset can serialize them
    df2 = df.copy()
    if "coords" in df2.columns:
        df2["coords"] = df2["coords"].apply(lambda x: np.asarray(x).tolist())
    return DatasetDict({split: Dataset.from_pandas(df2, preserve_index=False)})


def load_wings3d_dataset(dataset_id: str, local_path: Optional[str] = None) -> DatasetDict:
    if not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")

    try:
        return load_dataset(dataset_id)
    except Exception as e:
        if local_path is None:
            raise RuntimeError(
                f"Could not load Hugging Face dataset '{dataset_id}'.\n"
                f"- If it hasn't been uploaded yet, this is expected.\n"
                f"- If it's private, run: huggingface-cli login\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        p = Path(local_path)
        if not p.exists():
            raise RuntimeError(
                f"HF dataset '{dataset_id}' not available AND local_path not found: {local_path}"
            ) from e

        # Load local df
        if p.suffix == ".pkl":
            df = pd.read_pickle(p)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported local dataset format: {p.suffix} (use .pkl or .parquet)")

        return _pandas_to_datasetdict(df, split="train")