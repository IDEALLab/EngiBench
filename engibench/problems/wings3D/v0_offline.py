"""
Wings3D (offline) dataset-backed problem.

One sample corresponds to one slice (selected via slice_num) of a wing.
simulate() returns cd/cl stored in the dataset row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from engibench.core import ObjectiveDirection, Problem
from engibench.problems.wings3D.dataset_hf_wings3d import load_wings3d_dataset
from engibench.problems.wings3D.utils_wing3d import get_slice_coords

DesignType = dict[str, Any]


class Wings3DOffline(Problem[DesignType]):
    version = 0
    objectives = (("cd", ObjectiveDirection.MINIMIZE), ("cl", ObjectiveDirection.MAXIMIZE))

    design_space = spaces.Dict(
        {
            "coords": spaces.Box(low=-1e6, high=1e6, shape=(192, 2), dtype=np.float32),
            "alpha": spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32),
        }
    )

    dataset_id: str = "IDEALLab/wings3d_v0"  # replace with real HF id when available

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.dataset = load_wings3d_dataset(self.dataset_id)

    def random_design(self, dataset_split: str = "train") -> tuple[DesignType, dict[str, Any], int]:
        ds = self.dataset[dataset_split]
        idx = int(self.np_random.integers(low=0, high=len(ds), dtype=int))
        row = ds[idx]

        slice_num = int(row["slice_num"])
        coords_slice = get_slice_coords(row["coords"], slice_num)
        alpha = np.asarray([row.get("alpha", 0.0)], dtype=np.float32)

        design: DesignType = {"coords": coords_slice, "alpha": alpha}
        config = {"split": dataset_split, "idx": idx}
        return design, config, idx

    def simulate(self, design: DesignType, config: dict[str, Any] | None = None, **kwargs) -> npt.NDArray[np.float64]:
        if not config or "split" not in config or "idx" not in config:
            raise ValueError("simulate() requires config with keys {'split','idx'} (use random_design()).")

        row = self.dataset[config["split"]][int(config["idx"])]
        cd = float(row.get("cd_val", np.nan))
        cl = float(row.get("cl_val", np.nan))
        return np.array([cd, cl], dtype=np.float64)

        def render(self, design: DesignType, *, open_window: bool = False, save: bool = False) -> Figure:
        coords = np.asarray(design["coords"], dtype=np.float32)
        fig, ax = plt.subplots()
        ax.plot(coords[:, 0], coords[:, 1])
        ax.axis("equal")
        ax.axis("off")
        if open_window:
            plt.show()
        plt.close(fig)
        return fig
    