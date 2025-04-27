"""Set up the configuration for the Power Electronics problem."""
# ruff: noqa: N806, N815 # Upper case

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import os


def norm_join(dir1: str, dir2: str) -> str:
    """Join two paths and then normalize the result."""
    return os.path.normpath(os.path.join(dir1, dir2))


@dataclass
class Config:
    """Configuration for the Power Electronics problem.

    Accepts:
        - target_dir: Optional. Default to os.getcwd(). The user can manually change this.
                      All the files (except the original_netlist which already exists) will be saved in this directory,
                        including the rewritten netlist, log and raw files.
        - original_netlist_path: Optional. Default to "./data/netlist/5_4_3_6_10-dcdc_converter_1.net".

    Do not assign:
        - mode: Default to "control". That's the only mode supported in this version now.

    Does not receive due to init=False:
        - netlist_dir, raw_file_dir, log_file_dir
        - netlist_name
        - log_file_path, raw_file_path, rewrite_netlist_path
        - bucket_id, n_S, n_D, n_L, n_C
        - capacitor_val, inductor_val, switch_T1, switch_T2, switch_L1, switch_L2
    """

    target_dir: str = field(default_factory=str)  # all the other files will be saved in this directory.

    # These will be set from target_dir in __post_init__().
    netlist_dir: str = field(init=False)
    raw_file_dir: str = field(init=False)
    log_file_dir: str = field(init=False)

    # Accepts both absolute and relative paths.
    original_netlist_path: str = "./data/netlist/5_4_3_6_10-dcdc_converter_1.net"

    netlist_name: str = field(init=False)  # This will be set from original_netlist_path in __post_init__().
    mode: str = "control"  # Manually assign "mode=batch" when initializing the Config object will change log_file_path etc.

    log_file_path: str = field(init=False)  # depends on log_file_dir and netlist_name
    raw_file_path: str = field(init=False)  # depends on raw_file_dir and netlist_name
    rewrite_netlist_path: str = field(init=False)  # depends on netlist_dir, mode and netlist_name

    bucket_id: str = field(init=False)  # This will be retrevied from netlist_name. E.g. "5_4_3_6_10".
    n_S: int = field(init=False)
    n_D: int = field(init=False)
    n_L: int = field(init=False)
    n_C: int = field(init=False)

    # components of the design variable
    capacitor_val: list[float] = field(init=False)  # range: [1e-6, 2e-5]. This will be set in process_sweep_data.py.
    inductor_val: list[float] = field(init=False)  # range: [1e-6, 1e-3]
    switch_T1: list[float] = field(init=False)  # range: [0.1, 0.9]
    switch_T2: list[float] = field(init=False)  # Constant. All 1 for now
    switch_L1: list[float] = field(init=False)  # Binary.
    switch_L2: list[float] = field(init=False)  # Binary.

    def __post_init__(self):
        """Post-initialization of the Config object."""
        assert self.target_dir, "target_dir must be set before using the Config object."
        self.netlist_dir: str = os.path.normpath(os.path.join(self.target_dir, "./data/netlist"))
        self.raw_file_dir: str = os.path.normpath(os.path.join(self.target_dir, "./data/raw_file"))
        self.log_file_dir: str = os.path.normpath(os.path.join(self.target_dir, "./data/log_file"))
        if not os.path.exists(self.netlist_dir):
            os.makedirs(self.netlist_dir)
        if not os.path.exists(self.raw_file_dir):
            os.makedirs(self.raw_file_dir)
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

        # python 3.9 and newer
        self.netlist_name = (
            self.original_netlist_path.replace("\\", "/").split("/")[-1].removesuffix(".net")
        )  # E.g. 5_4_3_6_10-dcdc_converter_1

        self.log_file_path = os.path.normpath(os.path.join(self.log_file_dir, f"{self.netlist_name}.log"))
        self.raw_file_path = os.path.normpath(os.path.join(self.raw_file_dir, f"{self.netlist_name}.raw"))
        self.rewrite_netlist_path = os.path.join(self.netlist_dir, f"rewrite_{self.mode}_{self.netlist_name}.net")

        self.bucket_id = self.netlist_name.split("-")[0]  # E.g. "5_4_3_6_10"
        self.n_S = int(self.bucket_id.split("_")[0])
        self.n_D = int(self.bucket_id.split("_")[1])
        self.n_L = int(self.bucket_id.split("_")[2])
        self.n_C = int(self.bucket_id.split("_")[3])

        self.capacitor_val = []
        self.inductor_val = []
        self.switch_T1 = []
        self.switch_T2 = []
        self.switch_L1 = []
        self.switch_L2 = []

    def __str__(self):
        """More readable print()."""
        return f"""Config:
            - Target Directory: {self.target_dir}
            - Netlist Directory: {self.netlist_dir}
            - Raw File Directory: {self.raw_file_dir}
            - Log File Directory: {self.log_file_dir}

            - Original Netlist Path: {self.original_netlist_path}
            - Netlist Name (without .net): {self.netlist_name}
            - Log File Path: {self.log_file_path}
            - Raw File Path: {self.raw_file_path}

            - Mode: {self.mode}
            - Rewrite Netlist Path: {self.rewrite_netlist_path}

            - Bucket ID: {self.bucket_id}
            - Component Counts: S={self.n_S}, D={self.n_D}, L={self.n_L}, C={self.n_C}

            - Capacitor Value: {self.capacitor_val}
            - Inductor Value: {self.inductor_val}
            - Switches: T1={self.switch_T1}, T2={self.switch_T2}, L1={self.switch_L1}, L2={self.switch_L2}"""
