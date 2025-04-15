"""Set up the configuration for the Power Electronics problem."""
# ruff: noqa: N806, N815 # Upper case
# ruff: noqa: FIX002  # for TODO
# ruff: noqa: RUF009  # TODO: normpath

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Config:
    """Configuration for the Power Electronics problem."""

    source_dir: str = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    )  # The absolute path of power_electronics/
    # TODO: check if this works from another repo like EngiOpt

    netlist_dir: str = os.path.normpath(os.path.join(source_dir, "./data/netlist"))
    raw_file_dir: str = os.path.normpath(os.path.join(source_dir, "./data/raw_file"))
    log_file_dir: str = os.path.normpath(os.path.join(source_dir, "./data/log_file"))

    original_netlist_path: str = (
        "./data/netlist/5_4_3_6_10-dcdc_converter_1.net"  # Accepts both absolute and relative paths.
    )

    netlist_name: str = original_netlist_path.replace("\\", "/").split("/")[-1].removesuffix(".net")  # python 3.9 and newer
    log_file_path: str = os.path.normpath(os.path.join(log_file_dir, f"{netlist_name}.log"))
    raw_file_path: str = os.path.normpath(os.path.join(raw_file_dir, f"{netlist_name}.raw"))
    mode: str = "control"
    rewrite_netlist_path: str = os.path.join(netlist_dir, f"rewrite_{mode}_{netlist_name}.net")

    bucket_id: str = "5_4_3_6_10"  # Alternatively, we can get this from self.original_netlist_path.

    edge_map: dict[str, list[int]] | None = (
        None  # This will be turned into an empty dictionary in __posit_init__(). It will finally look like {"V0": [0, 1], "R0": [9, 12]}  # TODO: is this the correct way?
    )
    cmp_edg_str: str = ""  # The string that is used to rewrite the netlist.

    # components of the design variable
    capacitor_val: list[float] | None = None  # range: [1e-6, 2e-5]
    inductor_val: list[float] | None = None  # range: [1e-6, 1e-3]
    switch_T1: list[float] | None = None  # range: [0.1, 0.9]
    switch_T2: list[float] | None = None  # Constant. All 1 for now
    switch_L1: list[float] | None = None  # Binary.
    switch_L2: list[float] | None = None  # Binary.

    def __post_init__(self):
        """Component counts from buck_id. Set up config.edge_map."""
        # TODO: correct?
        self.n_S: int = int(self.bucket_id.split("_")[0])
        self.n_D: int = int(self.bucket_id.split("_")[1])
        self.n_L: int = int(self.bucket_id.split("_")[2])
        self.n_C: int = int(self.bucket_id.split("_")[3])

        self.edge_map = {}  # TODO: use field(default_factory=dict) instead of this?

    def __str__(self):
        """More readable print()."""
        return f"""Config:
            - Source Directory: {self.source_dir}
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

            - Edge Map: {self.edge_map}
            - Netlist String: {self.cmp_edg_str}

            - Capacitor Value: {self.capacitor_val}
            - Inductor Value: {self.inductor_val}
            - Switches: T1={self.switch_T1}, T2={self.switch_T2}, L1={self.switch_L1}, L2={self.switch_L2}"""
