"""Rewrite netlist for ngSpice simulation."""
# ruff: noqa: N806, N815 # Upper case
# ruff: noqa: PLR0912, PLR0915 # Too many branches. Too many statements

from __future__ import annotations

from engibench.problems.power_electronics.utils.config import Config


def rewrite_netlist(config: Config) -> None:
    """Rewrite the netlist based on the topology and the sweep data.

    It creates the direct input file sent to ngSpice.
    The main difference between this rewrite.netlist and the original netlist is the control section that contains simulation parameters.
    This function does not change config.
    """
    print(f"rewriting netlist to: {config.rewrite_netlist_path}")

    with open(config.rewrite_netlist_path, "w") as file:
        cmp_edg_str = "* rewrite netlist\n" + config.cmp_edg_str

        RC_str = ""
        for i in range(config.n_C):
            RC_str += f"RC{i} {config.edge_map[f'C{i}'][0]} {config.edge_map[f'C{i}'][1]} 100meg\n"

        cmp_edg_str += f"{RC_str}\n.PARAM V0_value=1000\n"

        for i in range(config.n_C):
            cmp_edg_str += f".PARAM C{i}_value = {config.capacitor_val[i]}\n"

        for i in range(config.n_L):
            cmp_edg_str += f".PARAM L{i}_value = {config.inductor_val[i]}\n"

        cmp_edg_str += ".PARAM R0_value = 10\n\n.model Ideal_switch SW (Ron=1m Roff=10Meg Vt=0.5 Vh=0 Lser=0 Vser=0)\n.model Ideal_D D\n\n"

        cmp_edg_str += "V_GSref_D  gs_ref_D 0 pwl(0 1 {GS0_T1-10e-9} 1 {GS0_T1} 0 {GS0_T2-10e-9} 0 {GS0_Ts} 1) r=0\nV_GSref_Dc  gs_ref_Dc 0 pwl(0 0 {GS0_T1-10e-9} 0 {GS0_T1} 1 {GS0_T2-10e-9} 1 {GS0_Ts} 0) r=0\n"

        for i in range(config.n_S):
            cmp_edg_str += f"V_GS{i} GS{i} 0 pwl(0 {{GS{i}_L1}} {{GS{i}_T1-10e-9}} {{GS{i}_L1}} {{GS{i}_T1}} {{GS{i}_L2}} {{GS{i}_T2-10e-9}} {{GS{i}_L2}} {{GS{i}_Ts}} {{GS{i}_L1}}) r=0\n"

        for i in range(config.n_S):
            cmp_edg_str += f".PARAM GS{i}_Ts = {config.switch_T2[i] * 5}e-06\n"
            cmp_edg_str += f".PARAM GS{i}_T1 = {config.switch_T1[i] * 5}e-06\n"
            cmp_edg_str += f".PARAM GS{i}_T2 = {config.switch_T2[i] * 5}e-06\n"
            cmp_edg_str += f".PARAM GS{i}_L1 = {config.switch_L1[i]}\n"
            cmp_edg_str += f".PARAM GS{i}_L2 = {config.switch_L2[i]}\n"

        if config.mode == "batch":
            """
            Here is an example:
            .save @C0[i] @RC0[i] @C1[i] @RC1[i] @C2[i] @RC2[i] @C3[i] @RC3[i] @C4[i] @RC4[i] @C5[i] @RC5[i] @L0[i] @L1[i] @L2[i] @S0[i] @S1[i] @S2[i] @S3[i] @S4[i] @D0[i] @D1[i] @D2[i] @D3[i] @R0[i]
            .save all
            """
            cmp_edg_str += "\n.save "
            for i in range(config.n_C):
                cmp_edg_str += f"@C{i}[i] @RC{i}[i] "
            for i in range(config.n_L):
                cmp_edg_str += f"@L{i}[i] "
            for i in range(config.n_S):
                cmp_edg_str += f"@S{i}[i] "
            for i in range(config.n_D):
                cmp_edg_str += f"@D{i}[i] "
            cmp_edg_str += "@R0[i]\n"  # This line should be outside the for loop, otherwise it will be duplicated.
            cmp_edg_str += ".save all\n"

            cmp_edg_str += ".tran 5n 1.06m 1m 5n uic\n"
            # The following .meas(ure) can be replaced by .print, although the former is tested and preferred, while the latter has not been tested in all cases.
            cmp_edg_str += f".meas TRAN Vo_mean avg par('V({config.edge_map['R0'][0]}) - V({config.edge_map['R0'][1]})') from = 1m to = 1.06m\n"
            cmp_edg_str = cmp_edg_str + ".meas TRAN gain param='Vo_mean/1000'\n"
            cmp_edg_str += f".meas TRAN Vpp pp par('V({config.edge_map['R0'][0]}) - V({config.edge_map['R0'][1]})') from = 1m to = 1.06m\n"
            cmp_edg_str += ".meas TRAN Vpp_ratio param = 'Vpp / Vo_mean'\n"  # Ripple voltage
            cmp_edg_str += "\n.end"

        elif config.mode == "control":
            cmp_edg_str += "\n.control\nsave "
            for i in range(config.n_C):
                cmp_edg_str += f"@C{i}[i] @RC{i}[i] "
            for i in range(config.n_L):
                cmp_edg_str += f"@L{i}[i] "
            for i in range(config.n_S):
                cmp_edg_str += f"@S{i}[i] "
            for i in range(config.n_D):
                cmp_edg_str += f"@D{i}[i] "
            cmp_edg_str += "@R0[i]\n"  # This line should be outside the for loop, otherwise it will be duplicated.
            cmp_edg_str += "save all\n"

            cmp_edg_str += "tran 5n 1.06m 1m 5n uic\n"
            cmp_edg_str += f"let Vdiff = V({config.edge_map['R0'][0]}) - V({config.edge_map['R0'][1]})\n"
            cmp_edg_str += "meas TRAN Vo_mean avg Vdiff from = 1m to = 1.06m\n"
            cmp_edg_str += "meas TRAN Vpp pp Vdiff from = 1m to = 1.06m\n"
            cmp_edg_str += "let Gain = Vo_mean / 1000\nlet Vpp_ratio = Vpp / Vo_mean\nprint Gain, Vpp_ratio\nrun\nset filetype = binary\n"
            cmp_edg_str += f"write {config.raw_file_path}\n"

            cmp_edg_str += "quit\n.endc\n\n.end"

        file.write(cmp_edg_str)
