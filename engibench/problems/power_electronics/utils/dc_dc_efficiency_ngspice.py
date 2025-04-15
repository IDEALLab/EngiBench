# ruff: noqa: N802, N803, N806  # Ignore uppercase names.
# ruff: noqa: PLR0912, PLR0913, PLR0915, FIX002  # TODO

"""Calculate the efficiency of a DC-DC converter using ngspice simulation data.

https://www.iosrjournals.org/iosr-jeee/Papers/Vol14%20Issue%203/Series-1/F1403014348.pdf
In order to determine the efficiency of a system it is rudimentary to understand the power loss in each
of its elements and the power delivered to the load. Power calculations in general are done under steady state
and the calculated power is the average power dissipation. However, the calculations get distorted with average
quantities of voltages and currents. Hence we use RMS values of the quantities involved to estimate the power
loss in a system.

@Author: Naga Siva Srinivas Putta <nagasiva@umd.edu>.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engibench.problems.power_electronics.utils import component as cmpt  # for type hint only
from engibench.problems.power_electronics.utils import raw_read as r


def calc_peak2peak(wav_arr: npt.NDArray) -> float:
    """Calculate the peak-to-peak value of a waveform."""
    return max(wav_arr) - min(wav_arr)


def calc_average(wav_arr: npt.NDArray) -> float:
    """Calculate the average value of a waveform."""
    return np.abs(np.mean(wav_arr))


def calc_rms(wav_arr: npt.NDArray) -> float:
    """Calculate the RMS (Root Mean Square) value of a waveform.

    Parameters:
        wav_arr (numpy.ndarray): Array containing the waveform data.

    Returns:
        float: The RMS value of the waveform.
    """
    return np.sqrt(np.mean(np.power(wav_arr, 2)))


def calc_mosfet_Coss(wav_arr: npt.NDArray, Coss_table: npt.NDArray) -> tuple[float, float]:
    """Calculate the Coss value of a MOSFET based on the maximum Vds.

    Parameters:
        wav_arr (numpy.ndarray): Array containing the waveform data.
        Coss_table (numpy.ndarray): 2D array containing the Coss values (2nd column) as a function of Vds_max (1st column).

    Returns:
        Vds_max (float): x. Maximum absolute value of the waveform.
        Coss_res (float): y. The "interpolated" Coss value from Coss_table.
    """
    Vds_max = max(np.abs(wav_arr))

    for i in range(len(Coss_table)):
        if (Vds_max > Coss_table[i][0]) and (i != len(Coss_table) - 1):
            continue
        elif Vds_max == Coss_table[i][0]:
            return Vds_max, Coss_table[i][1]
        elif i > 0:
            Coss_slope = (Coss_table[i][1] - Coss_table[i - 1][1]) / (Coss_table[i][0] - Coss_table[i - 1][0])
            Coss_res = Coss_slope * (Vds_max - Coss_table[i - 1][0]) + Coss_table[i - 1][1]
            return Vds_max, Coss_res
        else:  # when i == 0 and Vds_max <= Coss_table[0][0]
            return Vds_max, Coss_table[i][1]


def find_eff_calc_indx(
    num_cycles: int,
    skip_cycles: int,
    time_axis: npt.NDArray,
    Fsw: float,
    Vgs_mosfet_D: npt.NDArray,
    Vgs_mosfet_Dc: npt.NDArray,
) -> tuple[list[list[int]], list[list[int]]]:
    """Find the indices for efficiency calculation."""
    cycle_indx = []

    Ts = 1 / Fsw

    gl_cycle_num = 0
    gl_cycle_indx = 0

    while gl_cycle_num < skip_cycles:
        while time_axis[gl_cycle_indx] < (gl_cycle_num + 1) * Ts:
            gl_cycle_indx = gl_cycle_indx + 1

        gl_cycle_num = gl_cycle_num + 1

    cycle_indx.append(gl_cycle_indx)

    while gl_cycle_num < (skip_cycles + num_cycles):
        while time_axis[gl_cycle_indx] < (gl_cycle_num + 1) * Ts:
            gl_cycle_indx = gl_cycle_indx + 1

        cycle_indx.append(gl_cycle_indx)
        gl_cycle_num = gl_cycle_num + 1

    cycle_edg_det_D = []
    cycle_edg_det_Dc = []

    i = 0
    # For detections of rise, fall and sw time indices in 'DT' duty cycle Vgs pules
    while i < num_cycles:
        cycle_indx_edg = []
        j = cycle_indx[i]
        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j] == 1:
            j = j + 1

        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j] != 0:
            j = j + 1

        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j] == 0:
            j = j + 1

        cycle_indx_edg.append(j)
        cycle_indx_edg.append(cycle_indx[i + 1])

        cycle_edg_det_D.append(cycle_indx_edg)
        i = i + 1

    i = 0
    # For detections of rise, fall and sw time indices in '(1-D)T' duty cycle Vgs pules
    while i < num_cycles:
        cycle_indx_edg = []
        j = cycle_indx[i]
        cycle_indx_edg.append(j)

        while Vgs_mosfet_Dc[j] == 0:
            j = j + 1
        cycle_indx_edg.append(j)

        while Vgs_mosfet_Dc[j] != 1:
            j = j + 1

        cycle_indx_edg.append(j)

        while Vgs_mosfet_Dc[j] == 1:
            j = j + 1
        cycle_indx_edg.append(j)

        cycle_indx_edg.append(cycle_indx[i + 1])
        cycle_edg_det_Dc.append(cycle_indx_edg)
        i = i + 1

    return cycle_edg_det_D, cycle_edg_det_Dc


def metric_compute_DC_DC_efficiency_ngspice(
    file_path: str,
    sim_start: float,
    n_SW: int,
    n_C: int,
    n_L: int,
    n_D: int,
    edg_map: dict[str, list[int]],
    capacitor_model: list[cmpt.Capacitor],
    inductor_model: list[cmpt.Inductor],
    switch_model: list[cmpt.MOSFET],
    diode_model: list[cmpt.Diode],
    num_cycles: int,
    skip_cycles: int,
    gs_L1: list[float],
    gs_L2: list[float],
    t_sw_T1: list[float],
    t_sw_Ts: list[float],
    Fsw: float,
) -> tuple[int, float, float]:
    """Calculate the efficiency of a DC-DC converter using ngspice simulation data.

    Returns:
        err_report (int): Error report code (0 for no error, 8 if ).
        Power_loss (float): Total power loss in the system.
        P_src (float): Power delivered by the source.
    """
    arr, plt = r.rawread(file_path)
    data_arr = r.parse(arr, plt)

    err_report = 0

    time_axis = r.process_time(data_arr["time"], sim_start)

    """
    Vgs_ref are extra signal added in netlist to find the boundaries of ON & OFF states

    Vgs_ref_D - for GS0_L1 = 1 and GS0_L2 = 0
    Vgs_ref_Dc - for GS0_L1 = 0 and GS0_L2 = 1
    """
    Vgs_ref_D = data_arr["v(gs_ref_d)"]
    Vgs_ref_Dc = data_arr["v(gs_ref_dc)"]

    Ids_mosfet: list = []
    Vds_mosfet: list = []
    Vgs_mosfet: list = []
    I_cap: list = []
    I_Rcap: list = []
    V_cap: list = []
    I_L: list = []
    V_L: list = []
    I_D: list = []
    V_D: list = []

    for i in range(n_SW):
        Ids_mosfet.append([])
        Vds_mosfet.append([])
        Vgs_mosfet.append([])
        V_sw_edg = edg_map["S" + str(i)]
        Ids_mosfet[i] = data_arr["i(@s" + str(i) + "[i])"]

        decision_1 = 1 if V_sw_edg[0] > 0 else 0
        decision_2 = 1 if V_sw_edg[1] > 0 else 0
        if decision_1 + decision_2 == 1:
            if decision_2 == 0:
                Vds_mosfet[i] = data_arr["v(" + str(V_sw_edg[0]) + ")"]
            else:
                Vds_mosfet[i] = [-ii for ii in data_arr["v(" + str(V_sw_edg[1]) + ")"]]
        else:
            Vds_mosfet[i] = data_arr["v(" + str(V_sw_edg[0]) + ")"] - data_arr["v(" + str(V_sw_edg[1]) + ")"]

        Vgs_mosfet[i] = data_arr["v(gs" + str(i) + ")"]

    for i in range(n_C):
        I_cap.append([])
        I_Rcap.append([])
        V_cap.append([])
        V_C_edg = edg_map["C" + str(i)]
        I_cap[i] = data_arr["i(@c" + str(i) + "[i])"]
        I_Rcap[i] = data_arr["i(@rc" + str(i) + "[i])"]

        decision_1 = 1 if V_C_edg[0] > 0 else 0
        decision_2 = 1 if V_C_edg[1] > 0 else 0
        if decision_1 + decision_2 == 1:
            if decision_2 == 0:
                V_cap[i] = data_arr["v(" + str(V_C_edg[0]) + ")"]
            else:
                V_cap[i] = [-ii for ii in data_arr["v(" + str(V_C_edg[1]) + ")"]]
        else:
            V_cap[i] = data_arr["v(" + str(V_C_edg[0]) + ")"] - data_arr["v(" + str(V_C_edg[1]) + ")"]

    for i in range(n_L):
        I_L.append([])
        V_L.append([])
        V_L_edg = edg_map["L" + str(i)]
        I_L[i] = data_arr["i(@l" + str(i) + "[i])"]

        decision_1 = 1 if V_L_edg[0] > 0 else 0
        decision_2 = 1 if V_L_edg[1] > 0 else 0
        if decision_1 + decision_2 == 1:
            if decision_2 == 0:
                V_L[i] = data_arr["v(" + str(V_L_edg[0]) + ")"]
            else:
                V_L[i] = [-ii for ii in data_arr["v(" + str(V_L_edg[1]) + ")"]]
        else:
            V_L[i] = data_arr["v(" + str(V_L_edg[0]) + ")"] - data_arr["v(" + str(V_L_edg[1]) + ")"]

    for i in range(n_D):
        I_D.append([])
        V_D.append([])
        V_D_edg = edg_map["D" + str(i)]
        I_D[i] = data_arr["i(@d" + str(i) + "[i])"]

        decision_1 = 1 if V_D_edg[0] > 0 else 0
        decision_2 = 1 if V_D_edg[1] > 0 else 0
        if decision_1 + decision_2 == 1:
            if decision_2 == 0:
                V_D[i] = data_arr["v(" + str(V_D_edg[0]) + ")"]
            else:
                V_D[i] = [-ii for ii in data_arr["v(" + str(V_D_edg[1]) + ")"]]
        else:
            V_D[i] = data_arr["v(" + str(V_D_edg[0]) + ")"] - data_arr["v(" + str(V_D_edg[1]) + ")"]

    cycle_edg_det_D, cycle_edg_det_Dc = find_eff_calc_indx(num_cycles, skip_cycles, time_axis, Fsw, Vgs_ref_D, Vgs_ref_Dc)

    # ===========================MOSFET power loss calculations====================================================================

    P_cond_loss = np.zeros(n_SW)
    P_sw_loss = np.zeros(n_SW)
    P_coss = np.zeros(n_SW)
    P_gate_driver = np.zeros(n_SW)

    for i in range(n_SW):
        j = 0

        t_sw_fall = switch_model[i].td_off + switch_model[i].tf

        duty_cycle_type = 0

        if gs_L1[i] == 1 and gs_L2[i] == 0:
            duty_cycle_type = 1
        elif gs_L1[i] == 0 and gs_L2[i] == 1:
            duty_cycle_type = 0
        elif (gs_L1[i] + gs_L2[i]) != 1:
            if Ids_mosfet[i][int((cycle_edg_det_D[0][0] + cycle_edg_det_D[0][1]) / 2)] == 0:
                duty_cycle_type = 1
            elif Ids_mosfet[i][int((cycle_edg_det_D[0][2] + cycle_edg_det_D[0][3]) / 2)] == 0:
                duty_cycle_type = 0
            else:
                err_report = 8  # we will hit this condition if there is no cycle generation and current is increasing or decreasing curve
                duty_cycle_type = 0  # duty cycle typw will not matter anyway as it is not a cycle curve

        t_sw_on = t_sw_T1[i] - t_sw_fall if duty_cycle_type == 0 else t_sw_Ts[i] - t_sw_fall - t_sw_T1[i]

        while j < num_cycles:
            # Conduction Losses

            if duty_cycle_type == 0:
                I_mosfet_avg = calc_average(Ids_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])
                I_mosfet_pp = calc_peak2peak(Ids_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])
                P_cond_loss[i] += (Fsw * t_sw_on) * (I_mosfet_avg**2 + (I_mosfet_pp**2) / 12) * switch_model[i].Ron

            else:
                I_mosfet_avg = calc_average(Ids_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])
                I_mosfet_pp = calc_peak2peak(Ids_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])
                P_cond_loss[i] += (Fsw * t_sw_on) * (I_mosfet_avg**2 + (I_mosfet_pp**2) / 12) * switch_model[i].Ron

            # Off-state losses are neglected.

            # Switching losses

            if duty_cycle_type == 0:
                P_sw_max_fall = max(
                    np.abs(
                        np.multiply(
                            Ids_mosfet[i][cycle_edg_det_D[j][1] : cycle_edg_det_D[j][2]],
                            Vds_mosfet[i][cycle_edg_det_D[j][1] : cycle_edg_det_D[j][2]],
                        )
                    )
                )
                P_sw_max_rise = max(
                    np.abs(
                        np.multiply(
                            Ids_mosfet[i][cycle_edg_det_D[j][3] : cycle_edg_det_D[j][4]],
                            Vds_mosfet[i][cycle_edg_det_D[j][3] : cycle_edg_det_D[j][4]],
                        )
                    )
                )

            else:
                P_sw_max_rise = max(
                    np.abs(
                        np.multiply(
                            Ids_mosfet[i][cycle_edg_det_Dc[j][1] : cycle_edg_det_Dc[j][2]],
                            Vds_mosfet[i][cycle_edg_det_Dc[j][1] : cycle_edg_det_Dc[j][2]],
                        )
                    )
                )
                P_sw_max_fall = max(
                    np.abs(
                        np.multiply(
                            Ids_mosfet[i][cycle_edg_det_Dc[j][3] : cycle_edg_det_Dc[j][4]],
                            Vds_mosfet[i][cycle_edg_det_Dc[j][3] : cycle_edg_det_Dc[j][4]],
                        )
                    )
                )

            P_sw_loss[i] = (
                P_sw_loss[i]
                + 0.5
                * (
                    P_sw_max_rise * (switch_model[i].td_on + switch_model[i].tr)
                    + P_sw_max_fall * (switch_model[i].td_off + switch_model[i].tf)
                )
                * Fsw
            )

            # Coss losses
            if duty_cycle_type == 0:
                Vds_max, mosfet_Coss = calc_mosfet_Coss(
                    Vds_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]], switch_model[i].Coss_vs_Vds
                )
            else:
                Vds_max, mosfet_Coss = calc_mosfet_Coss(
                    Vds_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]], switch_model[i].Coss_vs_Vds
                )

            P_coss[i] += 0.5 * mosfet_Coss * (Vds_max**2) * Fsw

            # GateDriver Loss
            if duty_cycle_type == 0:
                P_gate_driver[i] += (
                    switch_model[i].Q_g_tot * Fsw * max(Vgs_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])
                )  # Qg - Total gate charge from data sheet
            else:
                P_gate_driver[i] += (
                    switch_model[i].Q_g_tot * Fsw * max(Vgs_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])
                )

            # Body Diode/ Dead-time losses
            j = j + 1

    # ====================Power disspiation input and Output==========================================================

    I_load = data_arr["i(@r0[i])"]
    decision_1 = 1 if edg_map["R0"][0] > 0 else 0
    decision_2 = 1 if edg_map["R0"][1] > 0 else 0

    I_src = data_arr["i(v0)"]
    decision_1 = 1 if edg_map["V0"][0] > 0 else 0
    decision_2 = 1 if edg_map["V0"][1] > 0 else 0
    if decision_1 + decision_2 == 1:
        if decision_2 == 0:
            V_src = data_arr["v(" + str(edg_map["V0"][0]) + ")"]
        else:
            V_src = [-ii for ii in data_arr["v(" + str(edg_map["V0"][1]) + ")"]]
    else:
        V_src = data_arr["v(" + str(edg_map["V0"][0]) + ")"] - data_arr["v(" + str(edg_map["V0"][1]) + ")"]

    R_load = 10
    P_load = 0
    P_src = 0
    j = 0
    while j < num_cycles:
        Irms_load = calc_rms(I_load[cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
        Irms_src = calc_rms(I_src[cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])

        P_load += (Irms_load**2) * R_load
        P_src += calc_average(V_src) * Irms_src
        j = j + 1

    # =====================Inductor Losses===============================================================================
    P_L = np.zeros(n_L)

    for i in range(n_L):
        j = 0
        while j < num_cycles:
            I_L_rms = calc_rms(I_L[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
            P_L[i] += inductor_model[i].DCR * (I_L_rms**2)  # Resistive Loss
            j = j + 1

    # =============================Capacitor losses============================================================================#
    P_Rcap = np.zeros(n_C)
    P_cap = np.zeros(n_C)

    for i in range(n_C):
        j = 0
        while j < num_cycles:
            Irms_Rcap = calc_rms(I_Rcap[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
            Irms_cap = calc_rms(I_cap[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
            P_Rcap[i] += capacitor_model[i].Rp * (Irms_Rcap**2)
            P_cap[i] += capacitor_model[i].ESR * (Irms_cap**2)
            j = j + 1

    # ==================== Diode Loss================================================#
    P_D_con = np.zeros(n_D)

    for i in range(n_D):
        j = 0
        while j < num_cycles:
            I_D_avg = calc_average(I_D[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
            I_D_rms = calc_rms(I_D[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])

            # On state losses
            P_D_con[i] += diode_model[i].Rd * (I_D_rms**2) + diode_model[i].Vt0 * I_D_avg

            j = j + 1

    Power_loss = (
        np.sum(P_D_con)
        + np.sum(P_Rcap)
        + np.sum(P_cap)
        + np.sum(P_L)
        + np.sum(P_cond_loss)
        + np.sum(P_sw_loss)
        + np.sum(P_coss)
        + np.sum(P_gate_driver)
    )

    return err_report, Power_loss, P_src
