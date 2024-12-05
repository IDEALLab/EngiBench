import numpy as np    
from . import raw_read as r

"""
https://www.iosrjournals.org/iosr-jeee/Papers/Vol14%20Issue%203/Series-1/F1403014348.pdf

In order to determine the efficiency of a system it is rudimentary to understand the power loss in each
of its elements and the power delivered to the load. Power calculations in general are done under steady state
and the calculated power is the average power dissipation. However, the calculations get distorted with average
quantities of voltages and currents. Hence we use RMS values of the quantities involved to estimate the power
loss in a system.
"""
def calc_peak2peak(wav_arr):
    return max(wav_arr) - min(wav_arr)

def calc_average(wav_arr):
    Ts = 5e6
    return np.abs(np.mean(wav_arr))

def calc_rms(wav_arr, Fsw):
    Ts = 1/Fsw
    return np.sqrt(np.mean(np.power(wav_arr, 2)))

def calc_mosfet_Coss(wav_arr, Coss_table):

    Vds_max = max(np.abs(wav_arr))

    for i in range(len(Coss_table)):

        if (Vds_max > Coss_table[i][0]) and (i!=len(Coss_table)-1):continue
        else:
            if Vds_max == Coss_table[i][0]: return Vds_max, Coss_table[i][1]
            else:
                if i>0:
                    Coss_slope = (Coss_table[i][1]-Coss_table[i-1][1])/(Coss_table[i][0]-Coss_table[i-1][0])
                    Coss_res = Coss_slope*(Vds_max - Coss_table[i-1][0]) + Coss_table[i-1][1] 
                    return Vds_max, Coss_res
                else:
                    return Vds_max, Coss_table[i][1]

def find_eff_calc_indx(num_cycles, skip_cycles, time_axis, Fsw, Vgs_mosfet_D, Vgs_mosfet_Dc):
    
    cycle_indx = []

    Ts = 1/Fsw

    gl_cycle_num = 0
    gl_cycle_indx = 0
    
    while(gl_cycle_num < skip_cycles):
        while( time_axis[gl_cycle_indx]< (gl_cycle_num+1)*Ts): gl_cycle_indx = gl_cycle_indx + 1

        gl_cycle_num = gl_cycle_num + 1

    cycle_indx.append(gl_cycle_indx)

    while gl_cycle_num < (skip_cycles+num_cycles):        
        while time_axis[gl_cycle_indx]< (gl_cycle_num+1)*Ts: gl_cycle_indx = gl_cycle_indx + 1 

        cycle_indx.append(gl_cycle_indx)
        gl_cycle_num = gl_cycle_num + 1 

    cycle_edg_det_D = []
    cycle_edg_det_Dc = []
    
    i = 0
    # For detections of rise, fall and sw time indexs in 'DT' duty cycle Vgs pules
    while i < num_cycles:
        cycle_indx_edg = []
        j = cycle_indx[i]
        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j] == 1: j = j+1

        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j] != 0: j = j+1

        cycle_indx_edg.append(j)

        while Vgs_mosfet_D[j]==0: j = j+1

        cycle_indx_edg.append(j)
        cycle_indx_edg.append(cycle_indx[i+1])

        cycle_edg_det_D.append(cycle_indx_edg)
        i = i+1

    i = 0       
    # For detections of rise, fall and sw time indexs in '(1-D)T' duty cycle Vgs pules  
    while i < num_cycles:

        cycle_indx_edg = []    
        j = cycle_indx[i]    
        cycle_indx_edg.append(j) 

        while Vgs_mosfet_Dc[j] == 0: j = j+1       
        cycle_indx_edg.append(j)    

        while Vgs_mosfet_Dc[j] != 1: j = j+1   

        cycle_indx_edg.append(j)

        while Vgs_mosfet_Dc[j] == 1: j = j+1   
        cycle_indx_edg.append(j)

        cycle_indx_edg.append(cycle_indx[i+1])    
        cycle_edg_det_Dc.append(cycle_indx_edg)  
        i = i+1 


    return cycle_edg_det_D, cycle_edg_det_Dc

    

def metric_compute_DC_DC_efficiency_ngspice(file_path, sim_start, comp_num, edg_map, capacitor_model, inductor_model, switch_model, diode_model, num_cycles, skip_cycles, gs_L1, gs_L2, t_sw_T1, t_sw_Ts, Fsw):

    arr,plt = r.rawread(file_path)
    data_arr = r.parse(arr,plt)

    err_report = 0

    time_axis = r.process_time(data_arr['time'], sim_start)
    
    n_SW = comp_num["S"]
    n_C = comp_num["C"]
    n_L = comp_num["L"]
    n_D = comp_num["D"]

    """
    Vgs_ref are extra signal added in netlist to find the boundaries of ON & OFF states

    Vgs_ref_D - for GS0_L1 = 1 and GS0_L2 = 0
    Vgs_ref_Dc - for GS0_L1 = 0 and GS0_L2 = 1
    """
    Vgs_ref_D = data_arr['v(gs_ref_d)']
    Vgs_ref_Dc = data_arr['v(gs_ref_dc)']

    Ids_mosfet, Vds_mosfet, Vgs_mosfet, I_cap, I_Rcap, V_cap, I_L, V_L, I_D, V_D = [], [], [], [], [], [], [], [], [], []

    for i in range(n_SW):
        Ids_mosfet.append([])
        Vds_mosfet.append([])
        Vgs_mosfet.append([])
        V_sw_edg = edg_map["S"+str(i)]  
        Ids_mosfet[i] = data_arr["i(@s"+str(i)+"[i])"]

        decision_1 = 1 if V_sw_edg[0]>0 else 0
        decision_2 = 1 if V_sw_edg[1]>0 else 0
        if decision_1+decision_2==1:
            if decision_2 == 0:
                Vds_mosfet[i] = data_arr["v("+str(V_sw_edg[0])+")"]
            else:
                Vds_mosfet[i] = [-ii for ii in data_arr["v("+str(V_sw_edg[1])+")"]]
        else:
            Vds_mosfet[i] = data_arr["v("+str(V_sw_edg[0])+")"] - data_arr["v("+str(V_sw_edg[1])+")"]
       
        Vgs_mosfet[i] = data_arr["v(gs"+str(i)+")"]


    for i in range(n_C):
        I_cap.append([])
        I_Rcap.append([])
        V_cap.append([])
        V_C_edg = edg_map["C"+str(i)]
        I_cap[i] = data_arr["i(@c"+str(i)+"[i])"]
        I_Rcap[i] = data_arr["i(@rc"+str(i)+"[i])"]

        decision_1 = 1 if V_C_edg[0]>0 else 0
        decision_2 = 1 if V_C_edg[1]>0 else 0
        if decision_1+decision_2==1:
            if decision_2 == 0:
                V_cap[i] = data_arr["v("+str(V_C_edg[0])+")"]
            else:
                V_cap[i] = [-ii for ii in data_arr["v("+str(V_C_edg[1])+")"]]
        else:
            V_cap[i] = data_arr["v("+str(V_C_edg[0])+")"] - data_arr["v("+str(V_C_edg[1])+")"]

    for i in range(n_L):
        I_L.append([])
        V_L.append([])
        V_L_edg = edg_map["L"+str(i)]
        I_L[i] = data_arr["i(@l"+str(i)+"[i])"]

        decision_1 = 1 if V_L_edg[0]>0 else 0
        decision_2 = 1 if V_L_edg[1]>0 else 0
        if decision_1+decision_2==1:
            if decision_2 == 0:
                V_L[i] = data_arr["v("+str(V_L_edg[0])+")"]
            else:
                V_L[i] = [-ii for ii in data_arr["v("+str(V_L_edg[1])+")"]]
        else:
            V_L[i] = data_arr["v("+str(V_L_edg[0])+")"] - data_arr["v("+str(V_L_edg[1])+")"]

    for i in range(n_D): 
        I_D.append([])
        V_D.append([])     
        V_D_edg = edg_map["D"+str(i)]     
        I_D[i] = data_arr["i(@d"+str(i)+"[i])"]   

        decision_1 = 1 if V_D_edg[0]>0 else 0
        decision_2 = 1 if V_D_edg[1]>0 else 0
        if decision_1+decision_2==1:
            if decision_2 == 0:
                V_D[i] = data_arr["v("+str(V_D_edg[0])+")"]
            else:
                V_D[i] = [-ii for ii in data_arr["v("+str(V_D_edg[1])+")"]]
        else:
            V_D[i] = data_arr["v("+str(V_D_edg[0])+")"] - data_arr["v("+str(V_D_edg[1])+")"]
   
    cycle_edg_det_D, cycle_edg_det_Dc = find_eff_calc_indx(num_cycles, skip_cycles, time_axis, Fsw, Vgs_ref_D, Vgs_ref_Dc)  
  
    #===========================MOSFET power loss calculations====================================================================

    P_cond_loss = np.zeros(n_SW)
    P_sw_loss   = np.zeros(n_SW)
    P_coss      = np.zeros(n_SW)
    P_gate_driver = np.zeros(n_SW)

    for i in range(n_SW):

        j = 0

        t_sw_rise = switch_model[i].td_on + switch_model[i].tr
        t_sw_fall = switch_model[i].td_off + switch_model[i].tf
        t_sw_tot  = t_sw_rise + t_sw_fall

        duty_cycle_type = 0

        if gs_L1[i]==1 and gs_L2[i]==0:
            duty_cycle_type = 1
        elif gs_L1[i]==0 and gs_L2[i]==1: 
            duty_cycle_type = 0
        elif (gs_L1[i] + gs_L2[i]) != 1:
            if Ids_mosfet[i][int((cycle_edg_det_D[0][0] + cycle_edg_det_D[0][1])/2)] == 0: duty_cycle_type = 1
            elif Ids_mosfet[i][int((cycle_edg_det_D[0][2] + cycle_edg_det_D[0][3])/2)] == 0: duty_cycle_type = 0
            else: 
                #err_report = 8 #we will hit this condition if there is no cycle generation and current is increasing or decreasing curve
                duty_cycle_type = 0 #duty cycle typw will not matter anyway as it is not a cycle curve

        if duty_cycle_type==0:
            t_sw_on   = t_sw_T1[i] - t_sw_fall
            t_sw_off  = t_sw_Ts[i] - t_sw_rise - t_sw_T1[i]
        else: 
            t_sw_on   = t_sw_Ts[i] - t_sw_fall - t_sw_T1[i]
            t_sw_off  = t_sw_T1[i] - t_sw_rise

        while j < num_cycles:
            #Conduction Losses    

            if duty_cycle_type==0:
                I_mosfet_avg = calc_average(Ids_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])
                I_mosfet_pp  = calc_peak2peak(Ids_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])
                P_cond_loss[i] += (Fsw*t_sw_on)*(I_mosfet_avg**2 + (I_mosfet_pp**2)/12)*switch_model[i].Ron

            else: 
                I_mosfet_avg = calc_average(Ids_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])    
                I_mosfet_pp  = calc_peak2peak(Ids_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])
                P_cond_loss[i] += (Fsw*t_sw_on)*(I_mosfet_avg**2 + (I_mosfet_pp**2)/12)*switch_model[i].Ron
            
            #off-state losses - neglected
            #if gs_L1[i]==1 and gs_L2[i]==0: 
            #    P_off_state[i] = P_off_state[i] + (Fsw*t_sw_off)*Idss[i]*calc_rms(Vds_mosfet[i][cycle_edg_det_D[j][2] : cycle_edg_det_D[j][3]], 1)
            #elif gs_L1[i]==0 and gs_L2[i]==1:
            #    P_off_state[i] = P_off_state[i] + (Fsw*t_sw_off)*Idss[i]*calc_rms(Vds_mosfet[i][cycle_edg_det_Dc[j][0] : cycle_edg_det_Dc[j][1]], 1)

            #Switching losses

            if duty_cycle_type==0:
                P_sw_max_fall = max(np.abs(np.multiply(Ids_mosfet[i][cycle_edg_det_D[j][1] : cycle_edg_det_D[j][2]],Vds_mosfet[i][cycle_edg_det_D[j][1] : cycle_edg_det_D[j][2]])))
                P_sw_max_rise = max(np.abs(np.multiply(Ids_mosfet[i][cycle_edg_det_D[j][3] : cycle_edg_det_D[j][4]],Vds_mosfet[i][cycle_edg_det_D[j][3] : cycle_edg_det_D[j][4]])))

            else:
                P_sw_max_rise = max(np.abs(np.multiply(Ids_mosfet[i][cycle_edg_det_Dc[j][1] : cycle_edg_det_Dc[j][2]],Vds_mosfet[i][cycle_edg_det_Dc[j][1] : cycle_edg_det_Dc[j][2]])))     
                P_sw_max_fall = max(np.abs(np.multiply(Ids_mosfet[i][cycle_edg_det_Dc[j][3] : cycle_edg_det_Dc[j][4]],Vds_mosfet[i][cycle_edg_det_Dc[j][3] : cycle_edg_det_Dc[j][4]])))    

            P_sw_loss[i] = P_sw_loss[i] + 0.5*(P_sw_max_rise*(switch_model[i].td_on+switch_model[i].tr) + P_sw_max_fall*(switch_model[i].td_off+switch_model[i].tf))*Fsw

            #Coss losses 
            if duty_cycle_type==0:
                Vds_max, mosfet_Coss = calc_mosfet_Coss(Vds_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]], switch_model[i].Coss_vs_Vds)
            else:
                Vds_max, mosfet_Coss = calc_mosfet_Coss(Vds_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]], switch_model[i].Coss_vs_Vds)

            P_coss[i] += 0.5*mosfet_Coss*(Vds_max**2)*Fsw

            #GateDriver Loss
            if duty_cycle_type==0:
                P_gate_driver[i] += switch_model[i].Q_g_tot*Fsw*max(Vgs_mosfet[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][1]])     #Qg - Total gate charge from data sheet
            else: 
                P_gate_driver[i] += switch_model[i].Q_g_tot*Fsw*max(Vgs_mosfet[i][cycle_edg_det_Dc[j][2] : cycle_edg_det_Dc[j][3]])

            #Body Diode/ Dead-time losses
            j = j+1

    #====================Power disspiation input and Output==========================================================
    
    I_load = data_arr["i(@r0[i])"]
    decision_1 = 1 if edg_map["R0"][0]>0 else 0
    decision_2 = 1 if edg_map["R0"][1]>0 else 0
    if decision_1+decision_2==1:
        if decision_2 == 0:
            V_load = data_arr["v("+str(edg_map["R0"][0])+")"]
        else:
            V_load = [-ii for ii in data_arr["v("+str(edg_map["R0"][1])+")"]]
    else:
        V_load = data_arr["v("+str(edg_map["R0"][0])+")"] - data_arr["v("+str(edg_map["R0"][1])+")"]

    I_src = data_arr["i(v0)"]
    decision_1 = 1 if edg_map["V0"][0]>0 else 0
    decision_2 = 1 if edg_map["V0"][1]>0 else 0
    if decision_1+decision_2==1:
        if decision_2 == 0:
            V_src = data_arr["v("+str(edg_map["V0"][0])+")"]
        else:
            V_src = [-ii for ii in data_arr["v("+str(edg_map["V0"][1])+")"]]
    else:
        V_src = data_arr["v("+str(edg_map["V0"][0])+")"] - data_arr["v("+str(edg_map["V0"][1])+")"]

    R_load = 10
    P_load = 0
    P_src  = 0
    j = 0
    while j < num_cycles:
        
        Irms_load = calc_rms(I_load[cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)
        Vrms_load = calc_rms(V_load[cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)
        Irms_src  = calc_rms(I_src[cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)

        P_load += (Irms_load**2)*R_load
        P_src  +=  calc_average(V_src)*Irms_src
        j = j+1

    

    

    #=====================Inductor Losses===============================================================================
    P_L = np.zeros(n_L)

    for i in range(n_L):

        j = 0
        while j<num_cycles:
            I_L_rms = calc_rms(I_L[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw) 
            P_L[i] += inductor_model[i].DCR*(I_L_rms**2)  #Resistive Loss
            j = j+1


    #=============================Capacitor losses============================================================================#
    P_Rcap = np.zeros(n_C)
    P_cap = np.zeros(n_C)

    for i in range(n_C):
        j = 0
        while j < num_cycles:
            Irms_Rcap = calc_rms(I_Rcap[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)
            Irms_cap  = calc_rms(I_cap[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)
            P_Rcap[i] += capacitor_model[i].Rp*(Irms_Rcap**2)
            P_cap[i]  += capacitor_model[i].ESR*(Irms_cap**2)
            j = j+1
    
    #==================== Diode Loss================================================#
    P_D_con = np.zeros(n_D)

    for i in range(n_D):
        j = 0
        while j < num_cycles:
            I_D_avg = calc_average(I_D[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]])
            I_D_rms = calc_rms(I_D[i][cycle_edg_det_D[j][0] : cycle_edg_det_D[j][4]], Fsw)

            #On state losses
            P_D_con[i] += diode_model[i].Rd *(I_D_rms**2)  + diode_model[i].Vt0* I_D_avg

            j = j+1

    Power_loss = np.sum(P_D_con) + np.sum(P_Rcap) + np.sum(P_cap) + np.sum(P_L) + np.sum(P_cond_loss) + np.sum(P_sw_loss) + np.sum(P_coss) + np.sum(P_gate_driver)

    if 0:
        print("P_Rcap "+str(np.sum(P_Rcap)))
        print("P_cap "+str(np.sum(P_cap)))
        print("P_cond_loss "+str(np.sum(P_cond_loss)))
        print("P_sw_loss "+str(np.sum(P_sw_loss)))
        print("P_coss "+str(np.sum(P_coss)))
        print("P_gate_driver "+str(np.sum(P_gate_driver)))
        print("P_src "+str(P_src))
        print("P_D_con "+str(np.sum(P_D_con)))

    return err_report, Power_loss, P_src






