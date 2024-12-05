"""Power Electronics problem."""

from __future__ import annotations
from typing import Any

import os
# import shutil
import subprocess
import numpy as np
# import networkx as nx  # TODO: for dataset

from utils.str_to_value import str_to_value
import utils.dc_dc_efficiency_ngspice as dc_lib_ng
import utils.data_sheet as ds
import utils.component as cmpt

from engibench.core import Problem


def build(**kwargs) -> PowerElectronics:
    """Builds an Power Electronics problem.

    Args:
        **kwargs: Arguments to pass to the constructor.
    """
    return PowerElectronics(**kwargs)


class PowerElectronics(Problem):
    r"""Power Electronics parameter optimization problem.

    ## Problem Description
    This problem tries to find the optimal circuit parameters for a given circuit topology. 
    The ngSpice simulator takes the topology and parameters as input and returns DcGain and Voltage Ripple.  Then Efficiency is calculated from
    As a single-objective optimization problem,

    ## Design space
    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates in [0., 1.) per design) that define the airfoil shape.
    Dependent on the dataset. In specific, TODO:

    ## Objectives
    The objectives are defined by the following parameters:
    - `DcGain`: Ratio of load vs. input voltage. It's a preset constant and the simulation result should be as close to the constant as possible.
    - `Voltage Ripple`: Fluctuation of voltage on the load side. The lower the better. 
    - `Efficiency`: Range is (0, 1). The higher the better.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    N/A

    ## Dataset
    TODO: The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/airfoil_2d).
    Networkx objects + netlists.
    New simulation data points can be appended locally. 

    ## Simulator
    Supports both Ubuntu ngspice package and Windows ngspice.exe.

    ## Lead
    Xuliang Dong @ liangXD523
    """

    # input_space = str
    # possible_objectives: frozenset[tuple[str, str]] = frozenset(
    #     {
    #         ("cd", "minimize"),
    #         ("cl", "maximize"),
    #     }
    # )
    # boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
    #     {
    #         ("s0", 3e-6),
    #         ("marchDist", 100.0),
    #     }
    # )
    # design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)  #TODO
    # dataset_id = "IDEALLab/airfoil_2d"
    # container_id = "mdolab/public:u22-gcc-ompi-stable"
    # _dataset = None



    def __init__(self, original_netlist_path: str, bucket_id: str) -> None:
        """Initializes the Power Electronics problem."""
        super().__init__()
        # self.seed = None  # may be useful in the future

        self.netlist_dir = '../data/netlist/'
        if not os.path.exists(self.netlist_dir):
            os.makedirs(self.netlist_dir)

        self.raw_file_dir = '../data/raw_file/'
        if not os.path.exists(self.raw_file_dir):
            os.makedirs(self.raw_file_dir)

        self.log_file_dir = '../data/log_file/'
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

        self.components = {'V':0, 'R':1, 'C':2, 'S':3, 'L':4, 'D':5}
        
        self.original_netlist_path = original_netlist_path  # E.g. '../data/netlist/5_4_3_6_10-dcdc_converter_1.net'
        self.rewrite_netlist_path: str = ""  # It will be modified in __rewrite_netlist() depending on mode='batch' or mode='control'. 
        self.netlist_name = self.original_netlist_path.split('/')[-1].removesuffix('.net')  # python 3.9 and newer
        self.log_file_path = f'../data/log_file/{self.netlist_name}.log'
        self.raw_file_path = f'../data/raw_file/{self.netlist_name}.raw'

        self.bucket_id = bucket_id  # Alternatively, we can get this from self.original_netlist_path.
        self.edge_map: dict[str, list[int]] = {'V0':[0,0], 'R0':[0,0]}  
        self.cmp_edg_str: str = ""
        
        # components of the design variable
        self.capacitor_val: list[float] = []
        self.inductor_val: list[float] = []
        self.switch_T1: list[float] = []  # TODO: Need a range
        self.switch_T2: list[float] = []  # Binary. TODO: might need to specify this.  
        self.switch_L1: list[float] = []  # Binary.
        self.switch_L2: list[float] = []  # Binary.
        # TODO: self.design_variable

        self.cmp_cnt: dict[str, int] = {'V':1, 'R':1, 'C':0, 'S':0, 'L':0, 'D':0}

        self.simulation_results: np.ndarray = np.array([])

    # def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
    #     """Resets the simulator and numpy random to a given seed.

    #     Args:
    #         seed (int, optional): The seed to reset to. If None, a random seed is used.
    #         cleanup (bool): Deletes the previous study directory if True.
    #     """
    #     # docker pull image if not already pulled
    #     subprocess.run(["docker", "pull", self.container_id], check=True)
    #     if cleanup:
    #         shutil.rmtree(self.__study_dir + f"_{self.seed}")

    #     super().reset(seed)
    #     self.current_study_dir = self.__study_dir + f"_{self.seed}/"
    #     clone_template(template_dir=self.__template_dir, study_dir=self.current_study_dir)

    def __process_topology(self, sweep_dict) -> None:
        """
        Read from self.original_netlist_path to get the topology. With the argument sweep_dict (C, L, T1, T2, L1, L2), set variables for rewriting. 

        It only keeps component lines and .PARAM lines. So the following lines are discarded in this process: .model, .tran, .save etc. 
        ----------
        sweep_dict : dict. variable.
        """
        print(f'Processing topology from original netlist path: {self.original_netlist_path}')

        self.capacitor_val = sweep_dict['C_val']
        self.inductor_val = sweep_dict['L_val']
        self.switch_T1 = sweep_dict['T1_val']
        self.switch_T2 = sweep_dict['T2_val']
        self.switch_L1 = sweep_dict['L1_val']
        self.switch_L2 = sweep_dict['L2_val']

        calc_comp_count = {'V':0, 'R':0, 'C':0, 'S':0, 'L':0, 'D':0}
        ref_comp_count = {'V':1, 'R':1, 'C':0, 'S':0, 'L':0, 'D':0}
        num_comp = self.bucket_id.split("_")
        ref_comp_count['S'] = int(num_comp[0])
        ref_comp_count['D'] = int(num_comp[1])
        ref_comp_count['L'] = int(num_comp[2])
        ref_comp_count['C'] = int(num_comp[3])

        file = open(self.original_netlist_path, "r") 
        for line in file:
            if line.strip()!= "":
                line = line.replace('=', ' ')  # to deal with .PARAM problems. See the comments below.
                                                # liang: Note that line still contains \n at the end of it!
                line_spl = line.split()
                if line_spl[0] == '.PARAM' and sweep_dict is None:
                    # e.g. .PARAM V0_value = 10
                    # e.g. .PARAM C0_value = 10u
                    # e.g. .PARAM L2_value=0.001. Note the whitespace! It appears in new files provided by RTRC.
                    # e.g. .PARAM GS2_L2=0
                    value = str_to_value(line_spl[-1])
                    # print(value)
                    if line_spl[1][0] == 'C':
                        self.capacitor_val.append(value)
                    elif line_spl[1][0] == 'L':
                        self.inductor_val.append(value)
                    elif 'T1' in line_spl[1]:
                        self.switch_T1.append(value)
                    elif 'T2' in line_spl[1]:
                        self.switch_T2.append(value)
                    elif 'L1' in line_spl[1]:
                        self.switch_L1.append(value)
                    elif 'L2' in line_spl[1]:
                        self.switch_L2.append(value)
                elif line[0] in self.components.keys():
                    line_spl = line.split(' ')[:3] 
                    
                    if 'RC' in line_spl[0] or '_GS' in line_spl[0]:
                        continue  # pass this line

                    self.edge_map[line_spl[0]] = [int(line_spl[1]), int(line_spl[2])]

                    # print(repr(line))  # this actually works for printing a raw string. Also consider removing the quotes. See https://www.geeksforgeeks.org/python-raw-strings/
                    self.cmp_edg_str = self.cmp_edg_str + line  # Liang: do not add a '\n' at the end of this line.
                    calc_comp_count[line[0]] = calc_comp_count[line[0]] + 1
                
        file.close()
        
        if ref_comp_count != calc_comp_count:
            print("Error - process_topology component check")
        self.cmp_cnt = ref_comp_count
    
    def __rewrite_netlist(self, mode='control') -> None:
        self.rewrite_netlist_path = f'../data/netlist/rewrite_{mode}_{self.netlist_name}.net'
        print(f'rewriting netlist to: {self.rewrite_netlist_path}')

        file = open(self.rewrite_netlist_path, "w")
        
        self.cmp_edg_str =  "* rewrite netlist\n" + self.cmp_edg_str

        RC_str = ""
        for i in range(len(self.capacitor_val)):
            RC_str = RC_str + "RC" + str(i) + " " + str(self.edge_map['C' + str(i)][0]) + " " + str(self.edge_map['C' + str(i)][1]) + " 100meg\n"

        self.cmp_edg_str = self.cmp_edg_str + RC_str + "\n" + ".PARAM V0_value=1000\n"
        
        for i in range(len(self.capacitor_val)):
            self.cmp_edg_str = self.cmp_edg_str + ".PARAM C" + str(i) + "_value=" + str(capacitor_val[i])+"\n"

        for i in range(len(self.inductor_val)):    
            self.cmp_edg_str = self.cmp_edg_str + ".PARAM L" + str(i) + "_value=" + str(inductor_val[i])+"\n"

        self.cmp_edg_str = self.cmp_edg_str + ".PARAM R0_value=10\n\n.model Ideal_switch SW (Ron=1m Roff=10Meg Vt=0.5 Vh=0 Lser=0 Vser=0)\n.model Ideal_D D\n\n"

        self.cmp_edg_str = self.cmp_edg_str + "V_GSref_D  gs_ref_D 0 pwl(0 1 {GS0_T1-10e-9} 1 {GS0_T1} 0 {GS0_T2-10e-9} 0 {GS0_Ts} 1) r=0\nV_GSref_Dc  gs_ref_Dc 0 pwl(0 0 {GS0_T1-10e-9} 0 {GS0_T1} 1 {GS0_T2-10e-9} 1 {GS0_Ts} 0) r=0\n"
        for i in range(len(self.switch_T1)): 
            self.cmp_edg_str = self.cmp_edg_str + "V_GS"+str(i)+" GS"+str(i)+ " 0 pwl(0 {GS"+str(i)+"_L1} {GS"+str(i)+"_T1-10e-9} {GS"+str(i)+"_L1} {GS"+str(i)+"_T1} {GS"+str(i)+"_L2}" \
                                                                        +" {GS"+str(i)+"_T2-10e-9} {GS"+str(i)+"_L2} {GS"+str(i)+"_Ts} {GS"+str(i)+"_L1}) r=0\n"

        for i in range(len(self.switch_T1)):
            self.cmp_edg_str = self.cmp_edg_str + ".PARAM GS"+str(i)+"_Ts="+str(self.switch_T2[i]*5)+"e-06\n"+ ".PARAM GS"+str(i)+"_T1="+str(self.switch_T1[i]*5)+"e-06\n"+ \
                            ".PARAM GS"+str(i)+"_T2="+str(self.switch_T2[i]*5)+"e-06\n" + ".PARAM GS"+str(i)+"_L1="+str(self.switch_L1[i])+"\n" + ".PARAM GS"+str(i)+"_L2="+str(self.switch_L2[i])+"\n"

        if mode == 'batch':
            """
            Here is an example:
            .save @C0[i] @RC0[i] @C1[i] @RC1[i] @C2[i] @RC2[i] @C3[i] @RC3[i] @C4[i] @RC4[i] @C5[i] @RC5[i] @L0[i] @L1[i] @L2[i] @S0[i] @S1[i] @S2[i] @S3[i] @S4[i] @D0[i] @D1[i] @D2[i] @D3[i] @R0[i] 
            .save all
            """
            self.cmp_edg_str = self.cmp_edg_str + "\n.save "
            for i in range(len(self.capacitor_val)):
                self.cmp_edg_str = self.cmp_edg_str + "@C"+str(i)+"[i] "
                self.cmp_edg_str = self.cmp_edg_str + "@RC"+str(i)+"[i] "
            for i in range(len(self.inductor_val)):
                self.cmp_edg_str = self.cmp_edg_str + "@L"+str(i)+"[i] "
            for i in range(len(self.switch_T1)): 
                self.cmp_edg_str = self.cmp_edg_str + "@S"+str(i)+"[i] "
            for i in range(self.cmp_cnt['D']): 
                self.cmp_edg_str = self.cmp_edg_str + "@D"+str(i)+"[i] "
                self.cmp_edg_str = self.cmp_edg_str + "@R0[i] " 
            self.cmp_edg_str = self.cmp_edg_str + "\n.save all\n"

            self.cmp_edg_str = self.cmp_edg_str + ".tran 5n 1.06m 1m 5n uic\n"
            # The following .meas(ure) can be replaced by .print, althogh the former is preferred.
            self.cmp_edg_str = self.cmp_edg_str + f".meas TRAN Vo_mean avg par('V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})') from=1m to=1.06m\n"
            self.cmp_edg_str = self.cmp_edg_str + ".meas TRAN gain param='Vo_mean/1000'\n"
            self.cmp_edg_str = self.cmp_edg_str + f".meas TRAN Vpp pp par('V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})') from=1m to=1.06m\n"
            self.cmp_edg_str = self.cmp_edg_str + ".meas TRAN Vpp_ratio param='Vpp/Vo_mean'\n" #Ripple voltage
            self.cmp_edg_str = self.cmp_edg_str + "\n.end"

        elif mode == 'control':
            self.cmp_edg_str = self.cmp_edg_str + "\n.control\nsave "
            for i in range(len(self.capacitor_val)):
                self.cmp_edg_str = self.cmp_edg_str + "@C"+str(i)+"[i] "
                self.cmp_edg_str = self.cmp_edg_str + "@RC"+str(i)+"[i] "
            for i in range(len(self.inductor_val)):
                self.cmp_edg_str = self.cmp_edg_str + "@L"+str(i)+"[i] "
            for i in range(len(self.switch_T1)): 
                self.cmp_edg_str = self.cmp_edg_str + "@S"+str(i)+"[i] "
            for i in range(self.cmp_cnt['D']): 
                self.cmp_edg_str = self.cmp_edg_str + "@D"+str(i)+"[i] "
                self.cmp_edg_str = self.cmp_edg_str + "@R0[i] " 
            self.cmp_edg_str = self.cmp_edg_str + "\n.save all\n"
            
            self.cmp_edg_str = self.cmp_edg_str + "tran 5n 1.06m 1m 5n uic\n"
            self.cmp_edg_str = self.cmp_edg_str + f"let Vdiff=V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})\n"
            self.cmp_edg_str = self.cmp_edg_str + f"meas TRAN Vo_mean avg Vdiff from=1m to=1.06m\n"
            self.cmp_edg_str = self.cmp_edg_str + f"meas TRAN Vpp pp Vdiff from=1m to=1.06m\n"
            self.cmp_edg_str = self.cmp_edg_str + "let Gain=Vo_mean/1000\nlet Vpp_ratio=Vpp/Vo_mean\nprint Gain, Vpp_ratio\nrun\nset filetype=binary\n"
            self.cmp_edg_str = self.cmp_edg_str + f"write {self.raw_file_path}\n"
            
            self.cmp_edg_str = self.cmp_edg_str + "quit\n.endc\n\n.end"
    
        file.write(self.cmp_edg_str)
        file.close()

    def __calculate_efficiency(self, exe=False) -> tuple[float, int, float, float]:
        capacitor_model, inductor_model, switch_model, diode_model = [], [], [], []
        max_comp_size = 12
        Ts = 5e-6
        Fs = 1/Ts

        for i in range(max_comp_size):
            switch_model.append(cmpt.MOSFET(dict(zip(ds.MOSFET_properties,ds.APT40SM120B))))
        for i in range(max_comp_size):
            diode_model.append(cmpt.Diode(dict(zip(ds.diode_properties,ds.APT30SCD65B))))

        try: 
            if exe:  # Windows: use the provided ngspice.exe
                cmd = ["../ngSpice64/bin/ngspice.exe", "-o", "../data/log_file/control_mode.log", self.rewrite_netlist_path]  # interactive mode with control section
            else:  # Ubuntu: use the ngspice package
                cmd = ["ngspice", "-o", "../data/log_file/control_mode.log", self.rewrite_netlist_path]  # interactive mode with control section

            subprocess.run(cmd,check = True, timeout =30) 

            for i in range(self.cmp_cnt['C']):
                #Assuming dissipiation factor = 5 at 200Khz freq;  ESR = Disspiation_factor/ 2*pi*f*C
                #Currenty we are using Parallel resistance (component "RC") as 100Meg
                #Using https://www.farnell.com/datasheets/2167237.pdf for ESR
                cap_model_values = [self.capacitor_val[i], 1/np.sqrt(10), 1e8]
                capacitor_model.append(cmpt.Capacitor(dict(zip(ds.capacitor_properties,cap_model_values))))
            for i in range(self.cmp_cnt['L']):
                #Using this inductor data sheet for now: https://www.eaton.com/content/dam/eaton/products/electronic-components/resources/data-sheet/eaton-fp1206-high-current-power-inductors-data-sheet.pdf
                ind_model_values = [self.inductor_val[i], 0.43e-3]
                inductor_model.append(cmpt.Inductor(dict(zip(ds.inductor_properties,ind_model_values))))

            try:
                err, P_loss, P_src = dc_lib_ng.metric_compute_DC_DC_efficiency_ngspice("../data/raw_file/control_mode.raw", 0.001, self.cmp_cnt, self.edge_map, capacitor_model, inductor_model, switch_model, diode_model, 10, 0, self.switch_L1, self.switch_L2, [float(ind)*Ts for ind in self.switch_T1], [float(ind)*Ts for ind in self.switch_T2], Fs)
                efficiency = (P_src - P_loss)/P_src

                error_report = err
                if efficiency < 0 or efficiency > 1:
                    error_report += 1 #bit 0 will be 1 to report invalid efficiency calculation
                #else:
                #    error_report = np.NAN if error_report==0 else error_report

            except ValueError as v: #In some conditions LTspice is not generating waveforms with invalid values
                efficiency = np.NAN   
                error_report = 16 #bit 4 will be 1 to report Process error such as invalid circuit
                P_loss = np.NAN
                P_src  = np.NAN

            except IndexError: #For some circuits the gate voltage is not created properly
                efficiency = np.NAN   
                error_report = 32 #bit 5 will be 1 to report Process error such as invalid circuit
                P_loss = np.NAN
                P_src  = np.NAN

        except subprocess.CalledProcessError as err:
            efficiency = np.NAN   
            error_report = 2 #bit 1 will be 1 to report Process error such as invalid circuit
            P_loss = np.NAN
            P_src  = np.NAN

        except subprocess.TimeoutExpired as e:   
            efficiency = np.NAN    
            error_report =  4 #bit 2 will be 1 to report Timeout 
            P_loss = np.NAN
            P_src  = np.NAN
        
        return efficiency, error_report, P_loss, P_src
    
    def __parse_log_file(self):
        with open(self.log_file_path, 'r') as log:
            lines = log.readlines()
            for line in lines:
                if line.strip()!= '':
                    parts = line.split()
                    if parts[0] == 'gain':
                        DcGain = float(parts[2])
                    elif parts[0] == 'vpp_ratio':
                        VoltageRipple = float(parts[2])
        return DcGain, VoltageRipple

    def simulate(self, design_variable: dict[str, float]) -> np.ndarray:
        """Simulates the performance of a Power Electronics design.

        Args:
            design (np.ndarray): The design to simulate.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        self.__process_topology(sweep_dict=design_variable)
        self.__rewrite_netlist(mode='control')
        Efficiency, _, _, _ = self.__calculate_efficiency(exe=False)
        DcGain, VoltageRipple = self.__parse_log_file()
        self.simulation_results = np.array([DcGain, VoltageRipple, Efficiency])
        
        return self.simulation_results
        

    def optimize(self):
        return NotImplementedError


if __name__ == "__main__":
    original_netlist_path = '../data/netlist/5_4_3_6_10-dcdc_converter_1.net'  # sweep 141
    bucket_id = '5_4_3_6_10'
    problem = PowerElectronics(original_netlist_path, bucket_id)

    sweep_data = {
        'C_val': [0.000015485, 0.00001948, 0.000015185, 0.000002442, 0.000009287, 0.000015377], 
        'L_val': [0.000354659, 0.000706596, 0.000195361],
        'T1_val': [0.867615857, 0.867615857, 0.867615857, 0.867615857, 0.867615857],
        'T2_val': [1, 1, 1, 1, 1],
        'L1_val': [1, 1, 0, 0, 1],
        'L2_val': [1, 1, 1, 0, 0],
        }
    problem.simulate(design_variable=sweep_data)   
    print(problem.simulation_results)

    # problem.reset(seed=0, cleanup=False)

    # dataset = problem.dataset

    # Get design and conditions from the dataset
    # design = np.array(dataset["initial"][0])  # type: ignore
    # config_keys = dataset.features.keys() - ["initial", "optimized"]
    # config = {key: dataset[key][0] for key in config_keys}

    # print(problem.optimize(design, config=config, mpicores=8))
