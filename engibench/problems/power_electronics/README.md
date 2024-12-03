# Title

## Problem Statement
no best solution

## Dataset
The dataset consists of input (netlist files) and output (DcGain, Voltage Ripple and Efficiency) of the ngSpice simulator. There is no ``optimal'' solutions of the circuit parameters provided. 

Since all the circuits in the same problem share the same circuit topology, we provide 2 forms of input:

1. Circuit parameters only. For exapmle, each circuit parameter of topology 0 is a __ dimensinal np.ndarray.
2. Networkx objects. Node labels TODO.

The direct input netlist files are stored in the folder data/netlist/, with the name of TODO. You can pick any one of them to test the installation of ngSpice.

In correspondence to the input, the output also has 2 forms:
1. np.ndarray. [DcGain, VoltageRipple, Efficiency]
2. Networkx objects. Graph attribute.

## Installation of ngSpice
### 1. Windows
Directly use the provided ngSpice64/ folder. Remember to keep the .log files and the /lib folder in .gitignore.

### 2. Ubuntu
sudo apt update
sudo apt install ngspice

Available in Ubuntu 20, 22 and 24. Not sure about earlier versions of Ubuntu.
``pip install`` won't work.
