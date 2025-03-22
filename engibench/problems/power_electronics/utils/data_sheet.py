"""@Author: Naga Siva Srinivas Putta <nagasiva@umd.edu>."""

import numpy as np

# Diode data sheet

diode_properties = ["Dynamic Resistance", "Threshold Voltage", "Qrr"]

APT30SCD65B = [0.05, 1.5, 0]
IDM10G120C5 = [0.15, 1.5, 0]
IDH20G120C5 = [0.075, 1.5, 0]

# Capacitor data sheet

capacitor_properties = ["Capacitance", "ESR", "Parallel Resistance"]

# Inductor data sheet

inductor_properties = ["Inductance", "DCR"]

# MOSFET data sheet

MOSFET_properties = [
    "R_DS_on",
    "Coss vs Vds",
    "Turn-on delay time",
    "Rise time",
    "Turn-off delay time",
    "Fall time",
    "Gate charge total",
]

# https://www.mouser.com/datasheet/2/268/APT40SM120B_C-1593296.pdf

APT40SM120B_Coss = np.array(
    [
        [0.990445794, 2.93456e-09],
        [1.27803343, 2.80893e-09],
        [1.507625272, 2.56975e-09],
        [1.700286842, 2.35048e-09],
        [1.917568906, 2.14991e-09],
        [2.037335569, 1.85182e-09],
        [2.298434897, 1.5718e-09],
        [2.592323296, 1.41635e-09],
        [2.837101261, 1.31484e-09],
        [3.448585293, 1.20303e-09],
        [4.451366245, 1.06859e-09],
        [5.410774855, 9.77722e-10],
        [7.757476187, 8.43252e-10],
        [10.47358152, 7.49159e-10],
        [13.51909357, 6.65435e-10],
        [16.18848618, 6.08812e-10],
        [22.1879228, 5.32886e-10],
        [29.51291544, 4.66369e-10],
        [35.33805216, 4.33112e-10],
        [47.00435253, 3.79049e-10],
        [55.44483544, 3.51996e-10],
        [69.44969528, 3.17327e-10],
        [83.16278206, 2.90325e-10],
        [107.3448877, 2.57879e-10],
        [130.4726039, 2.39506e-10],
        [160.9982315, 2.15902e-10],
        [217.3963328, 1.8616e-10],
        [276.4375914, 1.65345e-10],
        [367.6991006, 1.44705e-10],
        [427.303345, 1.32375e-10],
        [568.3339358, 1.17596e-10],
        [700.9841022, 1.17703e-10],
        [890.8973933, 1.17826e-10],
        [1004.355467, 1.17887e-10],
    ]
)

APT40SM120B = [0.080167053, APT40SM120B_Coss, 10e-9, 6e-9, 32e-9, 16e-9, 130e-9]
