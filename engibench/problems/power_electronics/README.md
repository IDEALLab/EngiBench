# Title

## Problem Statement

### Design Variables
Insert the wave plot of PWL switch and pulse switch.
[[Youtube] LTSpice Pulse Voltage Parameters](https://www.youtube.com/watch?v=5sYnePkanfU)

- `GS#_Ts`: The time of a specific point in a PWL voltage source. In our case, it is equivalent to `Tperiod` (LTSpice syntax).
  It is set to the **constant** `GS#_Ts` = 5e-6. So we do not count it as a design variable.
- `GS#_T1`: The time of a specific point in a PWL voltage source. In our case, this is the "off" time and equals (`Tdelay` + `Trise`).
  All switches share the same `T1`, meaning that all switches will switch their on/off states (including on to on and off to off) at the same time.
- `GS#_T2`: The time of a specific point in a PWL voltage source. In our case, `GS#_T2` = `GS#_T1` = 5e-6. So we do not count it as a design variable.
- `GS#_L1` and `GS#_L2`: Binary. Each switch has a pair of {`GS#_L1`, `GS#_L2`}, which is the on/off pattern of the switch. For example, {`GS3_L1`=1, `GS#_L2`=0} means switch 3 first turns on for (approximately) `GS3_T1` seconds, then turns off for (approximately) (`GS_Ts` - `GS_T1`) seconds.

In summary, the dimension of the design variable is `n_C + n_L + 1 (for T1) + 2 * n_S`.

## Dataset
Since all the circuits in the same problem share the same circuit topology,

## Installation of ngSpice
### 1. Windows
Directly use the provided `ngSpice64/` folder.

Remember to keep the `.log` files and the `/lib` folder in `.gitignore`.

### 2. Ubuntu
``` bash
sudo apt update
sudo apt install ngspice
```

Available in Ubuntu 20, 22 and 24. Not sure about earlier versions of Ubuntu.

Note: ``pip install`` won't work.
