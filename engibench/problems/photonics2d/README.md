# Photonics2D

**Lead**: Mark Fuge @markfuge

Photonics2D is an example photonics problem (a two wavelength Demultiplexer) that is essentially a EngiBench compatible wrapper for the open source [`ceviche` Finite Difference Frequency Domain (FDFD) solver](https://github.com/fancompute/ceviche). This problem essentially is just wrapping the Optical Inverse Design demo shown [in this GitHub Repo](https://github.com/fancompute/workshop-invdesign) to make it compatible with EngiBench and with little further modification therein.

## Constraints

### Theoretical constraints
- $\lambda_i > 0$
- $r_{blur} \geq 0$
- num_elems_x $> 0$
- num_elems_y $> 0$

### Implementation constraint, error
- $\lambda_i \in [0.5,\infty]$
- $r_{blur} \geq 0$
- num_elems_x $>60$, otherwise there is not space between input and output ports
- num_elems_y $\geq 105$, otherwise waveguide outputs merge together

### Implementation constraint, warning
- $\lambda_i \in [0.5,1.5]$
- $r_{blur} \in [0,5]$
- num_elems_x $\in [90,200]$
- num_elems_y $\in [110,300]$
- In general, as the gap between the two wavelengths gets very large (e.g., on the opposite ends of the wavelength bounds), the optimization can have difficulty converging to a reasonable result. This depends on the size of the domain and the specific wavelengths though, so it is tricky to put a hard number on. We could be conservative and just go with bounds on the largest and smallest n_elems and then compute sets of other numbers that work for all cases, but this could be quite restrictive. It might be easier to just compute the full range of wavelength differences and then just discard the simulations that fail.
- Likewise, if the blur radius is high, then many fine details will be lost and it will in general be difficult to optimize particularly small differences in wavelengths. Again, this is n_elems and wavelength dependent (e.g., smaller wavelengths require fine details)
- However, as you increase the size of the domain (num_elems_x, num_elems_y), the maximum blur radius before you face convergence issues also increases.
- Increasing num_elems_x and num_elems_y beyond the upper ranges will not break the simulator, but it will become very slow and, for the wavelength bounds considered here, there is not much benefit to increasing this size.
