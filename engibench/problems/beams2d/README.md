# Beams 2D (MBB Beam)

**Lead**: Arthur Drake @arthurdrake1

Beams 2D is a benchmark problem that aims to optimize a 2D MBB beam using the structural topology optimization (TO) approach. This is based on the 88-line code by Andreassen et al. (2011). We allow the user to specify the required set of boundary conditions and other parameters prior to optimization.

## Side notes

Here is the script I've used to generate the dataset conditions. Please note that `max_iter = 100` and it is assumed that `nelx = 2*nely`. This yields a total of 7623 samples, or 2541 samples for each of the three image resolutions.

```python
all_params = [
    np.array([25, 50, 100]),                        # nely (nelx = 2*nely)
    np.round(np.linspace(0.15, 0.4, 21), 4),        # volfrac
    np.round(np.linspace(1.5, 4.0, 11), 4),         # rmin
    np.round(np.linspace(0, 0.5, 11), 4)            # forcedist
]

params = np.array(np.meshgrid(*all_params)).T.reshape(-1, len(all_params))

```

Here is the script I've used to upload the data to HF:

```python

# TODO: Update this with the new script ASAP

```
