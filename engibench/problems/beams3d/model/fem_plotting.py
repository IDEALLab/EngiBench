"""Module for plotting a structural 3D Beams design with napari."""

import numpy as np


def plot_fem_3d(bcs, design) -> None:
    """Plot the structural design along with mechanical boundary conditions.

    Args:
        bcs (Dict[str, Any]): Dictionary specifying boundary conditions. Expected keys include:
            - "fixed_elements": Indices for fixed mechanical degrees of freedom.
            - "force_elements_x" (optional): Indices for x-direction force elements.
            - "force_elements_y" (optional): Indices for y-direction force elements.
            - "force_elements_z" (optional): Indices for z-direction force elements.
        design (npt.NDArray): The design array.

    Returns:
        None
    """
    fixed_elements = bcs.get("fixed_elements", np.zeros_like(design))

    force_elements_x = bcs.get("force_elements_x", np.zeros_like(design))
    force_elements_y = bcs.get("force_elements_y", np.zeros_like(design))
    force_elements_z = bcs.get("force_elements_z", np.zeros_like(design))
    force_elements = force_elements_x + force_elements_y + force_elements_z

    design = np.transpose(design, (2, 0, 1))
    fixed_elements = np.transpose(fixed_elements, (2, 1, 0))
    force_elements = np.transpose(force_elements, (2, 1, 0))

    import napari  # noqa: PLC0415

    viewer = napari.Viewer()
    viewer.add_image(design, name="rho", rendering="attenuated_mip")
    viewer.add_image(fixed_elements, name="fixed_elements", rendering="attenuated_mip", visible=False, colormap="green")
    viewer.add_image(force_elements, name="force_elements", rendering="attenuated_mip", visible=False, colormap="fire")

    viewer.dims.ndisplay = 3  # switch to 3D view
    napari.run()
