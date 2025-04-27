"""Utility functions for the airfoil problem."""

import numpy as np
import numpy.typing as npt
import pandas as pd


def _extract_connectivities(df_slice: pd.DataFrame) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Extract node connectivities from the dataframe slice.

    Args:
        df_slice (pd.DataFrame): A slice of a dataframe.

    Returns:
        tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: NodeC1 and NodeC2 arrays
    """
    node_c1 = np.array(df_slice["NodeC1"].dropna().values).astype(int)  # A list of node indices
    node_c2 = np.array(df_slice["NodeC2"].dropna().values).astype(int)  # A list of node indices
    return node_c1, node_c2


def _identify_segments(connectivities: npt.NDArray[np.int32]) -> tuple[list[int], list[int], npt.NDArray[np.float32]]:
    """Identify segment breaks and assign segment IDs.

    Args:
        connectivities (npt.NDArray[np.int32]): Array of node connections

    Returns:
        tuple[list[int], list[int], npt.NDArray[np.float32]]: Start indices, end indices, and segment IDs
    """
    id_breaks_start = [0]
    id_breaks_end = []
    prev_id = 0
    segment_ids = np.zeros(len(connectivities), dtype=np.float32)
    seg_id = 0

    for j in range(len(connectivities)):
        if connectivities[j][0] - 1 != prev_id:
            # This means that we have a new set of points
            id_breaks_start.append(connectivities[j][0] - 1)
            id_breaks_end.append(prev_id)
            seg_id += 1
        segment_ids[j] = seg_id
        prev_id = connectivities[j][1] - 1

    id_breaks_end.append(j)
    return id_breaks_start, id_breaks_end, segment_ids


def _order_segments(
    coords_x: npt.NDArray[np.float32],
    coords_y: npt.NDArray[np.float32],
    id_breaks_start: list[int],
    id_breaks_end: list[int],
    unique_segment_ids: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Order segments based on their spatial relationships.

    Args:
        coords_x (npt.NDArray[np.float32]): X coordinates
        coords_y (npt.NDArray[np.float32]): Y coordinates
        id_breaks_start (list[int]): Start indices of segments
        id_breaks_end (list[int]): End indices of segments
        unique_segment_ids (npt.NDArray[np.int32]): Unique segment identifiers

    Returns:
        npt.NDArray[np.int32]: Ordered segment IDs
    """
    seg_coords_start_x = coords_x[id_breaks_start]
    seg_coords_start_y = coords_y[id_breaks_start]
    seg_coords_end_x = coords_x[id_breaks_end]
    seg_coords_end_y = coords_y[id_breaks_end]

    ordered_ids = [unique_segment_ids[0]]
    seg_coords_end_x_idx = seg_coords_end_x[0]
    seg_coords_end_y_idx = seg_coords_end_y[0]
    seg_coords_start_x_idx = seg_coords_start_x[0]
    seg_coords_start_y_idx = seg_coords_start_y[0]

    # Loop through and find the end or start of a segment that matches the start of the current segment
    while len(ordered_ids) < len(unique_segment_ids):
        # Calculate the distance between the end of the current segment and the start of all other segments
        diff_end_idx_start_tot = np.sqrt(
            np.square(seg_coords_end_x_idx - seg_coords_start_x) + np.square(seg_coords_end_y_idx - seg_coords_start_y)
        )
        diff_start_idx_start_tot = np.sqrt(
            np.square(seg_coords_start_x_idx - seg_coords_start_x) + np.square(seg_coords_start_y_idx - seg_coords_start_y)
        )

        # Get the minimum distance excluding the current ordered segments
        diff_end_idx_start_tot[np.abs(ordered_ids)] = np.inf
        diff_start_idx_start_tot[np.abs(ordered_ids)] = np.inf
        diff_end_idx_start_tot_id = np.argmin(diff_end_idx_start_tot)
        diff_end_idx_start_tot_min = diff_end_idx_start_tot[diff_end_idx_start_tot_id]

        # Get the minimum distance excluding the current segment
        diff_end_idx_end_tot = np.sqrt(
            np.square(seg_coords_end_x_idx - seg_coords_end_x) + np.square(seg_coords_end_y_idx - seg_coords_end_y)
        )
        diff_end_idx_end_tot[np.abs(ordered_ids)] = np.inf
        diff_end_idx_end_tot_id = np.argmin(diff_end_idx_end_tot)
        diff_end_idx_end_tot_min = diff_end_idx_end_tot[diff_end_idx_end_tot_id]

        # If the end of the current segment matches the start of another segment,
        # we have found the correct order
        if diff_end_idx_start_tot_min < diff_end_idx_end_tot_min:
            # Append the matching segment id to the ordered ids
            ordered_ids.append(diff_end_idx_start_tot_id)
            # Update the current segment end coordinates
            seg_coords_end_x_idx = seg_coords_end_x[diff_end_idx_start_tot_id]
            seg_coords_end_y_idx = seg_coords_end_y[diff_end_idx_start_tot_id]
        else:
            # If the end of the current segment matches the end of another segment,
            # the segment we append must be in reverse order
            # We make the sign of the segment id negative to indicate reverse order
            ordered_ids.append(-diff_end_idx_end_tot_id)
            # Update the current segment end coordinates;
            # Because of reversal, we use the start of the segment we are appending as the new end coordinates
            seg_coords_end_x_idx = seg_coords_start_x[diff_end_idx_end_tot_id]
            seg_coords_end_y_idx = seg_coords_start_y[diff_end_idx_end_tot_id]

    return np.array(ordered_ids)


def _reorder_coordinates(  # noqa: PLR0913
    coords_x: npt.NDArray[np.float32],
    coords_y: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int32],
    connectivities: npt.NDArray[np.int32],
    segment_ids: npt.NDArray[np.float32],
    new_seg_order: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Reorder coordinates based on segment order.

    Args:
        coords_x (npt.NDArray[np.float32]): X coordinates
        coords_y (npt.NDArray[np.float32]): Y coordinates
        indices (npt.NDArray[np.int32]): Original indices
        connectivities (npt.NDArray[np.int32]): Node connections
        segment_ids (npt.NDArray[np.float32]): Segment identifiers
        new_seg_order (npt.NDArray[np.int32]): New segment order

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]: Reordered coordinates and indices
    """
    coords_x_reordered = np.array([])
    coords_y_reordered = np.array([])
    indices_reordered = np.array([])

    for j in range(len(new_seg_order)):
        if new_seg_order[j] < 0:
            segment = np.nonzero(segment_ids == -new_seg_order[j])[0]
            coords_x_segment = coords_x[connectivities[segment] - 1][:, 0][::-1]
            coords_y_segment = coords_y[connectivities[segment] - 1][:, 0][::-1]
            indices_segment = indices[connectivities[segment] - 1][:, 0][::-1]
        else:
            segment = np.nonzero(segment_ids == new_seg_order[j])[0]
            coords_x_segment = coords_x[connectivities[segment] - 1][:, 0]
            coords_y_segment = coords_y[connectivities[segment] - 1][:, 0]
            indices_segment = indices[connectivities[segment] - 1][:, 0]

        coords_x_reordered = np.concatenate((coords_x_reordered, coords_x_segment))
        coords_y_reordered = np.concatenate((coords_y_reordered, coords_y_segment))
        indices_reordered = np.concatenate((indices_reordered, indices_segment))

    return coords_x_reordered, coords_y_reordered, indices_reordered


def _align_coordinates(
    coords_x_reordered: npt.NDArray[np.float32],
    coords_y_reordered: npt.NDArray[np.float32],
    indices_reordered: npt.NDArray[np.int32],
    err: float = 1e-4,
    err_x: float = 0.90,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Align coordinates based on maximum x values and mean y values.

    Args:
        coords_x_reordered (npt.NDArray[np.float32]): Reordered x coordinates
        coords_y_reordered (npt.NDArray[np.float32]): Reordered y coordinates
        indices_reordered (npt.NDArray[np.int32]): Reordered indices
        err (float): Error tolerance
        err_x (float): X coordinate error factor

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]: Aligned coordinates and indices
    """
    max_x = np.amax(coords_x_reordered) * err_x
    max_x_ids = np.argwhere(np.abs(coords_x_reordered - np.amax(coords_x_reordered)) < err).reshape(-1, 1)
    # Maintain (N,1) shape by using boolean indexing and reshaping
    mask = coords_x_reordered[max_x_ids.ravel()] >= max_x
    max_x_ids = max_x_ids[mask].reshape(-1, 1)

    # Get the y values at the maximum x values
    max_x_y_values = coords_y_reordered[max_x_ids]
    max_y_value = np.max(max_x_y_values)
    min_y_value = np.min(max_x_y_values)
    mean_y_value = (max_y_value + min_y_value) / 2

    # Get the id of the value closest to the mean y value at the x value of the maximum y value
    mean_y_value_sub_id = np.argmin(np.abs(max_x_y_values - mean_y_value))
    mean_y_value_id = max_x_ids[mean_y_value_sub_id].item()

    # Now reorder the coordinates such that the mean y value is first
    coords_x_reordered = np.concatenate((coords_x_reordered[mean_y_value_id:], coords_x_reordered[:mean_y_value_id]))
    coords_y_reordered = np.concatenate((coords_y_reordered[mean_y_value_id:], coords_y_reordered[:mean_y_value_id]))
    indices_reordered = np.concatenate((indices_reordered[mean_y_value_id:], indices_reordered[:mean_y_value_id]))

    return coords_x_reordered, coords_y_reordered, indices_reordered


def _clean_coordinates(
    coords_x_reordered: npt.NDArray[np.float32],
    coords_y_reordered: npt.NDArray[np.float32],
    indices_reordered: npt.NDArray[np.int32],
    err_remove: float = 1e-6,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Remove duplicate coordinates and close the loop.

    Args:
        coords_x_reordered (npt.NDArray[np.float32]): Reordered x coordinates
        coords_y_reordered (npt.NDArray[np.float32]): Reordered y coordinates
        indices_reordered (npt.NDArray[np.int32]): Reordered indices
        err_remove (float): Error tolerance for removal

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]: Cleaned coordinates and indices
    """
    removal_ids = np.where(np.abs(np.diff(coords_x_reordered) + np.diff(coords_y_reordered)) < err_remove)[0]
    indices_reordered = np.delete(indices_reordered, removal_ids)
    coords_x_reordered = np.delete(coords_x_reordered, removal_ids)
    coords_y_reordered = np.delete(coords_y_reordered, removal_ids)

    coords_x_reordered = np.concatenate((coords_x_reordered, np.expand_dims(coords_x_reordered[0], axis=0)))
    coords_y_reordered = np.concatenate((coords_y_reordered, np.expand_dims(coords_y_reordered[0], axis=0)))
    indices_reordered = np.concatenate((indices_reordered, np.expand_dims(indices_reordered[0], axis=0)))

    return coords_x_reordered, coords_y_reordered, indices_reordered


def reorder_coords(df_slice: pd.DataFrame) -> npt.NDArray[np.float32]:
    """Reorder the coordinates of a slice of a dataframe.

    Args:
        df_slice (pd.DataFrame): A slice of a dataframe.

    Returns:
        npt.NDArray[np.float32]: The reordered coordinates.
    """
    # Extract connectivities
    node_c1, node_c2 = _extract_connectivities(df_slice)
    connectivities = np.concatenate((node_c1.reshape(-1, 1), node_c2.reshape(-1, 1)), axis=1)

    # Get coordinates
    coords_x = df_slice["CoordinateX"].to_numpy()
    coords_y = df_slice["CoordinateY"].to_numpy()
    indices = np.arange(len(df_slice))

    # Identify segments
    id_breaks_start, id_breaks_end, segment_ids = _identify_segments(connectivities)
    unique_segment_ids = np.arange(len(id_breaks_start))

    # Order segments
    new_seg_order = _order_segments(coords_x, coords_y, id_breaks_start, id_breaks_end, unique_segment_ids)

    # Reorder coordinates
    coords_x_reordered, coords_y_reordered, indices_reordered = _reorder_coordinates(
        coords_x, coords_y, indices, connectivities, segment_ids, new_seg_order
    )

    # Align coordinates
    coords_x_reordered, coords_y_reordered, indices_reordered = _align_coordinates(
        coords_x_reordered, coords_y_reordered, indices_reordered
    )

    # Clean coordinates
    coords_x_reordered, coords_y_reordered, indices_reordered = _clean_coordinates(
        coords_x_reordered, coords_y_reordered, indices_reordered
    )

    return np.array([coords_x_reordered, coords_y_reordered])


def scale_coords(
    coords: npt.NDArray[np.float64],
    blunted: bool = False,  # noqa: FBT001, FBT002
    xcut: float = 0.99,
    min_trailing_edge_indices: float = 6,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Scales the coordinates to fit in the design space.

    Args:
        coords (np.ndarray): The coordinates to scale.
        blunted (bool): If True, the coordinates are assumed to be blunted.
        xcut (float): The x coordinate of the cut, if the coordinates are blunted.
        min_trailing_edge_indices (int): The minimum number of trailing edge indices to remove.

    Returns:
        np.ndarray: The scaled coordinates.
    """
    # Test if the coordinates are blunted or not
    if not (blunted) and is_blunted(coords):
        blunted = True
        print(
            "The coordinates may be blunted. However, blunted was not set to True. Will set blunted to True and continue, but please check the coordinates."
        )

    if not (blunted):
        xcut = 1.0

    # Scale x coordinates to be xcut in length
    airfoil_length = np.abs(np.max(coords[0, :]) - np.min(coords[0, :]))

    # Center the coordinates around the leading edge and scale them
    coords[0, :] = xcut * (coords[0, :] - np.min(coords[0, :])) / airfoil_length
    airfoil_length = np.abs(np.max(coords[0, :]) - np.min(coords[0, :]))

    # Shift the coordinates to be centered at 0 at the leading edge
    leading_id = np.argmin(coords[0, :])
    y_dist = coords[1, leading_id]
    coords[1, :] += -y_dist
    # Ensure the first and last points are the same
    coords[0, 0] = xcut
    coords[0, -1] = xcut
    coords[1, -1] = coords[1, 0]
    # Set the leading edge location

    if blunted:
        coords_x = coords[0, :]
        # Get all of the trailing edge indices, i.e where consecutive x coordinates are the same
        err = 1e-5
        x_gt = np.max(coords_x) * 0.99
        trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < err)[0]
        trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < err)[0]
        # Include any indices that are in either list
        trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
        trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]

        err = 1e-4
        err_stop = 1e-3
        while len(trailing_edge_indices) < min_trailing_edge_indices:
            trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < err)[0]
            trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < err)[0]
            # Include any indices that are in either list
            trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
            trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]
            err *= 1.5
            if err > err_stop:
                break

        # Remove the trailing edge indices from the coordinates
        coords = np.delete(coords, trailing_edge_indices[1:-1], axis=1)

    return coords, blunted


def calc_off_wall_distance(  # noqa: PLR0913
    mach: float,
    reynolds: float,
    freestreamTemp: float = 300.0,  # noqa: N803
    reynoldsLength: float = 1.0,  # noqa: N803
    yplus: float = 1,
    R: float = 287.0,  # noqa: N803
    gamma: float = 1.4,
) -> float:
    """Estimation of the off-wall distance for a given design.

    The off-wall distance is calculated using the Reynolds number and the freestream temperature.
    """
    # ---------------------------
    a = np.sqrt(gamma * R * freestreamTemp)
    u = mach * a
    # ---------------------------
    # Viscosity from Sutherland's law
    ## Sutherland's law parameters
    mu0 = 1.716e-5
    T0 = 273.15  # noqa: N806
    S = 110.4  # noqa: N806
    mu = mu0 * ((freestreamTemp / T0) ** (3 / 2)) * (T0 + S) / (freestreamTemp + S)
    # ---------------------------
    # Density
    rho = reynolds * mu / (reynoldsLength * u)
    ## Skin friction coefficient
    Cf = (2 * np.log10(reynolds) - 0.65) ** (-2.3)  # noqa: N806
    # Wall shear stress
    tau = Cf * 0.5 * rho * (u**2)
    # Friction velocity
    uTau = np.sqrt(tau / rho)  # noqa: N806
    # Off wall distance
    return yplus * mu / (rho * uTau)


def is_blunted(coords: npt.NDArray[np.float64], delta_x_tol: float = 1e-5) -> bool:
    """Checks if the coordinates are blunted or not.

    Args:
        coords (np.ndarray): The coordinates to check.
        delta_x_tol (float): The tolerance for the x coordinate difference.

    Returns:
        bool: True if the coordinates are blunted, False otherwise.
    """
    # Check if the coordinates going away from the tip have a small delta y
    coords_x = coords[0, :]
    # Get all of the trailing edge indices, i.e where consecutive x coordinates are the same
    x_gt = np.max(coords_x) * 0.99
    trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < delta_x_tol)[0]
    trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < delta_x_tol)[0]
    # Include any indices that are in either list
    trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
    trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]

    # check if we have no trailing edge indices
    return not len(trailing_edge_indices) <= 1


def calc_area(coords: npt.NDArray[np.float32]) -> float:
    """Calculates the area of the airfoil.

    Args:
        coords (np.ndarray): The coordinates of the airfoil.

    Returns:
        float: The area of the airfoil.
    """
    return 0.5 * np.absolute(
        np.sum(coords[0, :] * np.roll(coords[1, :], -1)) - np.sum(coords[1, :] * np.roll(coords[0, :], -1))
    )
