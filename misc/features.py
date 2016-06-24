"""

Generate part distance features
mean and variance of the distance between parts

Benjamin Irving

"""

from __future__ import division, print_function

import numpy as np


def part_distance(supervoxel_labels, supervoxel_locations, part_labels, patfront=None):

    """
    Calculate the distance between each pair of additional for a single case

    Assuming a normal distribution
    http://mathworld.wolfram.com/NormalDifferenceDistribution.html

    Mean
    /mu_{x-y} = /mu_x - /mu_y
    Variance
    /sigma^2_{X-Y} = /sigma^2_X + /sigma^2_Y


    @param supervoxel_labels:
    @param supervoxel_locations:
    @param part_labels:
    @return: part_dist: distance between each part label
    """
    p = part_labels
    xyz = {"x": 0, "y": 1, "z": 2}

    # number of additional
    num_parts = len(p)+1

    # variance of the additional
    part_var = {"mean": np.zeros((num_parts, 3)),
                "var": np.zeros((num_parts, 3)),
                "mean_diff": np.zeros((num_parts, num_parts, 3)),
                "var_diff": np.zeros((num_parts, num_parts, 3))}

    for pp in part_labels.keys():
        # calculate mean and variance of the part
        part_coords = supervoxel_locations[supervoxel_labels == p[pp], :]
        part_var["mean"][p[pp], :] = np.mean(part_coords, axis=0)
        part_var["var"][p[pp], :] = np.var(part_coords, axis=0)

    part_var["tumour"] = {}
    part_var["part_var"] = {}
    part_var["front"] = {}
    part_var["uterus"] = {}

    for key, value in xyz.iteritems():
        # 1) Mean
        # Tumour base: Mean x, y, z
        part_var["tumour"][key] = np.around(part_var["mean"][1:, value] - part_var["mean"][p["tumour"], value], 2)
        # 2) Variance in the part based on supervoxel locations (not currently used)
        # Tumour base: Variance x, y, z
        part_var["tumour"][key + "var"] = np.around(part_var["var"][1:, value] + part_var["var"][p["tumour"], value], 2)

        # 3) Part var
        # Variance in the part based on supervoxel locations (not currently used)
        # Tumour base: Variance x, y, z
        part_var["part_var"][key] = np.around(part_var["var"][1:, value], 2)

        # 4) Find patient front
        if patfront is not None:
            # Tumour base: Mean x, y, z
            part_var["front"][key] = np.around(part_var["mean"][1:, value] - patfront[value], 2)

        # 5) Distance from uterus
        if 'uterus' in p.keys():
            # 1) Mean
            # Tumour base: Mean x, y, z
            part_var["uterus"][key] = np.around(part_var["mean"][1:, value] - part_var["mean"][p["uterus"], value], 2)
            # 2) Variance in the part based on supervoxel locations (not currently used)
            # Tumour base: Variance x, y, z
            part_var["uterus"][key + "var"] = np.around(part_var["var"][1:, value] + part_var["var"][p["uterus"], value], 2)

    # Distance
    return part_var