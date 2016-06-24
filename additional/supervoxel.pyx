"""
Benjamin Irving

Cython functions to speed up computation

Cython version of supervoxel.py

Openmp parallel version is blanked out
"""

from __future__ import division, print_function

#from cython.parallel import prange

import numpy as np
cimport numpy as np

DTYPE = np.int16

ctypedef np.int16_t DTYPE_t

def get_midpoints(np.ndarray[np.int16_t, ndim=3] super_im, np.ndarray[np.float32_t, ndim=1] voxel_size):
    """
    Find supervoxel mid points when a supervoxel image is input

    @param super_im: 3D image where each supervoxel region is labelled
    @param voxel_size: Array of length 3 with voxel size
    @return: mean1 (centrepoint of each supervoxel), extremaz1 (extrema in x and y direction?)

    Example:

        slic_reg = np.ascontiguousarray(labels, dtype=np.int16)
        vox_size = np.array([1, 1, 1], np.float32)
        supervoxel_coords, supervoxel_zmaxmin = ad.supervoxel_c.get_midpoints(slic_reg, voxel_size=vox_size)

    """

    # unique labels in the supervoxels >= 0
    # (exclude negative labels as they can be used to define regions that are not of interest)
    cdef np.ndarray labels = np.unique(super_im[super_im >= 0])

    # centre of each supervoxel (label x (x, y, z))
    cdef np.ndarray mean1 = np.zeros((len(labels), 3), dtype=np.float32)
    cdef np.ndarray extremaz1 = np.zeros((len(labels), 2), dtype=np.float32)
    cdef np.ndarray counter = np.zeros((len(labels)), dtype=np.float32)

    # Declare all variables
    cdef int a0, a1, a2, lab1
    cdef int xx, yy, zz, ii

    # Get image dimensions
    a0 = super_im.shape[0]
    a1 = super_im.shape[1]
    a2 = super_im.shape[2]

    # Declare memory views of each array for much faster access
    cdef short[:, :, :] super_im_view = super_im
    cdef float[:, :] mean1_view = mean1
    cdef float[:, :] extremaz_view = extremaz1
    cdef float[:] counter_view = counter

    for xx in range(a0):
        for yy in range(a1):
            for zz in range(a2):

                # Loop over every voxel in the image and add the coordinates to the able of interest
                lab1 = super_im_view[xx, yy, zz]

                # Ignore negative labels
                if lab1 < 0:
                    continue

                # values added
                mean1_view[lab1, 0] += xx
                mean1_view[lab1, 1] += yy
                mean1_view[lab1, 2] += zz

                # find max and min values
                if zz > extremaz_view[lab1, 0]:
                    extremaz_view[lab1, 0] = zz

                if zz < extremaz_view[lab1, 1] or extremaz_view[lab1, 1] == 0:
                    extremaz_view[lab1, 1] = zz

                # Count how many instances have been added to each label to calculate mean
                counter_view[lab1] += 1.0

    for ii in range(len(labels)):

        mean1[ii, 0] = mean1[ii, 0]/counter[ii]*voxel_size[0]
        mean1[ii, 1] = mean1[ii, 1]/counter[ii]*voxel_size[1]
        mean1[ii, 2] = mean1[ii, 2]/counter[ii]*voxel_size[2]

        extremaz1[ii, 0] = extremaz1[ii, 0] * voxel_size[2]
        extremaz1[ii, 1] = extremaz1[ii, 1] * voxel_size[2]

    # return the centre of mass of each supervoxel, min and max in z direction
    return np.asarray(mean1), np.asarray(extremaz1)




