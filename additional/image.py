
from __future__ import division, print_function, absolute_import

import numpy as np


def get_voxel_size_from_nifti(img):

    # Determining ratio of voxel sizes
    hdr = img.get_header()
    raw1 = hdr.structarr
    pixdim = raw1['pixdim']
    vox_size = np.around([pixdim[1], pixdim[2], pixdim[3]], 2)
    return vox_size