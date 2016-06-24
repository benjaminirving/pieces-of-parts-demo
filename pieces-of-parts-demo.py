from __future__ import print_function, division
import numpy as np
import nibabel as nib

import additional as ad
from part_class import PartVis as Parts

# Options
options1 = {}
options1["tumour_part_thresh"] = 0.1  # motion
options1["remove_small_regions"] = True
options1["roc_calculate"] = True
options1["roc_range"] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.18, 0.22,
                         0.26, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]

ptest = 'RIT006'
pvar = np.load('data/learned_part_variance.npz')
part_names = {"lumen": 0, "tumour": 1, "bladder": 2}
mean1 = pvar['mean1']
var1 = pvar['var1']

# Label probability (tumour) for each supervoxel using an LDA classifier before region processing
data2 = np.load('data/' + ptest + "_labels_lda.npz")
# tumour probabilities
super_prob_tumour = data2["tumour_prob"]
super_prob_tumour = super_prob_tumour[np.newaxis]
# lumen probabilities
super_prob_lumen = data2["lumen_prob"]
super_prob_lumen = super_prob_lumen[np.newaxis]
# bladder probabilities
super_prob_bladder = data2["bladder_prob2"]
super_prob_bladder = super_prob_bladder[np.newaxis]

# Get part locations
im1 = nib.load('data/' + ptest + "_slic_regions.nii")
slic_reg = im1.get_data()
slic_reg = np.asarray(slic_reg)
vox_size = ad.get_voxel_size_from_nifti(im1)

# 1 Extract locations of supervoxels
slic_reg = np.ascontiguousarray(slic_reg, dtype=np.int16)
supervoxel_coords, supervoxel_zmaxmin = ad.get_midpoints(slic_reg, voxel_size=vox_size)

# create model using part variances
# ~~~~~~~~~~~~~~ Parts Model Initialisation ~~~~~~~~~~~~~~~~
p1 = Parts(mean1, var1, part_names, weight_appearance=0.5, weight_original=0.5, weight_dist=0.5)

p1.plot_mv_scatter()

p1.get_test_candidate(locations=[supervoxel_coords],
                      part_potentials=[super_prob_lumen, super_prob_tumour, super_prob_bladder])

# Generate the part hypothesis for each supervoxel
p1.generate_part_hypothesis(supervoxel_coords=supervoxel_coords)

p1.create_probability_image(slic_reg)

# ~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# A) Features features and labels for the supervoxels
# just extracting tumour label

roi1 = nib.load('data/' + ptest + "_ground_truth.nii")
roi1 = roi1.get_data()

p1.labelled_tumour(thresh1=options1["tumour_part_thresh"],
                   tumour_constraint=options1["remove_small_regions"])

stats1 = p1.overlap_statistics(ground_roi=roi1)

roc1, roc1_s = p1.roc_statistics(thresh_vals_parts=options1["roc_range"],
                                 thresh_vals_super=options1["roc_range"],
                                 ground_roi=roi1,
                                 slic_reg=slic_reg)

slice1 = im1.shape[2] / 2.0

slice1 = np.around(slice1)
print("Slice: ", slice1)
p1.plot_probability_image(slice1)

print("Dice for ", ptest, ": ", stats1['dice'])


p1.show()

