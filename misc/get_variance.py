from __future__ import print_function, division
import numpy as np
import nibabel as nib
from features import part_distance
import additional_functionality_ben as ad


def find_part_variance(pat1str, options1, p):

    print("Training:")

    folder1 = options1["folder"]
    type1 = options1["type"]

    # Loading the data
    tum = {}
    tum_var = {}
    part_var = {}
    fr = {}
    ut = {}
    ut_var = {}

    tum["x"] = np.zeros((len(pat1str), len(p)+1))
    tum["y"] = np.zeros((len(pat1str), len(p)+1))
    tum["z"] = np.zeros((len(pat1str), len(p)+1))
    tum_var["x"] = np.zeros((len(pat1str), len(p)+1))
    tum_var["y"] = np.zeros((len(pat1str), len(p)+1))
    tum_var["z"] = np.zeros((len(pat1str), len(p)+1))
    part_var["x"] = np.zeros((len(pat1str), len(p)+1))
    part_var["y"] = np.zeros((len(pat1str), len(p)+1))
    part_var["z"] = np.zeros((len(pat1str), len(p)+1))
    fr["x"] = np.zeros((len(pat1str)))
    fr["y"] = np.zeros((len(pat1str)))
    fr["z"] = np.zeros((len(pat1str)))

    ut["x"] = np.zeros((len(pat1str), len(p)+1))
    ut["y"] = np.zeros((len(pat1str), len(p)+1))
    ut["z"] = np.zeros((len(pat1str), len(p)+1))
    ut_var["x"] = np.zeros((len(pat1str), len(p)+1))
    ut_var["y"] = np.zeros((len(pat1str), len(p)+1))
    ut_var["z"] = np.zeros((len(pat1str), len(p)+1))

    for ii in range(len(pat1str)):

        if (ii+1) % 4 != 0:
            print(pat1str[ii], " ", end="")
        else:
            print(pat1str[ii])

        folder_save = folder1 + pat1str[ii] + type1

        # load super label and feature images
        im1 = nib.load(folder_save + pat1str[ii] + "slic_reg.nii")
        slic_reg = im1.get_data()
        slic_reg = np.asarray(slic_reg)
        vox_size = ad.image.get_voxel_size_from_nifti(im1)

        # 1 Extract locations of supervoxels
        slic_reg = np.ascontiguousarray(slic_reg, dtype=np.int16)
        supervoxel_coords, supervoxel_zmaxmin = ad.supervoxel_c.get_midpoints(slic_reg, voxel_size=vox_size)

        # A) Features features and labels for the supervoxels
        data = np.load(folder1 + pat1str[ii] + type1 + pat1str[ii] + "features.npz")
        # super_feat1 = data["super_feat"]
        super_label1 = data["super_label"]
        # just extracting tumour label

        # distance and variance of additional between landmarks for each image
        patfront = [(slic_reg.shape[0]/2)*vox_size[0],
                    (slic_reg.shape[1]-1)*vox_size[1],
                    (slic_reg.shape[2]/2)*vox_size[2]]
        pdist = part_distance(super_label1, supervoxel_coords, part_labels=p, patfront=patfront)

        xyz = {"x": 0, "y": 1, "z": 2}
        for keys, value in xyz.iteritems():
            # Storing the distances for all cases
            tum[keys][ii, 0] = int(pat1str[ii][4:])
            tum[keys][ii, 1:] = pdist["tumour"][keys]
            tum_var[keys][ii, 0] = int(pat1str[ii][4:])
            tum_var[keys][ii, 1:] = pdist["tumour"][keys + "var"]
            part_var[keys][ii, 1:] = pdist["part_var"][keys]
            fr[keys][ii] = pdist["front"][keys][3]

    xyz = {"x": 0, "y": 1, "z": 2}

    # NB: part / value correspondence changes
    rel_reg = [p["lumen"], p["tumour"], p["bladder"]]

    rlen = len(rel_reg)
    mean1 = {}
    var1 = {}

    mean1["tumourb"], var1["tumourb"] = np.zeros((rlen, 3)), np.zeros((rlen, 3))

    for keys, value in xyz.iteritems():
        # Compute the mean and variance of the pairwise set of labels for the whole dataset
        mean1["tumourb"][:, value] = np.mean(tum[keys][:, rel_reg], axis=0)
        # Option 1: variance generated from inter case mean variation.
        var1["tumourb"][:, value] = np.var(tum[keys][:, rel_reg], axis=0)

    return mean1, var1


def main(options1):

    # Write parameters to a dataframe

    dice1 = np.zeros((len(options1["patient"]), 5))
    cc = 0

    # Get a dictionary of patient genders

    roc_data = []
    roc_s_data = []

    # Define the patient for testing

    p_testi = 6
    p_train = []
    p_test = 'RIT' + str(p_testi).zfill(3)
    print("Test case: ", p_test)

    for pp in options1["pat_loocv"]:
        candidate1 = 'RIT' + str(pp).zfill(3)
        p_train.append(candidate1)

    # remove test case from training
    p_train.remove(p_test)

    # ~~~~~~~~~~~~~~~ Main script ~~~~~~~~~~~~~~~~ #

    # find and save part variance for a dataset
    part_names = {"lumen": 1, "tumour": 2, "wall": 3, "bladder": 4}
    mean1, var1 = find_part_variance(p_train, options1, part_names)
    np.savez('learned_part_variance.npz', mean1=mean1['tumourb'], var1=var1['tumourb'])

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main Script
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Global definitions and options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    options1 = {}
    options1["plot"] = True  # Plot outputs
    options1["save"] = False  # Save images
    options1["print"] = False  # Print additional outputs in the terminal
    options1["tumour_part_thresh"] = 0.1  # motion
    options1["type"] = '/PRE/'
    options1["remove_small_regions"] = True

    options1["weight_appearance"] = 0.5  # appearance of child additional
    options1["weight_original"] = 0.5  # appearance of base part
    options1["weight_dist"] = 0.5  # distance of child additional
    options1["roc_calculate"] = True

    options1["folder"] = "/data1/Results/1_RIT_motion8/"


    main(options1)

