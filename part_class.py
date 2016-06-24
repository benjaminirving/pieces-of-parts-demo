from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import warnings
import timeit
import scipy.ndimage as ndi
from scipy.ndimage import morphology as mp
import pandas as pd

import additional_functionality_ben.create_im as crm
from additional_functionality_ben import radial_symmetry_weight


def normal_distribution_3d(dist_mean, part_sigma):
    """
    Generates probabilities for distances from the mean. Given sigma.

    :param dist_mean: (nx3) vector distance from mean (x - mu)
                     - rows are the number of points
                     - columns are the indices
    :param part_sigma: (3) vector part sigma (standard deviation)
    :return: prob: list of probabilities for each input
    """

    det1 = 1 / (part_sigma[0] * part_sigma[1] * part_sigma[2] * (2 * np.pi) ** (3 / 2))

    prob = det1 * np.exp(- 0.5 * (((dist_mean[:, 0] ** 2) / part_sigma[0] ** 2) +
                                  ((dist_mean[:, 1] ** 2) / part_sigma[1] ** 2) +
                                  ((dist_mean[:, 2] ** 2) / part_sigma[2] ** 2)))

    return prob


def log_normal_distribution_3d(dist_mean, part_sigma):

    """

    Generates log probabilities for distances from the mean. Given sigma.

    @param dist_mean:
    @param part_sigma:
    @return:
    """

    # determinant
    det1 = -np.log(part_sigma[0] * part_sigma[1] * part_sigma[2] * (2 * np.pi) ** (3 / 2))

    log_prob = det1 - 0.5 * (((dist_mean[:, 0] ** 2) / part_sigma[0] ** 2) +
                             ((dist_mean[:, 1] ** 2) / part_sigma[1] ** 2) +
                             ((dist_mean[:, 2] ** 2) / part_sigma[2] ** 2))
    return log_prob


class Parts(object):
    """

    Class to handle part based computations

    Benjamin Irving
    20141218

    """

    def __init__(self, part_means, part_var, part_names, weight_appearance=1.0, weight_original=1.0, weight_dist=1.0):
        """

        Labelling
        0) Lumen
        1) Tumour
        2) Bladder

        @param part_means: 3xn array where n is number of additional. 0 is base node.
        @param part_var: 3xn array where n is number of additional. 0 is base node.
        @param part_names: List of part names for record keeping
        @param weight_appearance: Weight the influence of appearance
                                    (larger weight leads to faster exponential drop off of appearance term)
        @param weight_original: separate weight for base appearance contribution
                                (should probably be the same as appearance term)
        @return:

        """

        # Sanity check (1)
        if len(part_names) != len(part_means[0]):
            warnings.warn("Dimensions are not equal")

        self.part_means = part_means
        self.part_var = part_var
        self.p = part_names

        # part weights
        self.w = {"appearance": weight_appearance, "original": weight_original, "distance": weight_dist}

        # number of additional
        self.n_parts = len(self.p)

        # convert to standard deviation (sigma)
        self.part_sigma = {}
        self.part_sigma = np.sqrt(self.part_var)

        # Test candidates
        self.candidate_locations = None
        self.candidate_potentials = None
        self.candidate_threshold = None

        self.neg_inf = -1e10

        # Probability images of child part location
        self._child_img_prob = []

        self.mean_fr_bladder = None
        self.var_fr_bladder = None

        # Part probabilities
        self.part_probabilities = {}
        self.part_img_prob = {}

    def get_test_candidate(self, locations, part_potentials, threshold=None):
        """
        Get the supervoxel potentials to be used as part of the classification method

        @param locations: x,y,z locations of each candidate for each part
                                    [part][candidates x dimensions]
                        units = (mm, mm, mm)
        @param part_potentials: Potential of candidate belonging to each part [part][candidates]
                 - [0] lumen
                 - [1] tumour
                 - [2] bladder
        @param threshold: only use a subset of candidates that are above a certain potential threshold
        @return:
        """
        if not isinstance(locations, list):
            warnings.warn("locations must be a list containing different additional. Skipping this method...")
            return

        # store candidate potentials
        # if only one set of candidates is given then append the same set for each part
        if len(locations) == 1:
            self.candidate_locations = []
            for ii in range(self.n_parts):
                self.candidate_locations.append(locations[0])
        else:
            self.candidate_locations = locations

        self.candidate_potentials = part_potentials
        self.candidate_threshold = threshold

    def generate_part_hypothesis(self, supervoxel_coords=None, lumen_restrain=True):
        """
        hypothesis2[part][base_candidate x child_candidate]
                    - list of n-additional
                    - Each part has a mxm array which corresponds to m supervoxels in the image
                    - rows: base part candidate
                    - columns: child part candidate

                    Math

                    Basic implementation of belief propagation for trees
                    (See Murphy, Chapter 20 - Exact inference for graphical models)

                    P(child_location | base_location)

                    Therefore rows should sum to 1. Given a base location, the probability of all
                    child locations must be 1.

                    --> P(child_location, base_location) = P(child_location | base_location)P(base_location)
                    --> P(child_location) = sum_base_location P(child_location, base_location)

        Notes: Maximum response is used but the location of each one causing the max response doesn't really matter
                because at each base candidate we're just looking for a max response generated. Not really on a single
                max response to determine the whole probability distribution. Could constrain this a bit more by
                including shape priors to determine neighbourhood inclusion in each supervoxel potential but not
                essential.
        @return:
        """

        # base part
        base_part = self.p["tumour"]

        # children
        p_child = {"lumen": self.p["lumen"], "bladder": self.p["bladder"]}

        # Sanity check (2)
        if base_part != np.nonzero(np.logical_not(self.part_means[:, 0]))[0][0]:
            warnings.warn("base part label does not match given part means")

        # Step 1 : forward belief propagation
        hypothesis2, sum_per_landmark, base_prob_norm = self.collect_step(p_child,
                                                                          base_part,
                                                                          self.candidate_potentials,
                                                                          self.candidate_locations,
                                                                          self.part_means,
                                                                          self.part_sigma,
                                                                          self.w)
        self.part_probabilities["tumour"] = base_prob_norm

        # Step 2: reverse the belief propagation
        # List provides a list of child probs equivalent to the list of self.p
        child_prob = self.distribute_step(p_child,
                                          hypothesis2,
                                          sum_per_landmark,
                                          base_prob_norm,
                                          lumen_restrain=lumen_restrain,
                                          supervoxel_coords=supervoxel_coords)

        self.part_probabilities[p_child.keys()[0]] = child_prob[0]
        self.part_probabilities[p_child.keys()[1]] = child_prob[1]


    @staticmethod
    def collect_step(p_child, base_part, candidate_potentials,
                     candidate_locations, part_means, part_sigma, w):
        """

        @param part_child: Numerical labels of child additional
        @param base_part: Numerical labels of base part
        @param candidate_potentials: Potentials for of each candidate per supervoxel
        @param candidate_locations: [n_parts][n_supervoxels x 3] Coordinate of each candidate supervoxel
                                    Generally just the same coordinates repeated for each part
        @param part_means: mean distance between base and each child
        @param part_sigma: variance between base and each part
        @param w: part model weights
        @return:
        """

        n_parts = len(p_child) + 1  # child additional + base part

        # Number of supervoxels for each part
        supervoxel_cnum = []
        for ii in range(len(candidate_locations)):
            supervoxel_cnum.append(candidate_locations[ii].shape[0])

        # hypothesis2 - like hypothesis one but includes a placeholder
        # for the base node supervoxel to keep the consistent order
        hypothesis2 = [None] * len(candidate_locations)

        sum_per_landmark = np.zeros((len(candidate_locations), supervoxel_cnum[base_part]))
        # loc_per_landmark = np.zeros((n_parts, self.supervoxel_cnum[base_part]))

        for key, ii in p_child.iteritems():

            # Hypothesis excluding current base node for collecting evidence stage of bp
            # Full hypothesis for distributing evidence stage of belief propagation
            hypothesis2[ii] = np.zeros((supervoxel_cnum[base_part], supervoxel_cnum[ii]))

            # For each base node candidate
            for jj in range(supervoxel_cnum[base_part]):

                # distance between base supervoxel candidate and all supervoxels of child
                loc_base1 = candidate_locations[base_part][jj, :]

                # candidate locations
                loc_child1 = candidate_locations[ii]

                # 1) distance between child and candidate
                dist_xyz = loc_child1 - np.tile(loc_base1, (loc_child1.shape[0], 1))
                # learned distances
                part_model_mean = part_means[ii, :]
                # variation of the distance from the mean part distance
                dist_mean = dist_xyz - np.tile(part_model_mean, (dist_xyz.shape[0], 1))
                # part variation
                part_model_sigma = part_sigma[ii, :]

                # 2) log potential for each base supervoxel candidate given child part ii
                # Part distance potential
                log_norm1 = log_normal_distribution_3d(dist_mean, part_model_sigma)
                # Posterior feature potential. Taking log so both are in the log domain
                log_cpot1 = np.log(candidate_potentials[ii])
                hypothesis2[ii][jj, :] = np.exp(w["distance"]*log_norm1[np.newaxis] + w["appearance"]*log_cpot1)
                # P(Child_jj|Base_jj) = 0

                # Supervoxel can't be both a Child and Base
                hypothesis2[ii][jj, jj] = 0

                # 4) Sum of part score contributions for each base landmark used
                # best_score = np.max(hypothesis2[ii][jj])
                sum_score = np.sum(hypothesis2[ii][jj])
                sum_per_landmark[ii, jj] = sum_score

        # Calculate the contribution of the base part
        prob_cpot2 = candidate_potentials[base_part]
        sum_per_landmark[base_part, :] = prob_cpot2 ** w["original"]

        # Concatenate
        all1 = p_child.values()[:]
        all1.append(base_part)
        # Calculate the total scores based on the product of contributions from base node and child nodes
        base_prob_norm = np.product(sum_per_landmark[all1], axis=0)

        # convert and normalised base probs
        # P(base)
        base_prob_norm = base_prob_norm / np.max(base_prob_norm)

        return hypothesis2, sum_per_landmark, base_prob_norm

    @staticmethod
    def distribute_step(p_child, hypothesis2, sum_per_landmark, base_prob_norm,
                        lumen_restrain=False, supervoxel_coords=None):
        """

        @param part_child:
        @param hypothesis2:

                Probabilities between all parent and child candidates for a given child

        @param sum_per_landmark: sum of contributions of a child to a particular landmark

                sum_score = np.sum(hypothesis2[ii][jj])
                sum_per_landmark[ii, jj] = sum_score

        @param base_prob_norm: Probability of each base part

                self.base_prob_norm = np.product(sum_per_landmark, axis=0)
                self.base_prob_norm = self.base_prob_norm / np.max(self.base_prob_norm)

        @param lumen_restrain: Extra processing step that constrains the lumen
        @param supervoxel_coords:
        @return:

        """

        #Use base probability to distribute back to child probabilities
        child_prob = []
        # find the child probabilities for the base probability distribution.
        for keys, ii in p_child.iteritems():
            # P(child_location | base_location)
            hyp_weight = hypothesis2[ii]
            # normalise the probabilities
            hyp_weight = hyp_weight / np.tile(hyp_weight.sum(axis=1)[np.newaxis].transpose(), (1, hyp_weight.shape[0]))
            #P(child_location, base_location)
            base_message = base_prob_norm
            # extracting the forward message from this landmark
            forward_message = sum_per_landmark[ii, :] / np.max(base_prob_norm)
            # normalising the base message by the forward message
            base_message = base_message / forward_message
            hyp_weight = hyp_weight * np.tile(base_message[np.newaxis].transpose(),
                                              (1, hypothesis2[ii].shape[1]))
            # P(child_location)
            #  Marginalising P(child_location, base_location) over the base_location
            if "lumen" in p_child.keys():
                if ii == p_child["lumen"] and lumen_restrain:  # just for the lumen
                    start = timeit.default_timer()
                    hw9 = np.ascontiguousarray(hyp_weight)
                    sc9 = np.ascontiguousarray(supervoxel_coords)
                    rad_hyp = radial_symmetry_weight(hw9, sc9, 6)
                    stop = timeit.default_timer()
                    print("Time c++: ", stop - start)
                    c1 = rad_hyp
                else:
                    c1 = np.sum(hyp_weight, axis=0)
            else:
                c1 = np.sum(hyp_weight, axis=0)

            c1 = c1 / np.max(c1)
            child_prob.append(c1)

        return child_prob


class PartEval(Parts):
    def create_probability_image(self, slic_reg):
        """
        Converting back to image space for plotting.

        :param slic_reg: Supervoxel image. Reference for converting supervoxel probabilities into images.
        :param use_log_probabilities: Create log probability images
        :return:
        """

        sreg = np.ascontiguousarray(slic_reg)
        # label probabilities
        lprob = np.ascontiguousarray(self.part_probabilities["tumour"])
        self.part_img_prob["tumour"] = crm.createlab(sreg, lprob)

        # TESTING
        lprob = np.ascontiguousarray(self.candidate_potentials[2][0])
        self.bladder_img_prob = crm.createlab(sreg, lprob)

        lprob = np.ascontiguousarray(self.candidate_potentials[self.p["tumour"]][0])
        self.tumour_img_prob = crm.createlab(sreg, lprob)

        lprob = np.ascontiguousarray(self.candidate_potentials[self.p["lumen"]][0])
        self.lumen_img_prob = crm.createlab(sreg, lprob)

        lprob = np.ascontiguousarray(self.part_probabilities["lumen"])
        self.part_img_prob["lumen"] = crm.createlab(sreg, lprob)

        lprob = np.ascontiguousarray(self.part_probabilities["bladder"])
        self.part_img_prob["bladder"] = crm.createlab(sreg, lprob)

    def labelled_tumour(self, thresh1, tumour_constraint=False):

        """
        @param: tumour_constraint = Tumour constraint in order to remove small regions

        """

        if tumour_constraint:
            # finding connected regions
            tumour = self.part_img_prob["tumour"] > thresh1
            self.tumour_img_pred = self.remove_small_regions(tumour)
        else:
            self.tumour_img_pred = self.part_img_prob["tumour"] > thresh1

    @staticmethod
    def remove_small_regions(tumour):
        """
        Remove small disconnected regions from a binary tumour segmentation
        """

        lregs, num1 = ndi.label(tumour)
        # size of each connected region
        size1 = np.zeros(num1 + 1)
        compac1 = np.zeros(num1 + 1)
        iner1 = np.zeros(num1 + 1)

        for ii in range(1, num1 + 1):
            size1[ii] = np.sum(lregs == ii)

            m1 = mp.binary_closing(lregs == ii)
            m2 = mp.binary_erosion(m1)
            diff1 = np.logical_and(m1, np.logical_not(m2))

            # compactness
            # surface area
            area1 = np.sum(diff1)
            # volume
            vol1 = np.sum(m1)
            compac1[ii] = area1 ** 2 / vol1

            #central normalised moment of inertia
            seg_ind = np.nonzero(lregs == ii)
            cm = np.mean(seg_ind, axis=1)
            diff_cm = seg_ind - np.tile(np.expand_dims(cm, axis=1), (1, np.shape(seg_ind)[1]))

            #Rotation invariant moment of inertia
            I1 = (np.sum((diff_cm[0, :]) ** 2) + np.sum((diff_cm[1, :]) ** 2) + np.sum((diff_cm[2, :]) ** 2)) / vol1
            iner1[ii] = I1

        max1 = np.argmax(size1)
        tumour_out = (lregs == max1)

        return tumour_out

    def overlap_statistics(self, ground_roi, print1=False):

        """
        Dice, sensitivity and specificity for a given threshold

        """

        # including the previous ground truth
        stats1 = self.overlap_evaluation(ground_roi == 1, self.tumour_img_pred == 1, print1=print1)

        return stats1

    @staticmethod
    def overlap_evaluation(truth1, prediction1, print1=False):
        """
        Evaluate the classification

        @param truth1: Ground truth
        @param prediction1: Predicted labels
        @param truthlabel:
        @param manlabel:
        @param print1: Print outputs in during running
        @return:

        """

        # DICE overlap
        dice1 = 2*np.sum(np.logical_and(truth1, prediction1))/(np.sum(prediction1)+np.sum(truth1))
        dice1 = np.around(dice1, decimals=4)

        if print1:
            print("Dice: ", dice1)

        sensitivity = np.around(np.sum(np.logical_and(prediction1, truth1)) / np.sum(truth1), decimals=4)
        specificity = np.around(np.sum(np.logical_and(np.logical_not(prediction1), np.logical_not(truth1))) /
                                     np.sum(np.logical_not(truth1)), decimals=4)

        stats = {'dice': dice1, 'sensitivity': sensitivity, 'specificity': specificity}

        return stats

    def roc_statistics(self, thresh_vals_parts, thresh_vals_super, ground_roi, slic_reg,  tumour_constraint=True):

        """
        Calculate the ROC values
        """

        index1 = np.array(thresh_vals_parts)
        df = pd.DataFrame(index=index1, columns=["dice", "sensitivity", "specificity"])
        for thresh1 in thresh_vals_parts:

            if tumour_constraint:
                # finding connected regions
                tumour = self.part_img_prob["tumour"] > thresh1
                tp = self.remove_small_regions(tumour)
            else:
                tp = self.part_img_prob["tumour"] > thresh1

            stats1 = self.overlap_evaluation(ground_roi == 1, tp)
            df.loc[thresh1] = [stats1["dice"], stats1["sensitivity"], stats1["specificity"]]

        index2 = np.array(thresh_vals_super)
        df_s = pd.DataFrame(index=index2, columns=["dice", "sensitivity", "specificity"])

        # Convert to image format again
        sreg = np.ascontiguousarray(slic_reg)
        lprob = np.ascontiguousarray(np.squeeze(self.candidate_potentials[1]))
        sprob = crm.createlab(sreg, lprob)
        for thresh1 in thresh_vals_super:
            if tumour_constraint:
                # finding connected regions
                tumour = sprob> thresh1
                tp = self.remove_small_regions(tumour)
            else:
                tp = self.candidate_potentials[1] > thresh1

            stats1 = self.overlap_evaluation(ground_roi == 1, tp)
            df_s.loc[thresh1] = [stats1["dice"], stats1["sensitivity"], stats1["specificity"]]

        # dfc = pd.concat([df, df_s], keys=["additional", "super"], axis=1)
        return df, df_s

    def generate_labelled_image(self, thresh1):

        labimg = np.zeros(self.part_img_prob["tumour"].shape)
        labimg[self.part_img_prob["lumen"] > thresh1] = 1
        labimg[self.tumour_img_pred] = 2

        return labimg


class PartVis(PartEval):
    """
    Add visualisation components to the additional model
    """

    def plot_dice_overlap(self, super_label, slic_reg, print1=False):

        """
        Display the DICE overlap graph before and after the additional implementation
        For all probability values [0.0, 1.0]
        @param super_label: Ground truth labels
        @param slic_reg: Supervoxel regions

        """

        # converting ground truth labels into an array
        sreg = np.ascontiguousarray(slic_reg)
        lprob = np.ascontiguousarray(super_label == 2, dtype=float)
        ground_truth = crm.createlab(sreg, lprob)

        dice1 = []
        dice2 = []

        for ii in np.arange(0, 1, 0.1):
            # Supervoxel method
            stats1 = self.overlap_evaluation(ground_truth, self.tumour_img_prob > ii, print1=print1)
            dice1.append(stats1['dice'])
            # Parts method
            stats2 = self.overlap_evaluation(ground_truth, self.part_img_prob["tumour"] > ii, print1=print1)
            dice2.append(stats2['dice'])

        plt.figure()
        d1, = plt.plot(np.arange(0, 1, 0.1), dice1, color='lightgreen', linewidth=4.0)
        d2, = plt.plot(np.arange(0, 1, 0.1), dice2, color='blue', linewidth=4.0)
        plt.legend([d1, d2], ['Supervoxels', 'Pieces-of-additional'])
        plt.xlabel('Threshold', fontsize=20)
        plt.ylabel('DSC', fontsize=20)
        plt.xticks(np.arange(0, 1.0, 0.1), fontsize=20)
        plt.yticks(fontsize=20)

    def plot_probability_image(self, slice1):
        """
        Plot image cross sections of supervoxel probabilities
        :param slice1:
        :return:
        """

        self.slice1 = slice1

        cmap1 = 'hot'  # jet, hsv

        plt.figure()
        # plt.subplot(2, 3, 1)
        plt.imshow(self.part_img_prob["tumour"][:, :, slice1])
        plt.colorbar()
        plt.title("Tumour (root)")
        plt.set_cmap(cmap1)

        plt.figure()
        plt.subplot(2, 2, 4)
        plt.imshow(self.bladder_img_prob[:, :, slice1])
        plt.colorbar()
        plt.title("Bladder (original)")
        plt.set_cmap(cmap1)

        plt.subplot(2, 2, 1)
        plt.imshow(self.tumour_img_prob[:, :, slice1])
        plt.colorbar()
        plt.title("Tumour (original)")
        plt.set_cmap(cmap1)

        plt.subplot(2, 2, 3)
        plt.imshow(self.lumen_img_prob[:, :, slice1])
        plt.colorbar()
        plt.title("Lumen (original)")
        plt.set_cmap(cmap1)

    def plot_ground_truth(self, super_label, slic_reg, slice1=None):
        """

        @param super_label:
        @param slic_reg:
        @param slice1:
        @return:
        """

        if slice1 is None:
            slice1 = self.slice1

        plt.figure()
        cmap1 = 'jet'

        sreg = np.ascontiguousarray(slic_reg)
        lprob = np.ascontiguousarray(super_label)
        ground_truth = crm.createlab(sreg, lprob)

        plt.imshow(ground_truth[:, :, slice1])
        plt.colorbar()
        plt.title("Ground truth")
        plt.set_cmap(cmap1)

    def plot_mv_scatter(self):
        """
        Visualise the part variance using a scatter plot
        @return:
        """
        list2 = [0]*2

        size1 = 200
        scat1 = np.zeros((size1, size1))

        list1 = np.where(scat1 == 0)
        list2[0] = list1[0] - size1/2
        list2[1] = list1[1] - size1/2

        view1 = [(0, 1), (2, 1)]

        l1 = ['y-axis (mm)', 'y-axis(mm)']
        l2 = ['x-axis (mm)', 'z-axis(mm)']

        for cc, jj in enumerate(view1):
            scat2 = np.zeros((size1, size1))
            p1 = np.zeros((len(list2[0]), 3))

            for ii in [0, 2]:

                p1[:, jj[0]] = list2[0] - self.part_means[ii, jj[0]]
                p1[:, jj[1]] = list2[1] - self.part_means[ii, jj[1]]
                # p1[:, 2] = self.part_means[ii, 2]

                psigma = self.part_sigma[ii, :]
                p = normal_distribution_3d(p1, psigma)
                p = p / np.max(p)

                scat1[list1[0], list1[1]] = p
                scat2 += scat1

            scat2 = scat2.transpose()
            scat2 = np.flipud(scat2)
            plt.figure()
            plt1 = plt.imshow(scat2, extent=[-100, 100, -100, 100])
            plt1.set_cmap('gray_r')
            plt.ylabel(l1[cc])
            plt.xlabel(l2[cc])



    def show(self):
        plt.show()







