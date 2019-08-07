from collections import defaultdict
from glob import glob
import itertools
import os 
from os.path import join
import subprocess

import cv2
import numpy as np

#Create log to ensure that only images that have not been analyzed are analyzed
from skimage import measure
import scipy.ndimage
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import measurements, morphology
from scipy.ndimage.filters import median_filter 
import skimage.morphology
import skimage.filters
from skimage.transform import resize

import multiprocess 

import fileutils
from imgutils import transforms
from imgutils.tifffile import imread, imsave
import mfileutils.makeconfig

import annotate
from analyses.automated_survival.objects import Neuron, ROI
import analyses.automated_survival.utils as utils
from analyses.automated_survival.output import Exporter

from ImageLogger import ImageLogger

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from math import isinf
import tensorflow as tf


tf.enable_eager_execution()

def f(img):
    plt.imshow(img)
    plt.show()

class Tracker():
    def __init__(self, exp_name, outdir, threshold_multiplier, magnification, microscope, binning, fiddle):
        self.image_logger = ImageLogger(join(outdir, 'image-logs'), disable=True)
        self.exp_name = exp_name
        self.binned = False if binning[0] == '1' else True
        self.outdir = outdir
        self.threshold_multiplier = threshold_multiplier
        # Turn into a real partial.
        self.um_to_px = lambda m: transforms.microns_to_pixels(m, magnification, microscope, binning, fiddle)
        #self.nn_model = tf.keras.models.load_model(r'\\joshua\josh2\rmdev\pipeline\analyses\automated_survival\model.hd5')
        self.nn_model = tf.keras.models.load_model(r'\\10.17.156.132\josh2\rmdev\pipeline\analyses\automated_survival\model.hd5')
        self.nn_model.compile(optimizer=tf.train.AdamOptimizer(),
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])

    def _label_and_slice(self, img):
        '''Label contiguous binary patches numerically and make list of smallest parallelpipeds that contain each.'''
        img = np.copy(img)
        img = scipy.ndimage.binary_fill_holes(img).astype(np.uint8)
        labeled_img, _ = measurements.label(img)
        return measurements.find_objects(labeled_img)

    def _remove_overlapping_slices(self, slices, img):
        ''' Remove overlapping slices'''
        # Sort slices by size so that small slices mostly overlapping larger ones are detected upon entry into bit array.
        area_of_2D_slice = lambda s: (s[0].stop - s[0].start) * (s[1].stop - s[1].start)
        slices = sorted(slices, key=area_of_2D_slice, reverse=True)

        # For each slice, check if any value in its interior is already 1. If so, that means it is overlapping
        # another slice. Then check if it overlaps any other slice. If so, remove it.
        num_array = np.empty_like(img, dtype=np.uint8)
        for s in slices[:]:
            sub_array = num_array[s]
            if 1 in sub_array:
                # Line that used to be use to keep an overlapper if it did not overlap to much.
                #if sum(sum(sub_array)) > .5 * sub_array.size:
                slices.remove(s)

            else:
                num_array[s] = 1

        return slices

    def _threshold(self, img):
        # Threshold, parameterized through use of a multiplier.
        img[img < np.mean(img) + self.threshold_multiplier * np.std(img)] = 0

    def _process_img(self, img):
        '''Process image.'''
        img = np.copy(img)
        self.image_logger.log('find-somas', 'original', img)

        self._threshold(img)

        #if not self.binned:
        disk = skimage.morphology.disk(2)
        # Use minimum filter in place.
        img = scipy.ndimage.filters.minimum_filter(img, footprint=disk)
        self.image_logger.log('find-somas', 'min-filter', img, box=True)

        # To further separate ROIs, use erosion to eliminate noise and processes.
        erode_kernel_dim = self.um_to_px(3)
        erode_kernel = np.ones((erode_kernel_dim,) * 2)
        eroded = morphology.binary_erosion(img, erode_kernel)

        # Use eroded image as mask to select pixels in original image.
        img[eroded == 0] = 0
        self.image_logger.log('find-somas', 'erode', img, box=True)

        rimg = skimage.filters.roberts(img)
        #Values will be between 0 and 1 now; scale and set to 16-bit
        rimg = (rimg * 2 ** 16).astype(np.uint16)

        # Apply scaled Sobel operators to image to enhance rois.
        sobx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1)
        soby = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1)
        img = np.hypot(sobx, soby).astype(np.uint16)
        self.image_logger.log('find-somas', 'rob+sobel', img, box=True)

        img = (img.astype(np.uint32) + rimg.astype(np.uint32))
        np.clip(img, 0, 2 ** 16 - 1, out=img)
        img = img.astype(np.uint16)
        self.image_logger.log('find-somas', 'clip', img, box=True)

        utils.nonzero_percentile_threshold(img, 40)
        if not self.binned:
            # Use minimum filter in place.
            img = scipy.ndimage.filters.minimum_filter(img, size=(2,2))
        else:
            self.image_logger.log('find-somas', 'threshold-2', img, box=True)

        return img

    def _find_graded_somas(self, img):

        img = self._process_img(img)

        slices = self._label_and_slice(img)
        self.image_logger.log('find-somas', 'post-labeling', img, slices=slices)

        # Remove slices under size threshold.
        px = self.um_to_px(5)
        slices = list(filter(lambda s: (s[0].stop - s[0].start) * (s[1].stop - s[1].start) > px ** 2, slices))
        self.image_logger.log('find-somas', 'size-thresholding', img, slices=slices)

        # After identifying the first batch of slices, run a tophat transform on each, and redo.
        tophat_kernel_dim = self.um_to_px(40)
        tophat_kernel=np.ones((tophat_kernel_dim,) * 2)

        erode_kernel_dim = self.um_to_px(3)
        erode_kernel = np.ones((erode_kernel_dim,) * 2)

        max_dim_accepted = self.um_to_px(100)

        for s in slices:
            subimg = img[s]
            subimg = cv2.morphologyEx(subimg, cv2.MORPH_TOPHAT, kernel=tophat_kernel)

            # Calculate area. If larger than certain value, then erode.
            max_subimg_dim = max(s[0].stop - s[0].start, s[1].stop - s[1].start)

            if max_subimg_dim > max_dim_accepted:
                subimg = np.clip(subimg, 0, 1)
                subimg = morphology.binary_dilation(subimg)
                subimg = morphology.binary_fill_holes(subimg)
                subimg = morphology.binary_erosion(subimg, structure=erode_kernel, iterations=3)

                # Scale subimg, which is currently binary, to a very high (arbitrary) value.
                subimg = subimg.astype(np.uint16) * (2 ** 14)
                img[s] = subimg
                #f(subimg)
                #print(subimg)
                #print(subimg.dtype)
                #print(subimg.shape)
                #utils.nonzero_percentile_threshold(subimg, 50)

                #subimg = morphology.grey_erosion(subimg, structure=erode_kernel)
                #subimg = cv2.erode(subimg, erode_kernel, iterations=2)
                #subimg = cv2.dilate(subimg, erode_kernel, iterations=2)

                #img[s] = subimg
                #f(img[s])

        slices = self._label_and_slice(img)
        self.image_logger.log('find-somas', 'erode', img, slices=slices)

        # Remove slices under size threshold.
        px = self.um_to_px(3)
        slices = list(filter(lambda s: (s[0].stop - s[0].start) * (s[1].stop - s[1].start) > px ** 2, slices))
        self.image_logger.log('find-somas', 'size-thresholding2', img, slices=slices)

        #slices = self._remove_overlapping_slices(slices, img) # This function sorts slices by area.
        #self.image_logger.log('find-somas', 'remove-overlappers', img, slices=slices)

        return slices

    def _estimate_centroids(self, slices):
        return [np.array((s[1].start + (s[1].stop - s[1].start) // 2, s[0].start + (s[0].stop - s[0].start) // 2))
                for s in slices]

    def _expand_slices(self, img, slices):
        expanded_slices = []

        # Everywhere there's a slice, set to 1.
        overlap_array = np.zeros(img.shape)
        for s in slices:
            overlap_array[s] = 1

        #Use slices fit to candidate neuron somas to center slices for collection of a neuron contour
        y_MAX, x_MAX = img.shape
        expand_slice = lambda s, px: (slice(max(0, s[0].start - px), min(y_MAX, s[0].stop + px)), 
                                      slice(max(0, s[1].start - px), min(x_MAX, s[1].stop + px)))
        for s in slices:
            # Set this slice's overlap slice to zero.
            overlap_array[s] = 0

            # Expand slice.
            px = self.um_to_px(10)
            exp_s = expand_slice(s, px)

            # Add 1 to the expanded overlap slice.
            overlap_array[exp_s] += 1

            # If any value inside the expanded overlap is greater than or equal to 2, we've made contact with 
            # another slice.
            while np.any(overlap_array[exp_s] >= 2):
                # Reset to 1, in case while expanding perturbed others.
                overlap_array[exp_s] = 1 
                # Reset to 0 so can begin process again.
                overlap_array[s] = 0

                px -= 1
                exp_s = expand_slice(s, px)
                overlap_array[exp_s] += 1

            expanded_slices.append(exp_s)

        return expanded_slices

    def _find_initial_candidates(self, img):
        img_copy = np.copy(img)
        candidates = self._find_candidates(img)
        #Build list of ROIs, then build list of Neurons and return

        # Area filter.
        area_min = self.um_to_px(10) ** 2
        area_max = self.um_to_px(150) ** 2
        A = cv2.contourArea
        candidates = [c for c in candidates if A(c[1]) > area_min and A(c[1]) < area_max]

        # Note that this can be done in batch; as a vector.
        _rois = []
        for cent, cont in candidates:
            x, y = cent
            # 83.2 microns was empirically determined to be equal to 64 pixels at our most commonly used resolutions.
            PX = int(self.um_to_px(83.2) / 2.0)
            if x < PX or y < PX or img.shape[1] - PX < x or img.shape[0] - PX < y:
                continue

            subimg = img_copy[y-PX:y+PX, x-PX:x+PX]
            subimg = transforms.to_8bit(subimg)
            subimg = resize(subimg, (64, 64), order=3, preserve_range=True).astype(np.uint16)
            subimg = subimg.astype('float32') / 255.0
            #plt.imshow(subimg)
            #plt.show()
            subimg = subimg.reshape(1, *subimg.shape, 1)

            alive_prob, dead_prob = self.nn_model.predict(subimg)[0]
            if alive_prob > .85:
                _rois.append((cent, cont))


        if len(_rois) == 0:
            print('No neurons found for well')
            return []

        rois = [ROI(img, centroid, contour) for centroid, contour in _rois]
        return [Neuron(ID=i, init_roi=rois[i]) for i in range(len(rois))]

    def _find_candidates(self, img):
        graded_slices = self._find_graded_somas(img)

        centroid_estimates = self._estimate_centroids(graded_slices)

        expanded_slices = self._expand_slices(img, graded_slices)
        self.image_logger.log('expand-slices', 'expand', img, slices=expanded_slices)

        self._threshold(img)

        # Begin processing image and building candidate neurons.
        candidates = []
        area_min = self.um_to_px(10) ** 2
        tophat_kernel_dim = self.um_to_px(40)
        tophat_kernel=np.ones((tophat_kernel_dim,) * 2)
        for ix, s in enumerate(expanded_slices):
            subimg = img[s]
            subimg = cv2.morphologyEx(subimg, cv2.MORPH_TOPHAT, kernel=tophat_kernel)
            #Threshold subimg
            utils.nonzero_percentile_threshold(subimg, 20)
            subimg = transforms.to_8bit(subimg)
            _, contours, _ = cv2.findContours(subimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Area filter.
            contours = filter(lambda c: cv2.contourArea(c) > area_min, contours)

            # Used to help eliminate selection of neuronal processes as valid ROIs.
            #contours = filter(lambda c: (cv2.contourArea(c) / cv2.arcLength(c, closed=True)) > 1.5, contours) 

            #Adjust coordinates to reflect location in actual image ; switch slice order as switching from numpy to opencv encoding 
            contours = [c[:, :, 0:2] + (s[1].start, s[0].start) for c in contours]

            for contour in contours:
                candidates.append((centroid_estimates[ix], contour))

        try: self.image_logger.log('filtration', 'size-and-circularity', img, contours=list(zip(*candidates))[1])
        except IndexError: self.image_logger.log('filtration', 'size-and-circularity', img)

        return candidates

    def _assign_rois(self, img, timepoint, neurons, candidates):
        '''Determine which neurons found in a consecutive timepoint correspond to previously found neurons.'''
        candidate_centroids, _ = zip(*candidates)

        try: unassigned_centroids, _ = zip(*self.unassigned_rois)
        except ValueError: unassigned_centroids = ()

        #Build map from neurons to candidate index containing ROI with shortest distance to a neuron
        neuron_to_cand_ix = {}

        #Exclude neurons that have already died from assignment
        living_neurons = [neuron for neuron in neurons if neuron.last_tp == None]

        # Create a map between a centroid and its index so it can be found.
        centroid_to_ix = {tuple(map(int, centroid)) : ix for ix, centroid in enumerate(candidate_centroids)}

        # Add each unassigned centroid to the map with an index of -1 as a sentinel.
        centroid_to_ix.update({tuple(map(int, centroid)) : -1 for centroid in unassigned_centroids})

        # Create 2D KDTree from all existing candidates and unassigned ROI centroids.
        tree = KDTree(candidate_centroids + unassigned_centroids)

        # Maximum distance that the (n+1)st centroid can be from the nth one if it's to be believed that it's the same neuron.
        max_dist = self.um_to_px(50)

        # Keep a set of candidate indices chosen. Will be used to determine which have not been selected.
        candidate_ixs_chosen = set()

        #self.image_logger.log('assignment', 'living-neurons', img, contours=[n.roi_data_as_dict(20)['contours'][-1] for n in living_neurons])
        ID_to_candidate = {}
        for neuron in living_neurons:
            # Find the distance of the nearest centroid and its index in the tree structure.
            distance, ix = tree.query(neuron.roi_series[-1].centroid, distance_upper_bound=max_dist)

            # If the nearest distance is infinity, then nothing was found. Go to next.
            if isinf(distance): continue

            nearest_centroid = tuple(tree.data[ix])
            centroid_ix = centroid_to_ix[nearest_centroid]

            # The nearest centroid belong to an unassigned ROI. Go to next.
            if centroid_ix == -1: continue

            # Map the neuron ID to this candidate for further evaluation.
            ID_to_candidate[neuron.ID] = candidates[centroid_ix]

            # Keep track of candidate index chosen.
            candidate_ixs_chosen.add(centroid_ix)

        unassigned_candidates = self._extend_roi_series(img, timepoint, living_neurons, ID_to_candidate)

        '''
        for ix, centroid in enumerate(centroids):
            #Function to compute Euclidean distance between a candidate neuron and an existing neuron from previous timepoint
            compute_dist = lambda _centroid: np.linalg.norm(centroid - _centroid)

            #Make list of tuples: (ID, distance)
            dists = [(neuron.ID, compute_dist(neuron.roi_series[-1].centroid)) for neuron in living_neurons]

            #Add distances of previously unassigned neurons, with a sentinel ID of -1
            dists += [(-1, compute_dist(cent)) for centroid, cent in self.unassigned_rois]

            #Choose tuple with minimum distance (closest neuron)
            neuron_ID, min_dist = min(dists, key=lambda x: x[1])

            #If the closest roi is a previously unassigned roi, continue to next ROI, as this candidate is likely not an existing neuron
            if neuron_ID == -1:
                continue

            #Map neuron to candidate index with minimum distance if distance if it is closer than current selection
            #10000 selected arbitrarily as an infeasibly large distance
            if neuron_to_cand_ix.get(neuron_ID, (None, 10000))[1] > min_dist:
                neuron_to_cand_ix[neuron_ID] = (ix, min_dist)
        '''

        # Set unassigned rois from this timepoint for continued tracking.
        self.unassigned_rois = filter(lambda c: c in unassigned_candidates, candidates)

    def _extend_roi_series(self, img, timepoint, neurons, ID_to_candidate):
        # Keep track of which candidates are determined unfit for assignation.
        unassigned_candidates = []

        #living_neurons = [neuron for neuron in neurons if neuron.last_tp == None]
        #self.image_logger.log('assignment', 'before-extension', img, contours=[n.roi_data_as_dict(20)['contours'][-1] for n in living_neurons])


        for neuron in neurons:
            if neuron.ID in ID_to_candidate:
                candidate = ID_to_candidate[neuron.ID]
                centroid, contour = candidate
                roi = ROI(np.copy(img), centroid, contour)

                '''
                #contours = filter(lambda c: 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, closed=True) ** 2 > .1, contours) 
                # Some death checks
                cand_area = roi.area
                cand_circularity = 4 * np.pi * roi.area / roi.perimeter ** 2 
                cand_max = roi.max
                cand_mean = roi.mean

                # Previous ROI.
                prev_roi = neuron.roi_series[-1]

                prev_area = prev_roi.area
                prev_circularity = 4 * np.pi * prev_roi.area / prev_roi.perimeter ** 2 
                prev_max = prev_roi.max
                prev_mean = prev_roi.mean
                '''


                # Note that this can be done in batch; as a vector.
                _rois = []

                #### NNN
                x, y = centroid
                # 83.2 microns was empirically determined to be equal to 64 pixels at our most commonly used resolutions.
                PX = int(self.um_to_px(83.2) / 2.0)
                if x < PX or y < PX or img.shape[1] - PX < x or img.shape[0] - PX < y:
                    continue

                subimg = img[y-PX:y+PX, x-PX:x+PX]
                subimg = transforms.to_8bit(subimg)
                subimg = resize(subimg, (64, 64), order=3, preserve_range=True).astype(np.uint16)
                subimg = subimg.astype('float32') / 255.0
                #plt.imshow(subimg)
                #plt.show()
                subimg = subimg.reshape(1, *subimg.shape, 1)

                alive_prob, dead_prob = self.nn_model.predict(subimg)[0]
                #### NNN

                # Death detection.
                #if (cand_circularity > prev_circularity or cand_circularity > .9) and cand_area < prev_area and cand_max < .8 * prev_max:
                if dead_prob > .8:
                    neuron.last_tp = timepoint - 1
                    neuron.censored = 1
                    neuron.death_cause = 'unfound'

                    # Keep track of unassigned candidate.
                    unassigned_candidates.append(candidate)

                else:
                    roi = ROI(np.copy(img), centroid, contour)
                    neuron.roi_series.append(roi)
            else:
                neuron.last_tp = timepoint - 1
                neuron.censored = 1
                neuron.death_cause = 'unfound'

        #living_neurons = [neuron for neuron in neurons if neuron.last_tp == None]
        #self.image_logger.log('assignment', 'after-extension', img, contours=[n.roi_data_as_dict(20)['contours'][-1] for n in living_neurons])

        return unassigned_candidates

    def track(self, data):
        '''Track neurons.'''
        well, stack, crop_val = data

        # Ensure stack is 3D even if it has a single timepoint.
        if len(stack.shape) == 2:
            stack = stack.reshape(1, *stack.shape)

        # Setup image logger to log images for this well.
        #self.image_logger.set_image_name(well)

        orig_stack = np.copy(stack)
        print(self.exp_name + ':: Automated Survival Analysis: T' + str(1) + '-' + well)

        #Crop stack edges as borders will be zero if images were shifted during stack registration
        stack = stack[:, crop_val:-crop_val, crop_val:-crop_val]

        # Set the first timepoint for image logger. 
        #self.image_logger.set_timepoint(1)
        neurons = self._find_initial_candidates(stack[0])

        # If stack has more than one timepoint and neurons were found, then iterate through remaining timepoints and track survival.
        if stack.shape[0] > 1 and neurons:
            #Variable will be used to track potentially viable ROIs that remained unassigned to avoid mistaken future assignments
            self.unassigned_rois = []
            for timepoint, img in enumerate(stack[1:], start=1):
                # Set timepoint for image logger. Add one as indices are zero-based.
                #self.image_logger.set_timepoint(timepoint+1)

                print(self.exp_name + ':: Automated Survival Analysis: T' + str(timepoint + 1) + '-' + well)

                candidates = self._find_candidates(img)

                if not candidates:
                    print('No neurons found after timepoint ' + str(timepoint))
                    #If no neurons were found, ensure that all remaining neurons are marked as dead
                    # ENCAPSULATE THIS BEHAVIOR WITHIN NEURON OBJECT.
                    for neuron in neurons:
                        if neuron.last_tp == None:
                            neuron.last_tp = timepoint - 1
                            neuron.censored = 1
                            neuron.death_cause = 'unfound'
                    break
                self._assign_rois(img, timepoint, neurons, candidates)

                # For debugging.
                #living_neurons = [neuron for neuron in neurons if neuron.last_tp == None]
                #self.image_logger.log('assignment', 'after-assignment', img, contours=[n.roi_data_as_dict(20)['contours'][-1] for n in living_neurons])

        # Save annotated stack.
        annotate_path = join(self.outdir, 'annotated')
        os.makedirs(annotate_path, exist_ok=True)

        neurons = [neuron for neuron in neurons if not neuron.excluded]
        #self.image_logger.log('final-exclusion', 'final-exclusion', img, contours=[n.roi_data_as_dict(20)['contours'][-1] for n in neurons 
        #    if n.last_tp == None])

        # Relabel neuron IDs according to centroid. Simplifies visual identification in annotated stacks.
        # First sort by y-axis, then by x-axis.
        neurons = sorted(neurons, key=lambda n: n.roi_series[0].centroid[0])
        neurons = sorted(neurons, key=lambda n: n.roi_series[0].centroid[1])
        # Now relabel neurons.
        for ix, neuron in enumerate(neurons):
            neuron.ID = ix

        rois = {n.ID : n.roi_data_as_dict(crop_val)['contours'] for n in neurons}
        annotated = annotate.annotate_survival(well, orig_stack, rois)
        imsave(f'{annotate_path}/{well}.tif', annotated, photometric='rgb')
        
        # Deleting now to aid memory usage.
        del stack
        return (well, neurons)

def run_cox_analysis(config, outdir):
    exp_name = config['experiment']['name']
    group_labels = config['experiment']['imaging']['group_labels']
    group_control_label = config['experiment']['imaging']['group_control_label']

    #Load R script template and fill in relevant variables
    dirpath = os.path.dirname(os.path.abspath(__file__))
    with open(join(dirpath, 'R_script_template.txt')) as f:
        text = f.read()
        workdir = join(os.path.dirname(outdir), 'results')
        workdir = workdir.replace('\\', '\\\\').replace('/', '\\\\')
        fname = exp_name + '_surv_data.csv'
        text = text.replace('WORKDIR', workdir).replace('FILENAME', fname).replace('EXPNAME', exp_name)

        try:
            group_labels.remove(group_control_label)
            group_labels.insert(0, group_control_label)
        except ValueError:
            print('Control group not found within list of groups. Proceeding without including control group in group labels.')

        text = text.replace('DOUBLE_QUOTED_COMMA_DELIMITED_GROUPS', ', '.join(["'" + label + "'" for label in group_labels]))
        colors = 'red blue green grey yellow aquamarine black orange cyan violet blueviolet bisque4'.split()
        text = text.replace('DOUBLE_QUOTED_COMMA_DELIMITED_COLORS', ', '.join(["'" + colors[i % len(colors)] + "'" for i in range(len(group_labels))]))

    #Save modified text as script
    scriptpath = join(os.path.dirname(outdir), 'results', exp_name + '_survival_analysis.R') 
    with open(scriptpath, 'w', newline='') as f:
        f.write(text)
    cox_analysis_results_file = open(join(os.path.dirname(outdir), 'results', exp_name + '_results.txt'), 'w')

    #Be sure to pipe out errors too
    subprocess.call(['rscript', scriptpath], stdout=cox_analysis_results_file)

class SurvivalAnalyzer:
    def __init__(self, workdir):
        fileutils.mkdir(join(workdir, 'analysis'))
        self.outdir = join(workdir, 'analysis')
        self.config = mfileutils.makeconfig.mfile_to_config(workdir, self.outdir) 
        self.microscope = self.config['experiment']['imaging']['microscope']
        self.magnification = self.config['experiment']['imaging']['magnification']
        self.binning = self.config['experiment']['imaging']['binning']
        self.fiddle = self.config['experiment']['imaging']['fiddle']
        #Identify directory containing images
        primary_channel = self.config['experiment']['imaging']['primary_channel']
        self.imgdir = join(workdir, 'processed_imgs', 'stacked', primary_channel) 

        fileutils.mkdir(join(workdir, 'analysis'))

        self.exporter = Exporter(self.config)
        self.exp_name = self.config['experiment']['name']
        self.surv_fname = self.exp_name + '_surv_data.csv'

    def readin_stacks(self):
        '''Read in and yield every TIF image file within images path as numpy array.'''
        stackpaths = glob(self.imgdir +'\*.tif')
        for stackpath in stackpaths:
            #Obtain well name
            well = os.path.basename(stackpath).split('.')[0]
            #if well != 'A11': continue
            stack = imread(stackpath)
            yield (well, stack, 20)
        
    def analyze(self, threshold_multiplier, parallel=True):
        exp_name = self.config['experiment']['name']
        tr = Tracker(exp_name, self.outdir, threshold_multiplier, self.magnification, self.microscope, self.binning, self.fiddle)
        gen = self.readin_stacks()
        multiprocess.freeze_support()
        pool = multiprocess.Pool()
        #This may need to be a function of memory
        offset = 10 
        self.resultdir = join(os.path.dirname(self.outdir), 'results')
        #Make directory in case it doesn't exist
        fileutils.mkdir(self.resultdir)
        self.exporter.prep_csv_file(self.resultdir, self.surv_fname)

        def output(well, neurons, crop_val):
            self.exporter.export(well, neurons, crop_val)

        if not parallel:
            for data in gen:
                well, neurons = tr.track(data)
                output(well, neurons, 20)

        else:
            stacks_left = True
            while stacks_left:
                try: 
                    it = itertools.chain([next(gen)], itertools.islice(gen, 0, offset - 1))
                    for well, neurons in pool.imap_unordered(func=tr.track, iterable=it):
                        output(well, neurons, 20)
                except StopIteration:
                    stacks_left = False
        group_labels = self.config['experiment']['imaging']['group_labels']
        group_control_label = self.config['experiment']['imaging']['group_control_label']
        run_cox_analysis(self.config, self.outdir)
        print('Survival analysis has concluded.\n')

if __name__ == '__main__':
    x = np.exp(1 - 1/(2**0))
    #workdir = r'g:/sb/ror2'
    workdir = r'z:/experiments/mrr08_mc'
    #workdir = r'x:/ror4'
    multiprocess.freeze_support()
    analyzer = SurvivalAnalyzer(workdir)
    analyzer.analyze(1.0)
