import cv2
import numpy as np

class ROI(object):
    def __init__(self, img, centroid, contour):
        # Passing entire image? This is madness.
        self.img = img
        self.centroid = centroid
        self.contour = contour
        #Delete image after computations are complete to preserve memory
        #Future version can pass substantially reduced size (subimg) to reduce computation time
        del self.img

    @property
    def contour(self):
        return self._contour

    @contour.setter
    def contour(self, val):
        self.area = cv2.contourArea(val)
        self.perimeter = cv2.arcLength(val, True)
        #Generate mask for computation purposes
        mask = np.zeros(self.img.shape, np.uint8)
        cv2.drawContours(mask, [val], 0, (255, 255, 255), -1)
        nonzero_mask = (mask != 0)
        self.mean, self.std = map(float, cv2.meanStdDev(self.img, mask=mask))
        self.median = np.median(self.img[nonzero_mask])
        self._95thpercentile = np.percentile(self.img[nonzero_mask], 95)
        self.min, self.max, minloc, maxloc = cv2.minMaxLoc(self.img, mask=mask)
        self._contour = val 

class Neuron(object):
    def __init__(self, ID, init_roi, last_tp=None, death_cause=None):
        self.ID = ID
        self.last_tp = last_tp
        self.death_cause = death_cause
        self.roi_series = [init_roi]
        self.censored = 0
        self.excluded = False
        self.excluded_cause = None
        #self.make_verification_structure()

    def roi_data_as_dict(self, crop_val):
        rs = self.roi_series
        return {'areas' : [r.area for r in rs],
                'centroids' : [r.centroid + crop_val for r in rs],
                'contours' : [r.contour + crop_val for r in rs],
                'CVs' : [(r.std / r.mean) if r.mean != 0 else 0 for r in rs],
                'means' : [r.mean for r in rs],
                'medians' : [r.median for r in rs],
                'perimeters' : [r.perimeter for r in rs],
                'stds' : [r.std for r in rs],
                '95thpercentiles' : [r._95thpercentile for r in rs]}
