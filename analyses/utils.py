import cv2
import numpy as np
import scipy

def comp_cent(contour):
    mom = cv2.moments(contour)
    return np.array([mom['m10'] / mom['m00'], mom['m01'] / mom['m00']], dtype=np.uint32)

def comp_lsq_circle(contour):
    '''Contour expected to be in OpenCV format'''
    contour = contour[:, 0]
    x, y = contour[:, 0], contour[:, 1]
    centroid = comp_cent(contour)
    cx, cy = centroid[0], centroid[1]
    theta, r = np.arctan2(y-cy, x-cx), np.hypot(x-cx, y-cy)
    theta += np.pi
    rr = scipy.optimize.leastsq(lambda rr: sum((rr - r) ** 2), x0=.5)[0]
    mse = np.abs(np.mean((rr - r) ** 2))
    return float(mse)

def find_hulls(img, size_threshold=50):
    img = np.copy(img)
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(contour) for contour in contours]
    return hulls
   
#Do not appreciate fact that an empty list can sneak in to the tuple
#Size threshold is not in use
def find_hulls_with_inner_contours(img, size_threshold=50):
   img = np.copy(img)
   hulls_inners = []
   _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
   if not contours: return hulls_inners
   hierarchy = hierarchy[0]
   for ix, contour in enumerate(contours):
       #Ensure this is outer contour of sufficient size
       if hierarchy[ix][3] != -1 or cv2.contourArea(contour) < size_threshold:
           continue
       #Now acquire corresponding inner contours, if they exist
       inner_contours = []
       #Index of first child contour, will be -1 if does not exist
       child_ix = hierarchy[ix][2]
       while child_ix != -1:
           inner_contours.append(contours[child_ix])
           #Set child index to the next index to check for more children
           child_ix = hierarchy[child_ix][0]
       hulls_inners.append((cv2.convexHull(contour), inner_contours))
   return hulls_inners

def nonzero_percentile_threshold(img, percentile):
    if img.max() != 0: 
        img[img <= np.percentile(img[img != 0], percentile)] = 0

from copyreg import pickle
from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try: 
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
