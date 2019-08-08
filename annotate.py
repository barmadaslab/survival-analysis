from imgutils import transforms
from os.path import join
import cv2
import fileutils
import numpy as np
import os

def compute_centroid(contour):
    m = cv2.moments(contour)
    try: cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    except ZeroDivisionError: cx, cy = 0, 0
    return cx, cy

# If a measurer function is passed, use this to compute a value and draw next to the ROI.
#val = None if measurer is None else measurer(roi.contour)

#cv2.putText(img=stack[timepoint], text=str(val), org=(cx + dx, cy + dy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            #fontScale=font_scale, color=(0, 255, 255), thickness=3)

def _transform_stack(stack):
    '''Enhance contrast and convert input stack to RGB for drawing purposes.'''
    stack = transforms.enhance_contrast(stack)
    return transforms.to_rgb_stack(stack)

def annotate_fractionation(well, stack, cell_rois, unknown_rois, nuclear_rois, 
                           nuclear_centroids=None, status_tags=None):
    # Text drawing parameters.
    FONT_SCALE = 1.6
    ## Used to offset numeric identifier from cell a bit. Should be resolution-dependent.
    DELTA = 25

    stack = _transform_stack(stack)

    for ID, cell_roi in cell_rois.items():
        for tp, contour in enumerate(cell_roi):
            cv2.drawContours(stack[tp], [contour], 0, (255, 0, 0), 1)

            # Draw numeric identifier.
            cx, cy = compute_centroid(contour)
            cv2.putText(img=stack[tp], text=str(ID+1), org=(cx + DELTA, cy + DELTA), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, color=(255, 255, 0), thickness=2)

            # Draw status tag.
            cv2.putText(img=stack[tp], text=status_tags[ID][tp], org=(cx - 2*DELTA, cy + DELTA), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 255, 0), thickness=2)

            if nuclear_centroids is not None:
                cv2.drawContours(stack[tp], [nuclear_centroids[ID][tp]], -1, (255, 0, 0), 2)

            # If the nuclear contour has no size, then it was not found, so continue to next.
            if nuclear_rois[ID][tp].size == 0: continue

            cv2.drawContours(stack[tp], [nuclear_rois[ID][tp]], -1, (0, 255, 0), 1)
            cv2.drawContours(stack[tp], [unknown_rois[ID][tp]], -1, (0, 0, 255), 1)

    return stack

def annotate_survival(well, stack, rois, width=2):
    # Text drawing parameters.
    FONT_SCALE = 1.6
    ## Used to offset numeric identifier from cell a bit. Should be resolution-dependent.
    DELTA = 25

    stack = _transform_stack(stack)

    # Draw data for each ROI.
    for ID, roi in rois.items():
        for tp, contour in enumerate(roi):
            cv2.drawContours(stack[tp], [contour], 0, (0, 255, 0), width)

            # Draw numeric identifier.
            cx, cy = compute_centroid(contour)
            cv2.putText(img=stack[tp], text=str(ID+1), org=(cx + DELTA, cy + DELTA), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, color=(255, 255, 0), thickness=2)

        # If total ROIs does not match total images/timepoints, then did not survive until end.
        if len(roi) != len(stack):
            #Draw final contour on first image where ROI was not re-identified.
            cv2.drawContours(stack[len(roi)], [roi[-1]], 0, (255, 0, 0), 2)

            # Draw numeric identifier.
            cx, cy = compute_centroid(roi[-1])
            cv2.putText(img=stack[len(roi)], text=str(ID+1), org=(cx + DELTA, cy + DELTA), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, color=(255, 0, 0), thickness=2)
    return stack
