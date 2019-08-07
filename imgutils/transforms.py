from math import ceil
import numpy as np

#Ratio was measured empirically and used as a basis for all other conversions
def microns_to_pixels(microns, magnification, microscope, binning, fiddle=1.0):
    microscope = microscope.lower().strip()
    if microscope in {'flo'}:
        pixels_per_micron_at_20x = 1.237 / float(binning[0])

    elif microscope in {'ds', 'deathstar'}:
        pixels_per_micron_at_20x = 3.077 / float(binning[0])

    elif microscope in {'flo2'}:
        pixels_per_micron_at_20x = 3.077 / float(binning[0])

    elif microscope in {'ds2', 'deathstar2'}:
        pixels_per_micron_at_20x = 1.237 / float(binning[0])

    elif microscope in {'ds3', 'deathstar3'}:
        pixels_per_micron_at_20x = 3.077 / float(binning[0])

    # THIS VALUE IS A DUMMY ONE BECAUSE IXM DATA IS NOT CURRENTLY USING SURVIVAL ANALYSIS, ONLY STITCHING/STACKING.
    elif microscope in {'ixm'}:
        pixels_per_micron_at_20x = 42

    else:
        raise ValueError('Unknown microscope name detected. Try "flo", "ds", "deathstar", "flo2", "ds2", "deathstar2", "ds3", "deathstar3", or "ixm".')

    return ceil(microns * fiddle * magnification / 20.0 * pixels_per_micron_at_20x)

def linear_map(extrema, stack):
    for i, img in enumerate(stack):
        _min, _max = extrema[i]
        if _max != 0:
            np.clip(img, _min, _max, stack[i])
            stack[i] = 255.0 * (stack[i] - _min) / (_max - _min)
    return stack

def enhance_contrast(stack, bins=256):
    # If incoming stack is 2D, then it corresponds to an image. Convert dimension to 3D.
    converted = False
    if len(stack.shape) == 2:
        stack = stack.reshape(1, *stack.shape)
        converted = True

    extrema = []
    _min, _max = stack.min(), stack.max()
    for img in stack:
        hist, bin_edges = np.histogram(img, bins)
        #find min
        for i, val in enumerate(hist):
            if val < .1 * img.size and val > .0005 * img.size:
                #Find average value for use as minimum
                _min = int((bin_edges[i] + bin_edges[i+1])/2.0)
                break
        #find max
        for i, val in enumerate(hist[-1::-1]):
            if val < .15 * img.size and val > .0005 * img.size:
                #Use only leftmost edge as value is guaranteed to exist
                _max = int(bin_edges[bins-i-1])
                break
        extrema.append((_min, _max))
    linear_map(extrema, stack)

    if converted:
        stack = stack.reshape(stack.shape[1:])
    return stack

def contour_CV_to_NP(contour):
    '''Contour expected in OpenCV format'''
    contour = contour[:, 0]
    return np.array([contour[:, 1], contour[:, 0]])

def to_xbit(img, bit):
    if img.size == 0: return img
    elif np.max(img) <= 2.0 ** bit - 1: return img
    else:
        imgf = img.astype(float)
        imgx = np.clip(imgf, img.min(), img.max())
        imgx -= img.min()
        imgx = (( (2.0 ** bit - 1) / (img.max() - img.min()) ) * imgx)
        return imgx

def to_8bit(img):
    img = np.copy(img)
    if img.dtype != np.uint8:
        img8 = to_xbit(img, 8)
        img = img8.astype(np.uint8)
    return img

def stack_to_8bit(stack):
    if stack.dtype != np.uint8:
        imgs = []
        for img in stack:
            copy = np.copy(img)
            imgs.append(to_8bit(copy))
            del copy
        stack = to_stack(imgs)
    return stack

def to_16bit(img):
    img = np.copy(img)
    if img.dtype != np.uint16:
        img16 = to_xbit(img, 16)
        img = img16.astype(np.uint16)
    return img

def to_stack(imgs):
    expand_first_dim = lambda img: np.expand_dims(img, axis=0)
    imgs = [expand_first_dim(img) for img in imgs]
    return np.concatenate(imgs, axis=0)

def to_rgb(img):
    s = np.expand_dims(img, axis=2)
    if img.dtype != np.uint8:
        s = to_8bit(s)
    return np.concatenate((s, s, s), axis=2)

def to_rgb_stack(stack):
    #Ensure stack is 8-bits
    if stack.dtype != np.uint8:
        imgs = [to_8bit(img) for img in stack]
        stack = to_stack(imgs)
    last_dim = len(stack.shape)
    s = np.expand_dims(stack, axis=last_dim)
    return np.concatenate((s, s, s), axis=last_dim)

def to_rgba_stack(stack):
    #Ensure stack is 8-bits
    if stack.dtype != np.uint8:
        imgs = [to_8bit(img) for img in stack]
        stack = to_stack(imgs)
    last_dim = len(stack.shape)
    s = np.expand_dims(stack, axis=last_dim)
    shape = tuple(list(stack.shape) + [1])
    z = np.zeros(shape, dtype=np.uint8)
    return np.concatenate((s, s, s, z), axis=last_dim)
