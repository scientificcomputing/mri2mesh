import numpy as np
import skimage.morphology as skim
import skimage
import scipy.ndimage as ndi


def get_closest_point(a, b):
    dist = ndi.distance_transform_edt(np.logical_not(a))
    assert isinstance(dist, np.ndarray)
    dist[np.logical_not(b)] = np.inf
    minidx = np.unravel_index(np.argmin(dist), a.shape)
    return minidx


def connect_by_line(m1, m2, footprint=skim.ball(1)):
    # compute connection between V3 and V4:
    pointa = get_closest_point(m1, m2)
    pointb = get_closest_point(m2, m1)

    # add a line between the shortest points to connect V3 and V4
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn = np.zeros(m1.shape, dtype=np.uint8)
    i, j, k = line
    conn[i, j, k] = 1
    return skim.binary_dilation(conn, footprint=footprint)


def binary_smoothing(img, footprint=skim.ball(1)):
    openend = skim.binary_opening(img, footprint=footprint)
    return skim.binary_closing(openend, footprint=footprint)
