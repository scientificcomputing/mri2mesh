import numpy as np
import skimage.morphology as skim

from mri2mesh import morphology as morph


def test_closest_point():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)

    a[:2, :2, :2] = True
    b[5:7, 5:7, 5:7] = True

    a_b_index = morph.get_closest_point(a, b)
    assert np.allclose(a_b_index, (5, 5, 5))

    b_a_index = morph.get_closest_point(b, a)
    assert np.allclose(b_a_index, (1, 1, 1))


def test_connect_by_line():
    m1 = np.zeros((3, 3, 3), dtype=bool)
    m2 = np.zeros((3, 3, 3), dtype=bool)

    m1[0, 0, 0] = True
    m2[2, 0, 0] = True

    conn = morph.connect_by_line(m1, m2, footprint=skim.ball(1))
    expected = np.zeros((3, 3, 3), dtype=np.bool)
    expected[:, 0, 0] = True
    expected[:, 1, 0] = True
    expected[:, 0, 1] = True

    assert np.allclose(conn, expected)
