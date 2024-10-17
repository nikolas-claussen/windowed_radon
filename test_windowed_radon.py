#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep  15:10:36 2020

@author: nikolas, nclaussen@ucsb.edu

Tests for the module windowed_radon.

"""

import numpy as np
import windowed_radon as wr
from skimage import transform
from skimage.data import shepp_logan_phantom
from skimage.morphology import disk

rng = np.random.default_rng()


radon_matrix_5 = wr.get_radon_tf_matrix(5)
radon_matrix_11 = wr.get_radon_tf_matrix(11)
radon_matrix_21 = wr.get_radon_tf_matrix(21)


def test_get_radon_tf_matrix():
    image = shepp_logan_phantom()
    image = transform.rescale(image, scale=0.4, mode='reflect',
                              multichannel=False)
    size = 21
    e_len = int((size-1)/2)
    image = image[50:50+size, 50:50+size] * disk(e_len)
    radon_ref = transform.radon(image,
                                theta=np.linspace(0, 180, 21, endpoint=False),
                                circle=True, preserve_range=True)
    radon_mat_method = radon_matrix_21.dot(image.flatten()).reshape(size, size)
    err = np.abs(radon_mat_method - radon_ref)

    assert np.max(err) < 1e-10


def test_runs_errorfree():
    # check whether there are any runtime errors and shapes are ok
    im = np.ones((20, 20))
    m = wr.windowed_radon(im, radon_matrix_5)

    assert (len(m.shape) == 4) and (m.shape[2] == m.shape[3] == 2)


def test_homogeneity():
    # radon tf should be homogeneity (not linear though due to maxima)
    im1 = rng.normal(size=(20, 20))
    im2 = 2*im1

    m1 = wr.windowed_radon(im1, radon_matrix_5, maxima_method='peak_local_max')
    m2 = wr.windowed_radon(im2, radon_matrix_5, maxima_method='peak_local_max')

    assert np.allclose(2*m1, m2)


def test_symmetric():
    # resulting tensor should be symmetric
    im = rng.normal(size=(50, 50))
    m = wr.windowed_radon(im, radon_matrix_11)

    assert np.allclose(m, m.transpose(0, 1, 3, 2))


def test_constant_im():
    # a constant image should return 0.
    im = np.ones((50, 50))
    m = wr.windowed_radon(im, radon_matrix_11)
    assert np.allclose(m, 0)


def test_xy_stripes():
    # simple constant stripes in x or y direction
    # Note: one needs to choose p>1 for anisotropy detection to work.
    im_x, im_y = (np.zeros((50, 50)), np.zeros((50, 50)))
    im_x[::5, :] = 1
    im_y[:, ::5] = 1
    m_x = wr.windowed_radon(im_x, radon_matrix_11,
                            maxima_method='h_maxima')
    n_x = np.linalg.eigh(m_x)[1][:, :, 1, :]  # get highest eigen vector
    m_y = wr.windowed_radon(im_y, radon_matrix_11,
                            maxima_method='h_maxima')
    n_y = np.linalg.eigh(m_y)[1][:, :, 1, :]

    assert (np.allclose(np.abs(n_x[:, :, 0]), np.ones(m_x.shape[:2]),
                        atol=0.01)
            and np.allclose(np.abs(n_y[:, :, 1]), np.ones(m_y.shape[:2]),
                            atol=0.01))


def test_trace_intensity():
    # the trace should be proportional to the local intensity
    im, _ = np.meshgrid(np.arange(50), np.arange(50))
    m = wr.windowed_radon(im, radon_matrix_5)
    # low sigma so no smoothing
    tr = np.einsum('abii', m)
    tr = tr/np.mean(tr)
    im_resized = transform.resize(im[:48, :48], m.shape[:2])
    im_resized = im_resized/np.mean(im_resized)
    assert np.abs(tr-im_resized).mean() < 0.1


def test_crossing_lines():
    # test a cross-shaped input
    im = np.zeros((50, 50))
    im[23:27, :] = 1
    im[:, 23:27] = 1

    m = wr.windowed_radon(im, radon_matrix_11)

    # check that m is 0 in corners and correctly oriented on sides
    m_corner = m[0, 0]
    m_top = m[0, 3] / m[0, 3].max()
    m_left = m[3, 0] / m[3, 0].max()
    
    corner = np.allclose(m_corner, np.zeros((2, 2)), atol=0.01)
    top = np.allclose(m_top, np.array([[0, 0], [0, 1]]), atol=0.01)
    left = np.allclose(m_left, np.array([[1, 0], [0, 0]]), atol=0.05)

    assert corner and top and left

"""
def test_smoothing():
    # test influence of smoothing parameter
    # take a criss-cross pattern and check that at high window size,
    # anisotropy is 0

    im = np.ones((50, 50))
    im[::2, ::2] = 0
    m = wr.windowed_radon(im, radon_matrix_11)
    # make traceless
    m_tr = m - np.einsum('abii,jk', m, np.eye(2))/2
    norm = np.linalg.norm(m, axis=(2, 3))
    norm_tr = np.linalg.norm(m_tr, axis=(2, 3));

    assert (norm_tr/norm).max() < 1e-5
"""