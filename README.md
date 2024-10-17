# windowed_radon
> Detect line-like elements in image with a windowed radon filter

This repository implements a simple image processing algorithm that locally detects line- or rod-like strutures in an image. This is useful for the [analysis of biological images](https://elifesciences.org/articles/27454) or in [astrophyics](https://doi.org/10.1051/0004-6361:20021571) as it allows one to estimate the local anisotropy in the image, and, for example, encode it as a nematic tensor. The image is partitioned into small patches and a [Radon transform](https://en.wikipedia.org/wiki/Radon_transform) is applied to each patch. The Radon transform computes the line integral of the brightness across all possible lines throgh the patch, parametrized by line orientation and offset. Line-like elements in the image are thus transformed into discrete peaks, which can be detected much more easily.

The provided code carries out the windowed Radon transform and can return either a list of detected line elements, or a nematic tensor that measures the smoothed-out image anisotropy. Some tools to visually check the results are also provided. The jupyter notebook provides a quick tutorial on how to use the module.

Dependencies: `numpy`, `matplotlib`, `scipy`, `sciki-image`, `scikit-learn`.

This code was written for the paper [Geometric control of myosin II orientation during axis elongation](https://elifesciences.org/articles/78787) by Lefebvre, Claussen, _et al._.
