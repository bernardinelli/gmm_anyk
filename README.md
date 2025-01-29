# gmm_anyk
 Gaussian Mixture Model package accounting for measurement uncertainties, selection effects, and an arbitrary number of components

## Introduction

This repository implements a few modifications of the standard Gaussian Mixture Model algorithm that allows one to use datasets with measurement uncertainties and selection effects, as well as determine, during training, the optimal number of Gaussians required to describe the data.

The following references detail the procedure:
-  _[Bovy, Hogg and Howeis, 2011, Ann. Appl. Stat. 5 (2B) 1657 - 1677](https://doi.org/10.1214/10-AOAS439)_ (inclusion of measurement uncertainties)
- _[Melchior & Goulding, 2018, Astronomy and Computing, 25, 183](http://doi.org/10.1016/j.ascom.2018.09.013)_ (inclusion of selection effects)
- _[Figueiredo & Jain, 2002, IEEE Transactions on Pattern Analysis and Machine Intelligence, 24, 381](https://ieeexplore.ieee.org/document/990138)_ (determination of number of components)

The full implementation will be described in [Bernardinelli et al., 2025, arXiv:2501.01551](https://ui.adsabs.harvard.edu/abs/2025arXiv250101551B/abstract). 

## Basic usage 
Suppose we have a 3D dataset (`x`), on the example, we're using a 3D dataset with one cluster:
```py
import numpy as np
import gmm_anyk as ga

x = np.random.multivariate_normal(np.array([0., 2., 3.]), np.identity(3), size=1000) # 3D Gaussian centered at (0,2,3) with the identity matrix as the covariance, 1000 samples
gmm = ga.GMM(1,3) #number of clusters and number of dimensions
gmm.fit(x, #your data 
		scale=1, # initial guess for the scale of the standard deviation 
		tolerance=1e-5, # tolerance for the log-likelihood until the training is considered successfgull 
		maxiter=10000, # max number of iterations
		miniter=1000) # min number of iterations
print(gmm.mean_best) # best fit mean, it should be close to the input
print(gmm.cov_best) # best fit covariance
```

`GMMNoise` includes measurement uncertainties, `AdaptiveGMM` uses an arbitrary number of components, and `IncompleteGMM` implements a stochastic selection effect. `IncompleteAdaptiveGMMNoise` implements all of these three effects at once. The classes also include options for numerical regularization and other initialization techniques.

## Dependencies
- `numpy`
- `numba`
- `scipy`
- Optional: `compress_pickle` (allows one to save and load the GMM class)
