import numpy as np
import numba

from scipy.special import logsumexp


@numba.njit
def gaussian(x, mean, sig, offset):
    d = x.shape[0]
    ndim = mean.shape[0]
    dx = x - mean
    # invsig = np.linalg.inv(sig)
    r = np.zeros(d)
    # cov = sig + offset
    for i in range(d):
        invsig = np.linalg.inv(sig + offset[i])
        r[i] = dx[i].T @ invsig @ dx[i]
    return np.exp(-r / 2) / (
        np.power(2 * np.pi, ndim / 2) * np.sqrt(np.linalg.det(sig + offset[i]))
    )


@numba.njit
def multigaussian(x, means, sigs, offset):
    d = x.shape[0]
    ngauss = means.shape[0]

    z = np.zeros((d, ngauss))

    for i in range(ngauss):
        z[:, i] = gaussian(x, means[:, i], sigs[i], offset)

    return z


@numba.njit
def log_gaussian(x, mean, sig, offset):
    d = x.shape[0]
    ndim = mean.shape[0]
    dx = x - mean
    # invsig = np.linalg.inv(sig)
    r = np.zeros(d)

    for i in range(d):
        invsig = np.linalg.inv(sig + offset[i])
        r[i] = dx[i].T @ invsig @ dx[i]
    return -r / 2


@numba.njit
def log_multigaussian(x, means, sigs, offset):
    d = x.shape[0]
    ngauss = means.shape[0]

    z = np.zeros((d, ngauss))

    for i in range(ngauss):
        z[:, i] = log_gaussian(x, means[i], sigs[i], offset)

    return z


@numba.njit("f8[:](f8[:,:],f8[:],f8[:,:])", fastmath=True)
def log_gaussian_no_off(x, mean, sig):
    dx = x - mean
    invsig = np.linalg.inv(sig)
    r = np.zeros(len(dx))
    for i in range(len(dx)):
        r[i] = dx[i].T @ invsig @ dx[i]
    return -r / 2


@numba.njit("f8[:,:](f8[:,:],f8[:,:],f8[:,:,:])", fastmath=True)
def log_multigaussian_no_off(x, means, sigs):
    d = x.shape[0]
    ngauss = means.shape[0]
    z = np.zeros((d, ngauss))
    for i in range(ngauss):
        z[:, i] = log_gaussian_no_off(x, means[i], sigs[i])
    return z


@numba.njit
def recompute_mean(q, x):
    ngauss = q.shape[-1]
    ndim = x.shape[-1]
    mean = np.zeros((ngauss, ndim))
    qk = np.sum(q, axis=0)

    for i in range(ngauss):
        mean[i] = np.sum(q[:, i] * x.T, axis=-1) / qk[i]
    return mean


# @numba.njit
def recompute_mean_multix(q, x):
    ngauss = q.shape[-1]
    ndim = x.shape[-1]
    mean = np.zeros((ngauss, ndim))
    qk = np.sum(q, axis=0)

    for i in range(ngauss):
        mean[i] = np.einsum("i,ij->j", q[:, i], x[i]) / qk[i]
    return mean


@numba.njit
def recompute_cov(q, x, means, reg):
    ngauss = means.shape[0]
    ndim = means.shape[-1]

    cov = np.zeros((ngauss, ndim, ndim))

    qk = np.sum(q, axis=0)

    if reg != 0.0:
        qk += 1
    regul = reg * np.identity(ndim)

    for i in range(ngauss):
        dx = x - means[i]
        # sc = q[:,i]
        cov[i] = ((q[:, i] * dx.T) @ dx) / qk[i] + regul / qk[i]

    return cov


# @numba.njit
def recompute_cov_multix(q, x, means, offset, reg):
    ngauss = means.shape[0]
    ndim = means.shape[-1]

    cov = np.zeros((ngauss, ndim, ndim))

    qk = np.sum(q, axis=0)

    if reg != 0.0:
        qk += 1
    regul = reg * np.identity(ndim)

    for i in range(ngauss):
        dx = x[i] - means[i]
        outer = np.einsum("...i,...j", dx, dx) + offset[i]
        outer = np.einsum("i, ijk -> jk", q[:, i], outer) + regul
        cov[i] = outer / qk[i]

    return cov


# @numba.njit
def modify_x_cov(x, cov_x, means, cov):
    ngauss = means.shape[0]
    ndim = means.shape[-1]

    b = np.zeros((ngauss, len(x), ndim))

    Bcov = np.zeros((ngauss, len(x), ndim, ndim))

    for i in range(ngauss):
        Tik = cov[i] + cov_x
        Tik = np.linalg.inv(Tik)
        dx = x - means[i]
        tmul = np.einsum("...jk,...k->...j", Tik, dx)
        xc = np.einsum("ij,...j->...i", cov[i], tmul)
        b[i] = means[i] + xc
        xcov = np.einsum("jk,...kl,lm", cov[i], Tik, cov[i])
        # xcov = np.matmul(cov[i],np.matmul(Tik, cov[i]))
        Bcov[i] = cov[i] - xcov

    return b, Bcov


def log_gaussian_no_off(x, mean, sig):
    dx = x - mean
    invsig = np.linalg.inv(sig)
    r = np.einsum("ij, ...i,...j", invsig, dx, dx)
    return -r / 2


class GMM:
    def __init__(self, k, ndim):
        self.K = k
        self.ndim = ndim

        self.mean = np.zeros((k, ndim))
        self.amp = np.zeros(k)
        self.cov = np.zeros((k, ndim, ndim))

        self.mean_best = None

    def write(self, filename):
        """
        Saves a pickle file (with a given filename) for the GMM, can be reloaded back with the read function
        """
        import compress_pickle as cp

        cp.dump(self, filename)

    @staticmethod
    def read(filename):
        """
        Loads a previously saved GMM. Usage:
                gmm = GMM.read(filename)
        """
        import compress_pickle as cp

        return cp.load(filename)

    def _simpleEStep(self, x, offset):
        logdet = np.array(
            [np.linalg.slogdet(self.cov[i] + offset)[1] for i in range(self.K)]
        ).T
        log_gauss = (
            log_multigaussian(x, self.mean, self.cov, offset)
            + np.log(self.amp)
            - logdet / 2
            - np.log(2 * np.pi) * self.ndim / 2
        )

        # self.like = -np.sum(log_gauss)

        summed = logsumexp(log_gauss, axis=-1)

        self.like = -np.sum(summed)

        self.q = np.exp(log_gauss.T - summed).T
        # self.q[mask] = 0

    def _EStep(self, x, offset):
        self._simpleEStep(x, offset)

    def _simpleMStep(self, x, offset):
        qk = np.sum(self.q, axis=0)

        self.amp = qk / self.n
        self.amp = self.amp / np.sum(self.amp)
        self.mean = recompute_mean(self.q, x)
        self.cov = recompute_cov(self.q, x, self.mean, self.reg)

    def _MStep(self, x, offset, selection=None):
        self._simpleMStep(x, offset)

    def _initialize(self, x, scale, selection):
        self.mean = (
            np.random.multivariate_normal(
                np.zeros(self.ndim), scale * scale * np.identity(self.ndim), self.K
            )
            + x[np.random.randint(0, self.n, self.K)]
        )
        self.cov = np.zeros((self.K, self.ndim, self.ndim))

        for i in range(self.ndim):
            self.cov[:, i, i] = scale * scale

        self.amp = np.ones((self.K)) / self.K

        self.mean_best = None
        self.K_best = None
        self.amp_best = None
        self.cov_best = None

        self.selection = selection

    def _fitloop(self, x, offset):
        self.like_last = self.like
        self._EStep(x, offset)
        self._converged = np.abs(
            (self.like_last - self.like)
        ) < self._tolerance * np.abs(self.like_last)
        self._stop = (
            self._converged and self._niter > self._miniter
        ) or self._niter > self._maxiter
        self._MStep(x, offset)
        self._niter += 1

    def fit(
        self,
        x,
        scale,
        reg=0,
        tolerance=1e-3,
        maxiter=1000,
        miniter=100,
        offset=None,
        selection=None,
        initialize=True,
    ):
        self.n = len(x)
        if initialize:
            self._initialize(x, scale, selection)

        self.reg = reg
        self.noise = True
        if offset is None:
            offset = np.zeros((self.n, self.ndim, self.ndim))
            self.noise = False

        self._EStep(x, offset)

        self.like_last = self.like

        self._stop = False
        self._niter = 0
        self._miniter = miniter
        self._maxiter = maxiter
        self._tolerance = tolerance

        while not self._stop:
            self._fitloop(x, offset)

        if self.mean_best is None:
            self.mean_best = self.mean
            self.amp_best = self.amp
            self.cov_best = self.cov
            self.K_best = self.K

    def predict(self, x, offset=None):
        if offset is None:
            offset = np.zeros((len(x), self.ndim, self.ndim))

        logdet = np.array(
            [
                np.linalg.slogdet(self.cov_best[i] + offset)[1]
                for i in range(self.K_best)
            ]
        ).T
        log_gauss = (
            log_multigaussian(x, self.mean_best, self.cov_best, offset)
            + np.log(self.amp_best)
            - logdet / 2
            - np.log(2 * np.pi) * self.ndim / 2
        )

        summed = logsumexp(log_gauss, axis=-1)

        return np.exp(summed), log_gauss - np.log(self.amp_best)

    def predictComponent(self, comp, x, offset=None):
        if offset is None:
            offset = np.zeros((len(x), self.ndim, self.ndim))

        logdet = np.linalg.slogdet(self.cov_best[comp] + offset)[1]
        log_gauss = (
            log_gaussian(x, self.mean_best[comp], self.cov_best[comp], offset)
            - logdet / 2
            - np.log(2 * np.pi) * self.ndim / 2
        )

        return log_gauss

    def sample(self, size):
        choice = np.random.choice(self.K_best, size=size, p=self.amp_best)
        chol = np.linalg.cholesky(self.cov_best)
        rand = np.random.multivariate_normal(
            self.ndim * [0], np.identity(self.ndim), size=size
        )

        return self.mean_best[choice] + np.einsum("...ij,...j", chol[choice], rand)

    def sampleCurrent(self, size):
        choice = np.random.choice(self.K, size=size, p=self.amp)
        chol = np.linalg.cholesky(self.cov)
        rand = np.random.multivariate_normal(
            self.ndim * [0], np.identity(self.ndim), size=size
        )

        return self.mean[choice] + np.einsum("...ij,...j", chol[choice], rand)

    def sampleComponent(self, size, comp):
        chol = np.linalg.cholesky(self.cov_best[comp])
        rand = np.random.multivariate_normal(
            self.ndim * [0], np.identity(self.ndim), size=size
        )

        return self.mean_best[comp] + np.einsum("...ij,...j", chol, rand)


class GMMNoise(GMM):
    def __init__(self, k, ndim):
        super().__init__(k, ndim)

    def _noiseEStep(self, x, offset):
        self._simpleEStep(x, offset)

        self.b, self.bcov = modify_x_cov(x, offset, self.mean, self.cov)

    def _EStep(self, x, offset):
        self._noiseEStep(x, offset)

    def _noiseMStep(self, x, offset):
        qk = np.sum(self.q, axis=0)
        self.amp = qk / self.n
        self.amp = self.amp / np.sum(self.amp)

        self.mean = recompute_mean_multix(self.q, self.b)
        self.cov = recompute_cov_multix(self.q, self.b, self.mean, self.bcov, self.reg)

    def _MStep(self, x, offset):
        self._noiseMStep(x, offset)


class AdaptiveGMM(GMM):
    def __init__(self, kmax, kmin, ndim):
        self.Kmax = kmax
        self.Kmin = kmin
        self.ndim = ndim

        self.ncomp = ndim + ndim * (ndim + 1) / 2

        self.K = kmax

        self.like_min = np.inf

        self.mean = np.zeros((kmax, ndim))
        self.amp = np.zeros(kmax)
        self.cov = np.zeros((kmax, ndim, ndim))

        self.mean_best = None

    def _killComponent(self, index):
        """self.like += (self.ncomp/2) * np.sum(np.log(self.amp[self.amp > 0]))
        self.like += (self.K/2) * np.log(self.n) * (self.ncomp + 1)
        if self.like < self.like_min:
                self.like_min = self.like
                self.K_best = self.K
                self.mean_best = self.mean
                self.amp_best = self.amp
                self.cov_best = self.cov
        """
        ind = np.arange(self.K)
        ind = ind[ind != index]
        self.q = self.q[:, ind]
        self.mean = self.mean[ind]
        self.cov = self.cov[ind]
        self.amp = self.amp[ind]
        self.amp = self.amp / np.sum(self.amp)
        if hasattr(self, "b"):
            self.b = self.b[ind]
            self.bcov = self.bcov[ind]
        # self.amp = self.amp/np.sum(self.amp)
        self.K -= 1
        self._niter = 0

    def _compareBest(self, x, offset, redo=False, shift=0):
        if redo:
            logdet = np.array(
                [np.linalg.slogdet(self.cov[i] + offset)[1] for i in range(self.K)]
            ).T
            log_gauss = (
                log_multigaussian(x, self.mean, self.cov, offset)
                + np.log(self.amp)
                - logdet / 2
                - np.log(2 * np.pi) * self.ndim / 2
            )
            summed = logsumexp(log_gauss, axis=-1)

            self.like = -np.sum(summed)
        self.like += (self.ncomp / 2) * np.sum(np.log(self.amp[self.amp > 0]))
        self.like += (self.K / 2) * np.log(self.n) * (self.ncomp + 1)
        self.like += shift
        if self.like < self.like_min:
            self.like_min = self.like
            self.K_best = self.K
            self.mean_best = self.mean
            self.amp_best = self.amp
            self.cov_best = self.cov

    def _EStep(self, x, offset):
        self._simpleEStep(x, offset)
        self._compareBest(x, offset, False)

    def _adaptiveMStep(self, x, offset):
        qk = np.maximum(np.sum(self.q, axis=0) - self.ncomp / 2, 0)

        if len(qk[qk == 0]) > 0 and self.K >= self.Kmin:
            # kill only first component
            kill = np.where(qk == 0)[0]
            while len(kill) > 0:
                self._killComponent(kill[0])
                qk = np.maximum(np.sum(self.q, axis=0) - self.ncomp / 2, 0)
                kill = np.where(qk == 0)[0]
                self._EStep(x, offset)
            # self._simpleMStep(x, offset)
            self._stop = self.K < self.Kmin

            # self.like += (self.ncomp/2) * np.sum(np.log(self.amp)) + (self.K/2) * np.log(self.n) * (self.ncomp + 1)# + self.K*(self.ncomp + 1)/2

        self._simpleMStep(x, offset)
        if self.K > self.Kmin:
            self._stop = False

            # kill smallest component if converged but not yet at Kmin
            if self._niter > self._maxiter or (
                self._converged and self._niter > self._miniter
            ):
                kill = np.argmin(self.amp)
                self._killComponent(kill)

    def _MStep(self, x, offset):
        self._adaptiveMStep(x, offset)


class AdaptiveGMMNoise(AdaptiveGMM, GMMNoise):
    def __init__(self, kmax, kmin, ndim):
        self.Kmax = kmax
        self.Kmin = kmin
        self.ndim = ndim

        self.ncomp = ndim + ndim * (ndim + 1) / 2

        self.K = kmax

        self.like_min = np.inf

        self.mean = np.zeros((kmax, ndim))
        self.amp = np.zeros(kmax)
        self.cov = np.zeros((kmax, ndim, ndim))

        self.mean_best = None

    def _simpleMStep(self, x, offset):
        self._noiseMStep(x, offset)

    def _EStep(self, x, offset):
        self._noiseEStep(x, offset)

    def _MStep(self, x, offset):
        self._adaptiveMStep(x, offset)


class IncompleteGMM(GMM):
    def __init__(self, k, ndim):
        self.K = k
        self.ndim = ndim

        self.mean = np.zeros((k, ndim))
        self.amp = np.zeros(k)
        self.cov = np.zeros((k, ndim, ndim))

    def _sampleAndReject(self, n, offset):
        # oversample data
        x_sample = self.sampleCurrent(n)

        if self.noise:
            noise = np.random.multivariate_normal(
                self.ndim * [0], np.identity(self.ndim), size=n
            )
            chol = np.linalg.cholesky(offset)
            choice = np.random.choice(self.n, size=n)

            x_sample = x_sample + np.einsum("...ij,...j", chol[choice], noise)
            offset_sample = offset[choice]
        else:
            offset_sample = np.zeros((n, self.ndim, self.ndim))

        omega = self.selection(x_sample)
        samples = omega > np.random.uniform(0, 1, size=n)

        return x_sample, offset_sample, omega, samples

    def _initialize(self, x, offset, selection):
        super()._initialize(x, offset, selection)

        # Poisson interval - taken from pygmmis implementation

        from scipy.stats import chi2

        alpha = 0.32
        self.lower = 0.5 * chi2.ppf(alpha / 2, 2 * self.n)
        self.upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * self.n + 2)

        self.omega_x = selection(x)

    def _fitloop(self, x, offset):
        # sample
        new_n = 10 * self.n
        x_sample, offset_sample, omega_sample, samples = self._sampleAndReject(
            new_n, offset
        )

        while (
            len(x_sample[samples]) > self.upper or len(x_sample[samples]) < self.lower
        ):
            new_n = int(len(x_sample) / len(x_sample[samples]) * self.n)
            x_sample, offset_sample, omega_sample, samples = self._sampleAndReject(
                new_n, offset
            )

        rej_x = x_sample[~samples]
        rej_offset = offset_sample[~samples]
        joint_x = np.vstack([x, rej_x])
        joint_offset = np.vstack([offset, rej_offset])

        self.like_last = self.like
        self._EStep(joint_x, joint_offset)
        shift = -np.log(np.sum(omega_sample[~samples]) + np.sum(self.omega_x)) + np.log(
            len(joint_x)
        )
        self._compareBest(x, offset, True, shift)

        # self.like -= -np.sum(log_gauss)
        self._converged = np.abs(
            (self.like_last - self.like)
        ) < self._tolerance * np.abs(self.like_last)
        self._stop = (
            self._converged and self._niter > self._miniter
        ) or self._niter > self._maxiter
        Kcur = int(self.K)
        self._MStep(joint_x, joint_offset)
        self._niter += 1


class IncompleteGMMNoise(IncompleteGMM, GMMNoise):
    def __init__(self, k, ndim):
        self.K = k
        self.ndim = ndim
        self.mean = np.zeros((k, ndim))
        self.amp = np.zeros(k)
        self.cov = np.zeros((k, ndim, ndim))
        self.mean_best = None

    def _simpleMStep(self, x, offset):
        self._noiseMStep(x, offset)

    def _EStep(self, x, offset):
        self._noiseEStep(x, offset)

    def _MStep(self, x, offset):
        self._simpleMStep(x, offset)


class IncompleteAdaptiveGMMNoise(IncompleteGMMNoise, AdaptiveGMMNoise):
    def __init__(self, kmax, kmin, ndim):
        self.Kmax = kmax
        self.Kmin = kmin
        self.ndim = ndim

        self.ncomp = ndim + ndim * (ndim + 1) / 2

        self.K = kmax

        self.like_min = np.inf

        self.mean = np.zeros((kmax, ndim))
        self.amp = np.zeros(kmax)
        self.cov = np.zeros((kmax, ndim, ndim))
        self.mean_best = None

    def _simpleMStep(self, x, offset):
        self._noiseMStep(x, offset)

    def _EStep(self, x, offset):
        self._noiseEStep(x, offset)

    def _MStep(self, x, offset):
        self._adaptiveMStep(x, offset)
