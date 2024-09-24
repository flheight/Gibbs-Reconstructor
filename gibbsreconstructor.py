import numpy as np
from scipy.linalg.blas import dsyrk, dgemv
from scipy.linalg import solve


class _Rdige:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        X = np.array(X, order="F", dtype=np.float64)
        y = np.array(y, order="F", dtype=np.float64)

        XtX = dsyrk(1.0, X, trans=True)
        Xty = dgemv(1.0, X, y, trans=True)

        XtX.flat[:: XtX.shape[0] + 1] += self.alpha

        self.coef_ = np.asfortranarray(solve(XtX, Xty, assume_a="pos"))

    def predict(self, x):
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        x = np.array(x, order="F", dtype=np.float64)
        return dgemv(1.0, x, self.coef_)


class GibbsReconstructor:
    def __init__(self, alpha):
        self.alpha = alpha
        self.regs, self.stds = {}, {}

    def fit(self, X):
        for k in range(X.shape[1]):
            mask = np.arange(X.shape[1]) != k
            X_k = X[:, mask]
            y_k = X[:, k]

            self.regs[k] = _Rdige(alpha=self.alpha)
            self.regs[k].fit(X_k, y_k)

            residuals = y_k - self.regs[k].predict(X_k)
            self.stds[k] = np.std(residuals)

    def predict(self, z, n_samples):
        missing_idxs = np.where(np.isnan(z))[0]

        c = z
        c[missing_idxs] = 0
        s = np.zeros_like(c)

        for _ in range(n_samples):
            for k in missing_idxs:
                mask = np.arange(c.size) != k

                mu_k = self.regs[k].predict(c[None, mask])
                sigma_k = self.stds[k]
                c[k] = np.random.normal(mu_k, sigma_k)[0]

            s += c

        return s / n_samples
