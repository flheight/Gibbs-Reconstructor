import numpy as np
from scipy.linalg.lapack import dposv


class GibbsReconstructor:
    """
    A class to perform Gibbs reconstruction of missing data in a dataset.

    This class uses a regularized approach to estimate the coefficients for reconstructing
    missing values in the input data matrix using Gibbs sampling methods.

    Attributes:
        alpha (float): Regularization parameter for the Ridge regression.

    Methods:
        fit(X): Fits the model to the input data matrix X.
        predict(z): Predicts missing values in the input array z.
    """

    def __init__(self, alpha=1e-3):
        """
        Initializes the GibbsReconstructor with a given regularization parameter.

        Parameters:
            alpha (float): Regularization parameter. Default is 1e-3.
        """
        self.alpha = alpha

    def fit(self, X):
        """
        Fits the GibbsReconstructor model to the input data.

        This method computes the coefficients based on the provided dataset X,
        taking into account the regularization to handle overfitting.

        Parameters:
            X (ndarray): A 2D NumPy array of shape (n_samples, n_features) representing the input data.

        Returns:
            None: The coefficients are stored in the instance variable coef_.
        """
        n, p = X.shape

        X = np.hstack((X, np.ones((n, 1))))

        beta = np.random.randn(p + 1, p + 1)

        XtX = X.T @ X

        beta = np.zeros((p + 1, p + 1))

        for k in range(p):
            mask = np.ones(p + 1, dtype=bool)
            mask[k] = False

            LHS = np.array(XtX[mask][:, mask], order="F", dtype=np.float64)
            LHS.flat[:: p + 1] += n * self.alpha
            RHS = np.array(XtX[mask, k], order="F", dtype=np.float64)

            beta[k, mask] = dposv(LHS, RHS)[1]

        self.coef_ = beta

    def predict(self, z):
        """
        Predicts missing values in the input array z using the fitted model.

        This method handles missing values by initializing them with random draws from a normal distribution
        and reconstructs the data based on the learned coefficients.

        Parameters:
            z (ndarray): A 1D NumPy array containing the data with potential missing values (NaNs).

        Returns:
            ndarray: A reconstructed 1D NumPy array with estimated values for the missing entries.
        """
        p = z.size
        missing_idxs = np.where(np.isnan(z))[0]

        z[missing_idxs] = np.random.randn(missing_idxs.size)

        A = np.eye(p + 1)
        for k in missing_idxs:
            A[k] = np.dot(self.coef_[k], A)

        D, P = np.linalg.eig(A)
        D = np.diag(D)
        P_inv = np.linalg.inv(P)
        D[np.abs(D) < 1] = 0
        A_inf = P @ D @ P_inv

        A_inf = A_inf.real[:p, :p]

        return A_inf @ z
