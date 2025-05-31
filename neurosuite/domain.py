import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CORAL(BaseEstimator, TransformerMixin):
    """
    CORrelation ALignment (CORAL) domain adaptation.
    Aligns source to target by matching second-order statistics (covariances).
    """

    def fit(self, X, groups=None):
        # Compute the reference covariance (whiten target)
        if groups is not None:
            unique_groups = np.unique(groups)
            covs = [np.cov(X[groups == g].T) for g in unique_groups]
            self.cov_avg_ = np.mean(covs, axis=0)
        else:
            self.cov_avg_ = np.cov(X.T)
        return self

    def transform(self, X):
        cov_src = np.cov(X.T)
        cov_src_inv = np.linalg.pinv(cov_src)
        whitening = np.linalg.cholesky(cov_src_inv)
        coloring = np.linalg.cholesky(self.cov_avg_)
        X_aligned = (X @ whitening.T) @ coloring
        return X_aligned
