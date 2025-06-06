import numpy as np
from sklearn.preprocessing import StandardScaler

class CORAL:
    def fit_transform(self, X, groups):
        subjects = np.unique(groups)
        Xs = [X[groups == s] for s in subjects]

        if len(Xs) < 2:
            raise ValueError("CORAL requires at least two subjects to align.")

        X_src = Xs[0]
        scaler = StandardScaler()
        X_src = scaler.fit_transform(X_src)
        cov_src = np.cov(X_src, rowvar=False)

        Xt_all = []
        for Xt in Xs[1:]:
            if Xt.shape[0] < 2:
                continue  # Skip subjects with <2 samples

            Xt = scaler.fit_transform(Xt)
            cov_tgt = np.cov(Xt, rowvar=False)

            eps = 1e-6
            cov_src_reg = cov_src + np.eye(cov_src.shape[0]) * eps
            cov_tgt_reg = cov_tgt + np.eye(cov_tgt.shape[0]) * eps

            try:
                C_src_inv = np.linalg.inv(cov_src_reg)
                A_coral = np.dot(np.linalg.cholesky(C_src_inv), np.linalg.cholesky(cov_tgt_reg))
                Xt_aligned = np.dot(Xt, A_coral)
                Xt_all.append(Xt_aligned)
            except np.linalg.LinAlgError:
                print("Skipping subject due to non-invertible matrix.")
                continue

        X_combined = np.vstack([X_src] + Xt_all)
        return X_combined
