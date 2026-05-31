import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize


class PCAYieldCurveModel:
    def __init__(self, n_components: int = 3, rolling_window: int = 252, standardize: bool = False):
        self.n_components = n_components
        self.rolling_window = rolling_window
        self.standardize = standardize
        self.model = PCA(n_components=n_components)

    def fit(self, yield_df: pd.DataFrame):
        self.yield_df = yield_df.copy()
        self.dy_bp = self.yield_df.diff().dropna() * 100

        X = self.dy_bp.copy()

        if self.standardize:
            self.x_mean = X.mean()
            self.x_std = X.std()
            X = (X - self.x_mean) / self.x_std

        self.scores = self.model.fit_transform(X.values)

        self.explained = pd.Series(
            self.model.explained_variance_ratio_,
            index=[f"PC{i+1}" for i in range(self.n_components)],
            name="Explained Variance"
        )

        self.loadings = pd.DataFrame(
            self.model.components_.T,
            index=X.columns,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )

        self.score_df = pd.DataFrame(
            self.scores,
            index=X.index,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )

        for pc in [f"PC{i+1}" for i in range(self.n_components)]:
            self.score_df[f"{pc}_z"] = self._rolling_zscore(self.score_df[pc])

        return self

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        return (s - s.rolling(self.rolling_window).mean()) / s.rolling(self.rolling_window).std()

    def summary(self) -> dict:
        return {
            "explained": self.explained,
            "loadings": self.loadings,
            "scores": self.score_df,
        }


class NSSYieldCurveModel:
    """
    Nelson-Siegel-Svensson curve fitting.

    Input yield_df:
        index: dates
        columns: tenors such as 1Y, 2Y, 3Y, 5Y, 10Y, 20Y, 30Y
        values: yields in percent
    """

    def __init__(self):
        pass

    @staticmethod
    def _tenor_to_years(tenor: str) -> float:
        tenor = tenor.upper().replace("Y", "")
        return float(tenor)

    @staticmethod
    def nss_rate(t, beta0, beta1, beta2, beta3, tau1, tau2):
        t = np.asarray(t, dtype=float)

        term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
        term2 = term1 - np.exp(-t / tau1)

        term3 = (1 - np.exp(-t / tau2)) / (t / tau2)
        term4 = term3 - np.exp(-t / tau2)

        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4

    def fit(self, yield_df: pd.DataFrame):
        self.yield_df = yield_df.copy()
        self.tenors = np.array([self._tenor_to_years(c) for c in yield_df.columns])

        fitted_rows = []
        param_rows = []

        for dt, row in yield_df.iterrows():
            y = row.values.astype(float)

            def objective(params):
                beta0, beta1, beta2, beta3, tau1, tau2 = params
                if tau1 <= 0 or tau2 <= 0:
                    return 1e9
                fitted = self.nss_rate(self.tenors, *params)
                return np.mean((y - fitted) ** 2)

            initial = np.array([
                y[-1],
                y[0] - y[-1],
                0.0,
                0.0,
                2.0,
                10.0
            ])

            bounds = [
                (-10, 20),
                (-20, 20),
                (-20, 20),
                (-20, 20),
                (0.1, 30),
                (0.1, 50)
            ]

            res = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)

            params = res.x
            fitted = self.nss_rate(self.tenors, *params)

            param_rows.append(params)
            fitted_rows.append(fitted)

        self.params = pd.DataFrame(
            param_rows,
            index=yield_df.index,
            columns=["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
        )

        self.fitted_curve = pd.DataFrame(
            fitted_rows,
            index=yield_df.index,
            columns=yield_df.columns
        )

        self.residual = self.yield_df - self.fitted_curve

        return self

    def summary(self) -> dict:
        return {
            "params": self.params,
            "fitted_curve": self.fitted_curve,
            "residual": self.residual,
        }