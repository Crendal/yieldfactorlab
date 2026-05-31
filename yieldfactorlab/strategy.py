import numpy as np
import pandas as pd


class CurveStrategyBacktester:
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.3, transaction_cost_bp: float = 0.2):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.transaction_cost_bp = transaction_cost_bp

    def _mean_reversion_signal(self, z: pd.Series) -> pd.Series:
        signal = pd.Series(index=z.index, data=0.0)
        pos = 0

        for i, zi in enumerate(z):
            if np.isnan(zi):
                signal.iloc[i] = 0
                continue

            if pos == 0:
                if zi > self.entry_z:
                    pos = -1
                elif zi < -self.entry_z:
                    pos = 1
            else:
                if abs(zi) < self.exit_z:
                    pos = 0

            signal.iloc[i] = pos

        return signal

    def run_pca_strategy(self, yield_df: pd.DataFrame, score_df: pd.DataFrame):
        spread = pd.DataFrame(index=yield_df.index)

        required = {"2Y", "5Y", "10Y"}
        if not required.issubset(set(yield_df.columns)):
            raise ValueError("PCA strategy requires 2Y, 5Y, and 10Y columns.")

        spread["2s10s"] = yield_df["10Y"] - yield_df["2Y"]
        spread["2s5s10s_bfly"] = 2 * yield_df["5Y"] - yield_df["2Y"] - yield_df["10Y"]

        spread_change_bp = spread.diff() * 100

        pc2_signal = self._mean_reversion_signal(score_df["PC2_z"])
        pc3_signal = self._mean_reversion_signal(score_df["PC3_z"])

        pnl = pd.DataFrame(index=score_df.index)
        pnl["PC2_2s10s"] = pc2_signal.shift(1) * spread_change_bp["2s10s"]
        pnl["PC3_bfly"] = pc3_signal.shift(1) * spread_change_bp["2s5s10s_bfly"]

        pnl["PC2_2s10s"] -= pc2_signal.diff().abs() * self.transaction_cost_bp
        pnl["PC3_bfly"] -= pc3_signal.diff().abs() * self.transaction_cost_bp

        pnl["Total"] = pnl["PC2_2s10s"] + pnl["PC3_bfly"]
        pnl = pnl.dropna()

        self.pnl = pnl
        self.metrics = self._metrics(pnl)

        return self

    def run_nss_strategy(self, residual_df: pd.DataFrame, holding_period: int = 20):
        """
        Simple NSS relative value strategy:
        - positive residual: market yield > fitted yield = cheap bond point → long
        - negative residual: market yield < fitted yield = rich bond point → short
        """

        signal = -residual_df.apply(lambda x: x.rank(pct=True) - 0.5, axis=1)
        future_change_bp = residual_df.diff(holding_period).shift(-holding_period) * 100

        pnl = (signal * future_change_bp).mean(axis=1).dropna()
        pnl = pnl.to_frame("NSS_RV")

        self.pnl = pnl
        self.metrics = self._metrics(pnl)

        return self

    def _metrics(self, pnl: pd.DataFrame) -> pd.DataFrame:
        rows = {}

        for col in pnl.columns:
            s = pnl[col].dropna()
            if len(s) == 0:
                continue

            cum = s.cumsum()
            rows[col] = {
                "Total PnL(bp)": cum.iloc[-1],
                "Avg Daily PnL(bp)": s.mean(),
                "Daily Vol(bp)": s.std(),
                "Sharpe": s.mean() / s.std() * np.sqrt(252) if s.std() != 0 else np.nan,
                "MDD(bp)": (cum - cum.cummax()).min(),
                "Win Rate": (s > 0).mean(),
                "N": len(s),
            }

        return pd.DataFrame(rows).T