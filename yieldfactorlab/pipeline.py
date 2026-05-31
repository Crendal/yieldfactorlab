from .config import AnalysisConfig
from .loaders import KoreaRatesLoader, USRatesLoader
from .models import PCAYieldCurveModel, NSSYieldCurveModel
from .strategy import CurveStrategyBacktester


class YieldFactorLab:
    def __init__(
        self,
        method: str = "PCA",
        factors: int = 3,
        country: str = "Korea",
        korea_file_path: str | None = None,
        korea_sheet_name: str = "Sheet1",
        start_date: str = "2016-01-01",
        end_date: str | None = "2025-12-31",
        rolling_window: int = 252,
        standardize: bool = False,
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        transaction_cost_bp: float = 0.2,
    ):
        self.config = AnalysisConfig(
            method=method.upper(),
            country=country,
            factors=factors,
            start_date=start_date,
            end_date=end_date,
            rolling_window=rolling_window,
            standardize=standardize,
            entry_z=entry_z,
            exit_z=exit_z,
            transaction_cost_bp=transaction_cost_bp,
        )

        self.korea_file_path = korea_file_path
        self.korea_sheet_name = korea_sheet_name

    def _get_loader(self, country: str):
        country_lower = country.lower()

        if country_lower == "korea":
            if self.korea_file_path is None:
                raise ValueError("korea_file_path is required for Korea data.")
            return KoreaRatesLoader(
                file_path=self.korea_file_path,
                sheet_name=self.korea_sheet_name
            )

        if country_lower in ["us", "usa", "unitedstates"]:
            return USRatesLoader()

        raise ValueError(f"Unsupported country: {country}")

    def run_single_country(self, country: str):
        loader = self._get_loader(country)
        yield_df = loader.load(
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )

        if self.config.method == "PCA":
            model = PCAYieldCurveModel(
                n_components=self.config.factors,
                rolling_window=self.config.rolling_window,
                standardize=self.config.standardize
            ).fit(yield_df)

            backtester = CurveStrategyBacktester(
                entry_z=self.config.entry_z,
                exit_z=self.config.exit_z,
                transaction_cost_bp=self.config.transaction_cost_bp
            ).run_pca_strategy(yield_df, model.score_df)

            return {
                "country": country,
                "yield": yield_df,
                "model": model,
                "summary": model.summary(),
                "pnl": backtester.pnl,
                "metrics": backtester.metrics,
            }

        if self.config.method == "NSS":
            model = NSSYieldCurveModel().fit(yield_df)

            backtester = CurveStrategyBacktester(
                entry_z=self.config.entry_z,
                exit_z=self.config.exit_z,
                transaction_cost_bp=self.config.transaction_cost_bp
            ).run_nss_strategy(model.residual)

            return {
                "country": country,
                "yield": yield_df,
                "model": model,
                "summary": model.summary(),
                "pnl": backtester.pnl,
                "metrics": backtester.metrics,
            }

        raise ValueError(f"Unsupported method: {self.config.method}")

    def run(self):
        country = self.config.country.lower()

        if country == "both":
            return {
                "Korea": self.run_single_country("Korea"),
                "US": self.run_single_country("US"),
            }

        return self.run_single_country(self.config.country)