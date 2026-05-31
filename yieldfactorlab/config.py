from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    method: str = "PCA"          # "PCA" or "NSS"
    country: str = "Korea"       # "Korea", "US", "Both"
    factors: int = 3
    start_date: str = "2016-01-01"
    end_date: str | None = "2025-12-31"
    rolling_window: int = 252
    standardize: bool = False
    entry_z: float = 2.0
    exit_z: float = 0.3
    transaction_cost_bp: float = 0.2