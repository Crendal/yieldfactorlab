from abc import ABC, abstractmethod
import pandas as pd
from pandas_datareader import data as web


class BaseRatesLoader(ABC):
    @abstractmethod
    def load(self, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        pass


class KoreaRatesLoader(BaseRatesLoader):
    def __init__(self, file_path: str, sheet_name: str = "Sheet1"):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.tenor_map = {
            "국고채(1년) [D]": "1Y",
            "국고채(2년) [D]": "2Y",
            "국고채(3년) [D]": "3Y",
            "국고채(5년) [D]": "5Y",
            "국고채(10년) [D]": "10Y",
            "국고채(20년) [D]": "20Y",
            "국고채(30년) [D]": "30Y",
            "국고채(50년) [D]": "50Y",
        }

    def load(self, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
        df = df.sort_index()

        cols = [c for c in self.tenor_map if c in df.columns]
        if len(cols) < 5:
            raise ValueError(f"Not enough Korean Treasury columns. Found: {df.columns.tolist()}")

        y = df[cols].rename(columns=self.tenor_map)
        y = y.apply(pd.to_numeric, errors="coerce")
        y = y.loc[start_date:end_date].dropna()

        return y


class USRatesLoader(BaseRatesLoader):
    def __init__(self):
        self.series_map = {
            "DGS1": "1Y",
            "DGS2": "2Y",
            "DGS3": "3Y",
            "DGS5": "5Y",
            "DGS7": "7Y",
            "DGS10": "10Y",
            "DGS20": "20Y",
            "DGS30": "30Y",
        }

    def load(self, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        raw = web.DataReader(
            list(self.series_map.keys()),
            "fred",
            start_date,
            end_date
        )

        y = raw.rename(columns=self.series_map)
        y = y.dropna()

        return y