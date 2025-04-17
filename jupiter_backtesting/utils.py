import pandas as pd
from datetime import timedelta
from typing import Tuple

class Utils:
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            if not df.index.is_monotonic_increasing:
                raise ValueError("Data cannot be sorted chronologically")
        return df

    def time_based_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.validate_data(df)
        split_idx = int(len(df) * (1 - test_size))
        split_date = df.index[split_idx]
        return df.loc[:split_date - timedelta(minutes=1)], df.loc[split_date:]
