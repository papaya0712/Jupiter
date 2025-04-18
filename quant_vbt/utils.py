import pandas as pd
from datetime import timedelta
from typing import Tuple

class Utils:
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.has_duplicates:
            raise ValueError("Duplicate timestamps found in index")
        
        # remove duplicates
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep='first')]

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            if not df.index.is_monotonic_increasing:
                raise ValueError("Data cannot be sorted chronologically")
        return df
    
    def time_based_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.validate_data(df)
        
        delta = df.index.to_series().diff().min()
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            raise ValueError("Cannot determine a positive time delta from index")
        
        split_idx = int(len(df) * (1 - test_size))
        split_date = df.index[split_idx]
        
        train = df.loc[: split_date - delta]
        test  = df.loc[ split_date :]

        if not train.empty and not test.empty:
            if train.index.max() >= test.index.min():
                raise RuntimeError("Overlap detected between train and test sets")

        return train, test
