import re
import pandas as pd


def clean_column_names(df: pd.DataFrame):
    """
    Replace [ or ] or < characters with "_" in column names
    This is useful to avoid problems with models such as xgboost
    """
    # Normalize tag names to avoid problems with xgboost
    # (it has problems with [ or ] or <)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in df.columns.values
    ]  # type: ignore
