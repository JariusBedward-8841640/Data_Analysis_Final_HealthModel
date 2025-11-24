#metrics EDA summary like value coutns and missing diagonostics

import pandas as pd
import numpy as np

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="number").T
# Returns descrirptive numeric stats transposed for veiwingb

#Reports miswsing values per column in a dataframe
def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = miss / len(df) *100
    report = pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_count", ascending=False)
    return report

def categorical_counts(df: pd.DataFrame, cols: list) -> dict:
    #Return dict of series with value counts for each cateogrical column
    result = {}
    for cols in cols:
        if cols in df.columns:
            result[cols] = df[cols].value_counts(dropana=False)
    return result


def correlation_matrix(df: pd.DataFrame, method: str="pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    return numeric.corr(method=method)


def eda_pipeline(df: pd.DataFrame, categorical_cols: list = None) -> dict:

    #run the eda inspections and return dict containing different stas
    if categorical_cols is None:
        categorical_cols = ["Gender", "Activity Level", "Dietary Preference"]
    out = {}
    out["summary_stas"] = summary_stats(df)
    out["missing_report"] = missing_values(df)
    out["correlation_matrix"] = correlation_matrix(df)
    out["categorical_value_counts"] = categorical_counts(df, categorical_cols)
