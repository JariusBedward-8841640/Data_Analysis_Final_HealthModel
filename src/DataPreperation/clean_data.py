import pandas as pd
import numpy as np
from typing import Tuple


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    #Strip whitespace from the column nmames and string values in objects

    df = df.rename(columns=lambda c: c.strip() if isinstance(c,str) else c)
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA})
    return df


#Convert candiadte numeric columns to numeric, if numeric_candiatge is none then attempt sensible deafault is set
def type_fix(df: pd.DataFrame, numeric_candidates: list = None) -> pd.DataFrame:

    if numeric_candidates is None:
        numeric_candidates = ["Ages", "Height", "Weight", "Daily Calorie Target",
                              "Protein", "Sugar", "Sodium", "Calories", "Carbohydrates", "Fiber"]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def handle_basic_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # removes rows with impossible or rextreme values

    if "Height" in df.columns:
     df = df[(df["Height"].isna())  | ((df["Height"] >= 80) & (df["Height"]<=250()))]

    if "Weight" in df.columns:
     df = df[(df["Weight"].isna())  | ((df["Weight"] >= 30) & (df["Weight"]<=500()))]

    if "Ages" in df.columns:
     df = df[(df["Ages"].isna())  | ((df["Ages"] >= 5) & (df["Height"]<=120()))]

    return df

def drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
#drops entire empty rows anc cols
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    return df
#Drop duplicate rows
def drop_duplicate_row(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first") #(keeps first)

def simple_impution(df: pd.DataFrame) -> pd.DataFrame:
    # Impution for numeric columns fillna with the mdeidan and for cetefgorical columns fillna with "unkown"

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    for cols in num_cols:
        df[cols] = df[cols].fillna(df[cols].median())
    for cols in cat_cols:
        df[cols] = df[cols].fillna("Unknown")
    return df


#Now we run the entire cleaning pipline on df and return cleaned df
# It goes in order

def clean_pipeline(df: pd.DataFrame, do_impute: bool = True) -> pd.DataFrame:

    df = strip_whitespace(df)
    df = drop_empty_rows_cols(df)
    df = drop_duplicate_row(df)
    df = type_fix(df)
    df = handle_basic_outliers(df)
    if do_impute:
        df = simple_impution(df)

    #is the final pass which drops rows that are still entirley NaN after impution
    df = df.dropna(axis=0, how="all")
    return df