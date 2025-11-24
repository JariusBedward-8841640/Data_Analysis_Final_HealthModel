import pandas as pd
import numpy as np
from typing import Tuple




def encode_gender(df: pd.DataFrame, gender_col: str = "Gender") -> pd.DataFrame:
#Encodes gender to numeric male = 1 and female - 0 and any unknown is -1

    def norm_gender(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if s in ("m", "male", "man"):
            return "Male"
        if s in ("f", "female", "female"):
            return "Female"
        return "Other"

    if gender_col in df.columns:
        df[gender_col] = df[gender_col].apply(norm_gender)
        df["Gender_Male"] = (df[gender_col] == "Male").astype(int)
        df["Gender_Female"] = (df[gender_col] == "Female").astype(int)
    else:
        df["Gender_Male"] = 0
        df["Gender_Female"] = 0
    return df


def add_bmi(df: pd.DataFrame, weight_col: str = "Weight", height_col: str= "Height") -> pd.DataFrame:
    #Assumes weight in kg and height in cm it then adds "BMI" as float

    if weight_col in df.columns and height_col in df.columns:
        height_m = df[height_col] / 100.0
        df["BMI"] = df[weight_col] / (height_m **2)
    else:
        df["BMI"] = pd.NA
    return df


def add_obesity(df: pd.DataFrame, bmi_col: str ="BMI") -> pd.DataFrame:
#Male: BMI >= 30 -> Obese
#Female: BMI >- 28 > obese

    def obese_row(row)
        bmi = row.get(bmi_col, np.nan)
        gender = row.get("Gender", "Other")
        try:
            bmi = float(bmi)
        except Exception:
            return 0
        if gender =="Male":
            return int(bmi >=30)
        if gender =="Female":
            return int(bmi >= 28)
        return int(bmi >= 30) #default

    df["Obese"] = df.apply(obese_row, axis=1)
    return df


def  encode_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    #one hot encode listed cols and drop origngal cols and return df
    for cols in cols:
        if cols in df.columns:
            dummies = pd.get_dummies(df[cols].astype(str).str.strip(), prefix=cols.replace(" ", "_"), dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[cols])
        return df


    # do smoker and alchol inference


    # infer dibaetes by sugar

    #final pipeline feature engineer