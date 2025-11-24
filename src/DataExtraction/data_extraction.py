#For if we take dataset from api or db
import pandas as pd
from pathlib import Path

raw_path = Path("data/raw/detailed_meals_macros_.csv")


def load_raw_data(path: Path = raw_path) -> pd.DataFrame:
    #load raw csv into dataframe

    try:
        df = pd.read_csv(path)

    except Exception:

        df = pd.read_csv(path, encoding="latin1")
    return df
