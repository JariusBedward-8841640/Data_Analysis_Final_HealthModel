# Split dataset into train val and test



import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional


processed_directory = Path("data/processed")

def detect_target_col(df: pd.DataFrame, candidates: list = None) -> Optional[str]:
    """
    Attempt to find a target column to stratify on. Returns column name or None.
    Common candidate names included.
    """
    if candidates is None:
        candidates = ["HeartAttack", "Heart_Attack", "HeartAttackFlag", "HasHeartAttack", "Target"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

#splits df into train/val/test and save the csvs into respective direcotry
#if target_col is present, stratify splits on it for classification and returns paths to the saved csvs
def split_save(df: pd.DataFrame, target_col: str =None, train_size: float =0.3, val_size: float = 0.15, test_size: float= 0.15, random_state: int = 42 ) -> Tuple[Path, Path, Path]:

    assert abs(train_size + val_size + test_size -1.0) < 1e-6 #spltits must  add to 1.0
    stratify_col = df[target_col] if target_col and target_col in df.columns else None

    #First split train
    df_train, df_remain = train_test_split(df, train_size=train_size, test_size=test_size, random_state=random_state, stratify=stratify_col)
    #Reaming use compute relative sizes
    if stratify_col is not None:
        stratify_rem = df_remain[target_col]
    else:
        stratify_rem = None
    val_rel = val_size / (val_size +test_size)
    df_val, df_test = train_test_split(df_remain, test_size=val_rel, random_state=random_state, statify=stratify_rem)


    #we must make sure direcotieies exist
    train_dir = processed_directory / "train"
    val_dir = processed_directory / "val"
    test_dir = processed_directory / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_path = train_dir / "train.csv"
    val_path = val_dir / "val.csv"
    test_path = test_dir / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    return train_path, val_path, test_path