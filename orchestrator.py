    # Connects everything
    # High-level pipeline orchetrator for the model
import pprint

from src.DataExtraction import load_raw_data
from src.DataPreperation.clean_data import clean_pipeline
from src.DataPreperation.feature_engineering import feature_pipeline
from src.Evaluation.metrics import eda_pipeline
from src.ModelTraining.train_model import split_save, detect_target_col


def main():
    #load raw set
    df_raw = load_raw_data()
    print(f"Raw shape: {df_raw.shape}")


    #clean data
    df_clean = clean_pipeline(df_raw, do_impute=True)
    print(f"Cleaned shape: {df_clean.shape}")

    #EDA (reports)
    eda_results = eda_pipeline(df_clean, categorical_cols=["Gender", "Activity Level", "Dietary Preference"])
    print("Top missing columns:")
    print(eda_results["missing_report"].head(10))

    #Feature engineering
    df_feature = feature_pipeline(df_clean)
    print(f"Feature shape: {df_feature.shape}")

    # Detect target col to decide stratify
    target_col = detect_target_col(df_feature, candidates=["HeartAttack", "Heart_Attack", "HasHeartAttack", "Target"])
    print("Detected target col:",  target_col)

    #Split datasets and save csv
    train_path, val_path, test_path = split_save(df=df_feature, target_col=target_col)
    print("Saved train/val/test to:", train_path, val_path, test_path)

    # pretty print small eda results for fast access
    pp = pprint.PrettyPrinter(indent=4)
    print("Cateogircal count: ")
    for k, val in eda_results["categorical_value_counts"].items():
        print(f"\n==={k} ===")
        print(val.head(10))

if __name__ == "__main__":
    main()