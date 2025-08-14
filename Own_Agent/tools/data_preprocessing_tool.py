from crewai.tools import BaseTool
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os


class DataPreprocessingTool(BaseTool):
    name: str = "DataPreprocessingTool"
    description: str = """
    Preprocesses only user-selected columns from a dataset.
    Capabilities:
    1. Loads CSV from given file path.
    2. Applies preprocessing ONLY to selected columns:
        - Fills missing numeric values using mean (if symmetrical) or median (if skewed).
        - Drops rows with excessive missing data if not reliably imputed.
        - Applies Label Encoding to binary categorical columns.
        - Applies One-Hot Encoding for multi-class categoricals (3–20 classes).
        - Drops high-cardinality categoricals (>20 classes).
    3. Preserves untouched columns.
    4. Combines processed + untouched columns into one dataset.
    5. Saves cleaned dataset as CSV to './data/csv_cleaned_data/<filename>_cleaned.csv'.
    """

    def _run(self, file_path: str, selected_columns: list) -> str:
        try:
            df = pd.read_csv(file_path)
            original_shape = df.shape
            summary = []

            invalid_columns = [col for col in selected_columns if col not in df.columns]
            if invalid_columns:
                return f"Invalid column(s): {invalid_columns}. Please select only valid columns."

            untouched_df = df.drop(columns=selected_columns)
            selected_df = df[selected_columns].copy()

            threshold = len(selected_df.columns) // 2
            before_drop = selected_df.shape[0]
            selected_df = selected_df[selected_df.isnull().sum(axis=1) <= threshold]
            after_drop = selected_df.shape[0]
            if before_drop != after_drop:
                summary.append(
                    f"Dropped {before_drop - after_drop} rows with excessive missing values."
                )

            num_cols = selected_df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            for col in num_cols:
                if selected_df[col].isnull().sum() > 0:
                    method = "median" if abs(selected_df[col].skew()) > 1 else "mean"
                    value = (
                        selected_df[col].median()
                        if method == "median"
                        else selected_df[col].mean()
                    )
                    selected_df[col].fillna(value, inplace=True)
                    summary.append(
                        f"Filled missing values in numeric column '{col}' using {method}."
                    )

            cat_cols = selected_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            for col in cat_cols:
                unique_vals = selected_df[col].nunique()
                if unique_vals == 2:
                    selected_df[col] = LabelEncoder().fit_transform(
                        selected_df[col].astype(str)
                    )
                    summary.append(f"Label encoded binary categorical column '{col}'.")
                elif 2 < unique_vals <= 20:
                    dummies = pd.get_dummies(selected_df[col], prefix=col, dtype=int)
                    selected_df.drop(columns=col, inplace=True)
                    selected_df = pd.concat([selected_df, dummies], axis=1)
                    summary.append(
                        f"One-hot encoded multi-class column '{col}' with {unique_vals} categories."
                    )
                else:
                    selected_df.drop(columns=col, inplace=True)
                    summary.append(
                        f"Dropped high-cardinality column '{col}' with {unique_vals} unique values."
                    )

            # dup_count = selected_df.duplicated().sum()
            # if dup_count > 0:
            #     selected_df.drop_duplicates(inplace=True)
            #     summary.append(f"Removed {dup_count} duplicate rows from processed data.")

            final_df = pd.concat(
                [
                    untouched_df.reset_index(drop=True),
                    selected_df.reset_index(drop=True),
                ],
                axis=1,
            )

            input_filename = Path(file_path).stem
            output_dir = Path("data/csv_cleaned_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            cleaned_path = output_dir / f"{input_filename}_cleaned.csv"

            if cleaned_path.exists():
                os.remove(cleaned_path)

            final_df.to_csv(cleaned_path, index=False)
            summary.append(f"✅ Cleaned dataset saved to: {cleaned_path}")
            summary.append(
                f"Original shape: {original_shape}, Final shape: {final_df.shape}"
            )

            return "\n".join(summary)

        except Exception as e:
            return f"Error during preprocessing: {str(e)}"
