from pathlib import Path
from typing import Any, Union

import cloudpickle
import pandas as pd


class FileHelper:

    @staticmethod
    def save_to_pickle_file(obj: Any, path: Union[str, Path]) -> None:
        with open(path, mode="wb") as file:
            cloudpickle.dump(obj, file)

    @staticmethod
    def load_from_pickle_file(path: Union[str, Path]) -> Any:
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    @staticmethod
    def save_to_csv_file(df: pd.DataFrame, path: Union[str, Path]) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        df.to_csv(path, index=False)

    @staticmethod
    def load_from_csv_file(path: Union[str, Path], dtype=None) -> pd.DataFrame:
        return pd.read_csv(path, dtype=dtype)

    @staticmethod
    def export_to_excel(file_path: str, df_list: list, sheet_name_list: str):
        # === Sanity check ===
        if len(df_list) != len(sheet_name_list):
            raise ValueError("The number of DataFrames and sheet names must be equal")

        # == Export the DataFrames to an Excel file ==
        # Ensure the parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # Create a Pandas Excel writer using XlsxWriter as the engine
        with pd.ExcelWriter(file_path) as writer:
            for i, df in enumerate(df_list):
                df.to_excel(writer, sheet_name=sheet_name_list[i])
