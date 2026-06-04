"""Temporal data splitting with 4-way split for CQR calibration."""

import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSplit:
    """Container for a 4-way temporal data split."""
    train: pd.DataFrame
    calibration: pd.DataFrame  # For CQR conformity scores
    validation: pd.DataFrame   # For early stopping
    test: pd.DataFrame         # Touched ONCE at final evaluation

    def __repr__(self) -> str:
        return (
            f"DataSplit(\n"
            f"  train:       {len(self.train):>7} rows  "
            f"[{self.train['timestamps'].iloc[0].date()} → {self.train['timestamps'].iloc[-1].date()}]\n"
            f"  calibration: {len(self.calibration):>7} rows  "
            f"[{self.calibration['timestamps'].iloc[0].date()} → {self.calibration['timestamps'].iloc[-1].date()}]\n"
            f"  validation:  {len(self.validation):>7} rows  "
            f"[{self.validation['timestamps'].iloc[0].date()} → {self.validation['timestamps'].iloc[-1].date()}]\n"
            f"  test:        {len(self.test):>7} rows  "
            f"[{self.test['timestamps'].iloc[0].date()} → {self.test['timestamps'].iloc[-1].date()}]\n"
            f")"
        )


def temporal_split(
    df: pd.DataFrame,
    test_months: int = 3,
    val_months: int = 3,
    cal_months: int = 3,
) -> DataSplit:
    """
    Split time series data into train/calibration/validation/test sets.

    Split is purely temporal (no shuffling) to prevent look-ahead bias.

    Timeline:
    |<── Train ──>|<── Cal ──>|<── Val ──>|<── Test ──>|
    | everything  | months    | months    | last N     |
    | before cal  | -9 to -6  | -6 to -3  | months     |

    Args:
        df: DataFrame with 'timestamps' column, sorted by time
        test_months: Months to reserve for test (touched once)
        val_months: Months for validation (early stopping)
        cal_months: Months for CQR calibration

    Returns:
        DataSplit with four non-overlapping temporal segments
    """
    df = df.sort_values("timestamps").reset_index(drop=True)
    end_date = df["timestamps"].max()

    # Compute split boundaries working backwards from end
    test_start = end_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)
    cal_start = val_start - pd.DateOffset(months=cal_months)

    # Split
    train = df[df["timestamps"] < cal_start].copy()
    calibration = df[(df["timestamps"] >= cal_start) & (df["timestamps"] < val_start)].copy()
    validation = df[(df["timestamps"] >= val_start) & (df["timestamps"] < test_start)].copy()
    test = df[df["timestamps"] >= test_start].copy()

    split = DataSplit(
        train=train.reset_index(drop=True),
        calibration=calibration.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )

    print(f"Temporal split:")
    print(split)

    # Warn if any split is too small
    for name, subset in [("train", train), ("calibration", calibration),
                          ("validation", validation), ("test", test)]:
        if len(subset) < 100:
            print(f"  WARNING: {name} has only {len(subset)} rows — may be too small!")

    return split


def save_split(split: DataSplit, output_dir: Path) -> None:
    """Save all split DataFrames as CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train", "calibration", "validation", "test"]:
        path = output_dir / f"{name}.csv"
        getattr(split, name).to_csv(path, index=False)
        print(f"Saved {name} → {path}")


def load_split(input_dir: Path) -> DataSplit:
    """Load a saved split from CSVs."""
    return DataSplit(
        train=pd.read_csv(input_dir / "train.csv", parse_dates=["timestamps"]),
        calibration=pd.read_csv(input_dir / "calibration.csv", parse_dates=["timestamps"]),
        validation=pd.read_csv(input_dir / "validation.csv", parse_dates=["timestamps"]),
        test=pd.read_csv(input_dir / "test.csv", parse_dates=["timestamps"]),
    )
