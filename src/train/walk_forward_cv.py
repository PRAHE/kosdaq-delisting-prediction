"""
Walk-Forward Cross-Validation splitter.

기존 단일 split (train=2015~2022 / valid=2023 / test=2024)을
연도 cutoff로 확장하여 4개 fold를 만든다.

Fold | Train years        | Valid year
-----+--------------------+-----------
  1  | 2015 ~ 2019        | 2020
  2  | 2015 ~ 2020        | 2021
  3  | 2015 ~ 2021        | 2022
  4  | 2015 ~ 2022        | 2023   (현재 단일 split의 valid)

Test (2024)는 단 한 번만 평가되며, fold 분할에 포함되지 않는다.
2025년 train row는 forward-looking 라벨이 미관측 미래(2026)를 참조하므로
walk-forward 폴드에서 제외한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.research.s0_diagnostic.baseline import (
    HIGH_MISSING_COLS,
    META_COLUMNS,
    TARGET_COLUMN,
    load_splits,
    signed_log1p,
)

WALK_FORWARD_FOLDS = [
    (1, (2015, 2019), 2020),
    (2, (2015, 2020), 2021),
    (3, (2015, 2021), 2022),
    (4, (2015, 2022), 2023),
]


@dataclass
class Fold:
    fold_id: int
    train_year_range: tuple[int, int]
    valid_year: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    feature_cols: list[str]
    train_meta: pd.DataFrame
    valid_meta: pd.DataFrame


def _combine_panel() -> pd.DataFrame:
    """train + valid (+ test의 2024 행은 제외, 2025도 제외)를 long form으로 반환.

    Walk-forward 폴드 분할용 raw 패널.
    """
    splits = load_splits(n=1, variant="exp-A")
    panel = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    panel = panel[panel["year"].isin(range(2015, 2024))].reset_index(drop=True)
    return panel


def _select_features(panel: pd.DataFrame) -> list[str]:
    exclude = META_COLUMNS | {TARGET_COLUMN} | HIGH_MISSING_COLS
    return [c for c in panel.columns if c not in exclude]


def build_walk_forward_folds(apply_signed_log1p: bool = True) -> list[Fold]:
    """4개 walk-forward fold를 빌드한다.

    각 fold마다 train으로 fit한 SimpleImputer를 valid에 동일 적용한다.
    signed_log1p는 imputation 이후 적용된다 (exp_018 패턴 일치).
    """
    panel = _combine_panel()
    feature_cols = _select_features(panel)
    meta_cols = [c for c in META_COLUMNS if c in panel.columns]

    folds: list[Fold] = []
    for fold_id, (yr_lo, yr_hi), valid_year in WALK_FORWARD_FOLDS:
        train_mask = panel["year"].between(yr_lo, yr_hi)
        valid_mask = panel["year"] == valid_year

        tr = panel[train_mask].reset_index(drop=True)
        vl = panel[valid_mask].reset_index(drop=True)

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(tr[feature_cols])
        X_vl = imputer.transform(vl[feature_cols])

        if apply_signed_log1p:
            X_tr = signed_log1p(X_tr)
            X_vl = signed_log1p(X_vl)

        folds.append(
            Fold(
                fold_id=fold_id,
                train_year_range=(yr_lo, yr_hi),
                valid_year=valid_year,
                X_train=X_tr.astype(np.float32),
                y_train=tr[TARGET_COLUMN].values.astype(int),
                X_valid=X_vl.astype(np.float32),
                y_valid=vl[TARGET_COLUMN].values.astype(int),
                feature_cols=feature_cols,
                train_meta=tr[meta_cols].reset_index(drop=True),
                valid_meta=vl[meta_cols].reset_index(drop=True),
            )
        )

    return folds


def summarize_folds(folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        rows.append({
            "fold": f.fold_id,
            "train_years": f"{f.train_year_range[0]}~{f.train_year_range[1]}",
            "valid_year": f.valid_year,
            "train_rows": len(f.y_train),
            "train_pos": int(f.y_train.sum()),
            "valid_rows": len(f.y_valid),
            "valid_pos": int(f.y_valid.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    folds = build_walk_forward_folds()
    print(summarize_folds(folds).to_string(index=False))
