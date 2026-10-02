"""
preprocess.py
Data loading, cleaning, encoding, and train/test splitting for the
XAI SME Threat Detection project.

Two datasets are supported:
  - IDS2025   (refined, class-rebalanced CICIDS2017 derivative; all-numeric features)
  - UNSW-NB15 (independent benchmark; three categorical feature columns)

Every function here mirrors the exact steps run during the study, so the
reported numbers are reproducible from a fresh checkout.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# IDS2025
# ----------------------------------------------------------------------
def load_ids2025(path="data/IDS2025.xlsx"):
    """Load IDS2025 from the Mendeley .xlsx release.

    Returns the raw DataFrame. Label column is 'newLabel'.
    Observed shape in the study: (91830, 80).
    """
    df = pd.read_excel(path)
    return df


def clean_ids2025(df):
    """Replace infinities with NaN and drop affected rows.

    In the study this removed a negligible number of rows
    (77 nulls + 177 infinities out of 91,830).
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def encode_and_split_ids2025(df, test_size=0.2, random_state=42):
    """Label-encode 'newLabel', build X/y, and produce a stratified split.

    random_state=42 reproduces the exact split used in the study
    (73,362 train / 18,341 test).

    Returns: X_train, X_test, y_train, y_test, label_encoder
    """
    le = LabelEncoder()
    df = df.copy()
    df["label_encoded"] = le.fit_transform(df["newLabel"])

    X = df.drop(columns=["newLabel", "label_encoded"])
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, le


# ----------------------------------------------------------------------
# UNSW-NB15
# ----------------------------------------------------------------------
CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "label", "attack_cat", "target"]


def load_unsw(train_path="data/UNSW_NB15_training-set.csv",
              test_path="data/UNSW_NB15_testing-set.csv"):
    """Load UNSW-NB15 pre-split partitions.

    Note: in the public release the file *contents* are swapped relative
    to their names (the larger 175,341-row partition is the training set).
    This function assigns train = the larger partition, matching the
    published convention and the study.

    Returns: train_df, test_df
    """
    a = pd.read_csv(train_path)
    b = pd.read_csv(test_path)
    train_df, test_df = (a, b) if len(a) > len(b) else (b, a)
    return train_df, test_df


def encode_unsw(train_df, test_df):
    """Encode the three categorical feature columns and the target.

    Categorical features (proto, service, state) are label-encoded on the
    union of train+test values so unseen categories do not error. The
    target column is 'attack_cat' (10 classes). Non-feature columns
    (id, label, attack_cat) are dropped from X.

    Returns: X_train, y_train, X_test, y_test, target_label_encoder
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in CATEGORICAL_COLS:
        enc = LabelEncoder()
        enc.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
        train_df[col] = enc.transform(train_df[col].astype(str))
        test_df[col] = enc.transform(test_df[col].astype(str))

    le_target = LabelEncoder()
    le_target.fit(pd.concat([train_df["attack_cat"], test_df["attack_cat"]]).astype(str))
    train_df["target"] = le_target.transform(train_df["attack_cat"].astype(str))
    test_df["target"] = le_target.transform(test_df["attack_cat"].astype(str))

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    return X_train, y_train, X_test, y_test, le_target
