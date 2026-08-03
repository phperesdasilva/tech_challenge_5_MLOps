import numpy as np
import pandas as pd
import pytest

from bandit.features import BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS, BankContextEncoder


def make_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 40, 60],
            "balance": [100, 200, 300],
            "pdays": [-1, 5, 10],
            "previous": [0, 1, 2],
            "housing": ["yes", "no", "yes"],
            "loan": ["no", "no", "yes"],
            "job": ["admin", "blue-collar", "admin"],
            "marital": ["single", "married", "married"],
            "education": ["primary", "secondary", "tertiary"],
            "poutcome": ["unknown", "success", "failure"],
        }
    )


def make_client(**overrides) -> dict:
    client = {
        "age": 30,
        "balance": 100,
        "pdays": -1,
        "previous": 0,
        "housing": "no",
        "loan": "no",
        "job": "admin",
        "marital": "single",
        "education": "primary",
        "poutcome": "unknown",
    }
    client.update(overrides)
    return client


def test_dim_sums_numeric_pdays_flag_binary_one_hot_and_bias():
    encoder = BankContextEncoder(make_train_df())

    expected_dim = len(NUMERIC_COLS) + 1 + len(BINARY_COLS) + 10 + 1  # 10 = soma dos vocabulários

    assert encoder.dim == expected_dim
    assert len(encoder.feature_names()) == encoder.dim


def test_encode_returns_1d_float64_vector_with_dim_length():
    encoder = BankContextEncoder(make_train_df())

    vec = encoder.encode(make_client())

    assert vec.shape == (encoder.dim,)
    assert vec.dtype == np.float64


def test_encode_applies_zscore_to_numeric_columns():
    df = make_train_df()
    encoder = BankContextEncoder(df)
    client = make_client(age=40, balance=200)

    vec = encoder.encode(client)

    expected_age_z = (40 - df["age"].mean()) / df["age"].std()
    expected_balance_z = (200 - df["balance"].mean()) / df["balance"].std()
    assert vec[0] == pytest.approx(expected_age_z)
    assert vec[1] == pytest.approx(expected_balance_z)


def test_encode_flags_pdays_was_contacted():
    encoder = BankContextEncoder(make_train_df())
    pdays_flag_index = len(NUMERIC_COLS)

    never_contacted = encoder.encode(make_client(pdays=-1))
    contacted = encoder.encode(make_client(pdays=5))

    assert never_contacted[pdays_flag_index] == 0.0
    assert contacted[pdays_flag_index] == 1.0


def test_encode_sets_binary_columns_only_on_yes():
    encoder = BankContextEncoder(make_train_df())
    binary_start = len(NUMERIC_COLS) + 1

    vec_yes = encoder.encode(make_client(housing="yes", loan="yes"))
    vec_no = encoder.encode(make_client(housing="no", loan="no"))

    assert vec_yes[binary_start] == 1.0
    assert vec_yes[binary_start + 1] == 1.0
    assert vec_no[binary_start] == 0.0
    assert vec_no[binary_start + 1] == 0.0


def test_encode_one_hot_activates_only_the_client_category():
    df = make_train_df()
    encoder = BankContextEncoder(df)
    job_vocab = sorted(df["job"].unique().tolist())
    job_start = len(NUMERIC_COLS) + 1 + len(BINARY_COLS)

    vec = encoder.encode(make_client(job="admin"))

    job_slice = vec[job_start : job_start + len(job_vocab)]
    assert job_slice.sum() == 1.0
    assert job_slice[job_vocab.index("admin")] == 1.0


def test_encode_unseen_category_yields_all_zero_one_hot_block():
    df = make_train_df()
    encoder = BankContextEncoder(df)
    job_vocab = sorted(df["job"].unique().tolist())
    job_start = len(NUMERIC_COLS) + 1 + len(BINARY_COLS)

    vec = encoder.encode(make_client(job="never-seen-job"))

    job_slice = vec[job_start : job_start + len(job_vocab)]
    assert job_slice.sum() == 0.0


def test_encode_appends_bias_term_as_last_position():
    encoder = BankContextEncoder(make_train_df())

    vec = encoder.encode(make_client())

    assert vec[-1] == 1.0


def test_feature_names_length_matches_categorical_vocab_sizes():
    df = make_train_df()
    encoder = BankContextEncoder(df)

    names = encoder.feature_names()

    for col in CATEGORICAL_COLS:
        vocab = sorted(df[col].unique().tolist())
        assert all(f"{col}_{cat}" in names for cat in vocab)
    assert names[-1] == "bias"
