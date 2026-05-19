# tests/conftest.py
"""Fixtures compartilhados para testes."""
import pandas as pd
import pytest

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Dados sintéticos para testes (nunca dados reais)."""
    return pd.DataFrame({
    "feature_1": [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4],
    "feature_2": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 3.5, 2.5],
    "feature_cat": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "target": [0, 1, 1, 0, 1, 0, 1, 0],
    })
