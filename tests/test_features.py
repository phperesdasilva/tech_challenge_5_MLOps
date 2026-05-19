# tests/test_features.py
"""Testes de feature engineering — schema contracts."""
import pandera as pa
from pandera import Column, DataFrameSchema
from src.features.feature_engineering import compute_features
FEATURE_SCHEMA = DataFrameSchema({
"feature_1": Column(float, pa.Check.between(0, 1)),
"feature_2": Column(float, pa.Check.gt(0)),
"feature_1_x_feature_2": Column(float),
})
def test_schema_contract(sample_data):
    """Features de saída devem respeitar o contrato de schema."""
    result = compute_features(sample_data)
    FEATURE_SCHEMA.validate(result)
def test_no_nulls(sample_data):
    """Nenhuma feature pode ter null após transformação."""
    result = compute_features(sample_data)
    assert result.isnull().sum().sum() == 0
def test_row_count_preserved(sample_data):
    """Número de registros deve ser preservado."""
    result = compute_features(sample_data)
    assert len(result) == len(sample_data)
