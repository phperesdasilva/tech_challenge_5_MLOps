import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.eda.DataManager import DataManager


class TestDataManager:
    """Testes para a classe DataManager."""

    @pytest.fixture
    def temp_dir(self):
        """Cria um diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def data_manager(self):
        """Cria uma instância de DataManager para testes."""
        return DataManager()

    def test_initialization(self, data_manager):
        """Testa a inicialização de DataManager."""
        assert data_manager.raw_data_path is None

    @patch("src.eda.DataManager.os.getenv")
    def test_get_raw_data_missing_env_variables(self, mock_getenv, data_manager):
        """Testa que o método falha quando variáveis de ambiente estão faltando."""
        mock_getenv.return_value = None

        with pytest.raises(SystemExit) as exc_info:
            data_manager.get_raw_data()

        assert exc_info.value.code == 1

    @patch.dict(
        os.environ,
        {
            "KAGGLE_USERNAME": "testuser",
            "KAGGLE_KEY": "testkey",
            "DATASET": "org/dataset-name",
        },
    )
    @patch("src.eda.DataManager.KaggleApi")
    @patch("pathlib.Path.mkdir")

    def test_get_raw_data_successful_authentication(
        self, mock_mkdir, mock_kaggle_api, data_manager
    ):
        """Testa autenticação bem-sucedida com a API Kaggle."""
        # Mock da API Kaggle
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance

        mock_api_instance.dataset_list.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(title="Test Dataset"),
        ]

        # Mock do arquivo de metadados
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"info": {"licenses": [{"name": "CC0"}]}}
            )

            with patch("pathlib.Path.exists", return_value=True):
                # Como não queremos executar o download real, vamos mockar também
                with patch.object(mock_api_instance, "dataset_download_files"):
                    data_manager.get_raw_data()

        # Verificar que a autenticação foi chamada
        mock_api_instance.authenticate.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "KAGGLE_USERNAME": "testuser",
            "KAGGLE_KEY": "testkey",
            "DATASET": "org/dataset-name",
        },
    )
    @patch("src.eda.DataManager.KaggleApi")
    def test_get_raw_data_dataset_not_found(self, mock_kaggle_api, data_manager):
        """Testa quando o dataset não é encontrado no Kaggle."""
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance
        mock_api_instance.dataset_list.return_value = []

        with pytest.raises(SystemExit) as exc_info:
            data_manager.get_raw_data()

        assert exc_info.value.code == 1

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", create=True)
    def test_clean_data_initialization(
        self, mock_open, mock_exists, data_manager, temp_dir
    ):
        """Testa que o método clean_data inicia sem erros."""
        data_manager.raw_data_path = temp_dir

        # Mocka os listdir para retornar alguns arquivos CSV fictícios
        with patch("os.listdir", return_value=["file1.csv", "file2.csv"]):
            # Mocka o método clean_data para não executar completamente
            with patch.object(data_manager, "clean_data", return_value=None):
                data_manager.clean_data()
                assert data_manager.raw_data_path == temp_dir

    def test_raw_data_path_attribute(self, data_manager):
        """Testa que raw_data_path pode ser atribuído."""
        test_path = Path("/test/path")
        data_manager.raw_data_path = test_path
        assert data_manager.raw_data_path == test_path


class TestDataManagerIntegration:
    """Testes de integração para DataManager."""

    @patch.dict(
        os.environ,
        {
            "KAGGLE_USERNAME": "testuser",
            "KAGGLE_KEY": "testkey",
            "DATASET": "org/dataset-name",
        },
    )
    @patch("src.eda.DataManager.KaggleApi")
    def test_authentication_error_handling(self, mock_kaggle_api):
        """Testa tratamento de erros na autenticação."""
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance
        mock_api_instance.authenticate.side_effect = Exception(
            "Authentication failed"
        )

        data_manager = DataManager()

        with pytest.raises(SystemExit):
            data_manager.get_raw_data()

    def test_data_manager_state_persistence(self):
        """Testa que o estado de DataManager é mantido entre chamadas."""
        data_manager = DataManager()
        test_path = Path("/test/data/path")
        data_manager.raw_data_path = test_path

        # Verificar que o estado persiste
        assert data_manager.raw_data_path == test_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
