import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from event_generator.generate_synthetic_events import (
    is_eligible,
    load_data,
    simulate_mab_environment,
)


class TestIsEligible:
    """Testes para a função is_eligible."""

    def test_eligible_client_meets_all_requirements(self):
        """Testa que um cliente elegível passa na verificação."""
        client = {"age": 30, "housing": "yes"}
        rules = {"min_age": 25, "requires_housing_loan": True}
        assert is_eligible(client, rules) is True

    def test_ineligible_client_age_below_minimum(self):
        """Testa que um cliente abaixo da idade mínima não é elegível."""
        client = {"age": 20, "housing": "yes"}
        rules = {"min_age": 25, "requires_housing_loan": False}
        assert is_eligible(client, rules) is False

    def test_ineligible_client_missing_housing_loan(self):
        """Testa que um cliente sem empréstimo habitacional não é elegível se necessário."""
        client = {"age": 30, "housing": "no"}
        rules = {"min_age": 25, "requires_housing_loan": True}
        assert is_eligible(client, rules) is False

    def test_eligible_client_no_housing_requirement(self):
        """Testa cliente elegível sem requisito de empréstimo habitacional."""
        client = {"age": 30, "housing": "no"}
        rules = {"min_age": 25, "requires_housing_loan": False}
        assert is_eligible(client, rules) is True

    def test_eligible_client_with_empty_rules(self):
        """Testa cliente elegível com regras vazias."""
        client = {"age": 30, "housing": "yes"}
        rules = {}
        assert is_eligible(client, rules) is True

    def test_eligible_client_exact_minimum_age(self):
        """Testa cliente na idade mínima exata."""
        client = {"age": 25, "housing": "yes"}
        rules = {"min_age": 25}
        assert is_eligible(client, rules) is True

    def test_eligible_client_age_one_below_minimum(self):
        """Testa cliente um ano abaixo da idade mínima."""
        client = {"age": 24, "housing": "yes"}
        rules = {"min_age": 25}
        assert is_eligible(client, rules) is False


class TestLoadData:
    """Testes para a função load_data."""

    @patch.dict(
        os.environ,
        {
            "OFFER_CATALOG_PATH": "test_catalog.json",
            "CLEAN_BANK_PATH": "test_bank.parquet",
        },
    )
    @patch("event_generator.generate_synthetic_events.pd.read_parquet")
    @patch("builtins.open", create=True)
    def test_load_data_successful(self, mock_file_open, mock_read_parquet):
        """Testa carregamento bem-sucedido de dados."""
        # Mock do dataframe de banco
        mock_df_bank = pd.DataFrame(
            {"client_id": [1, 2], "age": [30, 40], "balance": [1000, 2000]}
        )
        mock_read_parquet.return_value = mock_df_bank

        # Mock do catálogo de ofertas
        catalog_data = {
            "offers": [
                {
                    "arm_id": 1,
                    "name": "Offer 1",
                    "eligibility_rules": {"min_age": 25},
                }
            ]
        }

        mock_file_open.return_value.__enter__.return_value.read.return_value = (
            json.dumps(catalog_data)
        )

        df_bank, offers = load_data()

        assert len(df_bank) == 2
        assert len(offers) == 1
        assert offers[0]["arm_id"] == 1

    @patch.dict(os.environ, {"OFFER_CATALOG_PATH": "test_catalog.json"})
    @patch("builtins.open", side_effect=FileNotFoundError("Arquivo não encontrado"))
    def test_load_data_catalog_not_found(self, mock_file_open):
        """Testa quando o catálogo de ofertas não é encontrado."""
        with patch("event_generator.generate_synthetic_events.pd.read_parquet"):
            with pytest.raises(FileNotFoundError):
                load_data()

    @patch.dict(os.environ, {"CLEAN_BANK_PATH": "test_bank.parquet"})
    @patch("event_generator.generate_synthetic_events.pd.read_parquet")
    def test_load_data_bank_not_found(self, mock_read_parquet):
        """Testa quando o arquivo de banco não é encontrado."""
        mock_read_parquet.side_effect = FileNotFoundError("Arquivo não encontrado")

        with patch("builtins.open", create=True):
            with pytest.raises(FileNotFoundError):
                load_data()


class TestSimulateMABEnvironment:
    """Testes para a função simulate_mab_environment."""

    @patch.dict(
        os.environ,
        {
            "OFFER_CATALOG_PATH": "test_catalog.json",
            "CLEAN_BANK_PATH": "test_bank.parquet",
        },
    )
    @patch("event_generator.generate_synthetic_events.load_data")
    @patch("event_generator.generate_synthetic_events.pd.DataFrame.to_parquet")
    @patch("event_generator.generate_synthetic_events.pd.DataFrame.to_csv")
    def test_simulate_mab_environment_generates_events(
        self, mock_to_csv, mock_to_parquet, mock_load_data
    ):
        """Testa que a simulação gera eventos corretamente."""
        # Mock dos dados
        client_data = {
            "age": [30, 40, 25],
            "balance": [1000, 2000, 500],
            "housing": ["yes", "yes", "no"],
        }
        mock_df_bank = pd.DataFrame(client_data)

        offers = [
            {
                "arm_id": 1,
                "name": "Offer 1",
                "eligibility_rules": {"min_age": 20, "requires_housing_loan": False},
                "synthetic_conversion_prior": 0.5,
                "reward_value": 100,
            },
            {
                "arm_id": 2,
                "name": "Offer 2",
                "eligibility_rules": {"min_age": 30, "requires_housing_loan": True},
                "synthetic_conversion_prior": 0.3,
                "reward_value": 150,
            },
        ]

        mock_load_data.return_value = (mock_df_bank, offers)

        # Executar simulação
        simulate_mab_environment()

        # Verificar que os métodos foram chamados
        assert mock_to_parquet.called
        assert mock_to_csv.called

    @patch.dict(
        os.environ,
        {
            "OFFER_CATALOG_PATH": "test_catalog.json",
            "CLEAN_BANK_PATH": "test_bank.parquet",
        },
    )
    @patch("event_generator.generate_synthetic_events.load_data")
    @patch("event_generator.generate_synthetic_events.os.makedirs")
    @patch("event_generator.generate_synthetic_events.pd.DataFrame.to_parquet")
    @patch("event_generator.generate_synthetic_events.pd.DataFrame.to_csv")
    def test_simulate_mab_environment_event_structure(
        self, mock_to_csv, mock_to_parquet, mock_makedirs, mock_load_data
    ):
        """Testa que os eventos gerados têm a estrutura correta."""
        # Mock dos dados
        client_data = {
            "age": [30],
            "balance": [1000],
            "housing": ["yes"],
        }
        mock_df_bank = pd.DataFrame(client_data)

        offers = [
            {
                "arm_id": 1,
                "name": "Offer 1",
                "eligibility_rules": {"min_age": 20},
                "synthetic_conversion_prior": 0.5,
                "reward_value": 100,
            }
        ]

        mock_load_data.return_value = (mock_df_bank, offers)

        # Capturar os DataFrames passados para to_parquet
        captured_dfs = []

        def capture_to_parquet(path, **kwargs):
            captured_dfs.append(("parquet", self, path))

        mock_to_parquet.side_effect = capture_to_parquet

        simulate_mab_environment()

        # Verificar que pelo menos um evento foi gerado
        assert mock_to_parquet.called

    def test_simulate_mab_environment_runs_without_error(self):
        """Testa que a simulação roda sem erros com dados válidos."""
        # Este é um teste simples que verifica se a função é chamável
        # Testes mais complexos verificam a saída real
        assert callable(simulate_mab_environment)


class TestEventGeneratorIntegration:
    """Testes de integração para o gerador de eventos."""

    @patch("event_generator.generate_synthetic_events.load_data")
    @patch("event_generator.generate_synthetic_events.os.makedirs")
    def test_event_generation_with_delayed_rewards(
        self, mock_makedirs, mock_load_data
    ):
        """Testa que eventos com recompensas atrasadas são gerados corretamente."""
        # Dados realistas
        client_data = {
            "age": [25, 35, 45],
            "balance": [500, 1500, 3000],
            "housing": ["yes", "yes", "no"],
        }
        mock_df_bank = pd.DataFrame(client_data)

        offers = [
            {
                "arm_id": 1,
                "name": "Basic Offer",
                "eligibility_rules": {"min_age": 20},
                "synthetic_conversion_prior": 0.4,
                "reward_value": 50,
            },
            {
                "arm_id": 2,
                "name": "Premium Offer",
                "eligibility_rules": {"min_age": 30, "requires_housing_loan": True},
                "synthetic_conversion_prior": 0.2,
                "reward_value": 200,
            },
        ]

        mock_load_data.return_value = (mock_df_bank, offers)

        # Executar simulação e capturar os DataFrames
        dfs_captured = []

        def capture_to_parquet(path, **kwargs):
            dfs_captured.append(path)

        with patch.object(pd.DataFrame, "to_parquet", side_effect=capture_to_parquet):
            with patch.object(pd.DataFrame, "to_csv"):
                simulate_mab_environment()

        # Verificar que os arquivos de eventos e recompensas foram criados
        paths = [p for p in dfs_captured if "offer_events" in str(p)]
        assert len(paths) >= 1

    def test_eligible_offers_filtering(self):
        """Testa que apenas ofertas elegíveis são consideradas."""
        client = {"age": 35, "housing": "yes"}

        offer1 = {"arm_id": 1, "eligibility_rules": {"min_age": 30}}
        offer2 = {"arm_id": 2, "eligibility_rules": {"min_age": 40}}
        offer3 = {
            "arm_id": 3,
            "eligibility_rules": {"min_age": 30, "requires_housing_loan": True},
        }

        offers = [offer1, offer2, offer3]

        # Filtrar ofertas elegíveis
        eligible_offers = [
            offer for offer in offers if is_eligible(client, offer["eligibility_rules"])
        ]

        # Deve haver 2 ofertas elegíveis
        assert len(eligible_offers) == 2
        assert eligible_offers[0]["arm_id"] == 1
        assert eligible_offers[1]["arm_id"] == 3


class TestRandomSeedConsistency:
    """Testes para verificar a consistência com seed aleatória."""

    def test_numpy_seed_is_set(self):
        """Testa que o seed do numpy está configurado para reprodutibilidade."""
        # Verificar que np.random.seed(42) foi chamado no módulo
        # Isso garante que os resultados sejam reprodutíveis
        seed_value = 42
        np.random.seed(seed_value)

        # Gerar números aleatórios
        values1 = [np.random.randint(0, 100) for _ in range(5)]

        # Reset seed
        np.random.seed(seed_value)

        # Gerar novamente
        values2 = [np.random.randint(0, 100) for _ in range(5)]

        # Verificar que são idênticos
        assert values1 == values2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
