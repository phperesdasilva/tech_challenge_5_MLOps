import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


class DataManager:
    def __init__(self):
        self.raw_data_path = None

    def get_raw_data(self):
        try:
            # Load environment variables
            load_dotenv()

            kaggle_user = os.getenv("KAGGLE_USERNAME")
            kaggle_key = os.getenv("KAGGLE_KEY")
            dataset = os.getenv("DATASET")

            # Validate required environment variables
            if not kaggle_user or not kaggle_key or not dataset:
                print("❌ Erro: Variáveis de ambiente não configuradas!")
                print(f"   KAGGLE_USERNAME: {'✓' if kaggle_user else '✗'}")
                print(f"   KAGGLE_KEY: {'✓' if kaggle_key else '✗'}")
                print(f"   DATASET: {'✓' if dataset else '✗'}")
                sys.exit(1)

            os.environ["KAGGLE_USERNAME"] = kaggle_user
            os.environ["KAGGLE_KEY"] = kaggle_key

            # Authenticate with Kaggle API
            api = KaggleApi()
            api.authenticate()
            print("✓ Autenticação Kaggle bem-sucedida")

            # Download dataset - use absolute path based on script location
            script_dir = Path(__file__).parent  # src/eda
            project_root = script_dir.parent.parent
            dataset_name = dataset.split("/")[1]
            self.raw_data_path = project_root / "data" / "kaggle" / "raw" / dataset_name
            self.raw_data_path.mkdir(parents=True, exist_ok=True)

            print(f"Colhendo informações do dataset {dataset}...")
            dataset_info = api.dataset_list(search=dataset)

            if not dataset_info:
                print(f"❌ Erro: Dataset '{dataset}' não encontrado no Kaggle!")
                sys.exit(1)

            dataset_info = dataset_info[2]
            print(f"✓ Dataset encontrado: {dataset_info.title}")

            print("\nBaixando metadados...")
            api.dataset_metadata(dataset, path=self.raw_data_path)

            metadata_file = self.raw_data_path / "dataset-metadata.json"
            if metadata_file.exists():
                print(f"✓ Metadados baixados com sucesso.")
            else:
                print(f"❌ Erro: Metadados não encontrados após download.")
                sys.exit(1)

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_raw = json.load(f)

            metadata = metadata_raw.get("info")

            # A licença fica dentro de uma lista de dicionários
            licenses = metadata.get("licenses", [{}])
            license_name = (
                licenses[0].get("name", "Não informada")
                if licenses
                else "Não informada"
            )

            print(f"Licença: {license_name}\n")

            print(f"📥 Baixando dataset: {dataset}")

            api.dataset_download_files(
                dataset, path=str(self.raw_data_path), unzip=True
            )
            print(f"✓ Dataset baixado e descompactado com sucesso.")

        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def clean_data(self):

        print("\nLimpando dados...\n")
        csvs = os.listdir(self.raw_data_path)

        print(f"Arquivos encontrados:")
        for file in csvs:
            print(f" - {file}")

        csv_file = None

        while not csv_file:
            csv_file = str(
                input("\nDigite o nome do arquivo CSV a ser processado: ")
            ).strip()
            if csv_file not in csvs:
                print(f"❌ Arquivo '{csv_file}' não encontrado. Tente novamente.")
                csv_file = None

        df = pd.read_csv(f"{self.raw_data_path}/{csv_file}", sep=";")
        df = df.dropna()

        clean_df = df[
            [
                "age",
                "job",
                "marital",
                "education",
                "balance",
                "housing",
                "loan",
                "pdays",
                "previous",
                "poutcome",
            ]
        ]

        clean_dataset_path = str(os.getenv("CLEAN_DATASET_PATH"))

        os.makedirs(clean_dataset_path, exist_ok=True)
        clean_df.to_parquet(f"{clean_dataset_path}/clean_bank.parquet", index=False)

        removed_info = df.columns.difference(clean_df.columns)
        print(f"✓ Dados sensíveis removidos: {', '.join(removed_info)}")
