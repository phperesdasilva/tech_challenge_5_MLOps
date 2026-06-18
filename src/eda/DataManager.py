import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from pandas import api


class DataManager:
    def __init__(self):
        pass

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
            path = project_root / "data" / "kaggle" / "raw" / dataset_name
            path.mkdir(parents=True, exist_ok=True)

            print(f"Colhendo informações do dataset {dataset}...")
            dataset_info = api.dataset_list(search=dataset)

            if not dataset_info:
                print(f"❌ Erro: Dataset '{dataset}' não encontrado no Kaggle!")
                sys.exit(1)

            dataset_info = dataset_info[2]
            print(f"✓ Dataset encontrado: {dataset_info.title}")
            print(f"  URL: {dataset_info.url}")

            print("\nBaixando metadados...")
            api.dataset_metadata(dataset, path=path)

            metadata_file = path / "dataset-metadata.json"
            if metadata_file.exists():
                print(f"✓ Metadados baixados com sucesso: {metadata_file}")
            else:
                print(f"❌ Erro: Metadados não encontrados após download!")
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

            print(f"Licença: {license_name}")

            print(f"📥 Baixando dataset: {dataset}")
            print(f"   Para: {path}")

            api.dataset_download_files(dataset, path=str(path), unzip=True)
            print(f"✓ Dataset '{dataset}' baixado e descompactado com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
