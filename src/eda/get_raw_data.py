import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
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

        print(f"📥 Baixando dataset: {dataset}")
        print(f"   Para: {path}")

        api.dataset_download_files(dataset, path=str(path), unzip=True)
        print(f"✓ Dataset '{dataset}' baixado e descompactado com sucesso!")

    except Exception as e:
        print(f"❌ Erro ao baixar dataset: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
