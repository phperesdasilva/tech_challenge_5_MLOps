from eda.DataManager import DataManager


def main():
    print("🚀 Iniciando processo de EDA...")
    data_manager = DataManager()
    data_manager.get_raw_data()
    print("✅ EDA concluída com sucesso!")


if __name__ == "__main__":
    main()
