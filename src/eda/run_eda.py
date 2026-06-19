from eda.DataManager import DataManager


def main():
    print("Iniciando processo de EDA...\n")
    data_manager = DataManager()
    data_manager.get_raw_data()
    data_manager.clean_data()
    print("\n✅ EDA concluída com sucesso!")


if __name__ == "__main__":
    main()
